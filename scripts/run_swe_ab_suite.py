"""Run or validate a declared set of SWE A/B cells (spec §5, §8).

```bash
# Campaign cells on operator subscriptions (see benchmarks/swe_ab/RUNBOOK.md). The auth
# broker runs on this host over omp's own agent.db; containers reach it, and nothing else
# holds a login:
omp auth-broker serve --bind 127.0.0.1:8765 &
uv run python scripts/run_swe_ab_suite.py run --plan stage1.json --runtime docker \
    --output benchmarks/swe_ab/results/stage1 --work-root /scratch/swe-ab \
    --tasks-root SWE-bench_Pro-os/v2/tasks --omp-dir /opt/omp-linux-x64 \
    --omp-command "/opt/omp/bin/omp" --profile-dir ~/.omp/profiles/swebench/agent \
    --archex-wheel dist/archex-0.33.0-py3-none-any.whl --uv-binary /opt/uv/uv \
    --broker-url http://127.0.0.1:8765 --jobs 4

# No-spend rehearsal against the local stub provider:
uv run python scripts/run_swe_ab_suite.py run --plan dry-run.json --runtime local \
    --output /tmp/swe-ab/cells --work-root /tmp/swe-ab/work \
    --stub-script benchmarks/swe_ab/stub-script.json

uv run python scripts/run_swe_ab_suite.py validate --plan stage1.json \
    --input benchmarks/swe_ab/results/stage1
```

Resumable: a cell whose artifact exists is skipped and its recorded cost counts toward
the ceiling. The hard cumulative cost ceiling (the plan's, or a lower ``--cost-ceiling``)
is checked before every cell and aborts the run when reached; cost is omp's list-price
model of the tokens, a runaway guard rather than a bill. A cell whose runner produces no
artifact is recorded as a harness failure; nothing is dropped.

Subscription quota (docker runtime): before each cell attempt the suite reads the broker's
``/v1/usage`` and pauses while the cell's provider has less than ``--quota-min-headroom``
left, resuming when the reported window resets. A cell that ends in a quota block (before
its first tool call, or mid-run) is not a task outcome: its artifact is filed under
``<output>/quota-blocked/`` with its cost still counted, and the cell is re-run once the
quota clears, up to ``--quota-retries`` times per invocation. A pause longer than
``--quota-max-wait-seconds`` stops the run with exit status 4; resuming continues it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from archex.benchmark.swe_ab import (
    BROKER_ENV_NAMES,
    CLI_GUIDE_PATH,
    QUOTA_BLOCKED_DIR,
    QUOTA_MIN_HEADROOM,
    CellKey,
    FailureReason,
    PlanTask,
    QuotaReading,
    QuotaStatus,
    SweAbError,
    SweAbPlan,
    credential_files_in_profile,
    is_emulated,
    load_cell,
    load_plan,
    quota_blocked_relative_path,
    quota_reading,
    validate_swe_ab_directory,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swe_ab_cell as cell_runner  # noqa: E402 - sibling script, importable only via sys.path

_CELL_SCRIPT = Path(__file__).resolve().parent / "run_swe_ab_cell.py"
_STUB_SCRIPT = Path(__file__).resolve().parent / "swe_ab_stub_provider.py"
_CELL_TIMEOUT_SECONDS = 4 * 3600
QUOTA_EXIT = 4
"""Exit status when the quota guard paused longer than its wait budget; the run is resumable."""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _emit(event: Mapping[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def _start_stub(work_root: Path, script: Path) -> tuple[subprocess.Popen[str], Path, Path]:
    """Start the stub provider; return it, a profile pointing at it, and its capture dir."""
    port = _free_port()
    capture = work_root / "capture"
    process = subprocess.Popen(
        [
            sys.executable,
            str(_STUB_SCRIPT),
            "--port",
            str(port),
            "--script",
            str(script),
            "--capture-dir",
            str(capture),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    process.stdout.readline()  # "stub listening on ..."
    profile = work_root / "stub-profile"
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "models.yml").write_text(
        "providers:\n"
        "  stub:\n"
        f"    baseUrl: http://127.0.0.1:{port}/v1\n"
        "    api: openai-completions\n"
        "    auth: none\n"
        "    models:\n"
        "      - id: stub-model\n"
        "        name: Stub Model\n"
        "        contextWindow: 200000\n"
        "        maxTokens: 4096\n",
        encoding="utf-8",
    )
    return process, profile, capture


# --- the auth broker and the subscription quota guard ------------------------------------


class BrokerError(RuntimeError):
    """The auth broker could not be read (unreachable, refused, or malformed)."""


class QuotaPauseError(RuntimeError):
    """The subscription quota did not clear within the wait budget; the run is resumable."""


def _broker_request(url: str, token: str, path: str, *, method: str = "GET") -> dict[str, Any]:
    """One broker call. Errors name the endpoint and status, never the bearer token."""
    request = urllib.request.Request(  # noqa: S310 - scheme is checked by the caller's flags
        url.rstrip("/") + path, method=method, headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            body: object = json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raise BrokerError(f"broker {method} {path} answered HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise BrokerError(f"broker {method} {path} failed: {type(exc).__name__}") from None
    if not isinstance(body, dict):
        raise BrokerError(f"broker {method} {path} did not answer a JSON object")
    return cast("dict[str, Any]", body)


def fetch_usage(url: str, token: str) -> dict[str, Any]:
    """The broker's aggregate ``UsageReport[]`` (``GET /v1/usage``); makes no model request."""
    return _broker_request(url, token, "/v1/usage")


def invalidate_usage(url: str, token: str) -> None:
    """Drop the broker's cached usage (``POST /v1/usage/stale``) so the next read is fresh."""
    _broker_request(url, token, "/v1/usage/stale", method="POST")


def provider_of(model: str) -> str:
    """The omp provider of a ``provider/id`` selector."""
    return model.split("/", 1)[0]


@dataclass
class QuotaGuard:
    """Pauses the suite while a provider's subscription has too little headroom."""

    broker_url: str
    token: str = field(repr=False)
    min_headroom: float = QUOTA_MIN_HEADROOM
    poll_seconds: float = 60.0
    max_wait_seconds: float = 6 * 3600.0
    min_pause_seconds: float = 300.0
    fetch: Callable[[str, str], dict[str, Any]] = fetch_usage
    invalidate_cache: Callable[[str, str], None] = invalidate_usage
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time
    log: Callable[[Mapping[str, Any]], None] = _emit
    _unknown_logged: set[str] = field(default_factory=set[str])

    def reading(self, model: str) -> QuotaReading:
        usage = self.fetch(self.broker_url, self.token)
        return quota_reading(
            usage,
            provider_of(model),
            model_id=model.split("/", 1)[-1],
            min_headroom=self.min_headroom,
        )

    def invalidate(self) -> None:
        """Refresh the broker's view after a block; a broker that cannot is polled anyway."""
        with contextlib.suppress(BrokerError):
            self.invalidate_cache(self.broker_url, self.token)

    def wait(self, model: str, *, pause_first: float = 0.0) -> bool:
        """Block until ``model``'s provider has headroom; ``False`` if the wait budget ran out.

        ``pause_first`` sleeps before the first read: omp's own block for the credential can
        outlast what the usage report shows, so a re-run after a block waits a minimum first.
        """
        deadline = self.clock() + self.max_wait_seconds
        waited = pause_first > 0
        if waited:
            self.log({"quota": "cooling_down", "model": model, "seconds": pause_first})
            self.sleep(min(pause_first, self.max_wait_seconds))
        while True:
            reading: QuotaReading | None = None
            detail = ""
            try:
                reading = self.reading(model)
            except BrokerError as exc:
                detail = str(exc)
            if reading is not None and reading.status is not QuotaStatus.BLOCKED:
                self._note_clear(model, reading, waited)
                return True
            remaining = deadline - self.clock()
            if remaining <= 0:
                self.log({"quota": "wait_budget_exhausted", "model": model, "detail": detail})
                return False
            pause = self.poll_seconds
            if reading is not None and reading.resets_at_ms is not None:
                until_reset = reading.resets_at_ms / 1000.0 - self.clock() + 5.0
                pause = min(pause, max(until_reset, 1.0))
            self.log(
                {
                    "quota": "paused" if reading is not None else "broker_unreachable",
                    "model": model,
                    "detail": reading.detail if reading is not None else detail,
                    "resume_check_in_seconds": round(min(pause, remaining), 1),
                }
            )
            self.sleep(min(pause, remaining))
            waited = True

    def _note_clear(self, model: str, reading: QuotaReading, waited: bool) -> None:
        if reading.status is QuotaStatus.UNKNOWN:
            # The broker reports nothing usable for this provider (no usage endpoint, or a
            # login without limit data): proceed, and say so once rather than on every cell.
            provider = provider_of(model)
            if provider not in self._unknown_logged:
                self._unknown_logged.add(provider)
                self.log({"quota": "unknown", "provider": provider, "detail": reading.detail})
        elif waited:
            self.log({"quota": "cleared", "model": model, "detail": reading.detail})


def default_broker_token_file() -> Path:
    """``<config-dir>/auth-broker.token``; the config dir is ``~/.omp`` or ``$PI_CONFIG_DIR``."""
    return Path.home() / (os.environ.get("PI_CONFIG_DIR") or ".omp") / "auth-broker.token"


def read_broker_token(token_file: Path | None) -> str:
    """The broker bearer token, from ``$OMP_AUTH_BROKER_TOKEN`` or the token file; never printed."""
    if token := os.environ.get("OMP_AUTH_BROKER_TOKEN", "").strip():
        return token
    path = token_file or default_broker_token_file()
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SystemExit(f"cannot read the broker token from {path}: {exc.strerror}") from None


def container_broker_url(broker_url: str, override: str | None) -> str:
    """Where an agent container reaches the host's broker: ``host.docker.internal`` + its port."""
    if override:
        return override
    port = urllib.parse.urlparse(broker_url).port or 8765
    return f"http://host.docker.internal:{port}"


def cell_environment(container_url: str, token: str) -> dict[str, str]:
    """The cell runner's environment: this one's, minus stray broker variables, plus the two."""
    env = {k: v for k, v in os.environ.items() if k not in BROKER_ENV_NAMES}
    env[BROKER_ENV_NAMES[0]] = container_url
    env[BROKER_ENV_NAMES[1]] = token
    return env


# --- cells -------------------------------------------------------------------------------


def _spec(
    args: argparse.Namespace,
    plan: SweAbPlan,
    key: CellKey,
    *,
    profile: Path,
    capture: Path | None,
    prior_blocked_attempts: int = 0,
) -> dict[str, Any]:
    task = next(t for t in plan.tasks if t.task_id == key.task_id)
    cell_dir = f"{key.task_id}__{key.model.replace('/', '_')}__{key.arm}__rep{key.repetition}"
    spec: dict[str, Any] = {
        "task_id": key.task_id,
        "repo": task.repo,
        "model": key.model,
        "arm": key.arm.value,
        "repetition": key.repetition,
        "runtime": args.runtime,
        "output": str(args.output / key.relative_path),
        "work_dir": str(args.work_root / "cells" / cell_dir),
        "omp_command": shlex.split(args.omp_command),
        "profile_dir": str(profile),
        "cli_guide": str(args.cli_guide),
        "capture_dir": str(capture) if capture else None,
        "network": args.network,
        "prior_blocked_attempts": prior_blocked_attempts,
    }
    task_dir = (
        Path(task.task_dir)
        if task.task_dir
        else (args.tasks_root / key.task_id if args.tasks_root else None)
    )
    if task_dir is not None:
        spec["task_dir"] = str(task_dir)
        gold = task_dir / "solution" / "gold_patch.diff"
        if gold.exists():
            spec["gold_patch"] = str(gold)
    if args.runtime == "docker":
        spec.update(
            image=_task_image(task),
            omp_dir=str(args.omp_dir),
            archex_wheel=str(args.archex_wheel),
            uv_binary=str(args.uv_binary),
            broker_url=args.container_broker_url,
        )
    else:
        spec.update(
            local_repo=task.local_repo,
            prompt_file=task.prompt_file,
            archex_python=args.archex_python,
        )
    return spec


def _invoke_cell(spec: dict[str, Any], env: Mapping[str, str] | None) -> None:
    """Run the cell runner as a subprocess and guarantee an artifact exists afterwards."""
    output = Path(spec["output"])
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, str(_CELL_SCRIPT)],
            input=json.dumps(spec),
            text=True,
            capture_output=True,
            timeout=_CELL_TIMEOUT_SECONDS,
            check=False,
            env=dict(env) if env is not None else None,
        )
    if not output.exists():
        cell = cell_runner.failed_cell(
            cell_runner.CellSpec(spec),
            FailureReason.HARNESS_ERROR,
            "cell runner produced no artifact",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(cell.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _blocked_attempts(output_root: Path, key: CellKey) -> list[Path]:
    """Attempts of ``key`` already filed under ``quota-blocked/``."""
    first = output_root / quota_blocked_relative_path(key, 1)
    return sorted(first.parent.glob(first.name.replace(".attempt1.json", ".attempt*.json")))


def _file_quota_blocked(output_root: Path, key: CellKey) -> int:
    """Move ``key``'s artifact to the next ``quota-blocked/`` slot; return that attempt number."""
    attempt = len(_blocked_attempts(output_root, key)) + 1
    target = output_root / quota_blocked_relative_path(key, attempt)
    target.parent.mkdir(parents=True, exist_ok=True)
    (output_root / key.relative_path).replace(target)
    return attempt


def _run_one(
    spec: dict[str, Any],
    key: CellKey,
    output_root: Path,
    *,
    guard: QuotaGuard | None,
    env: Mapping[str, str] | None,
    quota_retries: int,
    invoke: Callable[[dict[str, Any], Mapping[str, str] | None], None] = _invoke_cell,
    log: Callable[[Mapping[str, Any]], None] = _emit,
) -> float:
    """Run one cell to a real outcome; return the recorded cost of every attempt.

    With a quota guard the provider's headroom is checked before each attempt. An attempt that
    ends in a quota block is filed under ``quota-blocked/`` and re-run after the quota clears
    (up to ``quota_retries`` times); past that budget the blocked artifact stays as the cell's
    record, which the validator refuses for publication.
    """
    output = Path(spec["output"])
    total = 0.0
    blocks = 0
    pause_first = 0.0
    while True:
        if guard is not None and not guard.wait(key.model, pause_first=pause_first):
            raise QuotaPauseError(f"{key.model}: subscription quota did not clear in time")
        invoke(spec, env)
        cell = load_cell(output)
        total += cell.usage.cost_usd
        blocked = cell.failure_reason is FailureReason.QUOTA_BLOCK
        log(
            {
                "cell": str(output),
                "status": cell.status,
                "reason": cell.failure_reason,
                "cost_usd": cell.usage.cost_usd,
                **({"quota_phase": cell.quota.block_phase} if blocked else {}),
            }
        )
        if not blocked or guard is None:
            return total
        blocks += 1
        if blocks > quota_retries:
            log({"quota": "retry_budget_exhausted", "cell": str(output), "attempts": blocks})
            return total
        attempt = _file_quota_blocked(output_root, key)
        spec["prior_blocked_attempts"] = attempt
        log({"quota": "cell_blocked_will_rerun", "cell": str(output), "attempt": attempt})
        guard.invalidate()
        pause_first = guard.min_pause_seconds


def _spent_on_blocked(output_root: Path) -> float:
    root = output_root / QUOTA_BLOCKED_DIR
    return (
        sum(load_cell(path).usage.cost_usd for path in root.rglob("*.json"))
        if root.is_dir()
        else 0.0
    )


def _settle(future: Future[float], errors: list[QuotaPauseError]) -> float:
    try:
        return future.result()
    except QuotaPauseError as exc:
        errors.append(exc)
        return 0.0


def _task_image(task: PlanTask) -> str:
    return task.image or f"ghcr.io/scaleapi/swe-bench_pro-v2:{task.task_id}"


def _prune_image(image: str, log: Callable[[Mapping[str, Any]], None] = _emit) -> None:
    """Remove a task image once every cell of its task has run (Pro images are multi-GB)."""
    done = subprocess.run(
        ["docker", "image", "rm", image], capture_output=True, text=True, check=False
    )
    log({"pruned_image": image, "removed": done.returncode == 0})


def run(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    ceiling = min(plan.cost_ceiling_usd, args.cost_ceiling or plan.cost_ceiling_usd)
    args.work_root.mkdir(parents=True, exist_ok=True)
    stub: subprocess.Popen[str] | None = None
    capture: Path | None = None
    profile = args.profile_dir
    if args.stub_script is not None:
        if args.runtime != "local" or args.jobs != 1:
            raise SystemExit("the stub provider runs only with --runtime local --jobs 1")
        stub, profile, capture = _start_stub(args.work_root, args.stub_script)
    if profile is None:
        raise SystemExit("--profile-dir is required unless --stub-script is given")
    guard: QuotaGuard | None = None
    env: dict[str, str] | None = None
    args.container_broker_url = None
    if args.runtime == "docker":
        if not args.broker_url:
            raise SystemExit("--broker-url is required with --runtime docker (subscription logins)")
        if leaked := credential_files_in_profile(profile):
            raise SystemExit(
                f"the profile carries credential stores {leaked}: logins reach containers only "
                "through the auth broker, so remove them (RUNBOOK §2)"
            )
        token = read_broker_token(args.broker_token_file)
        args.container_broker_url = container_broker_url(
            args.broker_url, args.container_broker_url_flag
        )
        env = cell_environment(args.container_broker_url, token)
        guard = QuotaGuard(
            broker_url=args.broker_url,
            token=token,
            min_headroom=args.quota_min_headroom,
            poll_seconds=args.quota_poll_seconds,
            max_wait_seconds=args.quota_max_wait_seconds,
        )
        _emit(
            {
                "broker": args.broker_url,
                "container_broker_url": args.container_broker_url,
                "agent_env_names": list(BROKER_ENV_NAMES),
                "emulated": is_emulated("docker", platform.machine()),
            }
        )
    spent = _spent_on_blocked(args.output)
    largest = 0.0
    pending: list[CellKey] = []
    for key in plan.cells():
        path = args.output / key.relative_path
        if path.exists() and load_cell(path).failure_reason is FailureReason.QUOTA_BLOCK:
            # A previous invocation ran out of quota retries: file the block, re-run the cell.
            _file_quota_blocked(args.output, key)
        if path.exists():
            cost = load_cell(path).usage.cost_usd
            spent += cost
            largest = max(largest, cost)
        else:
            pending.append(key)
    _emit({"declared": len(plan.cells()), "pending": len(pending), "spent_usd": spent})
    pauses: list[QuotaPauseError] = []
    remaining = Counter(key.task_id for key in pending)
    future_keys: dict[Future[float], CellKey] = {}

    def settle(future: Future[float]) -> float:
        key = future_keys.pop(future)
        completed = future.exception() is None
        cost = _settle(future, pauses)
        if completed:
            remaining[key.task_id] -= 1
            if args.prune_images and args.runtime == "docker" and remaining[key.task_id] == 0:
                _prune_image(_task_image(next(t for t in plan.tasks if t.task_id == key.task_id)))
        return cost

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            running: set[Future[float]] = set()
            for key in pending:
                while len(running) >= args.jobs:
                    finished, running = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        cost = settle(future)
                        spent += cost
                        largest = max(largest, cost)
                if pauses:
                    break
                # Reserve the largest recorded cell cost for every cell still in flight.
                if spent + largest * len(running) >= ceiling:
                    for future in running:
                        spent += settle(future)
                    _emit({"aborted": "cost ceiling", "spent_usd": spent, "ceiling_usd": ceiling})
                    return 3
                spec = _spec(
                    args,
                    plan,
                    key,
                    profile=profile,
                    capture=capture,
                    prior_blocked_attempts=len(_blocked_attempts(args.output, key)),
                )
                future = pool.submit(
                    _run_one,
                    spec,
                    key,
                    args.output,
                    guard=guard,
                    env=env,
                    quota_retries=args.quota_retries,
                )
                future_keys[future] = key
                running.add(future)
            for future in running:
                spent += settle(future)
    finally:
        if stub is not None:
            stub.terminate()
    if pauses:
        _emit({"aborted": "quota", "detail": str(pauses[0]), "spent_usd": round(spent, 4)})
        return QUOTA_EXIT
    _emit({"completed": True, "spent_usd": round(spent, 4)})
    return 0


def validate(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    try:
        coverage = validate_swe_ab_directory(args.input, plan, require_complete=not args.partial)
    except SweAbError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(coverage.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--work-root", type=Path, required=True)
    r.add_argument("--runtime", choices=("docker", "local"), required=True)
    r.add_argument(
        "--omp-command", default="omp", help="omp executable (and interpreter), shell-split"
    )
    r.add_argument(
        "--profile-dir", type=Path, help="provisioned `swebench` profile agent directory"
    )
    r.add_argument("--cli-guide", type=Path, default=Path(CLI_GUIDE_PATH))
    r.add_argument(
        "--cost-ceiling", type=float, default=None, help="lower the plan's ceiling (USD)"
    )
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--network", default="bridge", help="docker network for agent containers")
    r.add_argument("--tasks-root", type=Path, help="SWE-bench_Pro-os/v2/tasks")
    r.add_argument("--omp-dir", type=Path, help="host directory with the omp linux-x64 build")
    r.add_argument("--archex-wheel", type=Path)
    r.add_argument("--uv-binary", type=Path, help="static linux-x86_64 uv binary")
    r.add_argument(
        "--archex-python", default=sys.executable, help="local runtime: archex's interpreter"
    )
    r.add_argument(
        "--stub-script", type=Path, help="run against the local stub provider (no spend)"
    )
    r.add_argument(
        "--broker-url",
        help="the host's auth broker as this process reaches it (docker runtime), "
        "e.g. http://127.0.0.1:8765",
    )
    r.add_argument(
        "--container-broker-url",
        dest="container_broker_url_flag",
        help="the broker as agent containers reach it "
        "(default: http://host.docker.internal:<broker port>)",
    )
    r.add_argument(
        "--broker-token-file",
        type=Path,
        help="broker bearer token file (default: $OMP_AUTH_BROKER_TOKEN, else "
        "<omp config dir>/auth-broker.token)",
    )
    r.add_argument(
        "--quota-min-headroom",
        type=float,
        default=QUOTA_MIN_HEADROOM,
        help="pause a provider's cells below this fraction of subscription headroom",
    )
    r.add_argument("--quota-poll-seconds", type=float, default=60.0)
    r.add_argument(
        "--quota-max-wait-seconds",
        type=float,
        default=6 * 3600.0,
        help="longest single pause before the run stops with exit status 4 (resumable)",
    )
    r.add_argument(
        "--quota-retries",
        type=int,
        default=2,
        help="re-runs of a cell that ends in a quota block, per invocation",
    )
    r.add_argument(
        "--prune-images",
        action="store_true",
        help="remove a task's image once all its cells have run (Pro images are multi-GB)",
    )
    v = sub.add_parser("validate")
    v.add_argument("--plan", type=Path, required=True)
    v.add_argument("--input", type=Path, required=True)
    v.add_argument("--partial", action="store_true", help="allow missing cells (progress check)")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
