"""Run or validate a declared set of SWE A/B cells (spec §5, §8).

```bash
# Campaign cells on Muna-hosted models (see benchmarks/swe_ab/RUNBOOK.md). The key comes from
# $MUNA_ACCESS_KEY, else from the `MUNA_ACCESS_KEY=` line of --env-file (default: ./.env):
uv run python scripts/run_swe_ab_suite.py run --plan stage1.json --runtime docker \
    --output benchmarks/swe_ab/results/stage1 --work-root /scratch/swe-ab \
    --tasks-root SWE-bench_Pro-os/v2/tasks --omp-dir /opt/omp-linux-x64 \
    --omp-command "/opt/omp/bin/omp" --profile-dir ~/.omp/profiles/swebench/agent \
    --archex-wheel dist/archex-0.34.0-py3-none-any.whl --uv-binary /opt/uv/uv --jobs 4

# No-spend rehearsal against the local stub provider:
uv run python scripts/run_swe_ab_suite.py run --plan dry-run.json --runtime local \
    --output /tmp/swe-ab/cells --work-root /tmp/swe-ab/work \
    --stub-script benchmarks/swe_ab/stub-script.json

uv run python scripts/run_swe_ab_suite.py validate --plan stage1.json \
    --input benchmarks/swe_ab/results/stage1
```

Resumable: a cell whose artifact exists is skipped and its recorded cost counts toward
the ceiling. The hard cumulative cost ceiling (the plan's, or a lower ``--cost-ceiling``)
is checked before every cell and aborts the run when reached; cost is omp's model of the
tokens at the prices in ``benchmarks/swe_ab/muna-models.yml``, so a docker run refuses to
start while any of those prices is still 0. A cell whose runner produces no artifact is
recorded as a harness failure; nothing is dropped.

Order: cells run by (omp selector, task in plan order, configuration label, arm,
repetition), so one model family stays loaded in Muna's shared GPU capacity at a time;
``--prune-images`` removes a task's image once its cells for the current family are done.

Rate limits and capacity (Muna HTTP 429, including ``model_loading`` and
``model_capacity_exhausted``): a cell that ends in one (before its first tool call, or
mid-run) after omp's own in-run retries is not a task outcome. Its artifact is filed under
``<output>/quota-blocked/`` with its cost still counted, and the cell is re-run after
``--block-cooldown-seconds``, up to ``--quota-retries`` times per invocation; past that it
stays as the cell's record for a later resume.

Credit exhaustion (HTTP 402 and kin) is never retried and never scored: the attempt is
filed under ``quota-blocked/``, running cells finish, and the run exits with status 5; once
the account is topped up, run the same command again.
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
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from archex.benchmark.swe_ab import (
    CAMPAIGN_PROVIDER,
    CLI_GUIDE_PATH,
    CREDENTIAL_ENV_NAMES,
    MUNA_BASE_URL,
    OMP_CONFIG_PATH,
    PROVIDER_CONFIG_PATH,
    QUOTA_BLOCKED_DIR,
    CellKey,
    FailureReason,
    PlanTask,
    SweAbError,
    SweAbPlan,
    configuration,
    credential_files_in_profile,
    is_emulated,
    load_cell,
    load_plan,
    quota_blocked_relative_path,
    unpriced_models,
    validate_swe_ab_directory,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swe_ab_cell as cell_runner  # noqa: E402 - sibling script, importable only via sys.path

REPO_ROOT = Path(__file__).resolve().parents[1]
_CELL_SCRIPT = Path(__file__).resolve().parent / "run_swe_ab_cell.py"
_STUB_SCRIPT = Path(__file__).resolve().parent / "swe_ab_stub_provider.py"
_CELL_TIMEOUT_SECONDS = 4 * 3600
COST_EXIT = 3
"""Exit status when the cost ceiling was reached."""
CREDIT_EXIT = 5
"""Exit status when credits ran out; the run is resumable once the account is topped up."""
BLOCK_COOLDOWN_SECONDS = 300.0
"""Default wait before a rate-limit or capacity block is re-run."""
QUOTA_RETRIES = 2
"""Default re-runs of a blocked cell per invocation."""


class CreditExhaustedError(RuntimeError):
    """A cell ended in credit exhaustion; its attempt is filed and the run stops."""

    def __init__(self, message: str, cost: float) -> None:
        super().__init__(message)
        self.cost = cost


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _emit(event: Mapping[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def stub_provider_config(port: int) -> str:
    """The frozen Muna provider config with its endpoint pointed at the local stub.

    Provider name, model ids, and thinking settings stay exactly as campaign cells see them,
    so omp builds the same requests; only the base URL (and the key, which the stub does not
    check) differ, which is what the validator refuses a rehearsal cell for.
    """
    document = cast(
        "dict[str, Any]",
        yaml.safe_load((REPO_ROOT / PROVIDER_CONFIG_PATH).read_text(encoding="utf-8")),
    )
    provider = cast("dict[str, Any]", document["providers"][CAMPAIGN_PROVIDER])
    provider["baseUrl"] = f"http://127.0.0.1:{port}/v1"
    provider.pop("apiKey", None)
    provider["auth"] = "none"
    return yaml.safe_dump(document, sort_keys=False)


def _start_stub(work_root: Path, script: Path) -> tuple[subprocess.Popen[str], Path, Path, Path]:
    """Start the stub; return it, an empty profile, its provider config, and its capture dir."""
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
    provider_config = work_root / "stub-models.yml"
    provider_config.write_text(stub_provider_config(port), encoding="utf-8")
    return process, profile, provider_config, capture


# --- the campaign key --------------------------------------------------------------------


def read_env_file_key(path: Path) -> str | None:
    """The value of ``MUNA_ACCESS_KEY`` in a dotenv file; no other line is read into a result."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        name, separator, value = line.strip().removeprefix("export ").lstrip().partition("=")
        if separator and name.strip() == CREDENTIAL_ENV_NAMES[0]:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                return value[1:-1] or None
            return value.split(" #", 1)[0].strip() or None
    return None


def campaign_key(env_file: Path | None) -> str:
    """The campaign key from the environment, else from the dotenv file; never printed."""
    name = CREDENTIAL_ENV_NAMES[0]
    key = os.environ.get(name, "").strip() or (read_env_file_key(env_file) if env_file else None)
    if not key:
        raise SystemExit(
            f"{name} is neither in the environment nor in {env_file}; set it before any cell runs"
        )
    return key


def cell_environment(key: str) -> dict[str, str]:
    """The cell runner's environment: this one's, minus a stray campaign key, plus the key."""
    env = {k: v for k, v in os.environ.items() if k not in CREDENTIAL_ENV_NAMES}
    env[CREDENTIAL_ENV_NAMES[0]] = key
    return env


# --- cells -------------------------------------------------------------------------------


def _spec(
    args: argparse.Namespace,
    plan: SweAbPlan,
    key: CellKey,
    *,
    profile: Path,
    provider_config: Path,
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
        "provider_config": str(provider_config),
        "omp_config": str(REPO_ROOT / OMP_CONFIG_PATH),
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
        tests = task_dir / "tests" / "test_patch.patch"
        if tests.exists():
            spec["test_patch"] = str(tests)
    if args.runtime == "docker":
        spec.update(
            image=_task_image(task),
            omp_dir=str(args.omp_dir),
            archex_wheel=str(args.archex_wheel),
            uv_binary=str(args.uv_binary),
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


def _file_blocked(output_root: Path, key: CellKey) -> int:
    """Move ``key``'s artifact to the next ``quota-blocked/`` slot; return that attempt number."""
    attempt = len(_blocked_attempts(output_root, key)) + 1
    target = output_root / quota_blocked_relative_path(key, attempt)
    target.parent.mkdir(parents=True, exist_ok=True)
    (output_root / key.relative_path).replace(target)
    return attempt


_BLOCKS = (FailureReason.QUOTA_BLOCK, FailureReason.CREDIT_EXHAUSTED)


def _run_one(
    spec: dict[str, Any],
    key: CellKey,
    output_root: Path,
    *,
    env: Mapping[str, str] | None,
    quota_retries: int,
    cooldown_seconds: float,
    invoke: Callable[[dict[str, Any], Mapping[str, str] | None], None] = _invoke_cell,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[Mapping[str, Any]], None] = _emit,
) -> float:
    """Run one cell to a real outcome; return the recorded cost of every attempt.

    An attempt that ends in a rate-limit or capacity block is filed under ``quota-blocked/``
    and re-run after the cooldown (up to ``quota_retries`` times); past that budget the blocked
    artifact stays as the cell's record, which the validator refuses for publication. An attempt
    that ends in credit exhaustion is filed and raises `CreditExhaustedError`: never retried,
    never scored.
    """
    output = Path(spec["output"])
    total = 0.0
    blocks = 0
    while True:
        invoke(spec, env)
        cell = load_cell(output)
        total += cell.usage.cost_usd
        blocked = cell.failure_reason in _BLOCKS
        log(
            {
                "cell": str(output),
                "status": cell.status,
                "reason": cell.failure_reason,
                "cost_usd": cell.usage.cost_usd,
                **({"quota_phase": cell.quota.block_phase} if blocked else {}),
            }
        )
        if cell.failure_reason is FailureReason.CREDIT_EXHAUSTED:
            attempt = _file_blocked(output_root, key)
            log({"credit": "exhausted", "cell": str(output), "attempt": attempt})
            raise CreditExhaustedError(f"{key.model}: credits exhausted", total)
        if cell.failure_reason is not FailureReason.QUOTA_BLOCK:
            return total
        blocks += 1
        if blocks > quota_retries:
            log({"quota": "retry_budget_exhausted", "cell": str(output), "attempts": blocks})
            return total
        attempt = _file_blocked(output_root, key)
        spec["prior_blocked_attempts"] = attempt
        log(
            {
                "quota": "cell_blocked_will_rerun",
                "cell": str(output),
                "attempt": attempt,
                "cooldown_seconds": cooldown_seconds,
            }
        )
        sleep(cooldown_seconds)


def _spent_on_blocked(output_root: Path) -> float:
    root = output_root / QUOTA_BLOCKED_DIR
    return (
        sum(load_cell(path).usage.cost_usd for path in root.rglob("*.json"))
        if root.is_dir()
        else 0.0
    )


def _settle(future: Future[float], stops: list[CreditExhaustedError]) -> float:
    try:
        return future.result()
    except CreditExhaustedError as exc:
        stops.append(exc)
        return exc.cost


def _task_image(task: PlanTask) -> str:
    return task.image or f"ghcr.io/scaleapi/swe-bench_pro-v2:{task.task_id}"


def _prune_image(image: str, log: Callable[[Mapping[str, Any]], None] = _emit) -> None:
    """Remove a task image once its cells for the current model family have run."""
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
    provider_config = REPO_ROOT / PROVIDER_CONFIG_PATH
    env: dict[str, str] | None = None
    if args.runtime == "docker":
        env = cell_environment(campaign_key(args.env_file))
        if unpriced := unpriced_models(provider_config):
            raise SystemExit(
                f"{PROVIDER_CONFIG_PATH} leaves prices at 0: {unpriced}; the cost ceiling would "
                "be inert, so no docker run starts until every price is set (RUNBOOK §3)"
            )
    if args.stub_script is not None:
        if args.runtime != "local" or args.jobs != 1:
            raise SystemExit("the stub provider runs only with --runtime local --jobs 1")
        stub, profile, provider_config, capture = _start_stub(args.work_root, args.stub_script)
    if profile is None:
        raise SystemExit("--profile-dir is required unless --stub-script is given")
    if args.runtime == "docker":
        if leaked := credential_files_in_profile(profile):
            raise SystemExit(
                f"the profile carries credential stores {leaked}: the campaign key reaches "
                "containers only as an environment variable, so remove them (RUNBOOK §2.2)"
            )
        _emit(
            {
                "provider": CAMPAIGN_PROVIDER,
                "base_url": MUNA_BASE_URL,
                "agent_env_names": list(CREDENTIAL_ENV_NAMES),
                "emulated": is_emulated("docker", platform.machine()),
            }
        )
    scheduled = plan.scheduled_cells()
    spent = _spent_on_blocked(args.output)
    largest = 0.0
    pending: list[CellKey] = []
    for key in scheduled:
        path = args.output / key.relative_path
        if path.exists() and load_cell(path).failure_reason in _BLOCKS:
            # A previous invocation stopped on this block: file it, re-run the cell.
            _file_blocked(args.output, key)
        if path.exists():
            cost = load_cell(path).usage.cost_usd
            spent += cost
            largest = max(largest, cost)
        else:
            pending.append(key)
    _emit({"declared": len(scheduled), "pending": len(pending), "spent_usd": spent})
    stops: list[CreditExhaustedError] = []
    remaining = Counter((configuration(key.model).selector, key.task_id) for key in pending)
    future_keys: dict[Future[float], CellKey] = {}

    def settle(future: Future[float]) -> float:
        key = future_keys.pop(future)
        completed = future.exception() is None
        cost = _settle(future, stops)
        if completed:
            family = (configuration(key.model).selector, key.task_id)
            remaining[family] -= 1
            if args.prune_images and args.runtime == "docker" and remaining[family] == 0:
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
                if stops:
                    break
                # Reserve the largest recorded cell cost for every cell still in flight.
                if spent + largest * len(running) >= ceiling:
                    for future in running:
                        spent += settle(future)
                    _emit({"aborted": "cost ceiling", "spent_usd": spent, "ceiling_usd": ceiling})
                    return COST_EXIT
                spec = _spec(
                    args,
                    plan,
                    key,
                    profile=profile,
                    provider_config=provider_config,
                    capture=capture,
                    prior_blocked_attempts=len(_blocked_attempts(args.output, key)),
                )
                future = pool.submit(
                    _run_one,
                    spec,
                    key,
                    args.output,
                    env=env,
                    quota_retries=args.quota_retries,
                    cooldown_seconds=args.block_cooldown_seconds,
                )
                future_keys[future] = key
                running.add(future)
            for future in running:
                spent += settle(future)
    finally:
        if stub is not None:
            stub.terminate()
    if stops:
        _emit(
            {
                "aborted": "credits exhausted",
                "detail": str(stops[0]),
                "spent_usd": round(spent, 4),
                "resume": "top up the account, then run the same command again",
            }
        )
        return CREDIT_EXIT
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
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="dotenv file read for its MUNA_ACCESS_KEY= line when the variable is unset "
        "(docker runtime; default: <repo root>/.env)",
    )
    r.add_argument(
        "--block-cooldown-seconds",
        type=float,
        default=BLOCK_COOLDOWN_SECONDS,
        help="wait before a cell that ended in a rate-limit or capacity block is re-run",
    )
    r.add_argument(
        "--quota-retries",
        type=int,
        default=QUOTA_RETRIES,
        help="re-runs of a cell that ends in a rate-limit or capacity block, per invocation",
    )
    r.add_argument(
        "--prune-images",
        action="store_true",
        help="remove a task's image once its cells for the current model family have run "
        "(Pro images are multi-GB; docker re-pulls it for the next family)",
    )
    v = sub.add_parser("validate")
    v.add_argument("--plan", type=Path, required=True)
    v.add_argument("--input", type=Path, required=True)
    v.add_argument("--partial", action="store_true", help="allow missing cells (progress check)")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
