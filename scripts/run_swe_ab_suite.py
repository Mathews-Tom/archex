"""Run or validate a declared set of SWE A/B cells (spec §5, §8).

```bash
# Campaign cells (see benchmarks/swe_ab/RUNBOOK.md). The plan's `campaign` file names the
# provider and the configurations. The key comes from the campaign's key variable (for
# benchmarks/swe_ab/campaigns/muna.yml, $MUNA_ACCESS_KEY), else from that variable's `NAME=`
# line of --env-file (default: ./.env):
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

Resumable: a cell whose artifact exists is skipped and its recorded cost and billed tokens
count toward the ceilings. The hard cumulative cost ceiling (the plan's, or a lower
``--cost-ceiling``) is checked before every cell and aborts the run when reached; cost is
omp's model of the tokens at the prices in the campaign's provider config. A campaign model
priced at 0 (a free preview) makes that ceiling inert, so a docker run of such a campaign
refuses to start unless the plan sets ``token_ceiling``: a cap on cumulative billed tokens
(input + output + cache), the plan's or a lower ``--token-ceiling``, checked before every
cell exactly like the cost ceiling. Either ceiling aborts the run with exit status 3. A cell
whose runner produces no artifact is recorded as a harness failure; nothing is dropped.

Order: cells run by (omp selector, task in plan order, configuration label, arm,
repetition), so one model family stays loaded in a provider's shared capacity at a time
(Muna's GPU pool, say); ``--prune-images`` removes a task's image once its cells for the
current family are done.

Rate limits and capacity (HTTP 429, e.g. Muna's ``model_loading`` and
``model_capacity_exhausted``): a cell that ends in one (before its first tool call, or
mid-run) after omp's own in-run retries is not a task outcome. Its artifact is filed under
``<output>/quota-blocked/`` with its cost still counted, and the cell is re-run after
``--block-cooldown-seconds``, up to ``--quota-retries`` times per invocation; past that it
stays as the cell's record for a later resume. A re-run starts in an empty work directory:
the earlier attempt's files move to ``<work>/cells/<cell>/prior-attempts/<n>/``.

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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from archex.benchmark.swe_ab import (
    CLI_GUIDE_PATH,
    OMP_CONFIG_PATH,
    QUOTA_BLOCKED_DIR,
    Campaign,
    CellKey,
    FailureReason,
    PlanTask,
    SweAbCell,
    SweAbError,
    SweAbPlan,
    credential_files_in_profile,
    is_emulated,
    load_cell,
    load_plan,
    plan_campaign,
    quota_blocked_relative_path,
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
"""Exit status when the cost or token ceiling was reached."""
CREDIT_EXIT = 5
"""Exit status when credits ran out; the run is resumable once the account is topped up."""
BLOCK_COOLDOWN_SECONDS = 300.0
"""Default wait before a rate-limit or capacity block is re-run."""
QUOTA_RETRIES = 2
"""Default re-runs of a blocked cell per invocation."""


@dataclass(frozen=True)
class Spend:
    """What cells consumed: omp's dollar cost and the billed tokens (input + output + cache)."""

    cost_usd: float = 0.0
    tokens: int = 0

    def __add__(self, other: Spend) -> Spend:
        return Spend(self.cost_usd + other.cost_usd, self.tokens + other.tokens)


def _spend_of(cell: SweAbCell) -> Spend:
    return Spend(cell.usage.cost_usd, cell.usage.total_billed)


def _peak(a: Spend, b: Spend) -> Spend:
    """The larger cost and the larger token count seen so far (a per-cell reservation)."""
    return Spend(max(a.cost_usd, b.cost_usd), max(a.tokens, b.tokens))


class CreditExhaustedError(RuntimeError):
    """A cell ended in credit exhaustion; its attempt is filed and the run stops."""

    def __init__(self, message: str, spend: Spend) -> None:
        super().__init__(message)
        self.spend = spend


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _emit(event: Mapping[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def stub_provider_config(campaign: Campaign, port: int) -> str:
    """The campaign's frozen provider config with its endpoint pointed at the local stub.

    Provider name, model ids, and thinking settings stay exactly as campaign cells see them,
    so omp builds the same requests; only the base URL (and the key, which the stub does not
    check) differ, which is what the validator refuses a rehearsal cell for.
    """
    document = cast(
        "dict[str, Any]",
        yaml.safe_load((REPO_ROOT / campaign.provider_config).read_text(encoding="utf-8")),
    )
    provider = cast("dict[str, Any]", document["providers"][campaign.provider])
    provider["baseUrl"] = f"http://127.0.0.1:{port}/v1"
    provider.pop("apiKey", None)
    provider["auth"] = "none"
    return yaml.safe_dump(document, sort_keys=False)


def _start_stub(
    work_root: Path, script: Path, campaign: Campaign
) -> tuple[subprocess.Popen[str], Path, Path, Path]:
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
    provider_config.write_text(stub_provider_config(campaign, port), encoding="utf-8")
    return process, profile, provider_config, capture


# --- the campaign key --------------------------------------------------------------------


def read_env_file_key(path: Path, variable: str) -> str | None:
    """The value of ``variable`` in a dotenv file; no other line is read into a result."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        name, separator, value = line.strip().removeprefix("export ").lstrip().partition("=")
        if separator and name.strip() == variable:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                return value[1:-1] or None
            return value.split(" #", 1)[0].strip() or None
    return None


def campaign_key(campaign: Campaign, env_file: Path | None) -> str:
    """The campaign key from the environment, else from the dotenv file; never printed."""
    name = campaign.credential_env
    key = os.environ.get(name, "").strip() or (
        read_env_file_key(env_file, name) if env_file else None
    )
    if not key:
        raise SystemExit(
            f"{name} is neither in the environment nor in {env_file}; set it before any cell runs"
        )
    return key


def cell_environment(campaign: Campaign, key: str) -> dict[str, str]:
    """The cell runner's environment: this one's, minus a stray campaign key, plus the key."""
    env = {k: v for k, v in os.environ.items() if k != campaign.credential_env}
    env[campaign.credential_env] = key
    return env


# --- cells -------------------------------------------------------------------------------


def _spec(
    args: argparse.Namespace,
    plan: SweAbPlan,
    key: CellKey,
    *,
    campaign: Campaign,
    profile: Path,
    provider_config: Path | None,
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
        "campaign": campaign.path,
        "provider_config": str(provider_config) if provider_config else None,
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
) -> Spend:
    """Run one cell to a real outcome; return the recorded spend of every attempt.

    An attempt that ends in a rate-limit or capacity block is filed under ``quota-blocked/``
    and re-run after the cooldown (up to ``quota_retries`` times); past that budget the blocked
    artifact stays as the cell's record, which the validator refuses for publication. An attempt
    that ends in credit exhaustion is filed and raises `CreditExhaustedError`: never retried,
    never scored.
    """
    output = Path(spec["output"])
    total = Spend()
    blocks = 0
    while True:
        invoke(spec, env)
        cell = load_cell(output)
        total += _spend_of(cell)
        blocked = cell.failure_reason in _BLOCKS
        log(
            {
                "cell": str(output),
                "status": cell.status,
                "reason": cell.failure_reason,
                "cost_usd": cell.usage.cost_usd,
                "tokens": cell.usage.total_billed,
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


def _spent_on_blocked(output_root: Path) -> Spend:
    root = output_root / QUOTA_BLOCKED_DIR
    spent = Spend()
    if root.is_dir():
        for path in root.rglob("*.json"):
            spent += _spend_of(load_cell(path))
    return spent


def _settle(future: Future[Spend], stops: list[CreditExhaustedError]) -> Spend:
    try:
        return future.result()
    except CreditExhaustedError as exc:
        stops.append(exc)
        return exc.spend


def _task_image(task: PlanTask) -> str:
    return task.image or f"ghcr.io/scaleapi/swe-bench_pro-v2:{task.task_id}"


def _prune_image(image: str, log: Callable[[Mapping[str, Any]], None] = _emit) -> None:
    """Remove a task image once its cells for the current model family have run."""
    done = subprocess.run(
        ["docker", "image", "rm", image], capture_output=True, text=True, check=False
    )
    log({"pruned_image": image, "removed": done.returncode == 0})


def _token_ceiling(plan: SweAbPlan, lowered: int | None) -> int | None:
    """The plan's token ceiling, lowered (never raised, never invented) by ``--token-ceiling``."""
    if lowered is None:
        return plan.token_ceiling
    if plan.token_ceiling is None:
        raise SystemExit(
            "--token-ceiling only lowers the plan's token_ceiling, and the plan sets none"
        )
    return min(plan.token_ceiling, lowered)


def _ceiling_hit(
    spent: Spend, largest: Spend, running: int, ceiling: float, token_ceiling: int | None
) -> str | None:
    """The ceiling the next cell must not start under, if any.

    Every cell still in flight is reserved at the largest cell recorded so far, so ``spent``
    plus those reservations stays under each ceiling.
    """
    if spent.cost_usd + largest.cost_usd * running >= ceiling:
        return "cost ceiling"
    if token_ceiling is not None and spent.tokens + largest.tokens * running >= token_ceiling:
        return "token ceiling"
    return None


def run(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    campaign = plan_campaign(plan, root=REPO_ROOT)
    ceiling = min(plan.cost_ceiling_usd, args.cost_ceiling or plan.cost_ceiling_usd)
    token_ceiling = _token_ceiling(plan, args.token_ceiling)
    args.work_root.mkdir(parents=True, exist_ok=True)
    stub: subprocess.Popen[str] | None = None
    capture: Path | None = None
    profile = args.profile_dir
    provider_config: Path | None = None
    env: dict[str, str] | None = None
    if args.runtime == "docker":
        if campaign.unpriced and plan.token_ceiling is None:
            raise SystemExit(
                f"campaign {campaign.name!r} leaves prices at 0: {list(campaign.unpriced)}; the "
                "cost ceiling would be inert, so no docker run starts until the plan sets "
                "`token_ceiling` (RUNBOOK §3)"
            )
        env = cell_environment(campaign, campaign_key(campaign, args.env_file))
    if args.stub_script is not None:
        if args.runtime != "local" or args.jobs != 1:
            raise SystemExit("the stub provider runs only with --runtime local --jobs 1")
        stub, profile, provider_config, capture = _start_stub(
            args.work_root, args.stub_script, campaign
        )
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
                "campaign": campaign.name,
                "provider": campaign.provider,
                "base_url": campaign.base_url,
                "agent_env_names": [campaign.credential_env],
                "emulated": is_emulated("docker", platform.machine()),
            }
        )
    scheduled = plan.scheduled_cells(campaign)
    for key in scheduled:
        if (path := args.output / key.relative_path).exists() and load_cell(
            path
        ).failure_reason in _BLOCKS:
            # A previous invocation stopped on this block: file it, re-run the cell. Filed
            # first, so `spent` below counts it with every earlier attempt.
            _file_blocked(args.output, key)
    spent = _spent_on_blocked(args.output)
    largest = Spend()
    pending: list[CellKey] = []
    for key in scheduled:
        path = args.output / key.relative_path
        if path.exists():
            recorded = _spend_of(load_cell(path))
            spent += recorded
            largest = _peak(largest, recorded)
        else:
            pending.append(key)
    _emit(
        {
            "declared": len(scheduled),
            "pending": len(pending),
            "spent_usd": spent.cost_usd,
            "spent_tokens": spent.tokens,
        }
    )
    stops: list[CreditExhaustedError] = []

    def family_of(key: CellKey) -> tuple[str, str]:
        return (campaign.configuration(key.model).selector, key.task_id)

    remaining = Counter(family_of(key) for key in pending)
    future_keys: dict[Future[Spend], CellKey] = {}

    def settle(future: Future[Spend]) -> Spend:
        key = future_keys.pop(future)
        completed = future.exception() is None
        done = _settle(future, stops)
        if completed:
            family = family_of(key)
            remaining[family] -= 1
            if args.prune_images and args.runtime == "docker" and remaining[family] == 0:
                _prune_image(_task_image(next(t for t in plan.tasks if t.task_id == key.task_id)))
        return done

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            running: set[Future[Spend]] = set()
            for key in pending:
                while len(running) >= args.jobs:
                    finished, running = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        done = settle(future)
                        spent += done
                        largest = _peak(largest, done)
                if stops:
                    break
                if hit := _ceiling_hit(spent, largest, len(running), ceiling, token_ceiling):
                    for future in running:
                        spent += settle(future)
                    _emit(
                        {
                            "aborted": hit,
                            "spent_usd": spent.cost_usd,
                            "ceiling_usd": ceiling,
                            "spent_tokens": spent.tokens,
                            "ceiling_tokens": token_ceiling,
                        }
                    )
                    return COST_EXIT
                spec = _spec(
                    args,
                    plan,
                    key,
                    campaign=campaign,
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
                "spent_usd": round(spent.cost_usd, 4),
                "spent_tokens": spent.tokens,
                "resume": "top up the account, then run the same command again",
            }
        )
        return CREDIT_EXIT
    _emit({"completed": True, "spent_usd": round(spent.cost_usd, 4), "spent_tokens": spent.tokens})
    return 0


def validate(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    try:
        campaign = plan_campaign(plan, root=REPO_ROOT)
        coverage = validate_swe_ab_directory(
            args.input, plan, campaign, require_complete=not args.partial
        )
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
    r.add_argument(
        "--token-ceiling",
        type=int,
        default=None,
        help="lower the plan's token_ceiling (billed tokens); never raises it",
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
        help="dotenv file read for the campaign's key variable (`NAME=` line) when that "
        "variable is unset (docker runtime; default: <repo root>/.env)",
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
