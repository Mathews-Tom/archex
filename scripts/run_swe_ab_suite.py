"""Run or validate a declared set of SWE A/B cells (spec §5, §8).

```bash
# Campaign cells on an x86_64 Linux host (see benchmarks/swe_ab/RUNBOOK.md):
uv run python scripts/run_swe_ab_suite.py run --plan stage1.json --runtime docker \
    --output benchmarks/swe_ab/results/stage1 --work-root /scratch/swe-ab \
    --tasks-root SWE-bench_Pro-os/v2/tasks --omp-dir /opt/omp-linux-x64 \
    --omp-command "/opt/omp/bin/omp" --profile-dir ~/.omp/profiles/swebench/agent \
    --archex-wheel dist/archex-0.33.0-py3-none-any.whl --uv-binary /opt/uv/uv --jobs 8

# No-spend rehearsal against the local stub provider:
uv run python scripts/run_swe_ab_suite.py run --plan dry-run.json --runtime local \
    --output /tmp/swe-ab/cells --work-root /tmp/swe-ab/work \
    --stub-script benchmarks/swe_ab/stub-script.json

uv run python scripts/run_swe_ab_suite.py validate --plan stage1.json \
    --input benchmarks/swe_ab/results/stage1
```

Resumable: a cell whose artifact exists is skipped and its recorded cost
counts toward the ceiling. The hard cumulative cost ceiling (the plan's, or a
lower ``--cost-ceiling``) is checked before every cell and aborts the run when
reached. A cell whose runner produces no artifact is recorded as a harness
failure; nothing is dropped.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import socket
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from archex.benchmark.swe_ab import (
    CLI_GUIDE_PATH,
    CellKey,
    FailureReason,
    SweAbError,
    SweAbPlan,
    load_cell,
    load_plan,
    validate_swe_ab_directory,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_swe_ab_cell as cell_runner  # noqa: E402 - sibling script, importable only via sys.path

_CELL_SCRIPT = Path(__file__).resolve().parent / "run_swe_ab_cell.py"
_STUB_SCRIPT = Path(__file__).resolve().parent / "swe_ab_stub_provider.py"
_CELL_TIMEOUT_SECONDS = 4 * 3600


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _spec(
    args: argparse.Namespace, plan: SweAbPlan, key: CellKey, *, profile: Path, capture: Path | None
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
            image=task.image or f"ghcr.io/scaleapi/swe-bench_pro-v2:{key.task_id}",
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


def _run_one(spec: dict[str, Any]) -> float:
    """Run one cell; guarantee an artifact; return its recorded cost."""
    output = Path(spec["output"])
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, str(_CELL_SCRIPT)],
            input=json.dumps(spec),
            text=True,
            capture_output=True,
            timeout=_CELL_TIMEOUT_SECONDS,
            check=False,
        )
    if not output.exists():
        cell = cell_runner.failed_cell(
            cell_runner.CellSpec(spec),
            FailureReason.HARNESS_ERROR,
            "cell runner produced no artifact",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(cell.model_dump_json(indent=2) + "\n", encoding="utf-8")
    cell = load_cell(output)
    print(
        json.dumps(
            {
                "cell": str(output),
                "status": cell.status,
                "reason": cell.failure_reason,
                "cost_usd": cell.usage.cost_usd,
            }
        ),
        flush=True,
    )
    return cell.usage.cost_usd


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
    spent = 0.0
    largest = 0.0
    pending: list[CellKey] = []
    for key in plan.cells():
        path = args.output / key.relative_path
        if path.exists():
            cost = load_cell(path).usage.cost_usd
            spent += cost
            largest = max(largest, cost)
        else:
            pending.append(key)
    print(json.dumps({"declared": len(plan.cells()), "pending": len(pending), "spent_usd": spent}))
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            running: set[Future[float]] = set()
            for key in pending:
                while len(running) >= args.jobs:
                    finished, running = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        spent += future.result()
                        largest = max(largest, future.result())
                # Reserve the largest recorded cell cost for every cell still in flight.
                if spent + largest * len(running) >= ceiling:
                    for future in running:
                        spent += future.result()
                    print(
                        json.dumps(
                            {"aborted": "cost ceiling", "spent_usd": spent, "ceiling_usd": ceiling}
                        ),
                        flush=True,
                    )
                    return 3
                spec = _spec(args, plan, key, profile=profile, capture=capture)
                running.add(pool.submit(_run_one, spec))
            for future in running:
                spent += future.result()
    finally:
        if stub is not None:
            stub.terminate()
    print(json.dumps({"completed": True, "spent_usd": round(spent, 4)}))
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
    v = sub.add_parser("validate")
    v.add_argument("--plan", type=Path, required=True)
    v.add_argument("--input", type=Path, required=True)
    v.add_argument("--partial", action="store_true", help="allow missing cells (progress check)")
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
