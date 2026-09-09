"""Run every pinned Graft graph-memory cell for the head-to-head task subset.

Drives `scripts/run_graft_headtohead_lane.py` over the manifest's `task_subset`,
producing both Graft modes per task: the cold `build_plus_query` lane builds the
graph and asks, then the warm `query_warm` lane asks again against exactly that
graph. Every planned cell is written as one artifact — a measurement or a
recorded failure — so a cell is never missing from the published comparison.

Usage:

```bash
CI=1 DO_NOT_TRACK=1 uv run python scripts/run_graft_headtohead_suite.py \
    --manifest benchmarks/headtohead/manifest.yaml \
    --output benchmarks/headtohead/results \
    --binary /path/to/graft
```

The task checkout is prepared exactly as the other lanes prepare it (pinned
commit, manifest `include_paths` slice), the graph directory always lives
outside that checkout, and artifacts are rejected if they leak an absolute path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from archex.benchmark.graft import (
    GRAFT_NPM_INTEGRITY,
    GRAFT_RESULT_CARDINALITY,
    GRAFT_SOURCE_COMMIT,
)
from archex.benchmark.headtohead import load_headtohead_manifest, select_headtohead_tasks
from archex.benchmark.models import BenchmarkTask, GraphMemoryLaneConfig, GraphMemoryTool
from archex.benchmark.runner import repo_path_for_task

_CELL_RUNNER = Path(__file__).resolve().parent / "run_graft_headtohead_lane.py"


def _graft_lanes(manifest_path: Path) -> list[GraphMemoryLaneConfig]:
    manifest = load_headtohead_manifest(manifest_path)
    lanes = [lane for lane in manifest.graph_memory_lanes if lane.tool is GraphMemoryTool.GRAFT]
    if len(lanes) != 2:
        message = f"expected two Graft lanes in {manifest_path}, found {len(lanes)}"
        raise SystemExit(message)
    # Cold first: the warm lane answers from the graph the cold lane built.
    lanes.sort(key=lambda lane: not lane.includes_build_cost)
    return lanes


def _sanitize(document: str, *, repo_path: Path, graph_dir: Path) -> str:
    """Replace machine paths with the placeholders the public artifacts use."""
    sanitized = document.replace(str(repo_path), "<repo>").replace(str(graph_dir), "<graph-dir>")
    for leak in ("/Users/", "/home/", "/private/", "/tmp/"):
        if leak in sanitized:
            message = f"artifact leaks an absolute path containing {leak!r}"
            raise SystemExit(message)
    return sanitized


def _run_cell(
    lane: GraphMemoryLaneConfig,
    *,
    task: BenchmarkTask,
    repo_path: Path,
    graph_dir: Path,
    binary: str,
    output_dir: Path,
) -> dict[str, object]:
    payload = {
        "task": task.model_dump(mode="json"),
        "repo_path": str(repo_path),
        "graph_dir": str(graph_dir),
        "lane": lane.name,
        "tool": lane.tool.value,
        "mode": lane.mode.value,
        "graft": {
            "package_name": lane.package_name,
            "version": lane.version,
            "npm_integrity": GRAFT_NPM_INTEGRITY,
            "source_commit": GRAFT_SOURCE_COMMIT,
            "result_cardinality": GRAFT_RESULT_CARDINALITY,
        },
        "binary": binary,
    }
    completed = subprocess.run(
        [sys.executable, str(_CELL_RUNNER)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CI": "1", "DO_NOT_TRACK": "1"},
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        message = f"{lane.name} cell for {task.task_id} failed: {detail}"
        raise SystemExit(message)

    document = _sanitize(completed.stdout, repo_path=repo_path, graph_dir=graph_dir)
    lane_dir = output_dir / lane.name
    lane_dir.mkdir(parents=True, exist_ok=True)
    (lane_dir / f"{task.task_id}.json").write_text(document, encoding="utf-8")
    parsed: dict[str, object] = json.loads(document)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/headtohead/manifest.yaml")
    )
    parser.add_argument("--tasks-dir", type=Path, default=Path("benchmarks/tasks"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/headtohead/results"))
    parser.add_argument("--binary", default="graft")
    parser.add_argument("--task", action="append", default=None, help="run only these task ids")
    args = parser.parse_args()

    manifest = load_headtohead_manifest(args.manifest)
    lanes = _graft_lanes(args.manifest)
    tasks = select_headtohead_tasks(manifest, args.tasks_dir)
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if task.task_id in wanted]

    workspace = Path(tempfile.mkdtemp(prefix="archex-graft-suite-"))
    repo_cache: dict[tuple[str, str, tuple[str, ...]], Path] = {}
    cleanup_paths: list[Path] = []
    started = time.perf_counter()
    try:
        for index, task in enumerate(tasks, start=1):
            repo_path = repo_path_for_task(task, repo_cache, cleanup_paths)
            graph_dir = workspace / task.task_id / "graft-index"
            if graph_dir.exists():
                shutil.rmtree(graph_dir)
            for lane in lanes:
                cell = _run_cell(
                    lane,
                    task=task,
                    repo_path=repo_path,
                    graph_dir=graph_dir,
                    binary=args.binary,
                    output_dir=args.output,
                )
                print(
                    f"[{index}/{len(tasks)}] {task.task_id} {lane.name} "
                    f"status={cell['status']} tier={cell['extraction_tier']} "
                    f"required_file_recall={cell['required_file_recall']:.3f} "
                    f"units={cell['returned_units']} "
                    f"cold_ms={cell['cold_start_ms']:.0f} warm_ms={cell['warm_latency_ms']:.0f}",
                    flush=True,
                )
    finally:
        for path in cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)

    elapsed = time.perf_counter() - started
    print(f"wrote {len(tasks) * len(lanes)} cells in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
