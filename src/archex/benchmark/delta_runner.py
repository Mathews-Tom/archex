"""Delta benchmark execution engine: runs delta tasks and collects results."""

from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING

from archex.benchmark.delta_strategies import prepare_repo, run_delta_benchmark
from archex.benchmark.loader import load_delta_tasks

if TYPE_CHECKING:
    from pathlib import Path

    from archex.benchmark.models import DeltaBenchmarkResult, DeltaBenchmarkTask


def run_delta_benchmark_task(
    task: DeltaBenchmarkTask,
    repo_path: Path | None = None,
) -> DeltaBenchmarkResult:
    """Run a single delta benchmark task. Clones/copies repo if repo_path not provided."""
    needs_cleanup = False
    if repo_path is None:
        repo_path, needs_cleanup = prepare_repo(task.repo, task.base_commit)

    try:
        return run_delta_benchmark(task, repo_path)
    finally:
        if needs_cleanup:
            shutil.rmtree(repo_path, ignore_errors=True)


def run_all_delta(
    tasks_dir: Path,
    output_dir: Path,
    task_filter: str | None = None,
) -> list[DeltaBenchmarkResult]:
    """Load all delta tasks, run benchmarks, write results to output_dir."""
    tasks = load_delta_tasks(tasks_dir)
    if task_filter:
        tasks = [t for t in tasks if t.task_id == task_filter]
        if not tasks:
            raise ValueError(f"No delta task found with id '{task_filter}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[DeltaBenchmarkResult] = []

    for task in tasks:
        print(f"Running delta benchmark: {task.task_id} ({task.repo})", file=sys.stderr)
        result = run_delta_benchmark_task(task)
        results.append(result)

        result_path = output_dir / f"{task.task_id}.json"
        result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(
            f"  \u2192 {result.delta_files} delta files, "
            f"speedup={result.speedup_factor:.1f}x, "
            f"correct={result.correctness}, "
            f"{result.delta_time_ms:.0f}ms",
            file=sys.stderr,
        )

    return results
