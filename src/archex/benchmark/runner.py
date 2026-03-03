"""Benchmark execution engine: runs tasks across strategies and collects results."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, BenchmarkTask, Strategy
from archex.benchmark.strategies import STRATEGY_RUNNERS

logger = logging.getLogger(__name__)

AVAILABLE_STRATEGIES: list[Strategy] = [
    Strategy.RAW_FILES,
    Strategy.RAW_GREPPED,
    Strategy.ARCHEX_QUERY,
]


def clone_at_commit(repo_slug: str, commit: str) -> tuple[Path, bool]:
    """Clone a GitHub repo and checkout a specific commit. Returns (path, needs_cleanup)."""
    url = f"https://github.com/{repo_slug}.git"
    target = Path(tempfile.mkdtemp(prefix="archex-bench-"))
    subprocess.run(
        ["git", "clone", "--quiet", url, str(target)],
        check=True,
        capture_output=True,
        timeout=300,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", commit],
        cwd=target,
        check=True,
        capture_output=True,
        timeout=30,
    )
    return target, True


def run_benchmark(
    task: BenchmarkTask,
    strategies: list[Strategy] | None = None,
    repo_path: Path | None = None,
) -> BenchmarkReport:
    """Run a benchmark task across strategies. Clones repo if repo_path not provided."""
    if strategies is None:
        strategies = list(AVAILABLE_STRATEGIES)

    needs_cleanup = False
    if repo_path is None:
        repo_path, needs_cleanup = clone_at_commit(task.repo, task.commit)

    try:
        results: list[BenchmarkResult] = []
        for strategy in strategies:
            runner = STRATEGY_RUNNERS.get(strategy)
            if runner is None:
                logger.warning("No runner for strategy %s, skipping", strategy)
                continue
            try:
                result = runner(task, repo_path)
                results.append(result)
                print(
                    f"  [{strategy.value}] {result.tokens_total:,} tokens, "
                    f"recall={result.recall:.2f}, {result.wall_time_ms:.0f}ms",
                    file=sys.stderr,
                )
            except NotImplementedError as exc:
                logger.info("Skipping %s: %s", strategy.value, exc)
                print(f"  [{strategy.value}] skipped: {exc}", file=sys.stderr)

        # Compute baseline and backfill savings_vs_raw
        baseline_tokens = 0
        raw_result = next((r for r in results if r.strategy == Strategy.RAW_FILES), None)
        if raw_result is not None:
            baseline_tokens = raw_result.tokens_total

        if baseline_tokens > 0:
            for result in results:
                if result.strategy != Strategy.RAW_FILES:
                    result.savings_vs_raw = round(
                        (1 - result.tokens_total / baseline_tokens) * 100,
                        1,
                    )

        return BenchmarkReport(
            task_id=task.task_id,
            repo=task.repo,
            question=task.question,
            results=results,
            baseline_tokens=baseline_tokens,
        )
    finally:
        if needs_cleanup:
            shutil.rmtree(repo_path, ignore_errors=True)


def run_all(
    tasks_dir: Path,
    output_dir: Path,
    strategies: list[Strategy] | None = None,
    task_filter: str | None = None,
) -> list[BenchmarkReport]:
    """Load all tasks, run benchmarks, write results to output_dir."""
    tasks = load_tasks(tasks_dir)
    if task_filter:
        tasks = [t for t in tasks if t.task_id == task_filter]
        if not tasks:
            raise ValueError(f"No task found with id '{task_filter}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[BenchmarkReport] = []

    for task in tasks:
        print(f"Running benchmark: {task.task_id} ({task.repo})", file=sys.stderr)
        report = run_benchmark(task, strategies=strategies)
        reports.append(report)

        result_path = output_dir / f"{task.task_id}.json"
        result_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"  → Wrote {result_path}", file=sys.stderr)

    return reports
