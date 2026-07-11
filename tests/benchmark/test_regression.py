"""Regression test: run self-benchmark tasks and verify minimum quality."""

from __future__ import annotations

from pathlib import Path

import pytest

from archex.benchmark.loader import load_task
from archex.benchmark.models import Strategy
from archex.benchmark.runner import run_benchmark


@pytest.mark.slow
def test_self_benchmark_minimum_quality() -> None:
    """Run self-benchmark tasks against archex repo, assert minimum recall and MRR."""
    tasks_dir = Path.cwd() / "benchmarks" / "tasks"
    self_tasks = sorted(tasks_dir.glob("archex_*.yaml"))
    assert len(self_tasks) > 0, "No self-benchmark tasks found"

    for task_path in self_tasks:
        task = load_task(task_path)
        assert task.repo == ".", f"Self-benchmark task {task.task_id} must use repo='.'"

        report = run_benchmark(
            task,
            strategies=[Strategy.ARCHEX_QUERY],
            repo_path=Path.cwd(),
        )

        for result in report.results:
            assert result.recall >= 0.5, (
                f"{task.task_id}/{result.strategy.value}: recall {result.recall:.2f} < 0.5"
            )
            assert result.mrr > 0.0, (
                f"{task.task_id}/{result.strategy.value}: mrr {result.mrr:.2f} == 0.0"
            )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Control characterization: archex_query omits "
        "django/contrib/auth/validators.py for the pinned task."
    ),
)
def test_django_username_validator_required_file_is_returned() -> None:
    """Expose the pinned Django localization total miss before candidate repair."""
    task = load_task(Path.cwd() / "benchmarks/tasks/loc_django_username_validator.yaml")
    report = run_benchmark(
        task,
        strategies=[Strategy.ARCHEX_QUERY],
        repo_path=None,
    )
    result = report.results[0]

    assert "django/contrib/auth/validators.py" in result.result_files
