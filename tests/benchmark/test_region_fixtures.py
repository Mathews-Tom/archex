"""Tests for the seeded region-labelled benchmark fixture tasks.

These tasks declare deterministic expected regions against the in-repo
``python_simple`` fixture so region/line/ranking/context-efficiency metrics can
be exercised end-to-end without cloning external repositories.
"""

from __future__ import annotations

from pathlib import Path

from archex.benchmark.loader import load_tasks, validate_task
from archex.benchmark.strategies import run_archex_query

REGION_TASKS_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "region_tasks"


def test_region_fixture_tasks_load_with_regions() -> None:
    tasks = load_tasks(REGION_TASKS_DIR)
    assert tasks, "expected at least one seeded region task"
    for task in tasks:
        assert task.expected_regions, f"{task.task_id} declares no expected_regions"


def test_region_fixture_tasks_validate_against_fixture_repo(python_simple_repo: Path) -> None:
    for task in load_tasks(REGION_TASKS_DIR):
        errors = validate_task(task, python_simple_repo)
        assert errors == [], f"{task.task_id}: {errors}"


def test_region_fixture_task_computes_region_metrics(python_simple_repo: Path) -> None:
    tasks = {task.task_id: task for task in load_tasks(REGION_TASKS_DIR)}
    task = tasks["region_python_simple_auth"]

    result = run_archex_query(task, python_simple_repo)

    # File-level metrics keep working.
    assert 0.0 <= result.required_file_recall <= 1.0
    # Region metrics are populated because the task declares expected regions.
    assert result.region_recall is not None
    assert 0.0 <= result.region_recall <= 1.0
    assert result.region_precision is not None
    assert result.ranked_region_mrr is not None
    assert result.context_noise_ratio is not None
    assert result.useful_tokens is not None
    assert result.wasted_tokens is not None
    assert result.relevance_per_1k_tokens is not None
