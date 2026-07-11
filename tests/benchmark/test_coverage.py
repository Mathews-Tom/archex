"""Tests for benchmark required-file coverage evidence reading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archex.benchmark.coverage import read_required_file_coverage
from archex.benchmark.evidence import task_manifest_digest
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, Strategy


def _result(task_id: str, strategy: Strategy) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        strategy=strategy,
        tokens_total=100,
        tool_calls=1,
        files_accessed=2,
        recall=0.5,
        precision=0.5,
        savings_vs_raw=0.0,
        wall_time_ms=10.0,
        cached=True,
        timestamp="2026-07-11T00:00:00+00:00",
        result_files=["src/returned.py"],
        required_file_recall=0.5,
        required_files_missing=["src/missing.py"],
        token_efficiency_with_completion=0.25,
        warm_latency_ms=8.0,
        seed_files=["src/seed.py"],
        expanded_files=["src/returned.py"],
    )


def _report(task_id: str, strategies: list[Strategy]) -> BenchmarkReport:
    return BenchmarkReport(
        task_id=task_id,
        repo="owner/repo",
        question="Where is the required file?",
        results=[_result(task_id, strategy) for strategy in strategies],
        baseline_tokens=100,
    )


def test_reads_reported_file_coverage_without_task_oracle() -> None:
    rows = read_required_file_coverage(
        [_report("coverage_task", [Strategy.ARCHEX_QUERY, Strategy.RAW_FILES])],
        Strategy.ARCHEX_QUERY,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.task_id == "coverage_task"
    assert row.strategy is Strategy.ARCHEX_QUERY
    assert row.returned_files == ("src/returned.py",)
    assert row.missing_required_files == ("src/missing.py",)
    assert row.required_file_recall == 0.5
    assert row.completion_adjusted_token_efficiency == 0.25
    assert row.warm_latency_ms == 8.0
    assert row.seed_files == ("src/seed.py",)
    assert row.expanded_files == ("src/returned.py",)


def test_rejects_report_without_requested_strategy() -> None:
    with pytest.raises(ValueError, match="Expected exactly one archex_query result"):
        read_required_file_coverage(
            [_report("coverage_task", [Strategy.RAW_FILES])], Strategy.ARCHEX_QUERY
        )


def test_rejects_duplicate_report_task_ids() -> None:
    reports = [
        _report("coverage_task", [Strategy.ARCHEX_QUERY]),
        _report("coverage_task", [Strategy.ARCHEX_QUERY]),
    ]

    with pytest.raises(ValueError, match="Duplicate benchmark report task ID"):
        read_required_file_coverage(reports, Strategy.ARCHEX_QUERY)


def test_control_coverage_artifact_characterizes_five_reproducible_misses() -> None:
    root = Path(__file__).parents[2]
    artifact = root / "benchmarks" / "evidence" / "m0.2-control-coverage.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["strategy"] == Strategy.ARCHEX_QUERY.value
    assert payload["task_manifest_digest"] == task_manifest_digest(root / "benchmarks" / "tasks")
    observations = {entry["task_id"]: entry for entry in payload["tasks"]}
    assert set(observations) == {
        "archex_adapter_registry",
        "archex_project_status",
        "django_middleware",
        "loc_django_username_validator",
        "routing_pl_scoring",
    }
    assert {
        state
        for observation in observations.values()
        for state in observation["missing_file_admission_state"].values()
    } == {"not_admitted", "seed", "graph_expansion"}
    for observation in observations.values():
        assert set(observation["returned_files"]).isdisjoint(observation["missing_required_files"])
