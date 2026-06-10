"""Tests for benchmark readiness reporting."""

from __future__ import annotations

import json

from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTask,
    Strategy,
    TaskCategory,
)
from archex.benchmark.readiness import (
    build_readiness_report,
    format_readiness_json,
    format_readiness_markdown,
)


def _result(
    task_id: str,
    *,
    recall: float,
    precision: float,
    f1_score: float,
    mrr: float = 1.0,
    wall_time_ms: float = 10.0,
    tokens_total: int = 100,
    token_efficiency: float = 0.5,
    savings_vs_raw: float = 25.0,
    category: TaskCategory | None = None,
    strategy: Strategy = Strategy.ARCHEX_QUERY,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        strategy=strategy,
        tokens_total=tokens_total,
        token_efficiency=token_efficiency,
        tool_calls=1,
        files_accessed=1,
        recall=recall,
        precision=precision,
        f1_score=f1_score,
        mrr=mrr,
        savings_vs_raw=savings_vs_raw,
        wall_time_ms=wall_time_ms,
        cached=False,
        timestamp="2026-01-01T00:00:00Z",
        seed_files=["src/main.py"],
        category=category,
    )


def _report(task_id: str, result: BenchmarkResult) -> BenchmarkReport:
    return BenchmarkReport(
        task_id=task_id,
        repo="owner/repo",
        question="How does retrieval work?",
        results=[result],
        baseline_tokens=100,
    )


def _task(task_id: str, category: TaskCategory) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        repo="owner/repo",
        commit="abc123",
        question="How does retrieval work?",
        expected_files=["src/main.py", "src/context.py"],
        category=category,
    )


def test_build_readiness_report_tracks_targets_and_counts() -> None:
    reports = [
        _report(
            "good",
            _result(
                "good",
                recall=1.0,
                precision=0.8,
                f1_score=0.89,
                wall_time_ms=100.0,
                category=TaskCategory.SELF,
            ),
        ),
        _report(
            "miss",
            _result(
                "miss",
                recall=0.0,
                precision=0.0,
                f1_score=0.0,
                mrr=0.0,
                wall_time_ms=300.0,
                category=TaskCategory.EXTERNAL_LARGE,
            ),
        ),
    ]
    tasks = {
        "good": _task("good", TaskCategory.SELF),
        "miss": _task("miss", TaskCategory.EXTERNAL_LARGE),
    }

    readiness = build_readiness_report(reports, tasks)

    assert readiness.task_count == 2
    assert readiness.mean_recall == 0.5
    assert readiness.mean_precision == 0.4
    assert readiness.zero_recall_tasks == 1
    assert readiness.low_f1_tasks == 1
    assert readiness.low_precision_tasks == 1
    assert readiness.median_latency_ms == 200.0
    assert readiness.p95_latency_ms == 290.0
    assert readiness.tokens_total == 200
    assert readiness.token_efficiency == 0.5
    assert readiness.savings_vs_raw == 25.0
    assert readiness.ready is False
    assert [target.name for target in readiness.targets] == [
        "mean_recall",
        "mean_precision",
        "mean_f1_score",
        "zero_recall_tasks",
    ]
    assert readiness.blocking_tasks[0].task_id == "miss"


def test_build_readiness_report_groups_by_task_category_when_result_missing() -> None:
    report = _report(
        "semantic",
        _result("semantic", recall=0.7, precision=0.6, f1_score=0.65, category=None),
    )
    tasks = {"semantic": _task("semantic", TaskCategory.FRAMEWORK_SEMANTIC)}

    readiness = build_readiness_report([report], tasks)

    assert len(readiness.categories) == 1
    assert readiness.categories[0].category == "framework-semantic"


def test_build_readiness_report_handles_missing_strategy() -> None:
    report = _report(
        "raw",
        _result(
            "raw",
            recall=1.0,
            precision=1.0,
            f1_score=1.0,
            strategy=Strategy.RAW_FILES,
        ),
    )

    readiness = build_readiness_report([report], {}, strategy=Strategy.ARCHEX_QUERY)

    assert readiness.task_count == 0
    assert readiness.targets == []
    assert readiness.blocking_tasks == []


def test_format_readiness_outputs_are_stable() -> None:
    report = _report(
        "miss",
        _result(
            "miss",
            recall=0.0,
            precision=0.0,
            f1_score=0.0,
            category=TaskCategory.EXTERNAL_LARGE,
        ),
    )
    readiness = build_readiness_report(
        [report],
        {"miss": _task("miss", TaskCategory.EXTERNAL_LARGE)},
    )

    markdown = format_readiness_markdown(readiness)
    assert "# Benchmark Readiness" in markdown
    assert "mean_recall" in markdown
    assert "P95 latency" in markdown
    assert "Top Blocking Tasks" in markdown
    assert "Tokens Total" in markdown
    assert "Token Efficiency" in markdown
    assert "Savings vs Raw" in markdown
    assert "`miss`" in markdown
    assert "Expansion Reasons" in markdown

    payload = json.loads(format_readiness_json(readiness))
    assert payload["strategy"] == "archex_query"
    assert payload["ready"] is False
    assert payload["median_latency_ms"] == 10.0
    assert payload["p95_latency_ms"] == 10.0
    assert payload["tokens_total"] == 100
    assert payload["token_efficiency"] == 0.5
    assert payload["savings_vs_raw"] == 25.0
    assert payload["blocking_tasks"][0]["task_id"] == "miss"
