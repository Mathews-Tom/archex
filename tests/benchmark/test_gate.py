"""Tests for quality gate checks."""

from __future__ import annotations

from archex.benchmark.gate import QualityThresholds, check_gate
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, Strategy


def _make_report(
    recall: float = 0.8,
    precision: float = 0.5,
    f1_score: float = 0.6,
    mrr: float = 0.7,
    ndcg: float = 0.7,
    map_score: float = 0.6,
) -> BenchmarkReport:
    result = BenchmarkResult(
        task_id="test_task",
        strategy=Strategy.ARCHEX_QUERY,
        tokens_total=1000,
        tool_calls=1,
        files_accessed=3,
        recall=recall,
        precision=precision,
        f1_score=f1_score,
        mrr=mrr,
        ndcg=ndcg,
        map_score=map_score,
        savings_vs_raw=50.0,
        wall_time_ms=100.0,
        cached=False,
        timestamp="2026-01-01T00:00:00Z",
    )
    return BenchmarkReport(
        task_id="test_task",
        repo="test/repo",
        question="test question",
        results=[result],
        baseline_tokens=2000,
    )


def test_check_gate_all_pass() -> None:
    reports = [_make_report()]
    violations = check_gate(reports)
    assert violations == []


def test_check_gate_violation_detected() -> None:
    reports = [
        _make_report(
            recall=0.1,
            precision=0.05,
            f1_score=0.05,
            mrr=0.0,
            ndcg=0.0,
            map_score=0.0,
        )
    ]
    violations = check_gate(reports)
    assert len(violations) > 0
    violated_metrics = {v.metric for v in violations}
    assert "recall" in violated_metrics
    assert "f1_score" in violated_metrics
    assert "mrr" in violated_metrics


def test_check_gate_custom_thresholds() -> None:
    reports = [
        _make_report(
            recall=0.5,
            precision=0.4,
            f1_score=0.4,
            mrr=0.3,
            ndcg=0.4,
            map_score=0.4,
        )
    ]
    # With lower thresholds, should pass
    thresholds = QualityThresholds(
        min_recall=0.4,
        min_precision=0.3,
        min_f1=0.3,
        min_mrr=0.2,
        min_ndcg=0.3,
        min_map=0.3,
    )
    violations = check_gate(reports, thresholds)
    assert violations == []


def test_check_gate_token_efficiency_violation() -> None:
    reports = [_make_report()]
    thresholds = QualityThresholds(min_token_efficiency=0.5)
    violations = check_gate(reports, thresholds)
    violated_metrics = {v.metric for v in violations}
    assert "token_efficiency" in violated_metrics


def test_check_gate_token_efficiency_default_passes() -> None:
    """Default min_token_efficiency=0.0 never triggers a violation."""
    reports = [_make_report()]
    violations = check_gate(reports)
    violated_metrics = {v.metric for v in violations}
    assert "token_efficiency" not in violated_metrics


def _make_report_for_strategy(strategy: Strategy, recall: float = 0.1) -> BenchmarkReport:
    result = BenchmarkResult(
        task_id="test_task",
        strategy=strategy,
        tokens_total=1000,
        tool_calls=1,
        files_accessed=3,
        recall=recall,
        precision=0.05,
        f1_score=0.05,
        mrr=0.0,
        savings_vs_raw=0.0,
        wall_time_ms=100.0,
        cached=False,
        timestamp="2026-01-01T00:00:00Z",
    )
    return BenchmarkReport(
        task_id="test_task",
        repo="test/repo",
        question="test question",
        results=[result],
        baseline_tokens=2000,
    )


def test_check_gate_exempt_strategies_skipped() -> None:
    """Strategies in gate_exempt_strategies produce no violations even when below threshold."""
    for strategy in (
        Strategy.RAW_FILES,
        Strategy.RAW_GREPPED,
        Strategy.ARCHEX_QUERY_VECTOR,
        Strategy.ARCHEX_SYMBOL_LOOKUP,
    ):
        reports = [_make_report_for_strategy(strategy, recall=0.0)]
        violations = check_gate(reports)
        assert violations == [], f"Expected no violations for exempt strategy {strategy}"


def test_check_gate_non_exempt_strategy_still_checked() -> None:
    """Non-exempt strategies (e.g. archex_query) are still checked."""
    reports = [_make_report_for_strategy(Strategy.ARCHEX_QUERY, recall=0.0)]
    violations = check_gate(reports)
    assert any(v.metric == "recall" for v in violations)


def test_check_gate_strategy_thresholds_override() -> None:
    """Per-strategy threshold overrides apply instead of the default."""
    reports = [_make_report(recall=0.4, precision=0.4, f1_score=0.4, mrr=0.4)]
    # Default thresholds would flag recall=0.4 (min 0.60) and mrr=0.4 (min 0.55)
    per_strategy = QualityThresholds(
        min_recall=0.3,
        min_precision=0.3,
        min_f1=0.3,
        min_mrr=0.3,
    )
    thresholds = QualityThresholds(
        strategy_thresholds={"archex_query": per_strategy},
    )
    violations = check_gate(reports, thresholds)
    assert violations == []


def test_check_gate_custom_exempt_set() -> None:
    """A custom gate_exempt_strategies set overrides the default."""
    reports = [_make_report(recall=0.0, precision=0.0, f1_score=0.0, mrr=0.0)]
    thresholds = QualityThresholds(
        gate_exempt_strategies={"archex_query"},
    )
    violations = check_gate(reports, thresholds)
    assert violations == []
