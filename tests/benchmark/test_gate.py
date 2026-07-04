"""Tests for quality gate checks."""

from __future__ import annotations

from archex.benchmark.baseline import RankingSnapshotEntry
from archex.benchmark.gate import (
    PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR,
    QualityThresholds,
    RankingQualityThresholds,
    check_gate,
    check_ranking_stability,
    check_recall_regressions,
    non_token_quality_warnings,
    token_efficiency_violations,
)
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, Strategy


def _make_report(
    recall: float = 0.8,
    precision: float = 0.5,
    f1_score: float = 0.6,
    mrr: float = 0.7,
    ndcg: float = 0.7,
    map_score: float = 0.6,
    token_efficiency: float = PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR,
    required_file_recall: float = 1.0,
    missed_required_task_rate: float = 0.0,
    receipt_accuracy: bool | None = True,
    token_efficiency_with_completion: float = PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR,
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
        token_efficiency=token_efficiency,
        required_file_recall=required_file_recall,
        missed_required_task_rate=missed_required_task_rate,
        receipt_accuracy=receipt_accuracy,
        token_efficiency_with_completion=token_efficiency_with_completion,
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


def test_check_gate_token_efficiency_floor_fails_bloat() -> None:
    reports = [_make_report(token_efficiency=PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR - 0.001)]
    violations = check_gate(reports)
    assert [
        (v.metric, v.threshold, v.actual) for v in violations if v.metric == "token_efficiency"
    ] == [
        (
            "token_efficiency",
            PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR,
            PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR - 0.001,
        )
    ]


def test_check_gate_token_efficiency_floor_passes_at_floor() -> None:
    reports = [_make_report(token_efficiency=PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR)]
    violations = check_gate(reports)
    violated_metrics = {v.metric for v in violations}
    assert "token_efficiency" not in violated_metrics


def test_check_gate_trust_metrics_can_fail() -> None:
    reports = [
        _make_report(
            required_file_recall=0.75,
            missed_required_task_rate=1.0,
            receipt_accuracy=False,
            token_efficiency_with_completion=0.02,
        )
    ]
    thresholds = QualityThresholds(
        min_required_file_recall=0.9,
        max_missed_required_task_rate=0.0,
        min_receipt_accuracy=1.0,
        min_token_efficiency_with_completion=0.05,
    )

    violations = check_gate(reports, thresholds)

    assert {violation.metric for violation in violations} == {
        "required_file_recall",
        "missed_required_task_rate",
        "receipt_accuracy",
        "token_efficiency_with_completion",
    }


def test_check_gate_missing_receipt_accuracy_fails_when_required() -> None:
    reports = [_make_report(receipt_accuracy=None)]
    thresholds = QualityThresholds(min_receipt_accuracy=1.0)

    violations = check_gate(reports, thresholds)

    assert [(v.metric, v.threshold, v.actual) for v in violations] == [
        ("receipt_accuracy", 1.0, 0.0)
    ]


def test_baseline_gate_flags_completion_efficiency_regression() -> None:
    baseline = [_make_report(token_efficiency_with_completion=0.6)]
    current = [_make_report(token_efficiency_with_completion=0.5)]

    violations = check_recall_regressions(current, baseline)

    assert [(v.metric, v.baseline, v.actual) for v in violations] == [
        ("token_efficiency_with_completion", 0.6, 0.5)
    ]


def test_baseline_gate_flags_recall_regression() -> None:
    baseline = [_make_report(recall=0.8)]
    current = [_make_report(recall=0.6)]

    violations = check_recall_regressions(current, baseline)

    assert [(v.metric, v.baseline, v.actual) for v in violations] == [("recall", 0.8, 0.6)]


def test_baseline_gate_allows_non_recall_rank_drop() -> None:
    baseline = [_make_report(recall=0.6, mrr=1.0)]
    current = [_make_report(recall=0.7, mrr=0.3)]

    violations = check_recall_regressions(current, baseline)

    assert violations == []


def test_baseline_gate_reports_missing_baseline_result() -> None:
    baseline = [_make_report_for_strategy(Strategy.ARCHEX_QUERY_FUSION)]
    current = [_make_report_for_strategy(Strategy.ARCHEX_QUERY, recall=0.7)]

    violations = check_recall_regressions(current, baseline)

    assert [(v.strategy, v.metric) for v in violations] == [("archex_query", "baseline_missing")]


def test_absolute_gate_separates_token_failures_from_quality_warnings() -> None:
    reports = [
        _make_report(
            recall=0.1,
            token_efficiency=PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR - 0.001,
        )
    ]

    violations = check_gate(reports)

    assert [v.metric for v in token_efficiency_violations(violations)] == ["token_efficiency"]
    assert {v.metric for v in non_token_quality_warnings(violations)} == {"recall"}


def _make_report_for_strategy(
    strategy: Strategy,
    recall: float = 0.1,
    precision: float = 0.05,
    f1_score: float = 0.05,
    mrr: float = 0.0,
    token_efficiency: float = 0.0,
    token_efficiency_with_completion: float = 0.0,
) -> BenchmarkReport:
    result = BenchmarkResult(
        task_id="test_task",
        strategy=strategy,
        tokens_total=1000,
        tool_calls=1,
        files_accessed=3,
        recall=recall,
        precision=precision,
        f1_score=f1_score,
        mrr=mrr,
        savings_vs_raw=0.0,
        token_efficiency=token_efficiency,
        token_efficiency_with_completion=token_efficiency_with_completion,
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
        Strategy.RAW_RIPGREP,
        Strategy.ARCHEX_QUERY_VECTOR,
    ):
        reports = [_make_report_for_strategy(strategy, recall=0.0)]
        violations = check_gate(reports)
        assert violations == [], f"Expected no violations for exempt strategy {strategy}"


def test_check_gate_non_exempt_strategy_still_checked() -> None:
    """Non-exempt strategies (e.g. archex_query) are still checked."""
    reports = [_make_report_for_strategy(Strategy.ARCHEX_QUERY, recall=0.0)]
    violations = check_gate(reports)
    assert any(v.metric == "recall" for v in violations)


def test_check_gate_product_token_floor_only_applies_to_product_default() -> None:
    reports = [
        _make_report_for_strategy(
            Strategy.ARCHEX_QUERY_FUSION,
            recall=0.8,
            precision=0.5,
            f1_score=0.6,
            mrr=0.7,
            token_efficiency=PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR - 0.02,
        )
    ]
    violations = check_gate(reports)
    assert {v.metric for v in violations} == set()


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


_REGION_METRICS = {
    "region_recall",
    "line_recall",
    "context_noise_ratio",
    "relevance_per_1k_tokens",
}


def _region_report(**fields: float) -> BenchmarkReport:
    report = _make_report()
    report.results[0] = report.results[0].model_copy(update=fields)
    return report


def test_gate_ignores_absent_region_labels() -> None:
    # File-only result (region fields None): strict region thresholds must not fire.
    reports = [_make_report()]
    thresholds = QualityThresholds(
        min_region_recall=0.9,
        min_line_recall=0.9,
        max_context_noise_ratio=0.1,
        min_relevance_per_1k_tokens=100.0,
    )
    violations = check_gate(reports, thresholds)
    assert not any(v.metric in _REGION_METRICS for v in violations)


def test_gate_default_thresholds_pass_region_labeled_result() -> None:
    reports = [
        _region_report(
            region_recall=0.5,
            line_recall=0.5,
            context_noise_ratio=0.5,
            relevance_per_1k_tokens=5.0,
        )
    ]
    violations = check_gate(reports)
    assert not any(v.metric in _REGION_METRICS for v in violations)


def test_gate_fails_region_recall_below_threshold() -> None:
    reports = [
        _region_report(
            region_recall=0.2,
            line_recall=0.5,
            context_noise_ratio=0.3,
            relevance_per_1k_tokens=5.0,
        )
    ]
    thresholds = QualityThresholds(min_region_recall=0.5)
    violations = check_gate(reports, thresholds)
    assert any(v.metric == "region_recall" and v.actual == 0.2 for v in violations)


def test_gate_fails_high_context_noise_ratio() -> None:
    reports = [
        _region_report(
            region_recall=0.8,
            line_recall=0.8,
            context_noise_ratio=0.9,
            relevance_per_1k_tokens=5.0,
        )
    ]
    thresholds = QualityThresholds(max_context_noise_ratio=0.5)
    violations = check_gate(reports, thresholds)
    assert any(v.metric == "context_noise_ratio" and v.actual == 0.9 for v in violations)


def test_gate_region_violations_are_advisory_warnings() -> None:
    reports = [
        _region_report(
            region_recall=0.2,
            line_recall=0.5,
            context_noise_ratio=0.3,
            relevance_per_1k_tokens=5.0,
        )
    ]
    thresholds = QualityThresholds(min_region_recall=0.5)
    violations = check_gate(reports, thresholds)
    # Region failures are non-token quality warnings, not hard token failures.
    assert any(v.metric == "region_recall" for v in non_token_quality_warnings(violations))
    assert token_efficiency_violations(violations) == []


def test_check_ranking_stability_passes_when_unchanged() -> None:
    entries = [
        RankingSnapshotEntry(file_path="src/a.py", centrality=0.1, symbol_count=5),
        RankingSnapshotEntry(file_path="src/b.py", centrality=0.4, symbol_count=12),
        RankingSnapshotEntry(file_path="src/c.py", centrality=0.9, symbol_count=3),
        RankingSnapshotEntry(file_path="src/d.py", centrality=0.2, symbol_count=40),
        RankingSnapshotEntry(file_path="src/e.py", centrality=0.6, symbol_count=8),
    ]
    assert check_ranking_stability(entries, entries) == []


def test_check_ranking_stability_detects_symbol_flooding_regression() -> None:
    """A symbol-flooding conversion reorders symbol_count ranks while leaving
    structural_centrality untouched — proving the two metrics are checked
    independently and that the gate has teeth.

    This class of regression is invisible to recall/precision/F1/MRR: it never
    touches a `BenchmarkReport`/`BenchmarkResult` at all, and `RankingSnapshotEntry`
    (the only type flowing through `check_ranking_stability`) carries no
    retrieval-quality fields whatsoever. A baseline comparison over the same
    underlying retrieval results (`compare_baseline`) would report zero drift
    for this exact scenario, so `violations != []` below is the only signal
    that could ever catch it — a no-op stand-in for `check_ranking_stability`
    (always returning `[]`) would silently pass this regression through.
    """
    baseline = [
        RankingSnapshotEntry(file_path="src/a.py", centrality=0.05, symbol_count=10),
        RankingSnapshotEntry(file_path="src/b.py", centrality=0.30, symbol_count=20),
        RankingSnapshotEntry(file_path="src/c.py", centrality=0.10, symbol_count=30),
        RankingSnapshotEntry(file_path="src/d.py", centrality=0.50, symbol_count=40),
        RankingSnapshotEntry(file_path="src/e.py", centrality=0.20, symbol_count=50),
        RankingSnapshotEntry(file_path="src/f.py", centrality=0.40, symbol_count=60),
    ]
    # Simulate an overzealous chunk-only -> full conversion that floods the two
    # lowest-symbol_count files with many new low-value symbols. Every file's
    # structural centrality (PageRank over import edges) is left unchanged, so
    # only the symbol_count ranking should regress.
    current = [
        RankingSnapshotEntry(file_path="src/a.py", centrality=0.05, symbol_count=10_000),
        RankingSnapshotEntry(file_path="src/b.py", centrality=0.30, symbol_count=8_000),
        RankingSnapshotEntry(file_path="src/c.py", centrality=0.10, symbol_count=30),
        RankingSnapshotEntry(file_path="src/d.py", centrality=0.50, symbol_count=40),
        RankingSnapshotEntry(file_path="src/e.py", centrality=0.20, symbol_count=50),
        RankingSnapshotEntry(file_path="src/f.py", centrality=0.40, symbol_count=60),
    ]

    violations = check_ranking_stability(current, baseline)

    symbol_violations = [v for v in violations if v.metric == "symbol_count"]
    assert symbol_violations, "symbol-flooding must reorder the symbol_count ranking"
    assert (
        symbol_violations[0].correlation
        < RankingQualityThresholds().min_symbol_count_rank_correlation
    )
    assert not any(v.metric == "structural_centrality" for v in violations)


def test_check_ranking_stability_ignores_files_absent_from_one_snapshot() -> None:
    baseline = [RankingSnapshotEntry(file_path="src/shared.py", centrality=0.1, symbol_count=5)]
    current = [RankingSnapshotEntry(file_path="src/shared.py", centrality=0.9, symbol_count=500)]
    # Only one file is common to both snapshots: correlation is undefined below
    # two points, so no violation can be raised even though the shared file's
    # values differ wildly.
    assert check_ranking_stability(current, baseline) == []


def test_check_ranking_stability_respects_custom_thresholds() -> None:
    baseline = [
        RankingSnapshotEntry(file_path="src/a.py", centrality=0.1, symbol_count=10),
        RankingSnapshotEntry(file_path="src/b.py", centrality=0.2, symbol_count=20),
        RankingSnapshotEntry(file_path="src/c.py", centrality=0.3, symbol_count=30),
        RankingSnapshotEntry(file_path="src/d.py", centrality=0.4, symbol_count=40),
        RankingSnapshotEntry(file_path="src/e.py", centrality=0.5, symbol_count=50),
    ]
    current = [
        RankingSnapshotEntry(file_path="src/a.py", centrality=0.1, symbol_count=10),
        RankingSnapshotEntry(file_path="src/b.py", centrality=0.2, symbol_count=20),
        RankingSnapshotEntry(file_path="src/c.py", centrality=0.3, symbol_count=50),
        RankingSnapshotEntry(file_path="src/d.py", centrality=0.4, symbol_count=40),
        RankingSnapshotEntry(file_path="src/e.py", centrality=0.5, symbol_count=30),
    ]
    # A mild symbol_count reorder (Spearman rho == 0.6) fails the default 0.8
    # floor...
    default_violations = check_ranking_stability(current, baseline)
    assert any(v.metric == "symbol_count" for v in default_violations)

    # ...but clears a lowered custom floor.
    lenient = RankingQualityThresholds(
        min_centrality_rank_correlation=0.0,
        min_symbol_count_rank_correlation=0.0,
    )
    assert check_ranking_stability(current, baseline, lenient) == []


def test_check_ranking_stability_zero_variance_metric_skipped() -> None:
    baseline = [
        RankingSnapshotEntry(file_path="src/a.py", centrality=0.1, symbol_count=10),
        RankingSnapshotEntry(file_path="src/b.py", centrality=0.2, symbol_count=10),
        RankingSnapshotEntry(file_path="src/c.py", centrality=0.3, symbol_count=10),
        RankingSnapshotEntry(file_path="src/d.py", centrality=0.4, symbol_count=10),
        RankingSnapshotEntry(file_path="src/e.py", centrality=0.5, symbol_count=10),
    ]
    # symbol_count is constant (10) in both snapshots, so its correlation is
    # undefined and must never fire — even though centrality is fully inverted
    # (a genuine regression) and must fire. Proves a zero-variance metric can't
    # crash the gate or mask a real regression on the other metric.
    current = [
        RankingSnapshotEntry(file_path="src/a.py", centrality=0.5, symbol_count=10),
        RankingSnapshotEntry(file_path="src/b.py", centrality=0.4, symbol_count=10),
        RankingSnapshotEntry(file_path="src/c.py", centrality=0.3, symbol_count=10),
        RankingSnapshotEntry(file_path="src/d.py", centrality=0.2, symbol_count=10),
        RankingSnapshotEntry(file_path="src/e.py", centrality=0.1, symbol_count=10),
    ]

    violations = check_ranking_stability(current, baseline)

    assert [v.metric for v in violations] == ["structural_centrality"]
