"""Quality gate: check benchmark results against minimum thresholds."""

from __future__ import annotations

from pydantic import BaseModel

from archex.benchmark.models import (  # noqa: TCH001 — Pydantic needs at runtime
    BenchmarkReport,
    DeltaBenchmarkResult,
)

# Tier 2.5 product-default `archex_query` minimum higher-is-better token
# efficiency recalculates to 0.095. The operational floor rounds down with
# headroom for token-count drift while still failing bundles with <8% savings.
PRODUCT_DEFAULT_STRATEGY = "archex_query"
PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR = 0.08


class QualityThresholds(BaseModel):
    min_recall: float = 0.60
    min_precision: float = 0.20
    min_f1: float = 0.30
    min_mrr: float = 0.55
    min_ndcg: float = 0.0
    min_map: float = 0.0
    min_token_efficiency: float = 0.0
    product_default_strategy: str = PRODUCT_DEFAULT_STRATEGY
    product_default_min_token_efficiency: float = PRODUCT_DEFAULT_TOKEN_EFFICIENCY_FLOOR
    min_required_file_recall: float = 0.0
    max_missed_required_task_rate: float = 1.0
    min_receipt_accuracy: float = 0.0
    min_token_efficiency_with_completion: float = 0.0
    max_completion_efficiency_regression: float = 0.0
    # Optional region/line retrieval-quality thresholds. Defaults are permissive
    # so they never fire unless explicitly configured. Results whose optional
    # region metric is None (no expected-region labels) are always ignored.
    min_region_recall: float = 0.0
    min_line_recall: float = 0.0
    max_context_noise_ratio: float = 1.0
    min_relevance_per_1k_tokens: float = 0.0
    # Latency: warn-only, does not fail the gate
    warn_latency_ms: float = 5000.0
    # Strategies exempt from gate checks (results are informational only)
    gate_exempt_strategies: set[str] = {
        "raw_files",
        "raw_grepped",
        "raw_ripgrep",
        "archex_query_vector",
    }
    # Per-strategy threshold overrides; keyed by strategy value string
    strategy_thresholds: dict[str, QualityThresholds] = {}


class GateViolation(BaseModel):
    task_id: str
    strategy: str
    metric: str
    threshold: float
    actual: float


class BaselineGateViolation(BaseModel):
    task_id: str
    strategy: str
    metric: str
    baseline: float
    actual: float


class LatencyWarning(BaseModel):
    task_id: str
    strategy: str
    threshold_ms: float
    actual_ms: float


def _minimum_gate_checks(t: QualityThresholds, strategy: str) -> list[tuple[str, float]]:
    token_efficiency_floor = (
        t.product_default_min_token_efficiency
        if strategy == t.product_default_strategy
        else t.min_token_efficiency
    )
    return [
        ("recall", t.min_recall),
        ("precision", t.min_precision),
        ("f1_score", t.min_f1),
        ("mrr", t.min_mrr),
        ("ndcg", t.min_ndcg),
        ("map_score", t.min_map),
        ("token_efficiency", token_efficiency_floor),
        ("required_file_recall", t.min_required_file_recall),
        ("token_efficiency_with_completion", t.min_token_efficiency_with_completion),
    ]


def _maximum_gate_checks(t: QualityThresholds) -> list[tuple[str, float]]:
    return [("missed_required_task_rate", t.max_missed_required_task_rate)]


def _optional_minimum_gate_checks(t: QualityThresholds) -> list[tuple[str, float]]:
    """Minimum checks for optional region metrics; skipped when the value is None."""
    return [
        ("region_recall", t.min_region_recall),
        ("line_recall", t.min_line_recall),
        ("relevance_per_1k_tokens", t.min_relevance_per_1k_tokens),
    ]


def _optional_maximum_gate_checks(t: QualityThresholds) -> list[tuple[str, float]]:
    """Maximum checks for optional region metrics; skipped when the value is None."""
    return [("context_noise_ratio", t.max_context_noise_ratio)]


def _receipt_accuracy_score(value: bool | None) -> float:
    return 0.0 if value is None else 1.0 if value else 0.0


def check_gate(
    reports: list[BenchmarkReport],
    thresholds: QualityThresholds | None = None,
) -> list[GateViolation]:
    """Check all results against quality thresholds. Returns list of violations.

    Results whose strategy is in ``thresholds.gate_exempt_strategies`` are
    skipped entirely. When ``thresholds.strategy_thresholds`` contains an entry
    for a strategy, those per-strategy thresholds are used instead of the
    default ones.
    """
    if thresholds is None:
        thresholds = QualityThresholds()

    violations: list[GateViolation] = []
    for report in reports:
        for r in report.results:
            strategy_val = r.strategy.value
            if strategy_val in thresholds.gate_exempt_strategies:
                continue
            effective = thresholds.strategy_thresholds.get(strategy_val, thresholds)
            for metric_name, threshold_val in _minimum_gate_checks(effective, strategy_val):
                actual = getattr(r, metric_name)
                if actual < threshold_val:
                    violations.append(
                        GateViolation(
                            task_id=r.task_id,
                            strategy=strategy_val,
                            metric=metric_name,
                            threshold=threshold_val,
                            actual=actual,
                        )
                    )
            for metric_name, threshold_val in _maximum_gate_checks(effective):
                actual = getattr(r, metric_name)
                if actual > threshold_val:
                    violations.append(
                        GateViolation(
                            task_id=r.task_id,
                            strategy=strategy_val,
                            metric=metric_name,
                            threshold=threshold_val,
                            actual=actual,
                        )
                    )
            actual = _receipt_accuracy_score(r.receipt_accuracy)
            if actual < effective.min_receipt_accuracy:
                violations.append(
                    GateViolation(
                        task_id=r.task_id,
                        strategy=strategy_val,
                        metric="receipt_accuracy",
                        threshold=effective.min_receipt_accuracy,
                        actual=actual,
                    )
                )
            for metric_name, threshold_val in _optional_minimum_gate_checks(effective):
                actual = getattr(r, metric_name)
                if actual is None:
                    continue
                if actual < threshold_val:
                    violations.append(
                        GateViolation(
                            task_id=r.task_id,
                            strategy=strategy_val,
                            metric=metric_name,
                            threshold=threshold_val,
                            actual=actual,
                        )
                    )
            for metric_name, threshold_val in _optional_maximum_gate_checks(effective):
                actual = getattr(r, metric_name)
                if actual is None:
                    continue
                if actual > threshold_val:
                    violations.append(
                        GateViolation(
                            task_id=r.task_id,
                            strategy=strategy_val,
                            metric=metric_name,
                            threshold=threshold_val,
                            actual=actual,
                        )
                    )
    return violations


def check_recall_regressions(
    reports: list[BenchmarkReport],
    baseline_reports: list[BenchmarkReport],
    thresholds: QualityThresholds | None = None,
) -> list[BaselineGateViolation]:
    """Return recall regressions against a baseline run.

    Baseline-aware gating protects the retrieval contract without treating
    every accepted low-signal benchmark row as a hard absolute floor.
    """
    if thresholds is None:
        thresholds = QualityThresholds()

    baseline_by_result = {
        (report.task_id, result.strategy.value): result
        for report in baseline_reports
        for result in report.results
    }
    violations: list[BaselineGateViolation] = []
    for report in reports:
        for result in report.results:
            strategy_val = result.strategy.value
            if strategy_val in thresholds.gate_exempt_strategies:
                continue
            baseline = baseline_by_result.get((report.task_id, strategy_val))
            if baseline is None:
                violations.append(
                    BaselineGateViolation(
                        task_id=report.task_id,
                        strategy=strategy_val,
                        metric="baseline_missing",
                        baseline=1.0,
                        actual=0.0,
                    )
                )
                continue
            if result.recall < baseline.recall:
                violations.append(
                    BaselineGateViolation(
                        task_id=report.task_id,
                        strategy=strategy_val,
                        metric="recall",
                        baseline=baseline.recall,
                        actual=result.recall,
                    )
                )
            completion_efficiency_floor = (
                baseline.token_efficiency_with_completion
                - thresholds.max_completion_efficiency_regression
            )
            if result.token_efficiency_with_completion < completion_efficiency_floor:
                violations.append(
                    BaselineGateViolation(
                        task_id=report.task_id,
                        strategy=strategy_val,
                        metric="token_efficiency_with_completion",
                        baseline=completion_efficiency_floor,
                        actual=result.token_efficiency_with_completion,
                    )
                )
    return violations


def token_efficiency_violations(violations: list[GateViolation]) -> list[GateViolation]:
    """Return hard token-efficiency violations from absolute gate checks."""
    return [violation for violation in violations if violation.metric == "token_efficiency"]


def non_token_quality_warnings(violations: list[GateViolation]) -> list[GateViolation]:
    """Return advisory non-token absolute-threshold warnings."""
    return [violation for violation in violations if violation.metric != "token_efficiency"]


def check_latency_warnings(
    reports: list[BenchmarkReport],
    thresholds: QualityThresholds | None = None,
) -> list[LatencyWarning]:
    """Return latency warnings for results exceeding warn_latency_ms. Does not fail the gate."""
    if thresholds is None:
        thresholds = QualityThresholds()

    warnings: list[LatencyWarning] = []
    for report in reports:
        for r in report.results:
            if r.wall_time_ms > thresholds.warn_latency_ms:
                warnings.append(
                    LatencyWarning(
                        task_id=r.task_id,
                        strategy=r.strategy.value,
                        threshold_ms=thresholds.warn_latency_ms,
                        actual_ms=r.wall_time_ms,
                    )
                )
    return warnings


# ---------------------------------------------------------------------------
# Delta quality gate
# ---------------------------------------------------------------------------


class DeltaQualityThresholds(BaseModel):
    min_speedup: float = 1.5
    require_correctness: bool = True


class DeltaGateViolation(BaseModel):
    task_id: str
    metric: str
    threshold: float
    actual: float


def check_delta_gate(
    results: list[DeltaBenchmarkResult],
    thresholds: DeltaQualityThresholds | None = None,
) -> list[DeltaGateViolation]:
    """Check delta benchmark results against quality thresholds. Returns violations."""
    if thresholds is None:
        thresholds = DeltaQualityThresholds()

    violations: list[DeltaGateViolation] = []
    for r in results:
        if thresholds.require_correctness and not r.correctness:
            violations.append(
                DeltaGateViolation(
                    task_id=r.task_id,
                    metric="correctness",
                    threshold=1.0,
                    actual=0.0,
                )
            )
        if r.speedup_factor < thresholds.min_speedup:
            violations.append(
                DeltaGateViolation(
                    task_id=r.task_id,
                    metric="speedup_factor",
                    threshold=thresholds.min_speedup,
                    actual=r.speedup_factor,
                )
            )
    return violations
