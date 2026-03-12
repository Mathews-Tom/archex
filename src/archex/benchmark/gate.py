"""Quality gate: check benchmark results against minimum thresholds."""

from __future__ import annotations

from pydantic import BaseModel

from archex.benchmark.models import (  # noqa: TCH001 — Pydantic needs at runtime
    BenchmarkReport,
    DeltaBenchmarkResult,
)


class QualityThresholds(BaseModel):
    min_recall: float = 0.60
    min_precision: float = 0.20
    min_f1: float = 0.30
    min_mrr: float = 0.55
    min_ndcg: float = 0.0
    min_map: float = 0.0
    min_token_efficiency: float = 0.0
    # Latency: warn-only, does not fail the gate
    warn_latency_ms: float = 5000.0
    # Strategies exempt from gate checks (results are informational only)
    gate_exempt_strategies: set[str] = {
        "raw_files",
        "raw_grepped",
        "archex_query_vector",
        "archex_symbol_lookup",
    }
    # Per-strategy threshold overrides; keyed by strategy value string
    strategy_thresholds: dict[str, QualityThresholds] = {}


class GateViolation(BaseModel):
    task_id: str
    strategy: str
    metric: str
    threshold: float
    actual: float


class LatencyWarning(BaseModel):
    task_id: str
    strategy: str
    threshold_ms: float
    actual_ms: float


def _gate_checks(t: QualityThresholds) -> list[tuple[str, float]]:
    return [
        ("recall", t.min_recall),
        ("precision", t.min_precision),
        ("f1_score", t.min_f1),
        ("mrr", t.min_mrr),
        ("ndcg", t.min_ndcg),
        ("map_score", t.min_map),
        ("token_efficiency", t.min_token_efficiency),
    ]


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
            for metric_name, threshold_val in _gate_checks(effective):
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
    return violations


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
