"""Quality gate: check benchmark results against minimum thresholds."""

from __future__ import annotations

from pydantic import BaseModel

from archex.benchmark.models import (  # noqa: TCH001 — Pydantic needs at runtime
    BenchmarkReport,
    DeltaBenchmarkResult,
)


class QualityThresholds(BaseModel):
    min_recall: float = 0.6
    min_precision: float = 0.3
    min_f1: float = 0.4
    min_mrr: float = 0.3


class GateViolation(BaseModel):
    task_id: str
    strategy: str
    metric: str
    threshold: float
    actual: float


def check_gate(
    reports: list[BenchmarkReport],
    thresholds: QualityThresholds | None = None,
) -> list[GateViolation]:
    """Check all results against quality thresholds. Returns list of violations."""
    if thresholds is None:
        thresholds = QualityThresholds()

    checks = [
        ("recall", thresholds.min_recall),
        ("precision", thresholds.min_precision),
        ("f1_score", thresholds.min_f1),
        ("mrr", thresholds.min_mrr),
    ]

    violations: list[GateViolation] = []
    for report in reports:
        for r in report.results:
            for metric_name, threshold_val in checks:
                actual = getattr(r, metric_name)
                if actual < threshold_val:
                    violations.append(
                        GateViolation(
                            task_id=r.task_id,
                            strategy=r.strategy.value,
                            metric=metric_name,
                            threshold=threshold_val,
                            actual=actual,
                        )
                    )
    return violations


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
