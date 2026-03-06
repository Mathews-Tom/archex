"""Baseline snapshot: save, load, and compare benchmark baselines for regression detection."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from archex.benchmark.models import BenchmarkReport  # noqa: TCH001 — Pydantic needs at runtime


class BaselineEntry(BaseModel):
    task_id: str
    strategy: str
    recall: float
    precision: float
    f1_score: float
    mrr: float
    ndcg: float = 0.0
    map_score: float = 0.0
    token_efficiency: float = 0.0


class Baseline(BaseModel):
    entries: list[BaselineEntry] = []
    created_at: str = ""
    archex_version: str = ""


class BaselineComparison(BaseModel):
    task_id: str
    strategy: str
    metric: str
    baseline_value: float
    current_value: float
    delta: float
    regression: bool


def save_baseline(
    reports: list[BenchmarkReport],
    archex_version: str = "",
) -> Baseline:
    """Extract metrics from reports into a Baseline snapshot."""
    entries: list[BaselineEntry] = []
    for report in reports:
        for r in report.results:
            entries.append(
                BaselineEntry(
                    task_id=r.task_id,
                    strategy=r.strategy.value,
                    recall=r.recall,
                    precision=r.precision,
                    f1_score=r.f1_score,
                    mrr=r.mrr,
                    ndcg=r.ndcg,
                    map_score=r.map_score,
                    token_efficiency=r.token_efficiency,
                )
            )
    return Baseline(
        entries=entries,
        created_at=datetime.now(tz=UTC).isoformat(),
        archex_version=archex_version,
    )


def load_baseline(data: dict[str, object]) -> Baseline:
    """Validate and load a baseline from parsed JSON data."""
    return Baseline.model_validate(data)


_METRICS = ("recall", "precision", "f1_score", "mrr", "ndcg", "map_score", "token_efficiency")
_DEFAULT_TOLERANCE = 0.05


def compare_baseline(
    reports: list[BenchmarkReport],
    baseline: Baseline,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> list[BaselineComparison]:
    """Compare current reports against a baseline. Flag regressions beyond tolerance."""
    baseline_lookup: dict[tuple[str, str], BaselineEntry] = {
        (e.task_id, e.strategy): e for e in baseline.entries
    }
    comparisons: list[BaselineComparison] = []
    for report in reports:
        for r in report.results:
            key = (r.task_id, r.strategy.value)
            entry = baseline_lookup.get(key)
            if entry is None:
                continue
            for metric in _METRICS:
                baseline_val = getattr(entry, metric)
                current_val = getattr(r, metric)
                delta = current_val - baseline_val
                regression = current_val < baseline_val - tolerance
                comparisons.append(
                    BaselineComparison(
                        task_id=r.task_id,
                        strategy=r.strategy.value,
                        metric=metric,
                        baseline_value=baseline_val,
                        current_value=current_val,
                        delta=delta,
                        regression=regression,
                    )
                )
    return comparisons
