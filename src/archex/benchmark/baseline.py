"""Baseline snapshot: save, load, and compare benchmark baselines for regression detection."""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path

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


class RankingSnapshotEntry(BaseModel):
    """Per-file PageRank centrality and symbol count for ranking-stability gating."""

    file_path: str
    centrality: float
    symbol_count: int


class Baseline(BaseModel):
    entries: list[BaselineEntry] = []
    ranking: list[RankingSnapshotEntry] = []
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


def build_ranking_snapshot(repo_root: Path) -> list[RankingSnapshotEntry]:
    """Index *repo_root* and snapshot per-file PageRank centrality and symbol count.

    Used to populate ``Baseline.ranking`` and, later, to compare a promotion
    candidate's index against that baseline for ranking-stability regressions
    (see ``archex.benchmark.gate.check_ranking_stability``). Files with no import
    edges get centrality ``0.0``; every indexed file gets its chunk-derived
    symbol count regardless of edges.
    """
    from archex.api import index_repository
    from archex.config import load_config
    from archex.index.graph import DependencyGraph
    from archex.models import RepoSource

    source = RepoSource(local_path=str(repo_root))
    config = load_config(source)
    store = index_repository(source, config=config)
    try:
        centrality = DependencyGraph.from_edges(store.get_edges()).structural_centrality()
        file_metadata = store.get_file_metadata()
    finally:
        store.close()

    return [
        RankingSnapshotEntry(
            file_path=str(item["file_path"]),
            centrality=centrality.get(str(item["file_path"]), 0.0),
            symbol_count=int(item["symbol_count"]),
        )
        for item in file_metadata
    ]


_METRICS = ("recall", "precision", "f1_score", "mrr", "ndcg", "map_score", "token_efficiency")
_DEFAULT_TOLERANCE = 0.05


def compare_baseline(
    reports: list[BenchmarkReport],
    baseline: Baseline,
    tolerance: float = _DEFAULT_TOLERANCE,
    excluded_strategies: Collection[str] = (),
) -> list[BaselineComparison]:
    """Compare current reports against a baseline. Flag regressions beyond tolerance."""
    baseline_lookup: dict[tuple[str, str], BaselineEntry] = {
        (e.task_id, e.strategy): e for e in baseline.entries
    }
    comparisons: list[BaselineComparison] = []
    for report in reports:
        for r in report.results:
            strategy = r.strategy.value
            if strategy in excluded_strategies:
                continue
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
