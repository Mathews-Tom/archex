"""Thin helpers that convert existing CLI/MCP outputs into UsageEvent inputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from archex.metrics.recorder import MetricsRecorder, Surface, TraceDetails, UsageEvent

if TYPE_CHECKING:
    from archex.models import ContextBundle, ContextSkippedCandidate, RepoSource
    from archex.scout import ScoutResult


def record_query_usage(
    source: RepoSource,
    bundle: ContextBundle,
    *,
    surface: Surface,
    tokens_raw_equivalent: int,
    whole_repo_tokens: int | None,
    tool_name: str = "query",
    db_path: Path | None = None,
) -> None:
    repo_root = _local_repo_root(source)
    if repo_root is None:
        return
    files = sorted({chunk.chunk.file_path for chunk in bundle.chunks})
    symbols = sorted(
        {chunk.chunk.symbol_name for chunk in bundle.chunks if chunk.chunk.symbol_name}
    )
    receipt = bundle.receipt
    MetricsRecorder(db_path).record(
        UsageEvent(
            repo_root=repo_root,
            surface=surface,
            tool_name=tool_name,
            category="context_retrieval",
            tokens_returned=bundle.token_count,
            tokens_raw_equivalent=tokens_raw_equivalent,
            whole_repo_tokens=whole_repo_tokens,
            file_count=len(files),
            freshness=str(receipt.freshness) if receipt is not None else None,
            index_revision=receipt.index_revision if receipt is not None else None,
            trace=TraceDetails(
                query_text=bundle.query,
                returned_file_paths=files,
                symbols=symbols,
                skipped_counts=_skipped_counts(receipt.skipped_candidates if receipt else []),
            ),
        )
    )


def record_scout_usage(
    source: RepoSource,
    result: ScoutResult,
    *,
    surface: Surface,
    tokens_returned: int,
    tokens_raw_equivalent: int,
    whole_repo_tokens: int | None,
    tool_name: str = "scout",
    db_path: Path | None = None,
) -> None:
    repo_root = _local_repo_root(source)
    if repo_root is None:
        return
    receipt = result.receipt
    MetricsRecorder(db_path).record(
        UsageEvent(
            repo_root=repo_root,
            surface=surface,
            tool_name=tool_name,
            category="context_retrieval",
            tokens_returned=tokens_returned,
            tokens_raw_equivalent=tokens_raw_equivalent,
            whole_repo_tokens=whole_repo_tokens,
            file_count=len(result.ranked_files),
            freshness=str(receipt.freshness) if receipt is not None else None,
            index_revision=receipt.index_revision if receipt is not None else None,
            trace=TraceDetails(
                query_text=result.query,
                returned_file_paths=[item.path for item in result.ranked_files],
                symbols=[item.name for item in result.symbols],
                handles=list(result.fetch_plan.handles),
                skipped_counts={
                    "omitted_files": result.budget.omitted_files,
                    "omitted_symbols": result.budget.omitted_symbols,
                    "omitted_modules": result.budget.omitted_modules,
                    "omitted_graph_edges": result.budget.omitted_graph_edges,
                },
            ),
        )
    )


def _local_repo_root(source: RepoSource) -> Path | None:
    if source.local_path is None:
        return None
    return Path(source.local_path)


def _skipped_counts(candidates: list[ContextSkippedCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        reason = str(candidate.reason)
        counts[reason] = counts.get(reason, 0) + 1
    return counts
