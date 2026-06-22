"""Thin helpers that convert existing CLI/MCP outputs into UsageEvent inputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from archex.metrics.categories import category_for_tool
from archex.metrics.recorder import MetricsRecorder, Surface, TraceDetails, UsageEvent
from archex.reporting import count_tokens

if TYPE_CHECKING:
    from archex.models import (
        CodeChunk,
        ContextBundle,
        ContextSkippedCandidate,
        RankedChunk,
        RepoSource,
    )
    from archex.scout import ScoutResult

_TARGETED_CONTEXT_LINES = 5


def record_query_usage(
    source: RepoSource,
    bundle: ContextBundle,
    *,
    surface: Surface,
    tokens_raw_equivalent: int | None,
    whole_repo_tokens: int | None,
    tool_name: str = "query",
    db_path: Path | None = None,
) -> None:
    repo_root = _local_repo_root(source)
    if repo_root is None:
        return
    if tokens_raw_equivalent is None:
        return
    files = sorted({chunk.chunk.file_path for chunk in bundle.chunks})
    symbols = sorted(
        {chunk.chunk.symbol_name for chunk in bundle.chunks if chunk.chunk.symbol_name}
    )
    targeted_read = _targeted_read_tokens(
        bundle.chunks,
        full_file_tokens=tokens_raw_equivalent,
        returned_tokens=bundle.token_count,
    )
    receipt = bundle.receipt
    MetricsRecorder(db_path).record(
        UsageEvent(
            repo_root=repo_root,
            surface=surface,
            tool_name=tool_name,
            category=category_for_tool(tool_name),
            tokens_returned=bundle.token_count,
            tokens_raw_equivalent=tokens_raw_equivalent,
            whole_repo_tokens=whole_repo_tokens,
            tokens_targeted_read=targeted_read,
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
    tokens_raw_equivalent: int | None,
    whole_repo_tokens: int | None,
    tool_name: str = "scout",
    db_path: Path | None = None,
) -> None:
    repo_root = _local_repo_root(source)
    if repo_root is None:
        return
    if tokens_raw_equivalent is None:
        return
    receipt = result.receipt
    MetricsRecorder(db_path).record(
        UsageEvent(
            repo_root=repo_root,
            surface=surface,
            tool_name=tool_name,
            category=category_for_tool(tool_name),
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


def record_structural_usage(
    source: RepoSource,
    *,
    surface: Surface,
    tool_name: str,
    tokens_returned: int,
    tokens_raw_equivalent: int | None,
    whole_repo_tokens: int | None = None,
    file_count: int = 0,
    db_path: Path | None = None,
) -> None:
    repo_root = _local_repo_root(source)
    if repo_root is None:
        return
    if tokens_raw_equivalent is None:
        return
    MetricsRecorder(db_path).record(
        UsageEvent(
            repo_root=repo_root,
            surface=surface,
            tool_name=tool_name,
            category=category_for_tool(tool_name),
            tokens_returned=tokens_returned,
            tokens_raw_equivalent=tokens_raw_equivalent,
            whole_repo_tokens=whole_repo_tokens,
            file_count=file_count,
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


def _targeted_read_tokens(
    chunks: list[RankedChunk],
    *,
    full_file_tokens: int,
    returned_tokens: int,
    context_lines: int = _TARGETED_CONTEXT_LINES,
) -> int | None:
    """Estimate the realistic targeted-read baseline from returned chunk spans.

    Per returned file, take the union of ``[start_line - K, end_line + K]`` line
    spans (merged, deduped) and estimate their cost from the indexed chunk content's
    per-line token density. Deterministic and index-only: no file-system read and no
    model call. The estimate is clamped to ``<= full_file`` and floored at
    ``returned`` (the lower bound is prioritized), so ``returned <= targeted_read <=
    full_file`` holds for every realistic input where ``returned <= full_file``.
    """
    by_file: dict[str, list[CodeChunk]] = {}
    for ranked in chunks:
        by_file.setdefault(ranked.chunk.file_path, []).append(ranked.chunk)
    if not by_file:
        return None
    estimate = sum(
        _targeted_file_tokens(file_chunks, context_lines) for file_chunks in by_file.values()
    )
    return max(returned_tokens, min(estimate, full_file_tokens))


def _targeted_file_tokens(file_chunks: list[CodeChunk], context_lines: int) -> int:
    line_text: dict[int, str] = {}
    expanded_lines: set[int] = set()
    for chunk in file_chunks:
        start = chunk.start_line
        end = chunk.end_line
        for offset, text in enumerate(chunk.content.split("\n")):
            line_no = start + offset
            if line_no > end:
                break
            line_text[line_no] = text
        expanded_lines.update(range(max(1, start - context_lines), end + context_lines + 1))
    if not line_text:
        return 0
    matched_tokens = count_tokens("\n".join(line_text[ln] for ln in sorted(line_text)))
    density = matched_tokens / len(line_text)
    return round(density * len(expanded_lines))
