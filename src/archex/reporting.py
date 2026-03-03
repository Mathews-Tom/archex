"""Token efficiency reporting for MCP tool responses."""

from __future__ import annotations

import sys

import tiktoken

from archex.models import DeltaMeta, PipelineTiming, TokenMeta

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding."""
    return len(_encoder.encode(text))


def compute_meta(
    *,
    tool_name: str,
    response_text: str,
    raw_file_tokens: int,
    strategy: str,
    cached: bool = False,
    index_time_ms: float = 0.0,
    query_time_ms: float = 0.0,
    delta: DeltaMeta | None = None,
) -> TokenMeta:
    """Compute token efficiency metrics for a tool response."""
    returned = count_tokens(response_text)
    savings = (1 - returned / raw_file_tokens) * 100 if raw_file_tokens > 0 else 0.0
    return TokenMeta(
        tokens_returned=returned,
        tokens_raw_equivalent=raw_file_tokens,
        savings_pct=round(savings, 1),
        strategy=strategy,
        tool_name=tool_name,
        cached=cached,
        index_time_ms=round(index_time_ms, 1),
        query_time_ms=round(query_time_ms, 1),
        delta=delta,
    )


def print_timing(timing: PipelineTiming) -> None:
    """Print per-phase timing breakdown to stderr."""
    if timing.cached:
        print("[timing] Cache hit -- skipped parse", file=sys.stderr)
    else:
        if timing.acquire_ms > 0:
            print(f"[timing] Acquired repo in {timing.acquire_ms:.0f}ms", file=sys.stderr)
        if timing.parse_ms > 0:
            print(f"[timing] Parsed + indexed in {timing.parse_ms:.0f}ms", file=sys.stderr)
    if timing.search_ms > 0 or timing.assemble_ms > 0:
        sa = timing.search_ms + timing.assemble_ms
        print(f"[timing] Search + assemble in {sa:.0f}ms", file=sys.stderr)
    if timing.delta_meta is not None:
        dm = timing.delta_meta
        print(
            f"[timing] Delta index: {dm.files_modified}M/{dm.files_added}A/"
            f"{dm.files_deleted}D files in {dm.delta_time_ms:.0f}ms "
            f"(full reindex avoided)",
            file=sys.stderr,
        )


def print_savings(
    returned: int,
    raw: int,
    elapsed_ms: float,
    *,
    budget: int | None = None,
    file_count: int | None = None,
) -> None:
    """Print token savings summary to stderr."""
    savings_pct = (1 - returned / raw) * 100 if raw > 0 else 0.0
    budget_suffix = f" (budget: {budget:,})" if budget is not None else ""
    print(f"[savings] {returned:,} tokens returned{budget_suffix}", file=sys.stderr)
    files_suffix = f" across {file_count} files" if file_count is not None else ""
    print(f"[savings] Raw equivalent: {raw:,} tokens{files_suffix}", file=sys.stderr)
    print(f"[savings] Saved {savings_pct:.1f}% vs raw file access", file=sys.stderr)
    print(f"[timing] {elapsed_ms:.0f}ms total", file=sys.stderr)
