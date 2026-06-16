"""Central metrics category mapping for instrumented archex surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.metrics.recorder import Category

CONTEXT_RETRIEVAL = "context_retrieval"
STRUCTURAL_TOOLS = "structural_tools"

_CONTEXT_RETRIEVAL_TOOLS = frozenset({"query", "scout", "query_repo", "scout_repo"})
_STRUCTURAL_TOOLS = frozenset(
    {
        "analyze",
        "analyze_repo",
        "compare",
        "compare_repos",
        "get_file_outline",
        "get_file_tree",
        "get_symbol",
        "get_symbols_batch",
        "outline",
        "search_symbols",
        "symbol",
        "symbols",
        "tree",
    }
)


def category_for_tool(tool_name: str) -> Category:
    """Return the metrics category for a known instrumented tool name."""
    if tool_name in _CONTEXT_RETRIEVAL_TOOLS:
        return CONTEXT_RETRIEVAL
    if tool_name in _STRUCTURAL_TOOLS:
        return STRUCTURAL_TOOLS
    raise ValueError(f"unknown metrics tool category for {tool_name!r}")
