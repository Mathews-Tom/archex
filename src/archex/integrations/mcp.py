"""MCP integration: expose archex capabilities as Model Context Protocol tools."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from archex.api import (
    analyze,
    compare,
    file_outline,
    file_tree,
    get_file_token_count,
    get_files_token_count,
    get_repo_total_tokens,
    get_symbol,
    get_symbols_batch,
    index_repository,
    query,
    search_symbols,
)
from archex.graph_artifact import GraphArtifactError
from archex.graph_query import (
    GraphDirection,
    GraphEdgeSummary,
    GraphHubsResult,
    GraphNeighborsResult,
    GraphNodeLookupResult,
    GraphNodeSummary,
    GraphPathResult,
    GraphQuery,
    GraphQueryError,
    GraphStatsResult,
)
from archex.models import PipelineTiming, RepoSource
from archex.reporting import compute_meta, count_tokens
from archex.serve.compare import validate_dimensions
from archex.serve.intent import DEFAULT_TOKEN_BUDGET
from archex.serve.renderers.xml import render_xml, render_xml_envelope
from archex.utils import resolve_source

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"json", "markdown"}
DEFAULT_GRAPH_TOKEN_BUDGET = 2000


def handle_analyze_repo(repo_url: str, output_format: str = "json") -> str:
    """Analyze a repository and return an architecture profile.

    Args:
        repo_url: Local path or HTTP(S) URL of the repository to analyze.
        output_format: Output format — 'json' or 'markdown'. Defaults to 'json'.

    Returns:
        JSON envelope with ArchProfile content and _meta efficiency block.
    """
    if output_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"format must be one of {sorted(_SUPPORTED_FORMATS)}, got {output_format!r}"
        )

    source = resolve_source(repo_url)
    pt = PipelineTiming()
    profile = analyze(source, timing=pt)

    content = profile.to_markdown() if output_format == "markdown" else profile.to_json()

    raw_tokens = get_repo_total_tokens(source)
    meta = compute_meta(
        tool_name="analyze_repo",
        response_text=content,
        raw_file_tokens=max(raw_tokens, 1),
        strategy="full_analysis",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    return json.dumps({"content": content, "_meta": meta.model_dump()}, indent=2)


def handle_query_repo(repo_url: str, question: str, budget: int | None = None) -> str:
    """Retrieve context from a repository for a natural-language question.

    Args:
        repo_url: Local path or HTTP(S) URL of the repository to query.
        question: Natural-language question to answer from the codebase.
        budget: Optional explicit token budget override. Defaults to adaptive
            intent routing with a product ceiling of 8192.

    Returns:
        JSON envelope with ContextBundle content and _meta efficiency block.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if budget is not None and budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    source = resolve_source(repo_url)
    pt = PipelineTiming()
    token_budget = DEFAULT_TOKEN_BUDGET if budget is None else budget
    bundle = query(
        source,
        question,
        token_budget=token_budget,
        timing=pt,
        explicit_token_budget=budget is not None,
    )

    content = render_xml(bundle)
    metadata = bundle.retrieval_metadata
    raw_file_paths = sorted(
        {
            *metadata.seed_file_paths,
            *metadata.expanded_file_paths,
            *(c.chunk.file_path for c in bundle.chunks),
        }
    )
    raw_tokens = get_files_token_count(source, raw_file_paths)
    envelope_overhead = count_tokens(render_xml_envelope(bundle))
    meta = compute_meta(
        tool_name="query_repo",
        response_text=content,
        raw_file_tokens=max(raw_tokens, 1),
        envelope_overhead_tokens=envelope_overhead,
        strategy="bm25+graph",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
        delta=pt.delta_meta,
    )
    return json.dumps({"content": content, "_meta": meta.model_dump()}, indent=2)


def handle_compare_repos(
    repo_a: str,
    repo_b: str,
    dimensions: str = "api_surface,error_handling",
) -> str:
    """Compare two repositories across architectural dimensions.

    Args:
        repo_a: Local path or HTTP(S) URL of the first repository.
        repo_b: Local path or HTTP(S) URL of the second repository.
        dimensions: Comma-separated list of dimensions to compare.
            Supported values: error_handling, api_surface, state_management,
            concurrency, testing, configuration.
            Defaults to 'api_surface,error_handling'.

    Returns:
        JSON envelope with ComparisonResult content and _meta efficiency block.
    """
    dim_list = [d.strip() for d in dimensions.split(",") if d.strip()]
    if not dim_list:
        raise ValueError("dimensions must be a non-empty comma-separated list")
    validate_dimensions(dim_list)

    source_a = resolve_source(repo_a)
    source_b = resolve_source(repo_b)
    t0 = time.perf_counter()
    result = compare(source_a, source_b, dimensions=dim_list)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    content = result.model_dump_json(indent=2)
    raw_a = get_repo_total_tokens(source_a)
    raw_b = get_repo_total_tokens(source_b)
    meta = compute_meta(
        tool_name="compare_repos",
        response_text=content,
        raw_file_tokens=max(raw_a + raw_b, 1),
        strategy="full_comparison",
        query_time_ms=elapsed_ms,
    )
    return json.dumps({"content": json.loads(content), "_meta": meta.model_dump()}, indent=2)


def handle_get_file_tree(repo_url: str, max_depth: int = 5, language: str | None = None) -> str:
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    result = file_tree(source, max_depth=max_depth, language=language, timing=pt)
    content = result.model_dump_json(indent=2)
    raw_tokens = get_repo_total_tokens(source)
    meta = compute_meta(
        tool_name="get_file_tree",
        response_text=content,
        raw_file_tokens=max(raw_tokens, 1),
        strategy="file_tree",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    return json.dumps({"content": json.loads(content), "_meta": meta.model_dump()}, indent=2)


def handle_get_file_outline(repo_url: str, file_path: str) -> str:
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    result = file_outline(source, file_path=file_path, timing=pt)
    content = result.model_dump_json(indent=2)
    meta = compute_meta(
        tool_name="get_file_outline",
        response_text=content,
        raw_file_tokens=result.token_count_raw,
        strategy="file_outline",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    return json.dumps({"content": json.loads(content), "_meta": meta.model_dump()}, indent=2)


def handle_search_symbols(
    repo_url: str,
    query_text: str,
    kind: str | None = None,
    language: str | None = None,
    limit: int = 20,
) -> str:
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    matches = search_symbols(
        source, query=query_text, kind=kind, language=language, limit=limit, timing=pt
    )
    match_data = [m.model_dump() for m in matches]
    content = json.dumps(match_data, indent=2)
    unique_files = list({m.file_path for m in matches})
    raw_tokens = get_files_token_count(source, unique_files) if unique_files else 0
    meta = compute_meta(
        tool_name="search_symbols",
        response_text=content,
        raw_file_tokens=max(raw_tokens, 1),
        strategy="symbol_search",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    return json.dumps({"content": match_data, "_meta": meta.model_dump()}, indent=2)


def handle_get_symbol(repo_url: str, symbol_id: str) -> str:
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    result = get_symbol(source, symbol_id=symbol_id, timing=pt)
    if result is None:
        return json.dumps({"error": "Symbol not found", "symbol_id": symbol_id})
    content = result.model_dump_json(indent=2)
    raw_tokens = get_file_token_count(source, result.file_path)
    meta = compute_meta(
        tool_name="get_symbol",
        response_text=content,
        raw_file_tokens=max(raw_tokens, 1),
        strategy="symbol_lookup",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    return json.dumps({"content": json.loads(content), "_meta": meta.model_dump()}, indent=2)


def handle_get_symbols_batch(repo_url: str, symbol_ids: list[str]) -> str:
    if len(symbol_ids) > 50:
        raise ValueError(f"symbol_ids must contain at most 50 entries, got {len(symbol_ids)}")
    source = resolve_source(repo_url)
    pt = PipelineTiming()
    results = get_symbols_batch(source, symbol_ids=symbol_ids, timing=pt)
    result_data = [s.model_dump() if s else None for s in results]
    content = json.dumps(result_data, indent=2)
    unique_files = list({s.file_path for s in results if s is not None})
    raw_tokens = get_files_token_count(source, unique_files) if unique_files else 0
    meta = compute_meta(
        tool_name="get_symbols_batch",
        response_text=content,
        raw_file_tokens=max(raw_tokens, 1),
        strategy="symbol_batch",
        cached=pt.cached,
        index_time_ms=pt.index_ms,
        query_time_ms=pt.total_ms,
    )
    return json.dumps({"content": result_data, "_meta": meta.model_dump()}, indent=2)


def handle_graph_lookup(
    graph_path: str,
    node: str,
    output_format: str = "json",
    limit: int = 25,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.lookup(node, limit=limit)
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path})
    return _graph_tool_response(
        "graph_lookup",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_lookup_markdown,
    )


def handle_graph_neighbors(
    graph_path: str,
    node: str,
    output_format: str = "json",
    direction: GraphDirection = "both",
    depth: int = 1,
    limit: int = 25,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.neighbors(node, direction=direction, depth=depth, limit=limit)
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path, "node": node})
    return _graph_tool_response(
        "graph_neighbors",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_neighbors_markdown,
    )


def handle_graph_path(
    graph_path: str,
    source: str,
    target: str,
    output_format: str = "json",
    direction: GraphDirection = "both",
    max_edges: int = 100,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.shortest_path(
            source,
            target,
            direction=direction,
            max_edges=max_edges,
        )
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path})
    return _graph_tool_response(
        "graph_path",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_path_markdown,
    )


def handle_graph_stats(
    graph_path: str,
    output_format: str = "json",
    hub_limit: int = 10,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.stats(hub_limit=hub_limit)
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path})
    return _graph_tool_response(
        "graph_stats",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_stats_markdown,
    )


def handle_graph_hubs(
    graph_path: str,
    output_format: str = "json",
    limit: int = 25,
    threshold: int | None = None,
    hub_degree: int = 50,
    token_budget: int = DEFAULT_GRAPH_TOKEN_BUDGET,
) -> str:
    started = time.perf_counter()
    try:
        graph_query = _cached_graph_query(graph_path, hub_degree)
        result = graph_query.hubs(limit=limit, threshold=threshold)
    except (GraphArtifactError, GraphQueryError) as exc:
        return json.dumps({"error": str(exc), "graph_path": graph_path})
    return _graph_tool_response(
        "graph_hubs",
        graph_query,
        result,
        output_format,
        token_budget,
        started,
        _render_hubs_markdown,
    )


@lru_cache(maxsize=16)
def _cached_graph_query(graph_path: str, hub_degree: int) -> GraphQuery:
    return GraphQuery.from_artifact(Path(graph_path).expanduser().resolve(), hub_degree=hub_degree)


def clear_graph_query_cache() -> None:
    """Clear cached graph artifact handles used by MCP graph tools."""
    _cached_graph_query.cache_clear()


def _graph_tool_response(
    tool_name: str,
    graph_query: GraphQuery,
    result: GraphNodeLookupResult
    | GraphNeighborsResult
    | GraphPathResult
    | GraphStatsResult
    | GraphHubsResult,
    output_format: str,
    token_budget: int,
    started: float,
    markdown_renderer: Any,
) -> str:
    _validate_output_format(output_format)
    if token_budget < 1:
        raise ValueError("token_budget must be at least 1")
    if output_format == "markdown":
        rendered_content, budget_truncated = _apply_token_budget(
            markdown_renderer(result),
            token_budget,
        )
        response_text = rendered_content
        content: str | dict[str, Any] = rendered_content
    else:
        content = result.model_dump(mode="json")
        response_text = json.dumps(content, indent=2, sort_keys=True)
        budget_truncated = False
    raw_tokens = max(count_tokens(graph_query.graph.to_json()), 1)
    query_time_ms = (time.perf_counter() - started) * 1000
    meta = compute_meta(
        tool_name=tool_name,
        response_text=response_text,
        raw_file_tokens=raw_tokens,
        strategy="graph_query",
        query_time_ms=query_time_ms,
    ).model_dump()
    meta["format"] = output_format
    meta["token_budget"] = token_budget
    meta["token_budget_truncated"] = budget_truncated
    return json.dumps({"content": content, "_meta": meta}, indent=2, sort_keys=True)


def _validate_output_format(output_format: str) -> None:
    if output_format not in _SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format {output_format!r}; expected json or markdown")


def _apply_token_budget(text: str, token_budget: int) -> tuple[str, bool]:
    if count_tokens(text) <= token_budget:
        return text, False
    marker = "\n\n[truncated: token budget reached]\n"
    lines: list[str] = []
    for line in text.splitlines():
        candidate = "\n".join([*lines, line]) + marker
        if count_tokens(candidate) > token_budget:
            break
        lines.append(line)
    if not lines:
        return marker.lstrip(), True
    return "\n".join(lines).rstrip() + marker, True


def _render_lookup_markdown(result: GraphNodeLookupResult) -> str:
    lines = [
        f"# Graph Lookup: {result.query}",
        "",
        f"- Match kind: `{result.match_kind or 'none'}`",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
    ]
    lines.extend(_render_node_list(result.matches))
    return _finish_markdown(lines)


def _render_neighbors_markdown(result: GraphNeighborsResult) -> str:
    lines = [
        f"# Graph Neighbors: {result.seed.id}",
        "",
        f"- Path: `{result.seed.path or result.seed.id}`",
        f"- Direction: `{result.direction}`",
        f"- Depth: {result.depth}",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
        "## Edges",
        "",
    ]
    lines.extend(_render_edge_list(result.edges))
    if result.hubs:
        lines.extend(["", "## Terminal Hubs", ""])
        lines.extend(_render_node_list(result.hubs))
    return _finish_markdown(lines)


def _render_path_markdown(result: GraphPathResult) -> str:
    source = result.source.path if result.source is not None else result.source_query
    target = result.target.path if result.target is not None else result.target_query
    lines = [
        f"# Graph Path: {source} -> {target}",
        "",
        f"- Found: {_yes_no(result.found)}",
        f"- Direction: `{result.direction}`",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
    ]
    if result.nodes:
        lines.extend(["## Nodes", ""])
        lines.extend(_render_node_list(result.nodes))
        lines.extend(["", "## Edges", ""])
        lines.extend(_render_edge_list(result.edges))
    if result.avoided_hubs:
        lines.extend(["", "## Avoided Hubs", ""])
        lines.extend(_render_node_list(result.avoided_hubs))
    return _finish_markdown(lines)


def _render_stats_markdown(result: GraphStatsResult) -> str:
    lines = [
        f"# Graph Stats: {result.project}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Nodes | {result.nodes} |",
        f"| Edges | {result.edges} |",
        f"| Files | {result.files} |",
        f"| Max degree | {result.max_degree} |",
        "",
    ]
    if result.edge_types:
        lines.extend(["## Edge Types", "", "| Type | Count |", "| --- | ---: |"])
        for edge_type, count in sorted(result.edge_types.items()):
            lines.append(f"| {edge_type} | {count} |")
        lines.append("")
    if result.hubs:
        lines.extend(["## Hubs", ""])
        lines.extend(_render_node_list(result.hubs))
    return _finish_markdown(lines)


def _render_hubs_markdown(result: GraphHubsResult) -> str:
    lines = [
        "# Graph Hubs",
        "",
        f"- Threshold: {result.threshold}",
        f"- Limit: {result.limit}",
        f"- Truncated: {_yes_no(result.truncated)}",
        "",
    ]
    lines.extend(_render_node_list(result.hubs))
    return _finish_markdown(lines)


def _render_edge_list(edges: list[GraphEdgeSummary]) -> list[str]:
    if not edges:
        return ["No edges."]
    lines = [
        "| Source path | Kind | Target path | Confidence | Evidence |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for edge in edges:
        evidence = "; ".join(edge.evidence) if edge.evidence else ""
        lines.append(
            "| "
            f"`{edge.source.path or edge.source.id}` | "
            f"{edge.type} | "
            f"`{edge.target.path or edge.target.id}` | "
            f"{edge.confidence} ({edge.confidence_score:.2f}) | "
            f"{evidence} |"
        )
    return lines


def _render_node_list(nodes: list[GraphNodeSummary]) -> list[str]:
    if not nodes:
        return ["No nodes."]
    lines = ["| Path | ID | Type | Degree |", "| --- | --- | --- | ---: |"]
    for node in nodes:
        lines.append(f"| `{node.path or ''}` | `{node.id}` | {node.type} | {node.degree} |")
    return lines


def _finish_markdown(lines: list[str]) -> str:
    return "\n".join(lines).rstrip() + "\n"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def build_server() -> Any:
    """Build and return a configured MCP Server instance.

    Raises:
        ImportError: If the `mcp` package is not installed.
    """
    try:
        import mcp.types as mcp_types
        from mcp.server import Server
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for MCP integration. Install it with: uv add mcp"
        ) from exc

    server: Server[None, Any] = Server("archex")  # type: ignore[type-arg]

    @server.list_tools()  # pyright: ignore[reportUnusedFunction]
    async def list_tools() -> list[mcp_types.Tool]:  # pyright: ignore[reportUnusedFunction]
        return [
            mcp_types.Tool(
                name="analyze_repo",
                description=(
                    "Analyze a code repository and return an architecture profile including "
                    "modules, design patterns, interfaces, dependency graph, and architectural "
                    "decisions. Works with local paths and remote Git URLs."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": (
                                "Local filesystem path or HTTP(S) Git URL of the repository."
                            ),
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown"],
                            "default": "json",
                            "description": "Output format for the architecture profile.",
                        },
                    },
                    "required": ["repo_url"],
                },
            ),
            mcp_types.Tool(
                name="query_repo",
                description=(
                    "Retrieve relevant code context from a repository to answer a "
                    "natural-language question. Returns a ranked set of code chunks "
                    "within the specified token budget, suitable for use as LLM context."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": (
                                "Local filesystem path or HTTP(S) Git URL of the repository."
                            ),
                        },
                        "question": {
                            "type": "string",
                            "description": (
                                "Natural-language question to answer from the codebase."
                            ),
                        },
                        "budget": {
                            "type": "integer",
                            "description": (
                                "Optional explicit token budget override. Omit to use "
                                "adaptive intent routing with the 8192 product ceiling."
                            ),
                        },
                    },
                    "required": ["repo_url", "question"],
                },
            ),
            mcp_types.Tool(
                name="compare_repos",
                description=(
                    "Compare two code repositories across architectural dimensions such as "
                    "API surface, error handling, concurrency model, testing, "
                    "state management, and configuration."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_a": {
                            "type": "string",
                            "description": "Local path or HTTP(S) URL of the first repository.",
                        },
                        "repo_b": {
                            "type": "string",
                            "description": "Local path or HTTP(S) URL of the second repository.",
                        },
                        "dimensions": {
                            "type": "string",
                            "default": "api_surface,error_handling",
                            "description": (
                                "Comma-separated dimensions to compare. "
                                "Supported: api_surface, error_handling, concurrency, "
                                "testing, state_management, configuration."
                            ),
                        },
                    },
                    "required": ["repo_a", "repo_b"],
                },
            ),
            mcp_types.Tool(
                name="get_file_tree",
                description=(
                    "Return a hierarchical file tree for a repository, optionally filtered "
                    "by language and depth."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": "Local path or HTTP(S) URL of the repository.",
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum directory depth to traverse.",
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter results to files of this language.",
                        },
                    },
                    "required": ["repo_url"],
                },
            ),
            mcp_types.Tool(
                name="get_file_outline",
                description=(
                    "Return a structural outline of a single file — symbols, classes, "
                    "functions, and their locations."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": "Local path or HTTP(S) URL of the repository.",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Relative path of the file within the repository.",
                        },
                    },
                    "required": ["repo_url", "file_path"],
                },
            ),
            mcp_types.Tool(
                name="search_symbols",
                description=(
                    "Search for symbols (functions, classes, variables) in a repository "
                    "by name, kind, and/or language."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": "Local path or HTTP(S) URL of the repository.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query to match against symbol names.",
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by symbol kind (e.g. function, class).",
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by programming language.",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 20,
                            "description": "Maximum number of results to return.",
                        },
                    },
                    "required": ["repo_url", "query"],
                },
            ),
            mcp_types.Tool(
                name="get_symbol",
                description="Retrieve a single symbol by its stable symbol ID.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": "Local path or HTTP(S) URL of the repository.",
                        },
                        "symbol_id": {
                            "type": "string",
                            "description": "Stable symbol identifier.",
                        },
                    },
                    "required": ["repo_url", "symbol_id"],
                },
            ),
            mcp_types.Tool(
                name="get_symbols_batch",
                description=(
                    "Retrieve multiple symbols by their stable symbol IDs in a single call. "
                    "Maximum 50 IDs per request."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_url": {
                            "type": "string",
                            "description": "Local path or HTTP(S) URL of the repository.",
                        },
                        "symbol_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stable symbol identifiers (max 50).",
                        },
                    },
                    "required": ["repo_url", "symbol_ids"],
                },
            ),
            mcp_types.Tool(
                name="graph_lookup",
                description=(
                    "Look up graph nodes in an exported architecture graph artifact without "
                    "indexing source files. Exact node IDs and paths win over fuzzy matches."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                        "node": {
                            "type": "string",
                            "description": "Node ID, path, label, or query.",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown"],
                            "default": "json",
                        },
                        "limit": {"type": "integer", "default": 25},
                        "hub_degree": {"type": "integer", "default": 50},
                        "token_budget": {
                            "type": "integer",
                            "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                            "description": "Maximum markdown content tokens to return.",
                        },
                    },
                    "required": ["graph_path", "node"],
                },
            ),
            mcp_types.Tool(
                name="graph_neighbors",
                description=(
                    "Return graph neighbors for a node from an exported artifact without "
                    "indexing source files. Edges include kind, confidence, and evidence."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                        "node": {
                            "type": "string",
                            "description": "Node ID, path, label, or query.",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown"],
                            "default": "json",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["out", "in", "both"],
                            "default": "both",
                        },
                        "depth": {"type": "integer", "default": 1},
                        "limit": {"type": "integer", "default": 25},
                        "hub_degree": {"type": "integer", "default": 50},
                        "token_budget": {
                            "type": "integer",
                            "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                            "description": "Maximum markdown content tokens to return.",
                        },
                    },
                    "required": ["graph_path", "node"],
                },
            ),
            mcp_types.Tool(
                name="graph_path",
                description=(
                    "Find a shortest structural path between two graph nodes from an exported "
                    "artifact without indexing source files."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                        "source": {"type": "string", "description": "Source node ID or path."},
                        "target": {"type": "string", "description": "Target node ID or path."},
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown"],
                            "default": "json",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["out", "in", "both"],
                            "default": "both",
                        },
                        "max_edges": {"type": "integer", "default": 100},
                        "hub_degree": {"type": "integer", "default": 50},
                        "token_budget": {
                            "type": "integer",
                            "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                            "description": "Maximum markdown content tokens to return.",
                        },
                    },
                    "required": ["graph_path", "source", "target"],
                },
            ),
            mcp_types.Tool(
                name="graph_stats",
                description=(
                    "Return deterministic graph stats and hub summaries from an exported "
                    "artifact without indexing source files."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown"],
                            "default": "json",
                        },
                        "hub_limit": {"type": "integer", "default": 10},
                        "hub_degree": {"type": "integer", "default": 50},
                        "token_budget": {
                            "type": "integer",
                            "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                            "description": "Maximum markdown content tokens to return.",
                        },
                    },
                    "required": ["graph_path"],
                },
            ),
            mcp_types.Tool(
                name="graph_hubs",
                description=(
                    "Return high-degree graph hubs from an exported artifact without indexing "
                    "source files."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "graph_path": {"type": "string", "description": "Path to archgraph JSON."},
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown"],
                            "default": "json",
                        },
                        "limit": {"type": "integer", "default": 25},
                        "threshold": {"type": "integer"},
                        "hub_degree": {"type": "integer", "default": 50},
                        "token_budget": {
                            "type": "integer",
                            "default": DEFAULT_GRAPH_TOKEN_BUDGET,
                            "description": "Maximum markdown content tokens to return.",
                        },
                    },
                    "required": ["graph_path"],
                },
            ),
        ]

    @server.call_tool()  # pyright: ignore[reportUnusedFunction]
    async def call_tool(  # pyright: ignore[reportUnusedFunction]
        name: str,
        arguments: dict[str, Any],
    ) -> list[mcp_types.TextContent]:
        loop = asyncio.get_running_loop()

        if name == "analyze_repo":
            repo_url: str = arguments["repo_url"]
            fmt: str = arguments.get("format", "json")
            result_text = await loop.run_in_executor(None, handle_analyze_repo, repo_url, fmt)
        elif name == "query_repo":
            repo_url = arguments["repo_url"]
            question: str = arguments["question"]
            budget_arg = arguments.get("budget")
            budget = int(budget_arg) if budget_arg is not None else None
            result_text = await loop.run_in_executor(
                None, handle_query_repo, repo_url, question, budget
            )
        elif name == "compare_repos":
            repo_a: str = arguments["repo_a"]
            repo_b: str = arguments["repo_b"]
            dims: str = arguments.get("dimensions", "api_surface,error_handling")
            result_text = await loop.run_in_executor(
                None, handle_compare_repos, repo_a, repo_b, dims
            )
        elif name == "get_file_tree":
            repo_url = arguments["repo_url"]
            max_depth: int = int(arguments.get("max_depth", 5))
            language: str | None = arguments.get("language")
            result_text = await loop.run_in_executor(
                None, handle_get_file_tree, repo_url, max_depth, language
            )
        elif name == "get_file_outline":
            repo_url = arguments["repo_url"]
            file_path: str = arguments["file_path"]
            result_text = await loop.run_in_executor(
                None, handle_get_file_outline, repo_url, file_path
            )
        elif name == "search_symbols":
            repo_url = arguments["repo_url"]
            sym_query: str = arguments["query"]
            kind: str | None = arguments.get("kind")
            language = arguments.get("language")
            limit: int = int(arguments.get("limit", 20))
            result_text = await loop.run_in_executor(
                None, handle_search_symbols, repo_url, sym_query, kind, language, limit
            )
        elif name == "get_symbol":
            repo_url = arguments["repo_url"]
            symbol_id: str = arguments["symbol_id"]
            result_text = await loop.run_in_executor(None, handle_get_symbol, repo_url, symbol_id)
        elif name == "get_symbols_batch":
            repo_url = arguments["repo_url"]
            symbol_ids: list[str] = arguments["symbol_ids"]
            result_text = await loop.run_in_executor(
                None, handle_get_symbols_batch, repo_url, symbol_ids
            )
        elif name == "graph_lookup":
            graph_path = arguments["graph_path"]
            node: str = arguments["node"]
            fmt = arguments.get("format", "json")
            limit = int(arguments.get("limit", 25))
            hub_degree = int(arguments.get("hub_degree", 50))
            token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
            result_text = await loop.run_in_executor(
                None,
                handle_graph_lookup,
                graph_path,
                node,
                fmt,
                limit,
                hub_degree,
                token_budget,
            )
        elif name == "graph_neighbors":
            graph_path = arguments["graph_path"]
            node = arguments["node"]
            fmt = arguments.get("format", "json")
            direction: GraphDirection = arguments.get("direction", "both")
            depth = int(arguments.get("depth", 1))
            limit = int(arguments.get("limit", 25))
            hub_degree = int(arguments.get("hub_degree", 50))
            token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
            result_text = await loop.run_in_executor(
                None,
                handle_graph_neighbors,
                graph_path,
                node,
                fmt,
                direction,
                depth,
                limit,
                hub_degree,
                token_budget,
            )
        elif name == "graph_path":
            graph_path = arguments["graph_path"]
            source: str = arguments["source"]
            target: str = arguments["target"]
            fmt = arguments.get("format", "json")
            direction = arguments.get("direction", "both")
            max_edges = int(arguments.get("max_edges", 100))
            hub_degree = int(arguments.get("hub_degree", 50))
            token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
            result_text = await loop.run_in_executor(
                None,
                handle_graph_path,
                graph_path,
                source,
                target,
                fmt,
                direction,
                max_edges,
                hub_degree,
                token_budget,
            )
        elif name == "graph_stats":
            graph_path = arguments["graph_path"]
            fmt = arguments.get("format", "json")
            hub_limit = int(arguments.get("hub_limit", 10))
            hub_degree = int(arguments.get("hub_degree", 50))
            token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
            result_text = await loop.run_in_executor(
                None,
                handle_graph_stats,
                graph_path,
                fmt,
                hub_limit,
                hub_degree,
                token_budget,
            )
        elif name == "graph_hubs":
            graph_path = arguments["graph_path"]
            fmt = arguments.get("format", "json")
            limit = int(arguments.get("limit", 25))
            threshold_arg = arguments.get("threshold")
            threshold = int(threshold_arg) if threshold_arg is not None else None
            hub_degree = int(arguments.get("hub_degree", 50))
            token_budget = int(arguments.get("token_budget", DEFAULT_GRAPH_TOKEN_BUDGET))
            result_text = await loop.run_in_executor(
                None,
                handle_graph_hubs,
                graph_path,
                fmt,
                limit,
                threshold,
                hub_degree,
                token_budget,
            )
        else:
            raise ValueError(f"Unknown tool: {name!r}")

        return [mcp_types.TextContent(type="text", text=result_text)]

    return server


def _ignored_watch_path(path: str) -> bool:
    parts = Path(path).parts
    return any(part in {".git", ".archex", "__pycache__", ".pytest_cache"} for part in parts)


def _start_index_watch(repo_path: Path, debounce_ms: int) -> Any:
    try:
        from watchdog.events import FileSystemEvent, FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:
        raise ImportError("watch mode requires the 'watchdog' package") from exc

    class DebouncedIndexHandler(FileSystemEventHandler):
        def __init__(self) -> None:
            self._timer: threading.Timer | None = None
            self._lock = threading.Lock()
            self._pending_refresh = False
            self._refreshing = False

        def on_any_event(self, event: FileSystemEvent) -> None:
            src_path = str(event.src_path)
            if event.is_directory or _ignored_watch_path(src_path):
                return
            self._schedule()

        def _schedule(self) -> None:
            with self._lock:
                if self._refreshing:
                    self._pending_refresh = True
                    return
                self._arm_timer_locked()

        def _arm_timer_locked(self) -> None:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(debounce_ms / 1000.0, self._refresh)
            self._timer.daemon = True
            self._timer.start()

        def _refresh(self) -> None:
            with self._lock:
                if self._refreshing:
                    return
                self._refreshing = True
            try:
                source = RepoSource(local_path=str(repo_path))
                timing = PipelineTiming()
                store = index_repository(source, timing=timing)
                try:
                    logger.info("MCP watch refreshed %s via %s", repo_path, timing.strategy)
                finally:
                    store.close()
            finally:
                with self._lock:
                    self._refreshing = False
                    if self._pending_refresh:
                        self._pending_refresh = False
                        self._arm_timer_locked()

    observer = Observer()
    observer.schedule(DebouncedIndexHandler(), str(repo_path), recursive=True)
    observer.start()
    logger.info("MCP watch enabled for %s", repo_path)
    return observer


async def run_stdio_server(
    *,
    watch: bool = False,
    watch_path: str = ".",
    watch_debounce_ms: int = 300,
) -> None:
    """Run the archex MCP server over stdio."""
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for MCP integration. Install it with: uv add mcp"
        ) from exc

    observer: Any | None = None
    if watch:
        observer = _start_index_watch(Path(watch_path).expanduser().resolve(), watch_debounce_ms)

    server = build_server()
    try:
        async with stdio_server() as (read_stream, write_stream):
            init_opts = server.create_initialization_options()
            await server.run(read_stream, write_stream, init_opts, raise_exceptions=True)
    finally:
        if observer is not None:
            observer.stop()
            observer.join(timeout=5)
