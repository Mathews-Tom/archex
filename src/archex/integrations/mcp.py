"""MCP integration: expose archex capabilities as Model Context Protocol tools."""

from __future__ import annotations

import asyncio
import json
import logging
import time
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
    query,
    search_symbols,
)
from archex.models import PipelineTiming
from archex.reporting import compute_meta, count_tokens
from archex.serve.compare import validate_dimensions
from archex.serve.intent import DEFAULT_TOKEN_BUDGET
from archex.serve.renderers.xml import render_xml, render_xml_envelope
from archex.utils import resolve_source

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"json", "markdown"}


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
        else:
            raise ValueError(f"Unknown tool: {name!r}")

        return [mcp_types.TextContent(type="text", text=result_text)]

    return server


async def run_stdio_server() -> None:
    """Run the archex MCP server over stdio."""
    try:
        from mcp.server.stdio import stdio_server
    except ImportError as exc:
        raise ImportError(
            "The 'mcp' package is required for MCP integration. Install it with: uv add mcp"
        ) from exc

    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        init_opts = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_opts, raise_exceptions=True)
