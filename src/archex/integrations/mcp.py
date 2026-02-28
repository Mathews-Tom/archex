"""MCP integration: expose archex capabilities as Model Context Protocol tools."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from archex.api import analyze, compare, query
from archex.models import RepoSource

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"json", "markdown"}
_DEFAULT_DIMENSIONS = ["api_surface", "error_handling"]


def _resolve_source(path_or_url: str) -> RepoSource:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return RepoSource(url=path_or_url)
    return RepoSource(local_path=path_or_url)


def handle_analyze_repo(repo_url: str, output_format: str = "json") -> str:
    """Analyze a repository and return an architecture profile.

    Args:
        repo_url: Local path or HTTP(S) URL of the repository to analyze.
        output_format: Output format — 'json' or 'markdown'. Defaults to 'json'.

    Returns:
        Serialized ArchProfile in the requested format.
    """
    if output_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"format must be one of {sorted(_SUPPORTED_FORMATS)}, got {output_format!r}"
        )

    source = _resolve_source(repo_url)
    profile = analyze(source)
    if output_format == "markdown":
        return profile.to_markdown()
    return profile.to_json()


def handle_query_repo(repo_url: str, question: str, budget: int = 8000) -> str:
    """Retrieve context from a repository for a natural-language question.

    Args:
        repo_url: Local path or HTTP(S) URL of the repository to query.
        question: Natural-language question to answer from the codebase.
        budget: Maximum token budget for the returned context. Defaults to 8000.

    Returns:
        XML-formatted ContextBundle ready for use as an LLM prompt.
    """
    if not question.strip():
        raise ValueError("question must not be empty")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    source = _resolve_source(repo_url)
    bundle = query(source, question, token_budget=budget)
    return bundle.to_prompt(format="xml")


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
            Supported values: api_surface, error_handling, concurrency,
            testing_strategy, dependency_management, configuration_management.
            Defaults to 'api_surface,error_handling'.

    Returns:
        JSON-serialized ComparisonResult.
    """
    dim_list = [d.strip() for d in dimensions.split(",") if d.strip()]
    if not dim_list:
        raise ValueError("dimensions must be a non-empty comma-separated list")

    source_a = _resolve_source(repo_a)
    source_b = _resolve_source(repo_b)
    result = compare(source_a, source_b, dimensions=dim_list)
    return result.model_dump_json(indent=2)


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
            "The 'mcp' package is required for MCP integration. Install it with: pip install mcp"
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
                            "default": 8000,
                            "description": "Maximum token budget for the returned context.",
                        },
                    },
                    "required": ["repo_url", "question"],
                },
            ),
            mcp_types.Tool(
                name="compare_repos",
                description=(
                    "Compare two code repositories across architectural dimensions such as "
                    "API surface, error handling, concurrency model, testing strategy, "
                    "dependency management, and configuration management."
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
                                "testing_strategy, dependency_management, "
                                "configuration_management."
                            ),
                        },
                    },
                    "required": ["repo_a", "repo_b"],
                },
            ),
        ]

    @server.call_tool()  # pyright: ignore[reportUnusedFunction]
    async def call_tool(  # pyright: ignore[reportUnusedFunction]
        name: str,
        arguments: dict[str, Any],
    ) -> list[mcp_types.TextContent]:
        loop = asyncio.get_event_loop()

        if name == "analyze_repo":
            repo_url: str = arguments["repo_url"]
            fmt: str = arguments.get("format", "json")
            result_text = await loop.run_in_executor(None, handle_analyze_repo, repo_url, fmt)
        elif name == "query_repo":
            repo_url = arguments["repo_url"]
            question: str = arguments["question"]
            budget: int = int(arguments.get("budget", 8000))
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
            "The 'mcp' package is required for MCP integration. Install it with: pip install mcp"
        ) from exc

    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        init_opts = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_opts, raise_exceptions=True)
