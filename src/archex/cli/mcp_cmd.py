"""CLI mcp subcommand: start the archex MCP stdio server."""

from __future__ import annotations

import asyncio

import click


@click.command("mcp")
@click.option("--watch", is_flag=True, default=False, help="Watch the repo and refresh the index.")
@click.option(
    "--watch-path",
    default=".",
    show_default=True,
    help="Repository path to watch when --watch is enabled.",
)
@click.option(
    "--watch-debounce-ms",
    default=300,
    show_default=True,
    type=int,
    help="Debounce interval for filesystem refreshes.",
)
def mcp_cmd(watch: bool, watch_path: str, watch_debounce_ms: int) -> None:
    """Start the archex MCP server (stdio transport).

    Exposes 17 MCP tools (analyze_repo, query_repo, scout_repo, compare_repos,
    get_file_tree, get_file_outline, search_symbols, get_symbol,
    get_symbols_batch, get_impact, explain_target, generate_onboarding, and
    the graph_* lookup/neighbors/path/stats/hubs tools). Connect an
    MCP-compatible client (e.g. Claude Code) to stdin/stdout.
    """
    try:
        from archex.integrations.mcp import run_stdio_server
    except ImportError as exc:
        raise click.ClickException(
            "MCP integration requires the 'mcp' package. Install it with: uv add mcp"
        ) from exc

    asyncio.run(
        run_stdio_server(
            watch=watch,
            watch_path=watch_path,
            watch_debounce_ms=watch_debounce_ms,
        )
    )
