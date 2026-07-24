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
@click.option(
    "--tools",
    default=None,
    help=(
        "Scope the tools this server advertises via list_tools(): 'all' "
        "(default, every tool), a named profile ('core' excludes the "
        "graph_* cluster, 'graph' is only the graph_* cluster), or a "
        "comma-separated explicit tool-name allowlist. Every tool name "
        "still dispatches regardless of scoping -- this only shrinks the "
        "advertised schema surface, see `archex mcp-schema-size`."
    ),
)
def mcp_cmd(watch: bool, watch_path: str, watch_debounce_ms: int, tools: str | None) -> None:
    """Start the archex MCP server (stdio transport).

    Exposes 18 MCP tools (analyze_repo, scout_repo, query_repo, context,
    compare_repos, get_file_tree, get_file_outline, search_symbols,
    get_symbol, get_symbols_batch, get_impact, explain_target,
    generate_onboarding, and the graph_* lookup/neighbors/path/stats/hubs
    tools) unless narrowed with --tools. Connect an MCP-compatible client
    (e.g. Claude Code) to stdin/stdout.
    """
    try:
        from archex.integrations.mcp import resolve_tool_scope, run_stdio_server
    except ImportError as exc:
        raise click.ClickException(
            "MCP integration requires the `mcp` Python package.\n\n"
            "If archex was installed as a uv tool:\n"
            "  uv tool install --force 'archex[mcp]'\n\n"
            "If archex is a project dependency:\n"
            "  uv add 'archex[mcp]'\n\n"
            "If running from a development checkout:\n"
            "  uv sync --extra mcp"
        ) from exc

    try:
        tool_names = resolve_tool_scope(tools)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    asyncio.run(
        run_stdio_server(
            watch=watch,
            watch_path=watch_path,
            watch_debounce_ms=watch_debounce_ms,
            tool_names=tool_names,
        )
    )
