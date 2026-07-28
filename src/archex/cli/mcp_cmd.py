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
        "(every tool), a named profile ('core' excludes the graph_* cluster, "
        "'graph' is only the graph_* cluster, 'disclosure' is the "
        "retrieval-gated minimum), or a comma-separated explicit tool-name "
        "allowlist. Every tool name still dispatches regardless of scoping -- "
        "this only shrinks the advertised schema surface, see "
        "`archex mcp-schema-size`."
    ),
)
@click.option(
    "--disclosure/--no-disclosure",
    default=True,
    show_default=True,
    help=(
        "Advertise only the retrieval entry points until the client retrieves, "
        "then advertise everything and send notifications/tools/list_changed. "
        "Cuts the fixed per-turn schema cost from 3859 to 765 tokens. Pass "
        "--no-disclosure for the pre-R5 behavior, which every client can use. "
        "Note --tools does not disable the gate: it bounds what is advertised "
        "once the gate opens, so --tools all still starts minimal. Tools remain "
        "callable either way, so a client with hardcoded tool names is unaffected."
    ),
)
def mcp_cmd(
    watch: bool,
    watch_path: str,
    watch_debounce_ms: int,
    tools: str | None,
    disclosure: bool,
) -> None:
    """Start the archex MCP server (stdio transport).

    Advertises the retrieval entry points (context, query_repo) up front and
    the remaining tools once the client retrieves; pass --no-disclosure to
    advertise all of them from the start.

    Exposes 19 MCP tools (analyze_repo, scout_repo, query_repo, context,
    compare_repos, get_file_tree, get_file_outline, search_symbols,
    get_symbol, get_symbols_batch, get_impact, explain_target,
    generate_onboarding, the graph_* lookup/neighbors/path/stats/hubs
    tools, and the consolidated graph_query dispatch tool) unless
    narrowed with --tools. Connect an MCP-compatible client (e.g. Claude
    Code) to stdin/stdout.
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
            disclosure=disclosure,
        )
    )
