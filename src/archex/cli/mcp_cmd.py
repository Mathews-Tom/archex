"""CLI mcp subcommand: start the archex MCP stdio server."""

from __future__ import annotations

import asyncio

import click


@click.command("mcp")
def mcp_cmd() -> None:
    """Start the archex MCP server (stdio transport).

    Exposes analyze_repo, query_repo, and compare_repos as MCP tools.
    Connect an MCP-compatible client (e.g. Claude Desktop) to stdin/stdout.
    """
    try:
        from archex.integrations.mcp import run_stdio_server
    except ImportError as exc:
        raise click.ClickException(
            "MCP integration requires the 'mcp' package. Install it with: uv add mcp"
        ) from exc

    asyncio.run(run_stdio_server())
