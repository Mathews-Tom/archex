"""CLI mcp-schema-size subcommand: measure serialized MCP tool-schema size."""

from __future__ import annotations

import json

import click


@click.command("mcp-schema-size")
@click.option(
    "--tools",
    default=None,
    help=(
        "Tool scope to measure: 'all' (default, every tool), a named "
        "profile ('core' excludes the graph_* cluster, 'graph' is only "
        "the graph_* cluster), or a comma-separated explicit tool-name "
        "allowlist."
    ),
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
def mcp_schema_size_cmd(tools: str | None, format_: str) -> None:
    """Measure the serialized MCP tool-schema size for a tool scope.

    Reports total and per-tool character and token counts of the JSON schema archex's
    MCP server would advertise via list_tools() for the given scope, so a
    client can compare 'all' against a narrower scope before choosing
    `archex mcp --tools ...` or `archex install-client --tool-scope ...`.
    """
    from archex.integrations.mcp import measure_tool_schema_size, resolve_tool_scope

    try:
        tool_names = resolve_tool_scope(tools)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    report = measure_tool_schema_size(tool_names)
    if format_ == "json":
        click.echo(json.dumps({"scope": tools or "all", **report}, indent=2, sort_keys=True))
        return

    click.echo(f"Scope: {tools or 'all'}")
    click.echo(f"Tools: {report['tool_count']}")
    click.echo(
        f"Total serialized schema size: {report['total_chars']} chars, "
        f"{report['total_tokens']} tokens"
    )
    click.echo("\nPer-tool:")
    tokens = report["per_tool_tokens"]
    for name, size in sorted(report["per_tool_chars"].items()):
        click.echo(f"  {name}: {size} chars, {tokens[name]} tokens")
