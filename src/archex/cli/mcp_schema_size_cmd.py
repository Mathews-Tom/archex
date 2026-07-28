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
    "--disclosure/--no-disclosure",
    "disclosure",
    default=True,
    show_default=True,
    help=(
        "Measure through the retrieval-disclosure gate, matching what "
        "`archex mcp` advertises to a fresh session. --no-disclosure measures "
        "the ungated surface. Ignored when --tools names an explicit scope."
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
def mcp_schema_size_cmd(tools: str | None, format_: str, disclosure: bool) -> None:
    """Measure the serialized MCP tool-schema size a client is actually charged.

    Reports total and per-tool character and token counts of the JSON schema
    archex's MCP server advertises via list_tools(), so a client can compare
    scopes before choosing `archex mcp --tools ...` or
    `archex install-client --tool-scope ...`.

    By default this measures what `archex mcp` really advertises to a fresh
    session, which since R5 is the retrieval-gated surface, not every registered
    tool. `--no-disclosure` measures the ungated surface -- and so does any
    explicit `--tools`, which asks about a specific scope rather than about the
    shipped default. The gated figure is the honest per-turn cost; the expanded
    figure is what the same session pays after it first retrieves, and both are
    reported so neither can be quoted alone.
    """
    from archex.integrations.mcp import (
        DisclosureGate,
        measure_tool_schema_size,
        resolve_tool_scope,
    )

    try:
        tool_names = resolve_tool_scope(tools)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    # An explicit scope is a question about that scope, so only the bare command
    # is answered through the gate.
    gated = disclosure and tools is None
    scope_label = tools or ("disclosure (gated default)" if gated else "all")
    measured = DisclosureGate(enabled=gated).visible(tool_names)
    report = measure_tool_schema_size(measured)

    expanded = measure_tool_schema_size(tool_names) if gated else None

    if format_ == "json":
        payload = {"scope": scope_label, "gated": gated, **report}
        if expanded is not None:
            payload["after_first_retrieval"] = {
                "tool_count": expanded["tool_count"],
                "total_chars": expanded["total_chars"],
                "total_tokens": expanded["total_tokens"],
            }
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Scope: {scope_label}")
    click.echo(f"Tools: {report['tool_count']}")
    click.echo(
        f"Total serialized schema size: {report['total_chars']} chars, "
        f"{report['total_tokens']} tokens"
    )
    if expanded is not None:
        click.echo(
            f"After the client first retrieves: {expanded['tool_count']} tools, "
            f"{expanded['total_chars']} chars, {expanded['total_tokens']} tokens"
        )
    click.echo("\nPer-tool:")
    tokens = report["per_tool_tokens"]
    for name, size in sorted(report["per_tool_chars"].items()):
        click.echo(f"  {name}: {size} chars, {tokens[name]} tokens")
