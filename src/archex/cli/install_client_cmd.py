"""Client install/bootstrap command."""

from __future__ import annotations

from typing import cast

import click

from archex.client_setup import (
    ClientName,
    ClientScope,
    build_client_install_plan,
    render_client_install_preview,
    write_client_install_plan,
)


@click.command("install-client")
@click.argument(
    "client",
    type=click.Choice(["claude-code", "codex", "cursor", "opencode", "pi"]),
)
@click.argument("source", required=False, default=".")
@click.option(
    "--scope",
    type=click.Choice(["project", "user"]),
    default=None,
    help="Install scope.",
)
@click.option(
    "--write",
    is_flag=True,
    default=False,
    help="Write config instead of previewing it.",
)
def install_client_cmd(client: str, source: str, scope: str | None, write: bool) -> None:
    """Preview or write MCP client configuration for archex."""
    try:
        plan = build_client_install_plan(
            cast("ClientName", client),
            source,
            scope=cast("ClientScope | None", scope),
        )
        if write:
            target = write_client_install_plan(plan)
            click.echo(f"Wrote {client} config: {target}")
        else:
            click.echo(render_client_install_preview(plan), nl=False)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
