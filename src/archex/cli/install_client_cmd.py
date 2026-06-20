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
    type=click.Choice(["claude-code", "codex", "cursor", "opencode", "pi", "omp"]),
)
@click.argument("source", required=False, default=None)
@click.option(
    "--scope",
    type=click.Choice(["project", "user"]),
    default=None,
    help="Install scope. Default user/global; a SOURCE path or --scope project is repo-local.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview the target path and config without writing anything.",
)
def install_client_cmd(client: str, source: str | None, scope: str | None, dry_run: bool) -> None:
    """Install MCP client configuration for archex (preview with --dry-run)."""
    try:
        plan = build_client_install_plan(
            cast("ClientName", client),
            source,
            scope=cast("ClientScope | None", scope),
        )
        if dry_run:
            click.echo(render_client_install_preview(plan), nl=False)
        else:
            target = write_client_install_plan(plan)
            click.echo(f"Wrote {client} config: {target}")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
