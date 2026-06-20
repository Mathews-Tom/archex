"""Client install/bootstrap command."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click

from archex.client_setup import (
    ClientName,
    ClientScope,
    append_agent_guidance,
    build_client_install_plan,
    render_agent_guidance_preview,
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
@click.option(
    "--agent-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Append the archex MCP guidance prompt to this agent file (CLAUDE.md, AGENTS.md).",
)
def install_client_cmd(
    client: str,
    source: str | None,
    scope: str | None,
    dry_run: bool,
    agent_file: Path | None,
) -> None:
    """Install MCP client configuration for archex (preview with --dry-run)."""
    try:
        plan = build_client_install_plan(
            cast("ClientName", client),
            source,
            scope=cast("ClientScope | None", scope),
        )
        if dry_run:
            click.echo(render_client_install_preview(plan), nl=False)
            if agent_file is not None:
                click.echo(render_agent_guidance_preview(agent_file.expanduser()), nl=False)
        else:
            target = write_client_install_plan(plan)
            click.echo(f"Wrote {client} config: {target}")
            if agent_file is not None:
                resolved = agent_file.expanduser()
                if append_agent_guidance(resolved):
                    click.echo(f"Appended archex MCP guidance: {resolved}")
                else:
                    click.echo(f"archex MCP guidance already present: {resolved}")
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
