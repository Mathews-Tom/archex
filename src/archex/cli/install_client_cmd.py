"""Client install/bootstrap command."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click

from archex.client_setup import (
    ClientName,
    ClientScope,
    HookAction,
    append_agent_guidance,
    build_client_install_plan,
    build_hook_install_plan,
    render_agent_guidance_preview,
    render_client_install_preview,
    render_hook_install_preview,
    write_client_install_plan,
    write_hook_install_plan,
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
@click.option(
    "--hooks",
    is_flag=True,
    default=False,
    help=(
        "Install the archex PreToolUse hook (opt-in, never installed without this "
        "flag). Augments Grep/Glob calls with archex context on claude-code, omp, pi; "
        "on codex ships a diagnostics-only fallback (no Grep/Glob-equivalent tool-call "
        "event exists there, so nothing is injected, only logged)."
    ),
)
@click.option(
    "--remove-hooks",
    is_flag=True,
    default=False,
    help="Remove the archex PreToolUse hook previously installed by --hooks.",
)
def install_client_cmd(
    client: str,
    source: str | None,
    scope: str | None,
    dry_run: bool,
    agent_file: Path | None,
    hooks: bool,
    remove_hooks: bool,
) -> None:
    """Install MCP client configuration for archex (preview with --dry-run)."""
    if hooks and remove_hooks:
        raise click.ClickException("--hooks and --remove-hooks are mutually exclusive")
    if hooks or remove_hooks:
        _run_hook_action(
            cast("ClientName", client),
            source,
            scope=cast("ClientScope | None", scope),
            dry_run=dry_run,
            action="install" if hooks else "remove",
        )
        return
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


def _run_hook_action(
    client: ClientName,
    source: str | None,
    *,
    scope: ClientScope | None,
    dry_run: bool,
    action: HookAction,
) -> None:
    try:
        plan = build_hook_install_plan(client, source, scope=scope, action=action)
        if dry_run:
            click.echo(render_hook_install_preview(plan), nl=False)
            return
        target = write_hook_install_plan(plan)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if action == "install":
        click.echo(f"Installed archex hook for {client}: {target}")
    else:
        click.echo(f"Removed archex hook for {client} (if present): {target}")
