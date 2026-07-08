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


def _is_interactive() -> bool:
    import sys

    return sys.stdin.isatty()


@click.command("install-client")
@click.argument("client_or_source", required=False)
@click.argument("source_opt", required=False)
@click.option(
    "--all-detected",
    is_flag=True,
    default=False,
    help="Install configuration for all detected possible clients.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip interactive confirmation prompts.",
)
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
        "Install the archex hook/plugin (opt-in, never installed without this "
        "flag). Augments grep/glob calls with archex context on claude-code, omp, "
        "pi, opencode; on codex and cursor ships a diagnostics-only fallback (no "
        "Grep/Glob-equivalent tool-call event exists on codex, and cursor's "
        "beforeSubmitPrompt hook has no context-injection output field at all, "
        "so nothing is injected on either, only logged)."
    ),
)
@click.option(
    "--remove-hooks",
    is_flag=True,
    default=False,
    help="Remove the archex PreToolUse hook previously installed by --hooks.",
)
@click.option(
    "--allow-missing-mcp",
    is_flag=True,
    default=False,
    help="Install client config even if the archex mcp runtime is missing.",
)
def install_client_cmd(
    client_or_source: str | None,
    source_opt: str | None,
    scope: str | None,
    dry_run: bool,
    agent_file: Path | None,
    hooks: bool,
    remove_hooks: bool,
    allow_missing_mcp: bool,
    all_detected: bool,
    yes: bool,
) -> None:
    """Install MCP client configuration for archex (preview with --dry-run)."""
    if hooks and remove_hooks:
        raise click.ClickException("--hooks and --remove-hooks are mutually exclusive")
    valid_clients = ["claude-code", "codex", "cursor", "opencode", "pi", "omp"]

    client: str | None = None
    source: str | None = None

    if client_or_source is not None:
        if client_or_source in valid_clients:
            client = client_or_source
            source = source_opt
        else:
            if source_opt is not None:
                raise click.ClickException(f"Invalid client: {client_or_source}")
            client = None
            source = client_or_source

    if hooks or remove_hooks:
        if client is None:
            raise click.ClickException("Must specify a client when using --hooks")
        _run_hook_action(
            cast("ClientName", client),
            source,
            scope=cast("ClientScope | None", scope),
            dry_run=dry_run,
            action="install" if hooks else "remove",
        )
        return
    import importlib.util

    if not allow_missing_mcp and importlib.util.find_spec("mcp") is None:
        raise click.ClickException(
            "Cannot register archex MCP because this archex installation "
            "cannot start `archex mcp`.\n\n"
            "Fix for uv tool users:\n  uv tool install --force 'archex[mcp]'\n\n"
            "Fix for project users:\n  uv add 'archex[mcp]'\n\n"
            "Or use --allow-missing-mcp to bypass this check."
        )
    try:
        if all_detected:
            if client:
                raise click.ClickException(
                    "Cannot specify a specific client when using --all-detected"
                )
            from archex.client_setup import (
                build_discovered_install_plans,
                discover_clients,
                render_multiple_install_preview,
            )

            discovered = discover_clients(source)
            plans = build_discovered_install_plans(discovered, source)
            if dry_run:
                click.echo(render_multiple_install_preview(plans), nl=False)
                if agent_file is not None:
                    click.echo(render_agent_guidance_preview(agent_file.expanduser()), nl=False)
                return

            if not yes and not _is_interactive():
                raise click.ClickException(
                    "Refusing to write to all detected clients in a non-TTY without --yes."
                )
            for plan in plans:
                target = write_client_install_plan(plan)
                click.echo(f"Wrote {plan.client} config: {target}")
            if agent_file is not None:
                resolved = agent_file.expanduser()
                if append_agent_guidance(resolved):
                    click.echo(f"Appended archex MCP guidance: {resolved}")
                else:
                    click.echo(f"archex MCP guidance already present: {resolved}")
            return

        if client is None:
            # Interactive flow
            if not _is_interactive() and not yes:
                raise click.ClickException(
                    "install-client is interactive by default, but stdin/stdout are not TTY.\n"
                    "Use --all-detected --yes, or specify a client explicitly."
                )

            from archex.client_setup import (
                build_discovered_install_plans,
                discover_clients,
                render_multiple_install_preview,
            )

            discovered = discover_clients(source)

            click.echo("Detected possible clients:\n")
            for d in discovered:
                checkbox = "[x]" if d.is_installed else "[ ]"
                click.echo(f"{checkbox} {d.client.ljust(12)} {d.evidence}")

            if not any(d.is_installed for d in discovered):
                click.echo("\nNo configured clients found.")
                return

            plans = build_discovered_install_plans(discovered, source)
            click.echo(
                f"\nInstall archex MCP registration for {len(plans)} detected clients? [y/N] ",
                nl=False,
            )
            if not yes:
                resp = input().strip().lower()
                if resp not in ("y", "yes"):
                    click.echo("Aborted.")
                    return

            click.echo(render_multiple_install_preview(plans), nl=False)
            if not yes:
                click.echo("Proceed? [Y/n] ", nl=False)
                resp = input().strip().lower()
                if resp in ("n", "no"):
                    click.echo("Aborted.")
                    return

            for plan in plans:
                target = write_client_install_plan(plan)
                click.echo(f"Wrote {plan.client} config: {target}")

            if agent_file is not None:
                resolved = agent_file.expanduser()
                if append_agent_guidance(resolved):
                    click.echo(f"Appended archex MCP guidance: {resolved}")
                else:
                    click.echo(f"archex MCP guidance already present: {resolved}")
            else:
                from archex.client_setup import discover_agent_files

                agent_files = discover_agent_files(
                    Path(source if source is not None else ".").expanduser().resolve()
                )
                if agent_files:
                    click.echo(
                        "\nAppend archex MCP guidance to detected agent instruction files? [Y/n] ",
                        nl=False,
                    )
                    if not yes:
                        resp = input().strip().lower()
                        if resp in ("n", "no"):
                            return
                    click.echo("Detected:")
                    for af in agent_files:
                        click.echo(f"- {af}")
                    for af in agent_files:
                        if append_agent_guidance(af):
                            click.echo(f"Appended archex MCP guidance: {af}")
                        else:
                            click.echo(f"archex MCP guidance already present: {af}")
            return

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
