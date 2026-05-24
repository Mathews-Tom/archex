"""Project lifecycle initialization command."""

from __future__ import annotations

import click

from archex.project import init_project


@click.command("init")
@click.argument("source")
@click.option("--force", is_flag=True, default=False, help="Rewrite project settings.")
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="Delete existing .archex state before initialization. Requires --force.",
)
def init_cmd(source: str, force: bool, reset: bool) -> None:
    """Initialize repo-local archex project state."""
    try:
        result = init_project(source, force=force, reset=reset)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if result.created:
        click.echo(f"Initialized archex project at {result.state.project_dir}")
    else:
        click.echo(f"archex project already initialized at {result.state.project_dir}")

    if result.settings_written:
        click.echo(f"Wrote settings: {result.state.settings_path}")
    else:
        click.echo(f"Preserved settings: {result.state.settings_path}")

    if result.gitignore_updated:
        click.echo(f"Updated ignore rules: {result.state.gitignore_path}")
