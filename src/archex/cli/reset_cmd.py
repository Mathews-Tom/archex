"""Project lifecycle reset command."""

from __future__ import annotations

import click

from archex.project import reset_project


@click.command("reset")
@click.argument("source", required=False, default=".")
@click.option("--force", is_flag=True, default=False, help="Delete generated state.")
@click.option(
    "--all",
    "all_state",
    is_flag=True,
    default=False,
    help="Delete the whole .archex directory.",
)
def reset_cmd(source: str, force: bool, all_state: bool) -> None:
    """Delete repo-local archex generated state."""
    try:
        result = reset_project(source, force=force, all_state=all_state)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if result.removed_all:
        if result.removed_paths:
            click.echo(f"Removed project state: {result.state.project_dir}")
        else:
            click.echo(f"No project state found at {result.state.project_dir}")
        return

    if not result.removed_paths:
        click.echo(f"No generated archex state found at {result.state.project_dir}")
        return

    click.echo(f"Removed {len(result.removed_paths)} generated path(s):")
    for path in result.removed_paths:
        click.echo(f"  {path}")
