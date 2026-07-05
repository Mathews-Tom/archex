"""Project lifecycle initialization command."""

from __future__ import annotations

from pathlib import Path

import click

from archex.exceptions import ArchexError
from archex.index.artifact import import_artifact
from archex.project import init_project


@click.command("init")
@click.argument("source", required=False, default=".")
@click.option("--force", is_flag=True, default=False, help="Rewrite project settings.")
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="Delete existing .archex state before initialization. Requires --force.",
)
@click.option(
    "--from-artifact",
    "from_artifact_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Bootstrap the local index from a portable index artifact instead of cold-start reindex.",
)
def init_cmd(source: str, force: bool, reset: bool, from_artifact_path: Path | None) -> None:
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

    if from_artifact_path is not None:
        try:
            header = import_artifact(from_artifact_path, result.state.index_path)
        except ArchexError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Imported index artifact: {from_artifact_path}")
        click.echo(f"Artifact revision:       {header.index_revision}")
        click.echo(
            f"Artifact corpus:         {header.file_count} files, {header.chunk_count} chunks"
        )
