"""Project lifecycle status command."""

from __future__ import annotations

import json

import click

from archex.status import ProjectStatus, inspect_project_status


@click.command("status")
@click.argument("source", required=False, default=".")
@click.option("--strict", is_flag=True, default=False, help="Fail unless the index is fresh.")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def status_cmd(source: str, strict: bool, output_format: str) -> None:
    """Inspect repo-local archex project status without indexing."""
    try:
        status = inspect_project_status(source)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        click.echo(json.dumps(_status_payload(status), indent=2, sort_keys=True))
    else:
        _render_text(status)

    if status.state == "corrupt" or (strict and status.state != "fresh"):
        raise click.exceptions.Exit(1)


def _status_payload(status: ProjectStatus) -> dict[str, object]:
    return {
        "repo_root": str(status.repo_root),
        "initialized": status.initialized,
        "state": status.state,
        "index_path": str(status.index_path),
        "current_commit": status.current_commit,
        "indexed_commit": status.indexed_commit,
        "working_tree": status.working_tree,
        "files_indexed": status.files_indexed,
        "chunks_indexed": status.chunks_indexed,
        "languages": status.languages,
        "vector_index_available": status.vector_index_available,
        "dogfood_latest_path": (
            str(status.dogfood_latest_path) if status.dogfood_latest_path is not None else ""
        ),
        "error": status.error,
    }


def _render_text(status: ProjectStatus) -> None:
    click.echo(f"Repository:         {status.repo_root}")
    click.echo(f"State:              {status.state}")
    click.echo(f"Initialized:        {'yes' if status.initialized else 'no'}")
    click.echo(f"Index path:         {status.index_path}")
    click.echo(f"Current commit:     {status.current_commit or 'none'}")
    click.echo(f"Indexed commit:     {status.indexed_commit or 'none'}")
    click.echo(f"Working tree:       {status.working_tree}")
    click.echo(f"Files indexed:      {status.files_indexed}")
    click.echo(f"Chunks indexed:     {status.chunks_indexed}")
    if status.languages:
        languages = ", ".join(f"{language}={count}" for language, count in status.languages.items())
    else:
        languages = "none"
    click.echo(f"Languages:          {languages}")
    click.echo(f"Vector index:       {'yes' if status.vector_index_available else 'no'}")
    if status.dogfood_latest_path is not None:
        click.echo(f"Dogfood latest:     {status.dogfood_latest_path}")
    if status.error:
        click.echo(f"Error:              {status.error}")
