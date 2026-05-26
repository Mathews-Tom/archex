"""Impact analysis command."""

from __future__ import annotations

from pathlib import Path

import click

from archex.api import index_repository
from archex.config import load_config, load_index_config
from archex.exceptions import ArchexError
from archex.impact import (
    ImpactError,
    ImpactFileChange,
    analyze_impact,
    git_changed_files,
    render_impact_report,
)
from archex.models import RepoSource


@click.command("impact")
@click.argument("source", required=False, default=".")
@click.option("--base", default="main", help="Base ref for git diff mode.")
@click.option(
    "--changed-file",
    "changed_files",
    multiple=True,
    help="Changed file path. Repeat to bypass git diff mode.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format.",
)
def impact_cmd(
    source: str,
    base: str,
    changed_files: tuple[str, ...],
    output_format: str,
) -> None:
    """Analyze deterministic blast radius for changed files."""
    repo_root = Path(source).expanduser().resolve()
    if changed_files:
        changes = [ImpactFileChange(path=path) for path in changed_files]
    else:
        try:
            changes = git_changed_files(repo_root, base)
        except ImpactError as exc:
            raise click.ClickException(str(exc)) from exc

    repo_source = RepoSource(local_path=source)
    config = load_config(repo_source)
    index_config = load_index_config(repo_source)
    try:
        store = index_repository(repo_source, config=config, index_config=index_config)
        try:
            report = analyze_impact(store, changes)
        finally:
            store.close()
    except (ArchexError, ImpactError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(render_impact_report(report, output_format), nl=False)
