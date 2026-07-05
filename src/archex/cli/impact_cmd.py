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
    analyze_diff_impact,
    analyze_impact,
    git_changed_files,
    git_diff_hunks,
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
    "--diff",
    "diff_ref",
    is_flag=False,
    flag_value="HEAD",
    help=(
        "Enable diff-scoped symbol impact: resolve the diff to touched symbols and "
        "classify each with a LOW/MEDIUM/HIGH risk tier from deterministic graph "
        "signals. Optional ref to diff against (default: HEAD, i.e. uncommitted "
        "working tree changes). Adds affected_symbols to the report; output is "
        "unchanged when omitted. Cannot be combined with --changed-file or --base."
    ),
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format.",
)
@click.pass_context
def impact_cmd(
    ctx: click.Context,
    source: str,
    base: str,
    changed_files: tuple[str, ...],
    diff_ref: str | None,
    output_format: str,
) -> None:
    """Analyze deterministic blast radius for changed files."""
    if diff_ref is not None:
        if changed_files:
            raise click.UsageError("--diff cannot be combined with --changed-file.")
        if ctx.get_parameter_source("base") is click.core.ParameterSource.COMMANDLINE:
            raise click.UsageError("--diff cannot be combined with --base.")

    repo_root = Path(source).expanduser().resolve()
    if changed_files:
        changes = [ImpactFileChange(path=path) for path in changed_files]
    else:
        try:
            changes = git_changed_files(repo_root, diff_ref if diff_ref is not None else base)
        except ImpactError as exc:
            raise click.ClickException(str(exc)) from exc

    repo_source = RepoSource(local_path=source)
    config = load_config(repo_source)
    index_config = load_index_config(repo_source)
    try:
        store = index_repository(repo_source, config=config, index_config=index_config)
        try:
            if diff_ref is not None:
                hunks = git_diff_hunks(repo_root, diff_ref)
                report = analyze_diff_impact(store, repo_root, changes, hunks, diff_ref)
            else:
                report = analyze_impact(store, changes)
        finally:
            store.close()
    except (ArchexError, ImpactError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(render_impact_report(report, output_format), nl=False)
