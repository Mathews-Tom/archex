"""Report command group: read-only diff-review artifact projections."""

from __future__ import annotations

import click

from archex.exceptions import ArchexError
from archex.report.artifact import ReportArtifactError, build_analysis_artifact


@click.group("report")
def report_cmd() -> None:
    """Read-only diff-review artifact projections."""


@report_cmd.command("diff")
@click.argument("source", required=False, default=".")
@click.option("--base", default="main", help="Base ref to diff against.")
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["json", "markdown", "html"]),
    help="Output format.",
)
def diff_cmd(source: str, base: str, output_format: str) -> None:
    """Build and render the canonical AnalysisArtifactV1 diff-review artifact for SOURCE."""
    try:
        artifact = build_analysis_artifact(source, base_ref=base)
    except (ArchexError, ReportArtifactError) as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        click.echo(artifact.to_json())
        return

    if output_format == "html":
        from archex.report.render_html import render_html

        click.echo(render_html(artifact), nl=False)
        return

    from archex.report.render_markdown import render_markdown

    click.echo(render_markdown(artifact), nl=False)
