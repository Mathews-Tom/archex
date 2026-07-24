"""Report command group: read-only diff-review artifact projections."""

from __future__ import annotations

import click

from archex.exceptions import ArchexError
from archex.report.artifact import ReportArtifactError, build_analysis_artifact
from archex.report.release_artifact import CompatibilityArtifactError, build_compatibility_artifact
from archex.report.status_card import StatusCardError, build_status_card


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


@report_cmd.command("delta")
@click.argument("source", required=False, default=".")
@click.option("--base", default="main", help="Base ref to diff against.")
@click.option(
    "--format",
    "output_format",
    default="json",
    type=click.Choice(["json", "markdown"]),
    help="Output format.",
)
def delta_cmd(source: str, base: str, output_format: str) -> None:
    """Build a bounded, CI-log-sized delta summary of the diff-review artifact for SOURCE."""
    try:
        artifact = build_analysis_artifact(source, base_ref=base)
    except (ArchexError, ReportArtifactError) as exc:
        raise click.ClickException(str(exc)) from exc

    from archex.report.delta import build_report_delta

    delta = build_report_delta(artifact)
    if output_format == "markdown":
        click.echo(delta.to_markdown(), nl=False)
        return
    click.echo(delta.to_json())


@report_cmd.command("status-card")
@click.argument("source", required=False, default=".")
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["json", "markdown"]),
    help="Output format.",
)
def status_card_cmd(source: str, output_format: str) -> None:
    """Build and render the dimensioned, evidence-linked documentation/release status card.

    Every dimension is UNKNOWN unless its corresponding provider is
    configured on the index (documentation_evidence_providers). Never
    computes a composite score or letter grade, and never writes the
    result back into SOURCE -- pipe the output into your own README by
    hand when you want to publish it.
    """
    try:
        card = build_status_card(source)
    except (ArchexError, StatusCardError) as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        click.echo(card.to_json())
        return

    from archex.report.render_status_card import render_status_card_markdown

    click.echo(render_status_card_markdown(card), nl=False)


@report_cmd.command("release-artifact")
@click.argument("source", required=False, default=".")
def release_artifact_cmd(source: str) -> None:
    """Build the per-release CompatibilityArtifact: version/schema facts plus SOURCE's status card.

    Bundles archex's own installed version, supported Python range, and
    report/index schema versions with SOURCE's dimensioned status card and
    a pointer to any checked-in benchmark evidence into one read-only
    artifact suitable for attaching to a GitHub release. Never runs a
    benchmark or mutates SOURCE.
    """
    try:
        artifact = build_compatibility_artifact(source)
    except (ArchexError, CompatibilityArtifactError, StatusCardError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(artifact.to_json())
