"""CLI doctor subcommand: report archex installation and project health."""

from __future__ import annotations

import json
from pathlib import Path

import click

from archex.doctor import inspect_doctor, render_doctor_text


@click.command("doctor")
@click.argument("source", required=False, default=".")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def doctor_cmd(source: str, output_format: str) -> None:
    """Run read-only diagnostics for an archex repo-local project."""
    try:
        report = inspect_doctor(Path(source))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        click.echo(json.dumps(report.to_payload(), indent=2, sort_keys=True))
    else:
        click.echo(render_doctor_text(report), nl=False)

    if report.has_errors():
        raise click.exceptions.Exit(1)
