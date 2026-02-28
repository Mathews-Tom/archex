"""CLI analyze subcommand: acquire and analyze a repository, writing an ArchProfile."""

from __future__ import annotations

import time

import click

from archex.api import analyze
from archex.exceptions import ArchexError
from archex.models import Config, RepoSource


@click.command("analyze")
@click.argument("source")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown"]),
    default="json",
    show_default=True,
    help="Output format.",
)
@click.option(
    "-l",
    "--language",
    "languages",
    multiple=True,
    help="Filter by language (may be repeated).",
)
@click.option("--timing", is_flag=True, default=False, help="Print timing breakdown.")
def analyze_cmd(source: str, output_format: str, languages: tuple[str, ...], timing: bool) -> None:
    """Analyze a repository and produce an architecture profile."""
    if source.startswith("http://") or source.startswith("https://"):
        source_obj = RepoSource(url=source)
    else:
        source_obj = RepoSource(local_path=source)

    lang_list: list[str] | None = list(languages) if languages else None
    config = Config(languages=lang_list)

    t0 = time.perf_counter()
    try:
        profile = analyze(source_obj, config)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if output_format == "json":
        click.echo(profile.to_json())
    else:
        click.echo(profile.to_markdown())

    if timing:
        click.echo(f"\n--- Timing: {elapsed_ms:.0f}ms total ---", err=True)
