"""CLI analyze subcommand: acquire and analyze a repository, writing an ArchProfile."""

from __future__ import annotations

import click

from archex.api import analyze, get_repo_total_tokens
from archex.exceptions import ArchexError
from archex.models import Config, PipelineTiming
from archex.reporting import count_tokens, print_savings, print_timing
from archex.utils import resolve_source


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
    source_obj = resolve_source(source)

    lang_list: list[str] | None = list(languages) if languages else None
    config = Config(languages=lang_list)

    pt = PipelineTiming() if timing else None
    try:
        profile = analyze(source_obj, config, timing=pt)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        click.echo(profile.to_json())
    else:
        click.echo(profile.to_markdown())

    if timing and pt is not None:
        print_timing(pt)
        output = profile.to_json() if output_format == "json" else profile.to_markdown()
        returned = count_tokens(output)
        raw = get_repo_total_tokens(source_obj, config)
        print_savings(returned, raw, pt.total_ms)
