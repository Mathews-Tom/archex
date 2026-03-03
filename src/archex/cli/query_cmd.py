"""CLI query subcommand: retrieve a ContextBundle from a repository."""

from __future__ import annotations

import click

from archex.api import get_files_token_count, query
from archex.exceptions import ArchexError
from archex.reporting import print_savings, print_timing
from archex.utils import resolve_source


@click.command("query")
@click.argument("source")
@click.argument("question")
@click.option("--budget", default=8192, type=int, help="Token budget for the context bundle.")
@click.option(
    "--format",
    "output_format",
    default="xml",
    type=click.Choice(["xml", "json", "markdown"]),
    help="Output format.",
)
@click.option("-l", "--language", multiple=True, help="Filter to specific languages.")
@click.option(
    "--strategy",
    type=click.Choice(["bm25", "hybrid"]),
    default="bm25",
    show_default=True,
    help="Retrieval strategy.",
)
@click.option("--timing", is_flag=True, default=False, help="Print timing breakdown.")
def query_cmd(
    source: str,
    question: str,
    budget: int,
    output_format: str,
    language: tuple[str, ...],
    strategy: str,
    timing: bool,
) -> None:
    """Query a repository and return a context bundle."""
    from archex.models import Config, IndexConfig, PipelineTiming

    repo_source = resolve_source(source)
    config = Config(languages=list(language) if language else None)
    index_config = IndexConfig(vector=(strategy == "hybrid"))

    pt = PipelineTiming() if timing else None
    try:
        bundle = query(
            repo_source,
            question,
            token_budget=budget,
            config=config,
            index_config=index_config,
            timing=pt,
        )
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(bundle.to_prompt(format=output_format))

    if timing and pt is not None:
        print_timing(pt)
        unique_files = list({c.chunk.file_path for c in bundle.chunks})
        raw = get_files_token_count(repo_source, unique_files, config)
        print_savings(
            bundle.token_count, raw, pt.total_ms, budget=budget, file_count=len(unique_files)
        )
