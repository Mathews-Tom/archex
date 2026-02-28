"""CLI query subcommand: retrieve a ContextBundle from a repository."""

from __future__ import annotations

import time

import click

from archex.api import query
from archex.exceptions import ArchexError
from archex.models import RepoSource


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
    from archex.models import Config, IndexConfig

    repo_source = RepoSource(
        url=source if source.startswith("http") else None,
        local_path=source if not source.startswith("http") else None,
    )
    config = Config(languages=list(language) if language else None)
    index_config = IndexConfig(vector=(strategy == "hybrid"))

    t0 = time.perf_counter()
    try:
        bundle = query(
            repo_source, question, token_budget=budget, config=config, index_config=index_config
        )
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000

    click.echo(bundle.to_prompt(format=output_format))

    if timing:
        click.echo(f"\n--- Timing: {elapsed_ms:.0f}ms total ---", err=True)
