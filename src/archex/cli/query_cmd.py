"""CLI query subcommand: retrieve a ContextBundle from a repository."""

from __future__ import annotations

import click

from archex.api import query
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
def query_cmd(
    source: str,
    question: str,
    budget: int,
    output_format: str,
    language: tuple[str, ...],
) -> None:
    """Query a repository and return a context bundle."""
    from archex.models import Config

    repo_source = RepoSource(
        url=source if source.startswith("http") else None,
        local_path=source if not source.startswith("http") else None,
    )
    config = Config(languages=list(language) if language else None)

    bundle = query(repo_source, question, token_budget=budget, config=config)
    click.echo(bundle.to_prompt(format=output_format))
