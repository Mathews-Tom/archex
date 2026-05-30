"""CLI query subcommand: retrieve a ContextBundle from a repository."""

from __future__ import annotations

import click

from archex.api import get_files_token_count, query
from archex.exceptions import ArchexError
from archex.reporting import print_savings, print_timing
from archex.utils import resolve_source


@click.command("query")
@click.argument("args", nargs=-1, required=True)
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
    default=None,
    help="Retrieval strategy.",
)
@click.option("--timing", is_flag=True, default=False, help="Print timing breakdown.")
@click.option("--metrics", is_flag=True, default=False, help="Print timing metrics as JSON.")
@click.option("--splade", is_flag=True, default=False, help="Use the opt-in SPLADE retrieval leg.")
def query_cmd(
    args: tuple[str, ...],
    budget: int,
    output_format: str,
    language: tuple[str, ...],
    strategy: str | None,
    timing: bool,
    metrics: bool,
    splade: bool,
) -> None:
    """Query a repository and return a context bundle."""
    from archex.config import load_config, load_index_config
    from archex.models import PipelineTiming

    source, question = _source_and_question(args)
    repo_source = resolve_source(source)
    config = load_config(repo_source)
    if language:
        config = config.model_copy(update={"languages": list(language)})
    index_config = load_index_config(repo_source)
    if strategy is not None:
        index_config = index_config.model_copy(update={"vector": strategy == "hybrid"})
    if splade:
        index_config = index_config.model_copy(update={"splade": True})

    pt = PipelineTiming() if (timing or metrics) else None
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

    if metrics and pt is not None:
        import json
        from dataclasses import asdict

        metrics_dict = asdict(pt)
        # Remove non-serializable delta_meta for JSON output
        if metrics_dict.get("delta_meta") is not None:
            dm = metrics_dict["delta_meta"]
            metrics_dict["delta_meta"] = {k: v for k, v in dm.items()}
        click.echo(json.dumps(metrics_dict, indent=2), err=True)


def _source_and_question(args: tuple[str, ...]) -> tuple[str, str]:
    if len(args) == 1:
        return ".", args[0]
    if len(args) >= 2:
        return args[0], " ".join(args[1:])
    raise click.UsageError("query requires a question")
