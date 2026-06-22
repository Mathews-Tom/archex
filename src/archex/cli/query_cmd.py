"""CLI query subcommand: retrieve a ContextBundle from a repository."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import click

from archex.api import get_files_token_count, get_repo_total_tokens, query
from archex.exceptions import ArchexError
from archex.metrics.capture import record_query_usage
from archex.metrics.health import note_metrics_recording_failure
from archex.metrics.policy import resolve_metrics_policy
from archex.reporting import print_savings, print_timing
from archex.serve.intent import DEFAULT_TOKEN_BUDGET
from archex.utils import resolve_source

if TYPE_CHECKING:
    from archex.models import Config, ContextBundle, RepoSource

logger = logging.getLogger(__name__)


@click.command("query")
@click.argument("args", nargs=-1, required=True)
@click.option(
    "--budget",
    default=None,
    show_default=str(DEFAULT_TOKEN_BUDGET),
    type=int,
    help="Override the intent-routed token budget for the context bundle.",
)
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
@click.option(
    "--module-prefilter",
    is_flag=True,
    default=False,
    help="Use opt-in module responsibility prefiltering.",
)
@click.option(
    "--allow-remote-code",
    is_flag=True,
    default=False,
    help="Allow explicitly selected pinned model paths that require Hugging Face remote code.",
)
@click.option(
    "--no-refresh",
    is_flag=True,
    default=False,
    help="Skip working-tree freshness checks and query the existing index as-is.",
)
@click.option(
    "--refresh-threshold",
    type=float,
    default=None,
    help="Maximum changed-file ratio eligible for inline refresh.",
)
def query_cmd(
    args: tuple[str, ...],
    budget: int | None,
    output_format: str,
    language: tuple[str, ...],
    strategy: str | None,
    timing: bool,
    metrics: bool,
    splade: bool,
    module_prefilter: bool,
    allow_remote_code: bool,
    no_refresh: bool,
    refresh_threshold: float | None,
) -> None:
    """Query a repository and return a context bundle."""
    from archex.config import load_config, load_index_config
    from archex.models import PipelineTiming

    source, question = _source_and_question(args)
    repo_source = resolve_source(source)
    config = load_config(repo_source)
    if refresh_threshold is not None:
        config = config.model_copy(update={"delta_threshold": refresh_threshold})
    if language:
        config = config.model_copy(update={"languages": list(language)})
    index_config = load_index_config(repo_source)
    if strategy is not None:
        index_config = index_config.model_copy(update={"vector": strategy == "hybrid"})
    if allow_remote_code:
        index_config = index_config.model_copy(update={"allow_remote_code": True})
    if splade:
        index_config = index_config.model_copy(update={"splade": True})
    if module_prefilter:
        index_config = index_config.model_copy(update={"module_prefilter": True})

    pt = PipelineTiming() if (timing or metrics) else None
    token_budget = DEFAULT_TOKEN_BUDGET if budget is None else budget
    try:
        bundle = query(
            repo_source,
            question,
            token_budget=token_budget,
            explicit_token_budget=budget is not None,
            config=config,
            index_config=index_config,
            timing=pt,
            refresh=not no_refresh,
        )
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(bundle.to_prompt(format=output_format))

    unique_files = list({c.chunk.file_path for c in bundle.chunks})
    raw_tokens: int | None = None
    if timing and pt is not None:
        print_timing(pt)
        raw_tokens = get_files_token_count(repo_source, unique_files, config)
        if raw_tokens is not None:
            print_savings(
                bundle.token_count,
                raw_tokens,
                pt.total_ms,
                budget=token_budget,
                file_count=len(unique_files),
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

    _record_metrics(repo_source, bundle, raw_tokens, unique_files, config)


def _record_metrics(
    repo_source: RepoSource,
    bundle: ContextBundle,
    raw_tokens: int | None,
    unique_files: list[str],
    config: Config,
) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        raw = raw_tokens
        if raw is None:
            raw = get_files_token_count(repo_source, unique_files, config)
        whole_repo_tokens = _repo_total_tokens(repo_source, config)
        record_query_usage(
            repo_source,
            bundle,
            surface="cli",
            tokens_raw_equivalent=raw,
            whole_repo_tokens=whole_repo_tokens,
        )
    except Exception as exc:
        note_metrics_recording_failure(exc)


def _repo_total_tokens(repo_source: RepoSource, config: Config) -> int | None:
    try:
        return get_repo_total_tokens(repo_source, config)
    except Exception:
        logger.debug("query metrics whole-repo tokens unavailable", exc_info=True)
        return None


def _source_and_question(args: tuple[str, ...]) -> tuple[str, str]:
    if len(args) == 1:
        return ".", args[0]
    if len(args) >= 2:
        return args[0], " ".join(args[1:])
    raise click.UsageError("query requires a question")
