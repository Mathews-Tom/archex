"""CLI analyze subcommand: acquire and analyze a repository, writing an ArchProfile."""

from __future__ import annotations

import logging

import click

from archex.api import analyze, get_repo_total_tokens
from archex.config import load_config
from archex.exceptions import ArchexError
from archex.metrics.capture import record_structural_usage
from archex.metrics.health import record_metrics_failure
from archex.metrics.policy import resolve_metrics_policy
from archex.models import Config, PipelineTiming, RepoSource
from archex.reporting import count_tokens, print_savings, print_timing
from archex.utils import resolve_source

logger = logging.getLogger(__name__)


@click.command("analyze")
@click.argument("source", required=False, default=".")
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

    config = load_config(source_obj)
    if languages:
        config = config.model_copy(update={"languages": list(languages)})

    pt = PipelineTiming() if timing else None
    try:
        profile = analyze(source_obj, config, timing=pt)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    output = profile.to_json() if output_format == "json" else profile.to_markdown()
    click.echo(output)

    raw_tokens: int | None = None
    if timing and pt is not None:
        print_timing(pt)
        returned = count_tokens(output)
        raw_tokens = get_repo_total_tokens(source_obj, config)
        print_savings(returned, raw_tokens, pt.total_ms)
    _record_metrics(source_obj, output, raw_tokens, config)


def _record_metrics(
    source: RepoSource,
    output: str,
    raw_tokens: int | None,
    config: Config,
) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        raw = raw_tokens if raw_tokens is not None else get_repo_total_tokens(source, config)
        record_structural_usage(
            source,
            surface="cli",
            tool_name="analyze",
            tokens_returned=count_tokens(output),
            tokens_raw_equivalent=raw,
            whole_repo_tokens=raw,
        )
    except Exception as exc:
        logger.debug("analyze metrics recording failed", exc_info=True)
        try:
            record_metrics_failure("record", str(exc))
        except Exception:
            logger.debug("analyze metrics health recording failed", exc_info=True)
