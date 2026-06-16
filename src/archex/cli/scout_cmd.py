"""CLI scout subcommand: emit a token-capped structural map without code bodies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import click

from archex.api import get_files_token_count, get_repo_total_tokens, scout
from archex.exceptions import ArchexError
from archex.metrics.capture import record_scout_usage
from archex.metrics.health import record_metrics_failure
from archex.metrics.policy import resolve_metrics_policy
from archex.reporting import count_tokens
from archex.scout import DEFAULT_SCOUT_TOKEN_BUDGET, ScoutFormat, render_scout
from archex.utils import resolve_source

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from archex.models import RepoSource
    from archex.scout import ScoutResult


@click.command("scout")
@click.argument("args", nargs=-1, required=True)
@click.option(
    "--budget",
    default=DEFAULT_SCOUT_TOKEN_BUDGET,
    show_default=True,
    type=int,
    help="Hard token cap for the structural scout map.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format.",
)
@click.option(
    "--no-refresh",
    is_flag=True,
    default=False,
    help="Skip working-tree freshness checks and scout the existing index as-is.",
)
def scout_cmd(
    args: tuple[str, ...],
    budget: int,
    output_format: ScoutFormat,
    no_refresh: bool,
) -> None:
    """Return a compact no-body structural map for a repository question."""
    source, question = _source_and_question(args)
    repo_source = resolve_source(source)
    try:
        result = scout(
            repo_source,
            question,
            token_budget=budget,
            output_format=output_format,
            refresh=not no_refresh,
        )
    except (ArchexError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    rendered = render_scout(result, output_format=output_format)
    click.echo(rendered, nl=False)
    _record_metrics(repo_source, result, rendered)


def _record_metrics(repo_source: RepoSource, result: ScoutResult, rendered: str) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        returned_file_paths = _scout_file_paths(result)
        raw = get_files_token_count(repo_source, returned_file_paths)
        whole_repo_tokens = get_repo_total_tokens(repo_source)
        record_scout_usage(
            repo_source,
            result,
            surface="cli",
            tokens_returned=count_tokens(rendered),
            tokens_raw_equivalent=raw,
            whole_repo_tokens=whole_repo_tokens,
        )
    except Exception as exc:
        logger.debug("scout metrics recording failed", exc_info=True)
        try:
            record_metrics_failure("record", str(exc))
        except Exception:
            logger.debug("scout metrics health recording failed", exc_info=True)


def _scout_file_paths(result: ScoutResult) -> list[str]:
    paths = {item.path for item in result.ranked_files}
    paths.update(symbol.file_path for symbol in result.symbols)
    for module in result.modules:
        paths.update(module.relevant_files)
    for edge in result.graph:
        if edge.source_path is not None:
            paths.add(edge.source_path)
        if edge.target_path is not None:
            paths.add(edge.target_path)
    return sorted(paths)


def _source_and_question(args: tuple[str, ...]) -> tuple[str, str]:
    if len(args) == 1:
        return ".", args[0]
    if len(args) >= 2:
        return args[0], " ".join(args[1:])
    raise click.UsageError("scout requires a question")
