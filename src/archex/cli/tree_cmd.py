"""CLI tree subcommand: display annotated file tree for a repository."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import click

from archex.api import file_tree, get_repo_total_tokens
from archex.exceptions import ArchexError
from archex.metrics.capture import record_structural_usage
from archex.metrics.health import record_metrics_failure
from archex.metrics.policy import resolve_metrics_policy
from archex.models import PipelineTiming, RepoSource
from archex.reporting import count_tokens, print_savings, print_timing
from archex.utils import resolve_source

if TYPE_CHECKING:
    from archex.models import FileTreeEntry

logger = logging.getLogger(__name__)


def _render_tree(
    entries: list[FileTreeEntry], prefix: str = "", is_last_list: list[bool] | None = None
) -> list[str]:
    lines: list[str] = []
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "+-- " if is_last else "|-- "
        if entry.is_directory:
            name = entry.path.split("/")[-1] + "/"
            lines.append(f"{prefix}{connector}{name}")
            child_prefix = prefix + ("    " if is_last else "|   ")
            lines.extend(_render_tree(entry.children, child_prefix))
        else:
            name = entry.path.split("/")[-1]
            lang = entry.language or "unknown"
            info = f"{lang}, {entry.lines} lines, {entry.symbol_count} symbols"
            lines.append(f"{prefix}{connector}{name} ({info})")
    return lines


@click.command("tree")
@click.argument("source", required=False, default=".")
@click.option("--depth", default=5, type=int, help="Maximum directory depth.")
@click.option("-l", "--language", default=None, help="Filter to specific language.")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON.")
@click.option("--timing", is_flag=True, default=False, help="Print timing breakdown.")
def tree_cmd(
    source: str, depth: int, language: str | None, output_json: bool, timing: bool
) -> None:
    """Display the annotated file tree of a repository."""
    source_obj = resolve_source(source)

    pt = PipelineTiming() if timing else None
    try:
        result = file_tree(source_obj, max_depth=depth, language=language, timing=pt)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        output = result.model_dump_json(indent=2)
    else:
        output = "\n".join([result.root, *_render_tree(result.entries)])
    click.echo(output)

    raw_tokens: int | None = None
    if timing and pt is not None:
        print_timing(pt)
        returned = count_tokens(output)
        raw_tokens = get_repo_total_tokens(source_obj)
        print_savings(returned, raw_tokens, pt.total_ms)
    _record_metrics(source_obj, output, raw_tokens)


def _record_metrics(source: RepoSource, output: str, raw_tokens: int | None) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        raw = raw_tokens if raw_tokens is not None else get_repo_total_tokens(source)
        record_structural_usage(
            source,
            surface="cli",
            tool_name="tree",
            tokens_returned=count_tokens(output),
            tokens_raw_equivalent=raw,
            whole_repo_tokens=raw,
        )
    except Exception as exc:
        logger.debug("tree metrics recording failed", exc_info=True)
        try:
            record_metrics_failure("record", str(exc))
        except Exception:
            logger.debug("tree metrics health recording failed", exc_info=True)
