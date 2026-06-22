"""CLI symbols subcommand: search symbols by name across a repository."""

from __future__ import annotations

import json
import logging

import click

from archex.api import get_files_token_count, search_symbols
from archex.exceptions import ArchexError
from archex.metrics.capture import record_structural_usage
from archex.metrics.health import note_metrics_recording_failure
from archex.metrics.policy import resolve_metrics_policy
from archex.models import PipelineTiming, RepoSource, SymbolMatch
from archex.reporting import count_tokens, print_savings, print_timing
from archex.utils import resolve_source

logger = logging.getLogger(__name__)


@click.command("symbols")
@click.argument("args", nargs=-1, required=True)
@click.option("--kind", default=None, help="Filter by kind: function, class, method, etc.")
@click.option("-l", "--language", default=None, help="Filter to specific language.")
@click.option("--limit", default=20, type=int, help="Maximum results.")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON.")
@click.option("--timing", is_flag=True, default=False, help="Print timing breakdown.")
def symbols_cmd(
    args: tuple[str, ...],
    kind: str | None,
    language: str | None,
    limit: int,
    output_json: bool,
    timing: bool,
) -> None:
    """Search symbols by name across a repository."""
    source, query = _source_and_query(args)
    source_obj = resolve_source(source)

    pt = PipelineTiming() if timing else None
    try:
        results = search_symbols(
            source_obj, query=query, kind=kind, language=language, limit=limit, timing=pt
        )
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        output = json.dumps([m.model_dump() for m in results], indent=2)
    else:
        output = _render_symbols(results)
    click.echo(output)

    raw_tokens: int | None = None
    unique_files = list({m.file_path for m in results})
    if timing and pt is not None:
        print_timing(pt)
        returned = count_tokens(output) if output_json else len(results)
        raw_tokens = get_files_token_count(source_obj, unique_files)
        if raw_tokens is not None:
            print_savings(returned, raw_tokens, pt.total_ms, file_count=len(unique_files))
    _record_metrics(source_obj, output, raw_tokens, unique_files)


def _render_symbols(results: list[SymbolMatch]) -> str:
    if not results:
        return "No symbols found."
    ck = max((len(m.kind) for m in results), default=4)
    cn = max((len(m.name) for m in results), default=4)
    cf = max((len(m.file_path) for m in results), default=9)
    hdr = f"{'kind':<{ck}}  {'name':<{cn}}  {'file_path':<{cf}}  {'line':<6}  symbol_id"
    lines = [hdr, "-" * len(hdr)]
    for m in results:
        lines.append(
            f"{m.kind:<{ck}}  {m.name:<{cn}}  {m.file_path:<{cf}}  {m.start_line:<6}  {m.symbol_id}"
        )
    return "\n".join(lines)


def _record_metrics(
    source: RepoSource,
    output: str,
    raw_tokens: int | None,
    unique_files: list[str],
) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        raw = raw_tokens if raw_tokens is not None else get_files_token_count(source, unique_files)
        returned = count_tokens(output)
        record_structural_usage(
            source,
            surface="cli",
            tool_name="symbols",
            tokens_returned=returned,
            tokens_raw_equivalent=raw,
            file_count=len(unique_files),
        )
    except Exception as exc:
        note_metrics_recording_failure(exc)


def _source_and_query(args: tuple[str, ...]) -> tuple[str, str]:
    if len(args) == 1:
        return ".", args[0]
    if len(args) >= 2:
        return args[0], " ".join(args[1:])
    raise click.UsageError("symbols requires a query")
