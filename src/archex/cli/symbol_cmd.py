"""CLI symbol subcommand: retrieve full source for a single symbol by ID."""

from __future__ import annotations

import logging

import click

from archex.api import get_file_token_count, get_symbol
from archex.exceptions import ArchexError
from archex.metrics.capture import record_structural_usage
from archex.metrics.health import note_metrics_recording_failure
from archex.metrics.policy import resolve_metrics_policy
from archex.models import PipelineTiming, RepoSource
from archex.reporting import count_tokens, print_savings, print_timing
from archex.utils import resolve_source

logger = logging.getLogger(__name__)


@click.command("symbol")
@click.argument("args", nargs=-1, required=True)
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON.")
@click.option("--timing", is_flag=True, default=False, help="Print timing breakdown.")
def symbol_cmd(args: tuple[str, ...], output_json: bool, timing: bool) -> None:
    """Retrieve the full source for a single symbol by its stable ID."""
    source, symbol_id = _source_and_symbol_id(args)
    source_obj = resolve_source(source)

    pt = PipelineTiming() if timing else None
    try:
        result = get_symbol(source_obj, symbol_id=symbol_id, timing=pt)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    if result is None:
        raise click.ClickException(f"Symbol not found: {symbol_id}")

    if output_json:
        output = result.model_dump_json(indent=2)
        click.echo(output)
    else:
        loc = f"{result.file_path}:{result.start_line}-{result.end_line}"
        output = f"# {result.name} ({result.kind}) — {loc}\n{result.source}"
        click.echo(f"# {result.name} ({result.kind}) — {loc}")
        click.echo(result.source)

    raw_tokens: int | None = None
    if timing and pt is not None:
        print_timing(pt)
        raw_tokens = get_file_token_count(source_obj, result.file_path)
        if raw_tokens is not None:
            print_savings(result.token_count, raw_tokens, pt.total_ms)
    _record_metrics(source_obj, output, result.token_count, raw_tokens, result.file_path)


def _record_metrics(
    source: RepoSource,
    output: str,
    returned_tokens: int,
    raw_tokens: int | None,
    file_path: str,
) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        raw = raw_tokens if raw_tokens is not None else get_file_token_count(source, file_path)
        record_structural_usage(
            source,
            surface="cli",
            tool_name="symbol",
            tokens_returned=returned_tokens if returned_tokens > 0 else count_tokens(output),
            tokens_raw_equivalent=raw,
            file_count=1,
        )
    except Exception as exc:
        note_metrics_recording_failure(exc)


def _source_and_symbol_id(args: tuple[str, ...]) -> tuple[str, str]:
    if len(args) == 1:
        return ".", args[0]
    if len(args) == 2:
        return args[0], args[1]
    raise click.UsageError("symbol requires a symbol ID, or source and symbol ID")
