"""CLI outline subcommand: display the symbol hierarchy for a single file."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import click

from archex.api import file_outline
from archex.exceptions import ArchexError
from archex.metrics.capture import record_structural_usage
from archex.metrics.health import record_metrics_failure
from archex.metrics.policy import resolve_metrics_policy
from archex.models import PipelineTiming, RepoSource
from archex.reporting import count_tokens, print_savings, print_timing
from archex.utils import resolve_source

if TYPE_CHECKING:
    from archex.models import SymbolOutline

logger = logging.getLogger(__name__)


def _render_symbols(symbols: list[SymbolOutline], indent: int = 0) -> list[str]:
    lines: list[str] = []
    pad = "  " * indent
    for sym in symbols:
        sig = f" {sym.signature}" if sym.signature else ""
        lines.append(f"{pad}{sym.kind} {sym.name} [L{sym.start_line}-{sym.end_line}]{sig}")
        lines.extend(_render_symbols(sym.children, indent + 1))
    return lines


@click.command("outline")
@click.argument("source")
@click.argument("file_path")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON.")
@click.option("--timing", is_flag=True, default=False, help="Print timing breakdown.")
def outline_cmd(source: str, file_path: str, output_json: bool, timing: bool) -> None:
    """Display the symbol outline for a single file in a repository."""
    source_obj = resolve_source(source)

    pt = PipelineTiming() if timing else None
    try:
        result = file_outline(source_obj, file_path=file_path, timing=pt)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        output = result.model_dump_json(indent=2)
    else:
        lines = [
            f"file: {result.file_path}",
            f"language: {result.language}",
            f"lines: {result.lines}",
        ]
        lines.extend(_render_symbols(result.symbols))
        output = "\n".join(lines)
    click.echo(output)

    if timing and pt is not None:
        print_timing(pt)
        returned = count_tokens(output)
        print_savings(returned, result.token_count_raw, pt.total_ms)
    _record_metrics(source_obj, output, result.token_count_raw)


def _record_metrics(source: RepoSource, output: str, raw_tokens: int) -> None:
    try:
        policy = resolve_metrics_policy()
        if not policy.metrics_enabled:
            return
        record_structural_usage(
            source,
            surface="cli",
            tool_name="outline",
            tokens_returned=count_tokens(output),
            tokens_raw_equivalent=raw_tokens,
            file_count=1,
        )
    except Exception as exc:
        logger.debug("outline metrics recording failed", exc_info=True)
        try:
            record_metrics_failure("record", str(exc))
        except Exception:
            logger.debug("outline metrics health recording failed", exc_info=True)
