"""CLI outline subcommand: display the symbol hierarchy for a single file."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from archex.api import file_outline
from archex.exceptions import ArchexError
from archex.models import PipelineTiming
from archex.reporting import count_tokens, print_savings, print_timing
from archex.utils import resolve_source

if TYPE_CHECKING:
    from archex.models import SymbolOutline


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
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(f"file: {result.file_path}")
        click.echo(f"language: {result.language}")
        click.echo(f"lines: {result.lines}")
        for line in _render_symbols(result.symbols):
            click.echo(line)

    if timing and pt is not None:
        print_timing(pt)
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
        returned = count_tokens(output)
        print_savings(returned, result.token_count_raw, pt.total_ms)
