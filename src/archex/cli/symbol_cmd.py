"""CLI symbol subcommand: retrieve full source for a single symbol by ID."""

from __future__ import annotations

import click

from archex.api import get_file_token_count, get_symbol
from archex.exceptions import ArchexError
from archex.models import PipelineTiming
from archex.reporting import print_savings, print_timing
from archex.utils import resolve_source


@click.command("symbol")
@click.argument("source")
@click.argument("symbol_id")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON.")
@click.option("--timing", is_flag=True, default=False, help="Print timing breakdown.")
def symbol_cmd(source: str, symbol_id: str, output_json: bool, timing: bool) -> None:
    """Retrieve the full source for a single symbol by its stable ID."""
    source_obj = resolve_source(source)

    pt = PipelineTiming() if timing else None
    try:
        result = get_symbol(source_obj, symbol_id=symbol_id, timing=pt)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    if result is None:
        raise click.ClickException(f"Symbol not found: {symbol_id}")

    if output_json:
        click.echo(result.model_dump_json(indent=2))
    else:
        loc = f"{result.file_path}:{result.start_line}-{result.end_line}"
        click.echo(f"# {result.name} ({result.kind}) — {loc}")
        click.echo(result.source)

    if timing and pt is not None:
        print_timing(pt)
        raw = get_file_token_count(source_obj, result.file_path)
        print_savings(result.token_count, raw, pt.total_ms)
