"""CLI symbols subcommand: search symbols by name across a repository."""

from __future__ import annotations

import json

import click

from archex.api import get_files_token_count, search_symbols
from archex.exceptions import ArchexError
from archex.models import PipelineTiming
from archex.reporting import count_tokens, print_savings, print_timing
from archex.utils import resolve_source


@click.command("symbols")
@click.argument("source")
@click.argument("query")
@click.option("--kind", default=None, help="Filter by kind: function, class, method, etc.")
@click.option("-l", "--language", default=None, help="Filter to specific language.")
@click.option("--limit", default=20, type=int, help="Maximum results.")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON.")
@click.option("--timing", is_flag=True, default=False, help="Print timing breakdown.")
def symbols_cmd(
    source: str,
    query: str,
    kind: str | None,
    language: str | None,
    limit: int,
    output_json: bool,
    timing: bool,
) -> None:
    """Search symbols by name across a repository."""
    source_obj = resolve_source(source)

    pt = PipelineTiming() if timing else None
    try:
        results = search_symbols(
            source_obj, query=query, kind=kind, language=language, limit=limit, timing=pt
        )
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_json:
        click.echo(json.dumps([m.model_dump() for m in results], indent=2))
    else:
        if not results:
            click.echo("No symbols found.")
        else:
            ck = max((len(m.kind) for m in results), default=4)
            cn = max((len(m.name) for m in results), default=4)
            cf = max((len(m.file_path) for m in results), default=9)
            hdr = f"{'kind':<{ck}}  {'name':<{cn}}  {'file_path':<{cf}}  {'line':<6}  symbol_id"
            click.echo(hdr)
            click.echo("-" * len(hdr))
            for m in results:
                click.echo(
                    f"{m.kind:<{ck}}  {m.name:<{cn}}  "
                    f"{m.file_path:<{cf}}  {m.start_line:<6}  "
                    f"{m.symbol_id}"
                )

    if timing and pt is not None:
        print_timing(pt)
        output_text = json.dumps([m.model_dump() for m in results], indent=2) if output_json else ""
        returned = count_tokens(output_text) if output_text else len(results)
        unique_files = list({m.file_path for m in results})
        raw = get_files_token_count(source_obj, unique_files)
        print_savings(returned, raw, pt.total_ms, file_count=len(unique_files))
