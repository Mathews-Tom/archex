"""CLI tree subcommand: display annotated file tree for a repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

from archex.api import file_tree, get_repo_total_tokens
from archex.exceptions import ArchexError
from archex.models import PipelineTiming
from archex.reporting import count_tokens, print_savings, print_timing
from archex.utils import resolve_source

if TYPE_CHECKING:
    from archex.models import FileTreeEntry


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
@click.argument("source")
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
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(result.root)
        for line in _render_tree(result.entries):
            click.echo(line)

    if timing and pt is not None:
        print_timing(pt)
        if output_json:
            output = result.model_dump_json(indent=2)
        else:
            output = "\n".join([result.root, *_render_tree(result.entries)])
        returned = count_tokens(output)
        raw = get_repo_total_tokens(source_obj)
        print_savings(returned, raw, pt.total_ms)
