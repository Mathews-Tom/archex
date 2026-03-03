"""CLI compare subcommand: compare two repositories and produce a ComparisonResult."""

from __future__ import annotations

import time

import click

from archex.api import compare, get_repo_total_tokens
from archex.exceptions import ArchexError
from archex.models import Config
from archex.reporting import count_tokens, print_savings
from archex.serve.compare import SUPPORTED_DIMENSIONS
from archex.utils import resolve_source


def render_comparison_markdown(result: object) -> str:
    """Render a ComparisonResult as markdown."""
    from archex.models import ComparisonResult

    if not isinstance(result, ComparisonResult):
        raise TypeError(f"Expected ComparisonResult, got {type(result).__name__}")

    name_a = result.repo_a.url or result.repo_a.local_path or "Repo A"
    name_b = result.repo_b.url or result.repo_b.local_path or "Repo B"

    lines: list[str] = [
        f"# Comparison: {name_a} vs {name_b}",
        "",
        f"**Repo A:** {result.repo_a.total_files} files, {result.repo_a.total_lines} lines",
        f"**Repo B:** {result.repo_b.total_files} files, {result.repo_b.total_lines} lines",
        "",
    ]

    if result.dimensions:
        lines.append("## Dimensions")
        lines.append("")

        for dim in result.dimensions:
            lines.append(f"### {dim.dimension.replace('_', ' ').title()}")
            lines.append("")
            lines.append("| | Repo A | Repo B |")
            lines.append("|---|--------|--------|")
            lines.append(f"| Approach | {dim.repo_a_approach} | {dim.repo_b_approach} |")
            lines.append(
                f"| Evidence | {len(dim.evidence_a)} item(s) | {len(dim.evidence_b)} item(s) |"
            )
            lines.append("")

            if dim.evidence_a:
                lines.append("**Repo A evidence:**")
                for ev in dim.evidence_a:
                    lines.append(f"- {ev}")
                lines.append("")

            if dim.evidence_b:
                lines.append("**Repo B evidence:**")
                for ev in dim.evidence_b:
                    lines.append(f"- {ev}")
                lines.append("")

            if dim.trade_offs:
                lines.append("**Trade-offs:**")
                for tf in dim.trade_offs:
                    lines.append(f"- {tf}")
                lines.append("")

    if result.summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(result.summary)
        lines.append("")

    return "\n".join(lines)


@click.command("compare")
@click.argument("source_a")
@click.argument("source_b")
@click.option(
    "--dimensions",
    "dimensions_str",
    default=None,
    help="Comma-separated dimensions. Supported: " + ", ".join(sorted(SUPPORTED_DIMENSIONS)),
)
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
def compare_cmd(
    source_a: str,
    source_b: str,
    dimensions_str: str | None,
    output_format: str,
    languages: tuple[str, ...],
    timing: bool,
) -> None:
    """Compare two repositories across architectural dimensions."""
    dims: list[str] | None = None
    if dimensions_str is not None:
        dims = [d.strip() for d in dimensions_str.split(",") if d.strip()]

    lang_list: list[str] | None = list(languages) if languages else None
    config = Config(languages=lang_list)

    source_a_obj = resolve_source(source_a)
    source_b_obj = resolve_source(source_b)

    t0 = time.perf_counter()
    try:
        result = compare(source_a_obj, source_b_obj, dimensions=dims, config=config)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if output_format == "json":
        output = result.model_dump_json(indent=2)
        click.echo(output)
    else:
        output = render_comparison_markdown(result)
        click.echo(output)

    if timing:
        returned = count_tokens(output)
        raw = get_repo_total_tokens(source_a_obj, config) + get_repo_total_tokens(
            source_b_obj, config
        )
        print_savings(returned, raw, elapsed_ms)
