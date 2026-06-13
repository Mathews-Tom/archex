"""CLI scout subcommand: emit a token-capped structural map without code bodies."""

from __future__ import annotations

import click

from archex.api import scout
from archex.exceptions import ArchexError
from archex.scout import DEFAULT_SCOUT_TOKEN_BUDGET, ScoutFormat, render_scout
from archex.utils import resolve_source


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
    click.echo(render_scout(result, output_format=output_format), nl=False)


def _source_and_question(args: tuple[str, ...]) -> tuple[str, str]:
    if len(args) == 1:
        return ".", args[0]
    if len(args) >= 2:
        return args[0], " ".join(args[1:])
    raise click.UsageError("scout requires a question")
