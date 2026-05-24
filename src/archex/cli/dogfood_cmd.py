"""Project dogfood command."""

from __future__ import annotations

import click

from archex.dogfood import run_dogfood


@click.command("dogfood")
@click.argument("source", required=False, default=".")
@click.option("--task", "task_id", default=None, help="Run a single dogfood task by task_id.")
@click.option(
    "--all",
    "run_all_self",
    is_flag=True,
    default=False,
    help="Run all self dogfood tasks.",
)
@click.option(
    "--include-external",
    is_flag=True,
    default=False,
    help="Include non-self benchmark tasks.",
)
@click.option(
    "--query-fusion",
    is_flag=True,
    default=False,
    help="Include the experimental archex_query_fusion strategy.",
)
@click.option(
    "--baseline",
    "baseline_path",
    default=None,
    type=click.Path(),
    help="Baseline JSON path. Defaults to local dogfood baseline, or committed CI baseline.",
)
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
def dogfood_cmd(
    source: str,
    task_id: str | None,
    run_all_self: bool,
    include_external: bool,
    query_fusion: bool,
    baseline_path: str | None,
    output_format: str,
) -> None:
    """Run self-benchmark dogfood tasks and gate against a stored baseline."""
    del run_all_self
    try:
        result = run_dogfood(
            source,
            task_id=task_id,
            include_external=include_external,
            include_query_fusion=query_fusion,
            baseline_path=baseline_path,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "json":
        click.echo(result.latest_json_path.read_text(encoding="utf-8").rstrip())
    else:
        click.echo(f"Dogfood tasks:      {len(result.tasks)}")
        click.echo(f"Reports:            {result.latest_json_path}")
        click.echo(f"Markdown:           {result.latest_markdown_path}")
        click.echo(f"History:            {result.history_json_path}")
        if result.baseline_path is None:
            click.echo("Baseline:           none")
        else:
            click.echo(f"Baseline:           {result.baseline_path}")
        if result.regressions:
            click.echo(f"Regressions:        {len(result.regressions)}")
            for regression in result.regressions:
                click.echo(
                    f"  {regression.task_id}/{regression.strategy} {regression.metric}: "
                    f"{regression.baseline_value:.3f} -> {regression.current_value:.3f} "
                    f"({regression.delta:+.3f})"
                )
        else:
            click.echo("Regressions:        0")

    if result.regressions:
        raise click.exceptions.Exit(1)
