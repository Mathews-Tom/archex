"""CLI benchmark subcommands: run, report, validate."""

from __future__ import annotations

import json
from pathlib import Path

import click

from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkReport, Strategy
from archex.benchmark.reporter import format_json, format_markdown, format_summary
from archex.benchmark.runner import run_all


@click.group("benchmark")
def benchmark_cmd() -> None:
    """Benchmark archex retrieval strategies against real repos."""


@benchmark_cmd.command("run")
@click.option(
    "--output",
    "output_dir",
    default="benchmarks/results",
    type=click.Path(),
    help="Directory for result JSON files.",
)
@click.option("--task", "task_id", default=None, help="Run a single task by task_id.")
@click.option(
    "--strategy",
    "strategy_names",
    multiple=True,
    type=click.Choice([s.value for s in Strategy]),
    help="Filter to specific strategy (repeatable).",
)
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(exists=True),
    help="Directory containing task YAML files.",
)
def run_cmd(
    output_dir: str,
    task_id: str | None,
    strategy_names: tuple[str, ...],
    tasks_dir: str,
) -> None:
    """Run benchmarks across strategies."""
    strategies: list[Strategy] | None = None
    if strategy_names:
        strategies = [Strategy(s) for s in strategy_names]

    reports = run_all(
        tasks_dir=Path(tasks_dir),
        output_dir=Path(output_dir),
        strategies=strategies,
        task_filter=task_id,
    )

    click.echo(f"\nCompleted {len(reports)} benchmark(s).", err=True)


@benchmark_cmd.command("report")
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format.",
)
@click.option(
    "--input",
    "input_dir",
    default="benchmarks/results",
    type=click.Path(exists=True),
    help="Directory containing result JSON files.",
)
def report_cmd(output_format: str, input_dir: str) -> None:
    """Generate formatted reports from benchmark results."""
    input_path = Path(input_dir)
    reports: list[BenchmarkReport] = []

    for json_file in sorted(input_path.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        reports.append(BenchmarkReport.model_validate(data))

    if not reports:
        raise click.ClickException(f"No result files found in {input_dir}")

    if output_format == "json":
        for report in reports:
            click.echo(format_json(report))
    else:
        for report in reports:
            click.echo(format_markdown(report))
        click.echo(format_summary(reports))


@benchmark_cmd.command("validate")
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(exists=True),
    help="Directory containing task YAML files.",
)
def validate_cmd(tasks_dir: str) -> None:
    """Validate benchmark task definitions."""
    tasks = load_tasks(Path(tasks_dir))

    if not tasks:
        raise click.ClickException(f"No task files found in {tasks_dir}")

    has_errors = False
    for task in tasks:
        click.echo(f"Validating: {task.task_id} ({task.repo})")
        # Structural validation only (no clone) — check fields are reasonable
        errors: list[str] = []
        if not task.expected_files:
            errors.append("No expected_files defined")
        if not task.question.strip():
            errors.append("Empty question")
        if not task.commit:
            errors.append("No commit hash")

        if errors:
            has_errors = True
            for err in errors:
                click.echo(f"  ERROR: {err}", err=True)
        else:
            click.echo(f"  OK ({len(task.expected_files)} expected files)")

    if has_errors:
        raise SystemExit(1)
    click.echo(f"\nAll {len(tasks)} task(s) valid.")
