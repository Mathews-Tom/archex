"""CLI benchmark subcommands: run, report, validate."""

from __future__ import annotations

import json
from pathlib import Path

import click

from archex.benchmark.baseline import compare_baseline, load_baseline, save_baseline
from archex.benchmark.delta_runner import run_all_delta
from archex.benchmark.gate import (
    DeltaQualityThresholds,
    QualityThresholds,
    check_delta_gate,
    check_gate,
)
from archex.benchmark.loader import load_tasks
from archex.benchmark.models import BenchmarkReport, DeltaBenchmarkResult, Strategy
from archex.benchmark.reporter import (
    format_bucketed_summary,
    format_delta_summary,
    format_json,
    format_markdown,
    format_summary,
)
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
        click.echo(format_bucketed_summary(reports))


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


@benchmark_cmd.group("baseline")
def baseline_cmd() -> None:
    """Manage benchmark baselines for regression detection."""


@baseline_cmd.command("save")
@click.option(
    "--input",
    "input_dir",
    default="benchmarks/results",
    type=click.Path(exists=True),
    help="Directory containing result JSON files.",
)
@click.option(
    "--output",
    "output_path",
    default="benchmarks/baseline.json",
    type=click.Path(),
    help="Output path for baseline JSON.",
)
def baseline_save_cmd(input_dir: str, output_path: str) -> None:
    """Save current benchmark results as a golden baseline."""
    input_path = Path(input_dir)
    reports: list[BenchmarkReport] = []
    for json_file in sorted(input_path.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        reports.append(BenchmarkReport.model_validate(data))

    if not reports:
        raise click.ClickException(f"No result files found in {input_dir}")

    baseline = save_baseline(reports)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"Saved baseline with {len(baseline.entries)} entries to {output_path}")


@baseline_cmd.command("compare")
@click.option(
    "--input",
    "input_dir",
    default="benchmarks/results",
    type=click.Path(exists=True),
    help="Directory containing result JSON files.",
)
@click.option(
    "--baseline",
    "baseline_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to baseline JSON file.",
)
def baseline_compare_cmd(input_dir: str, baseline_path: str) -> None:
    """Compare current results against a saved baseline."""
    input_path = Path(input_dir)
    reports: list[BenchmarkReport] = []
    for json_file in sorted(input_path.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        reports.append(BenchmarkReport.model_validate(data))

    if not reports:
        raise click.ClickException(f"No result files found in {input_dir}")

    baseline_data = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    baseline = load_baseline(baseline_data)

    comparisons = compare_baseline(reports, baseline)
    regressions = [c for c in comparisons if c.regression]

    click.echo(f"Compared {len(comparisons)} metric(s) against baseline.")
    if regressions:
        click.echo(f"\nREGRESSIONS DETECTED: {len(regressions)}")
        for r in regressions:
            click.echo(
                f"  {r.task_id}/{r.strategy} {r.metric}: "
                f"{r.baseline_value:.3f} -> {r.current_value:.3f} (delta: {r.delta:+.3f})"
            )
        raise SystemExit(1)
    else:
        click.echo("No regressions detected.")


@benchmark_cmd.command("gate")
@click.option(
    "--input",
    "input_dir",
    default="benchmarks/results",
    type=click.Path(exists=True),
    help="Directory containing result JSON files.",
)
@click.option("--min-recall", default=0.6, type=float, help="Minimum recall threshold.")
@click.option("--min-precision", default=0.3, type=float, help="Minimum precision threshold.")
@click.option("--min-f1", default=0.4, type=float, help="Minimum F1 threshold.")
@click.option("--min-mrr", default=0.3, type=float, help="Minimum MRR threshold.")
def gate_cmd(
    input_dir: str,
    min_recall: float,
    min_precision: float,
    min_f1: float,
    min_mrr: float,
) -> None:
    """Check benchmark results against quality thresholds."""
    input_path = Path(input_dir)
    reports: list[BenchmarkReport] = []
    for json_file in sorted(input_path.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        reports.append(BenchmarkReport.model_validate(data))

    if not reports:
        raise click.ClickException(f"No result files found in {input_dir}")

    thresholds = QualityThresholds(
        min_recall=min_recall,
        min_precision=min_precision,
        min_f1=min_f1,
        min_mrr=min_mrr,
    )
    violations = check_gate(reports, thresholds)

    if violations:
        click.echo(f"QUALITY GATE FAILED: {len(violations)} violation(s)")
        for v in violations:
            click.echo(f"  {v.task_id}/{v.strategy} {v.metric}: {v.actual:.3f} < {v.threshold:.3f}")
        raise SystemExit(1)
    else:
        click.echo("Quality gate passed.")


# ---------------------------------------------------------------------------
# Delta benchmark subcommands
# ---------------------------------------------------------------------------


@benchmark_cmd.group("delta")
def delta_cmd() -> None:
    """Delta indexing benchmarks: measure speedup and correctness."""


@delta_cmd.command("run")
@click.option(
    "--output",
    "output_dir",
    default="benchmarks/delta_results",
    type=click.Path(),
    help="Directory for delta result JSON files.",
)
@click.option("--task", "task_id", default=None, help="Run a single task by task_id.")
@click.option(
    "--tasks-dir",
    default="benchmarks/delta_tasks",
    type=click.Path(exists=True),
    help="Directory containing delta task YAML files.",
)
def delta_run_cmd(output_dir: str, task_id: str | None, tasks_dir: str) -> None:
    """Run delta indexing benchmarks."""
    results = run_all_delta(
        tasks_dir=Path(tasks_dir),
        output_dir=Path(output_dir),
        task_filter=task_id,
    )
    click.echo(f"\nCompleted {len(results)} delta benchmark(s).", err=True)


@delta_cmd.command("gate")
@click.option(
    "--input",
    "input_dir",
    default="benchmarks/delta_results",
    type=click.Path(exists=True),
    help="Directory containing delta result JSON files.",
)
@click.option("--min-speedup", default=1.5, type=float, help="Minimum speedup threshold.")
@click.option(
    "--require-correctness/--no-require-correctness",
    default=True,
    help="Require correctness (chunk/edge equivalence).",
)
def delta_gate_cmd(input_dir: str, min_speedup: float, require_correctness: bool) -> None:
    """Check delta benchmark results against quality thresholds."""
    input_path = Path(input_dir)
    results: list[DeltaBenchmarkResult] = []
    for json_file in sorted(input_path.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        results.append(DeltaBenchmarkResult.model_validate(data))

    if not results:
        raise click.ClickException(f"No delta result files found in {input_dir}")

    thresholds = DeltaQualityThresholds(
        min_speedup=min_speedup,
        require_correctness=require_correctness,
    )
    violations = check_delta_gate(results, thresholds)

    if violations:
        click.echo(f"DELTA QUALITY GATE FAILED: {len(violations)} violation(s)")
        for v in violations:
            click.echo(f"  {v.task_id} {v.metric}: {v.actual:.3f} < {v.threshold:.3f}")
        raise SystemExit(1)
    else:
        click.echo("Delta quality gate passed.")


@delta_cmd.command("report")
@click.option(
    "--input",
    "input_dir",
    default="benchmarks/delta_results",
    type=click.Path(exists=True),
    help="Directory containing delta result JSON files.",
)
def delta_report_cmd(input_dir: str) -> None:
    """Generate formatted reports from delta benchmark results."""
    input_path = Path(input_dir)
    results: list[DeltaBenchmarkResult] = []
    for json_file in sorted(input_path.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        results.append(DeltaBenchmarkResult.model_validate(data))

    if not results:
        raise click.ClickException(f"No delta result files found in {input_dir}")

    click.echo(format_delta_summary(results))
