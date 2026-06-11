"""CLI benchmark subcommands: run, report, validate."""

from __future__ import annotations

import json
from pathlib import Path

import click

from archex.benchmark.arch_quality import (
    DEFAULT_ARCHITECTURE_BASELINE_DIR,
    architecture_gate_warnings,
    format_architecture_summary,
    load_architecture_results,
    run_architecture_all,
)
from archex.benchmark.baseline import compare_baseline, load_baseline, save_baseline
from archex.benchmark.delta_runner import run_all_delta
from archex.benchmark.gate import (
    DeltaQualityThresholds,
    LatencyWarning,
    QualityThresholds,
    check_delta_gate,
    check_gate,
    check_latency_warnings,
    check_recall_regressions,
    non_token_quality_warnings,
    token_efficiency_violations,
)
from archex.benchmark.headtohead import (
    HeadToHeadManifestError,
    format_headtohead_markdown,
    load_headtohead_manifest,
    load_headtohead_results,
    run_headtohead,
)
from archex.benchmark.loader import load_arch_tasks, load_delta_tasks, load_tasks
from archex.benchmark.models import (
    ArchitectureBenchmarkResult,
    BenchmarkReport,
    BenchmarkRetrievalOptions,
    DeltaBenchmarkResult,
    Strategy,
)
from archex.benchmark.preflight import warm_benchmark_models
from archex.benchmark.progress import BenchmarkProgress
from archex.benchmark.readiness import (
    build_readiness_report,
    format_readiness_json,
    format_readiness_markdown,
)
from archex.benchmark.reporter import (
    format_bucketed_summary,
    format_delta_summary,
    format_json,
    format_markdown,
    format_summary,
)
from archex.benchmark.runner import DEFAULT_STRATEGIES, load_selected_tasks, run_all
from archex.benchmark.triage import (
    format_triage_json,
    format_triage_markdown,
    load_benchmark_reports,
    load_benchmark_tasks,
    triage_failures,
)

DEFAULT_BENCHMARK_RESULTS_DIR = ".archex/benchmark-results"
DEFAULT_DELTA_RESULTS_DIR = ".archex/delta-results"


@click.group("benchmark")
def benchmark_cmd() -> None:
    """Benchmark archex retrieval strategies against real repos."""


@benchmark_cmd.command("run")
@click.option(
    "--output",
    "output_dir",
    default=DEFAULT_BENCHMARK_RESULTS_DIR,
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
@click.option(
    "--query-fusion",
    is_flag=True,
    default=False,
    help="Include the experimental archex_query_fusion strategy.",
)
@click.option(
    "--cross_layer_fusion",
    is_flag=True,
    default=False,
    help="Include the experimental cross_layer_fusion strategy.",
)
@click.option(
    "--rerank",
    is_flag=True,
    default=False,
    help="Include the fusion+rerank strategy (archex_query_fusion_rerank).",
)
@click.option(
    "--splade",
    is_flag=True,
    default=False,
    help="Enable the opt-in SPLADE retrieval leg for archex benchmark strategies.",
)
@click.option(
    "--module-prefilter",
    is_flag=True,
    default=False,
    help="Enable the opt-in module responsibility prefilter for BM25-backed archex strategies.",
)
@click.option(
    "--embedder",
    default="jina-v2",
    show_default=True,
    help="Embedder to pin across every vector-backed benchmark strategy.",
)
@click.option(
    "--rerank-model",
    default=None,
    help="Cross-encoder model to use for the fusion+rerank benchmark strategy.",
)
@click.option(
    "--self-only",
    is_flag=True,
    default=False,
    help='Run only benchmark tasks whose repo is ".".',
)
@click.option(
    "--no-progress",
    is_flag=True,
    default=False,
    help="Disable the live progress display.",
)
def run_cmd(
    output_dir: str,
    task_id: str | None,
    strategy_names: tuple[str, ...],
    tasks_dir: str,
    query_fusion: bool,
    cross_layer_fusion: bool,
    rerank: bool,
    splade: bool,
    module_prefilter: bool,
    embedder: str,
    rerank_model: str | None,
    self_only: bool,
    no_progress: bool,
) -> None:
    """Run benchmarks across strategies."""
    strategies: list[Strategy] = list(DEFAULT_STRATEGIES)
    for name in strategy_names:
        strategy = Strategy(name)
        if strategy not in strategies:
            strategies.append(strategy)
    if query_fusion and Strategy.ARCHEX_QUERY_FUSION not in strategies:
        strategies.append(Strategy.ARCHEX_QUERY_FUSION)
    if cross_layer_fusion and Strategy.CROSS_LAYER_FUSION not in strategies:
        strategies.append(Strategy.CROSS_LAYER_FUSION)
    if rerank:
        if Strategy.ARCHEX_QUERY_FUSION not in strategies:
            strategies.append(Strategy.ARCHEX_QUERY_FUSION)
        if Strategy.ARCHEX_QUERY_FUSION_RERANK not in strategies:
            strategies.append(Strategy.ARCHEX_QUERY_FUSION_RERANK)

    retrieval_options = BenchmarkRetrievalOptions(
        splade=splade,
        module_prefilter=module_prefilter,
        embedder=embedder,
        rerank_model=rerank_model,
    )
    warmed_models = warm_benchmark_models(strategies, retrieval_options)
    if warmed_models:
        click.echo(
            f"Benchmark model preflight loaded {len(warmed_models)} model(s).",
            err=True,
        )

    tasks_path = Path(tasks_dir)
    tasks = load_selected_tasks(tasks_path, task_filter=task_id, self_only=self_only)
    with BenchmarkProgress(tasks, force_disable=no_progress) as progress:
        reports = run_all(
            tasks_dir=tasks_path,
            output_dir=Path(output_dir),
            strategies=strategies,
            task_filter=task_id,
            self_only=self_only,
            progress=progress,
            tasks=tasks,
            retrieval_options=retrieval_options,
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
    default=DEFAULT_BENCHMARK_RESULTS_DIR,
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


@benchmark_cmd.command("triage")
@click.option(
    "--input",
    "input_dir",
    default=DEFAULT_BENCHMARK_RESULTS_DIR,
    type=click.Path(exists=True),
    help="Directory containing result JSON files.",
)
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(exists=True),
    help="Directory containing task YAML files.",
)
@click.option(
    "--strategy",
    "strategy_name",
    default=Strategy.ARCHEX_QUERY.value,
    type=click.Choice([s.value for s in Strategy]),
    help="Strategy to triage.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format.",
)
def triage_cmd(input_dir: str, tasks_dir: str, strategy_name: str, output_format: str) -> None:
    """Rank benchmark retrieval failures for a strategy."""
    reports = load_benchmark_reports(Path(input_dir))
    if not reports:
        raise click.ClickException(f"No result files found in {input_dir}")

    tasks_by_id = load_benchmark_tasks(Path(tasks_dir))
    findings = triage_failures(reports, tasks_by_id, strategy=Strategy(strategy_name))
    if output_format == "json":
        click.echo(format_triage_json(findings))
    else:
        click.echo(format_triage_markdown(findings))


@benchmark_cmd.command("readiness")
@click.option(
    "--input",
    "input_dir",
    default=DEFAULT_BENCHMARK_RESULTS_DIR,
    type=click.Path(exists=True),
    help="Directory containing result JSON files.",
)
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(exists=True),
    help="Directory containing task YAML files.",
)
@click.option(
    "--strategy",
    "strategy_name",
    default=Strategy.ARCHEX_QUERY.value,
    type=click.Choice([s.value for s in Strategy]),
    help="Strategy to assess.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format.",
)
def readiness_cmd(input_dir: str, tasks_dir: str, strategy_name: str, output_format: str) -> None:
    """Generate a non-blocking benchmark readiness report."""
    reports = load_benchmark_reports(Path(input_dir))
    if not reports:
        raise click.ClickException(f"No result files found in {input_dir}")

    tasks_by_id = load_benchmark_tasks(Path(tasks_dir))
    readiness = build_readiness_report(reports, tasks_by_id, strategy=Strategy(strategy_name))
    if output_format == "json":
        click.echo(format_readiness_json(readiness))
    else:
        click.echo(format_readiness_markdown(readiness))


@benchmark_cmd.command("validate")
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory containing benchmark task YAML files.",
)
@click.option(
    "--arch-tasks-dir",
    default="benchmarks/arch_tasks",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory containing architecture task YAML files.",
)
@click.option(
    "--delta-tasks-dir",
    default="benchmarks/delta_tasks",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory containing delta task YAML files.",
)
@click.option(
    "--kind",
    default="tasks",
    type=click.Choice(["tasks", "arch", "delta", "all"]),
    show_default=True,
    help="Task definition family to validate.",
)
def validate_cmd(tasks_dir: str, arch_tasks_dir: str, delta_tasks_dir: str, kind: str) -> None:
    """Validate benchmark task definitions."""
    repo_root = Path.cwd()
    validated_counts: list[tuple[str, int]] = []

    if kind in {"tasks", "all"}:
        try:
            tasks = load_tasks(Path(tasks_dir))
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        if not tasks:
            raise click.ClickException(f"No task files found in {tasks_dir}")

        has_errors = False
        for task in tasks:
            click.echo(f"Validating: {task.task_id} ({task.repo})")
            errors: list[str] = []
            if task.repo == ".":
                for expected in task.expected_files:
                    if not (repo_root / expected).is_file():
                        errors.append(f"Expected file not found: {expected}")

            if errors:
                has_errors = True
                for err in errors:
                    click.echo(f"  ERROR: {err}", err=True)
            else:
                click.echo(f"  OK ({len(task.expected_files)} expected files)")

        if has_errors:
            raise SystemExit(1)
        validated_counts.append(("task", len(tasks)))

    if kind in {"arch", "all"}:
        try:
            arch_tasks = load_arch_tasks(Path(arch_tasks_dir))
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        if not arch_tasks:
            raise click.ClickException(f"No architecture task files found in {arch_tasks_dir}")
        validated_counts.append(("architecture task", len(arch_tasks)))

    if kind in {"delta", "all"}:
        try:
            delta_tasks = load_delta_tasks(Path(delta_tasks_dir))
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        if not delta_tasks:
            raise click.ClickException(f"No delta task files found in {delta_tasks_dir}")
        validated_counts.append(("delta task", len(delta_tasks)))

    if validated_counts == [("task", 1)]:
        click.echo("\nAll 1 task(s) valid.")
        return

    summary = ", ".join(
        f"{count} {label}{'' if count == 1 else 's'}" for label, count in validated_counts
    )
    click.echo(f"\nAll {summary} valid.")


@benchmark_cmd.group("baseline")
def baseline_cmd() -> None:
    """Manage benchmark baselines for regression detection."""


@baseline_cmd.command("save")
@click.option(
    "--input",
    "input_dir",
    default=DEFAULT_BENCHMARK_RESULTS_DIR,
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
    default=DEFAULT_BENCHMARK_RESULTS_DIR,
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
    default=DEFAULT_BENCHMARK_RESULTS_DIR,
    type=click.Path(exists=True),
    help="Directory containing result JSON files.",
)
@click.option("--min-recall", default=0.60, type=float, help="Minimum recall threshold.")
@click.option("--min-precision", default=0.20, type=float, help="Minimum precision threshold.")
@click.option("--min-f1", default=0.30, type=float, help="Minimum F1 threshold.")
@click.option("--min-mrr", default=0.55, type=float, help="Minimum MRR threshold.")
@click.option(
    "--baseline",
    "baseline_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Optional baseline result directory for recall-regression gating.",
)
@click.option(
    "--warn-latency-ms",
    default=5000.0,
    type=float,
    help="Warn (non-fatal) if mean task latency exceeds this value in ms.",
)
def gate_cmd(
    input_dir: str,
    min_recall: float,
    min_precision: float,
    min_f1: float,
    min_mrr: float,
    baseline_dir: str | None,
    warn_latency_ms: float,
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
        warn_latency_ms=warn_latency_ms,
    )

    latency_warnings: list[LatencyWarning] = check_latency_warnings(reports, thresholds)
    if latency_warnings:
        click.echo(
            f"LATENCY WARNING: {len(latency_warnings)} task(s) exceeded {warn_latency_ms:.0f}ms"
        )
        for w in latency_warnings:
            click.echo(
                f"  {w.task_id}/{w.strategy}: {w.actual_ms:.0f}ms"
                f" (threshold: {w.threshold_ms:.0f}ms)"
            )

    absolute_violations = check_gate(reports, thresholds)
    token_violations = token_efficiency_violations(absolute_violations)
    advisory_warnings = non_token_quality_warnings(absolute_violations)

    if baseline_dir is not None:
        baseline_path = Path(baseline_dir)
        baseline_reports: list[BenchmarkReport] = []
        for json_file in sorted(baseline_path.glob("*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            baseline_reports.append(BenchmarkReport.model_validate(data))
        if not baseline_reports:
            raise click.ClickException(f"No baseline result files found in {baseline_dir}")

        recall_regressions = check_recall_regressions(reports, baseline_reports, thresholds)
        if advisory_warnings:
            click.echo(
                f"QUALITY WARNING: {len(advisory_warnings)} non-token absolute-threshold warning(s)"
            )
            for v in advisory_warnings:
                click.echo(
                    f"  {v.task_id}/{v.strategy} {v.metric}: {v.actual:.3f} < {v.threshold:.3f}"
                )

        failures = token_violations
        if failures or recall_regressions:
            click.echo(
                f"QUALITY GATE FAILED: {len(failures) + len(recall_regressions)} violation(s)"
            )
            for v in failures:
                click.echo(
                    f"  {v.task_id}/{v.strategy} {v.metric}: {v.actual:.3f} < {v.threshold:.3f}"
                )
            for v in recall_regressions:
                if v.metric == "baseline_missing":
                    click.echo(f"  {v.task_id}/{v.strategy}: missing baseline result")
                else:
                    click.echo(
                        f"  {v.task_id}/{v.strategy} {v.metric}: "
                        f"{v.actual:.3f} < baseline {v.baseline:.3f}"
                    )
            raise SystemExit(1)

        click.echo("Quality gate passed.")
        return

    if absolute_violations:
        click.echo(f"QUALITY GATE FAILED: {len(absolute_violations)} violation(s)")
        for v in absolute_violations:
            click.echo(f"  {v.task_id}/{v.strategy} {v.metric}: {v.actual:.3f} < {v.threshold:.3f}")
        raise SystemExit(1)

    click.echo("Quality gate passed.")


def _load_architecture_baseline(
    baseline_dir: str | None,
) -> tuple[Path, list[ArchitectureBenchmarkResult] | None]:
    baseline_path = Path(baseline_dir) if baseline_dir else DEFAULT_ARCHITECTURE_BASELINE_DIR
    if not baseline_path.exists():
        if baseline_dir is not None:
            raise click.ClickException(
                f"No architecture baseline directory found at {baseline_path}"
            )
        return baseline_path, None
    if not baseline_path.is_dir():
        raise click.ClickException(
            f"Architecture baseline path is not a directory: {baseline_path}"
        )

    baseline_results = load_architecture_results(baseline_path)
    if not baseline_results:
        raise click.ClickException(
            f"No architecture baseline result files found in {baseline_path}"
        )
    return baseline_path, baseline_results


@benchmark_cmd.group("arch")
def arch_cmd() -> None:
    """Architecture-quality benchmarks for analyze/explain outputs."""


@arch_cmd.command("run")
@click.option(
    "--output",
    "output_dir",
    default="benchmarks/arch_results",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory to write architecture-quality result JSON files.",
)
@click.option("--task", "task_id", default=None, help="Run a single architecture task by task_id.")
@click.option(
    "--tasks-dir",
    default="benchmarks/arch_tasks",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing architecture task YAML files.",
)
def arch_run_cmd(output_dir: str, task_id: str | None, tasks_dir: str) -> None:
    """Run architecture-quality benchmarks against labeled local repo slices."""
    try:
        results = run_architecture_all(
            Path(tasks_dir),
            Path(output_dir),
            task_filter=task_id,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"\nCompleted {len(results)} architecture benchmark(s).", err=True)


@arch_cmd.command("report")
@click.option(
    "--input",
    "input_dir",
    default="benchmarks/arch_results",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing architecture-quality result JSON files.",
)
@click.option(
    "--baseline",
    "baseline_dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help=(
        "Optional accepted baseline directory. Defaults to "
        ".archex/arch-quality-baseline when present; absent default means seed mode."
    ),
)
def arch_report_cmd(input_dir: str, baseline_dir: str | None) -> None:
    """Generate a formatted architecture-quality report."""
    results = load_architecture_results(Path(input_dir))
    if not results:
        raise click.ClickException(f"No architecture result files found in {input_dir}")
    baseline_path, baseline_results = _load_architecture_baseline(baseline_dir)
    click.echo(
        format_architecture_summary(
            results,
            baseline_dir=baseline_path,
            baseline_results=baseline_results,
        )
    )


@arch_cmd.command("gate")
@click.option(
    "--input",
    "input_dir",
    default="benchmarks/arch_results",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory containing architecture-quality result JSON files.",
)
@click.option(
    "--baseline",
    "baseline_dir",
    default=None,
    type=click.Path(file_okay=False, dir_okay=True),
    help=(
        "Optional accepted baseline directory. Defaults to "
        ".archex/arch-quality-baseline when present; absent default means seed mode."
    ),
)
@click.option("--min-boundary-f1", default=0.80, type=float, help="Advisory boundary F1 floor.")
@click.option(
    "--min-pattern-precision",
    default=0.80,
    type=float,
    help="Advisory pattern precision floor.",
)
@click.option(
    "--min-pattern-recall", default=0.80, type=float, help="Advisory pattern recall floor."
)
@click.option(
    "--min-interface-completeness",
    default=0.80,
    type=float,
    help="Advisory interface completeness floor.",
)
def arch_gate_cmd(
    input_dir: str,
    baseline_dir: str | None,
    min_boundary_f1: float,
    min_pattern_precision: float,
    min_pattern_recall: float,
    min_interface_completeness: float,
) -> None:
    """Check architecture-quality scores in advisory mode."""
    results = load_architecture_results(Path(input_dir))
    if not results:
        raise click.ClickException(f"No architecture result files found in {input_dir}")
    baseline_path, baseline_results = _load_architecture_baseline(baseline_dir)
    warnings = architecture_gate_warnings(
        results,
        baseline_results=baseline_results,
        min_boundary_f1=min_boundary_f1,
        min_pattern_precision=min_pattern_precision,
        min_pattern_recall=min_pattern_recall,
        min_interface_completeness=min_interface_completeness,
    )
    if baseline_results is None:
        click.echo(
            "Architecture baseline mode: FIRST RUN / seed candidate "
            f"(no accepted baseline loaded from {baseline_path})"
        )
        click.echo(
            "Accepted baseline seeding remains an operator decision; "
            f"copy reviewed result JSON files into {baseline_path} to enable regression comparison."
        )
    else:
        click.echo(f"Architecture baseline mode: REGRESSION COMPARISON ({baseline_path})")
    if warnings:
        click.echo(f"ARCHITECTURE QUALITY ADVISORY: {len(warnings)} warning(s)")
        for warning in warnings:
            click.echo(f"  {warning}")
        return
    click.echo("Architecture quality advisory gate passed.")


# ---------------------------------------------------------------------------
# Head-to-head benchmark subcommands
# ---------------------------------------------------------------------------


@benchmark_cmd.group("headtohead")
def headtohead_cmd() -> None:
    """Public same-task comparison harness."""


@headtohead_cmd.command("run")
@click.option(
    "--manifest",
    "manifest_path",
    default="benchmarks/headtohead/manifest.yaml",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Pinned head-to-head comparison manifest.",
)
@click.option(
    "--output",
    "output_dir",
    default=".archex/headtohead",
    type=click.Path(file_okay=False),
    show_default=True,
    help="Directory for result JSON files.",
)
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(exists=True, file_okay=False),
    show_default=True,
    help="Directory containing benchmark task YAML files.",
)
def headtohead_run_cmd(manifest_path: str, output_dir: str, tasks_dir: str) -> None:
    """Run the public archex/ccc/raw comparison from a manifest."""
    try:
        reports = run_headtohead(Path(manifest_path), Path(output_dir), Path(tasks_dir))
    except HeadToHeadManifestError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"\nCompleted {len(reports)} head-to-head benchmark(s).", err=True)


@headtohead_cmd.command("report")
@click.option(
    "--input",
    "input_dir",
    default=".archex/headtohead",
    type=click.Path(exists=True, file_okay=False),
    show_default=True,
    help="Directory containing head-to-head result JSON files.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown"]),
    show_default=True,
    help="Report format.",
)
def headtohead_report_cmd(input_dir: str, output_format: str) -> None:
    """Render the public head-to-head comparison report."""
    del output_format
    input_path = Path(input_dir)
    manifest_path = input_path / "manifest.yaml"
    try:
        manifest = load_headtohead_manifest(manifest_path)
    except (FileNotFoundError, HeadToHeadManifestError) as exc:
        raise click.ClickException(str(exc)) from exc
    reports = load_headtohead_results(input_path)
    if not reports:
        raise click.ClickException(f"No result files found in {input_dir}")
    try:
        click.echo(format_headtohead_markdown(manifest, reports))
    except HeadToHeadManifestError as exc:
        raise click.ClickException(str(exc)) from exc


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
    default=DEFAULT_DELTA_RESULTS_DIR,
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
    default=DEFAULT_DELTA_RESULTS_DIR,
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
    default=DEFAULT_DELTA_RESULTS_DIR,
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
