"""CLI benchmark subcommands: run, report, validate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import click

from archex.benchmark.arch_quality import (
    DEFAULT_ARCHITECTURE_BASELINE_DIR,
    architecture_gate_warnings,
    format_architecture_summary,
    load_architecture_results,
    run_architecture_all,
)
from archex.benchmark.baseline import (
    build_ranking_snapshot,
    compare_baseline,
    load_baseline,
    save_baseline,
)
from archex.benchmark.bundle_eval import BundleOnlyEvaluatorError, run_bundle_only_eval_all
from archex.benchmark.competitive import format_competitive_markdown, load_compression_results
from archex.benchmark.corpus_audit import (
    CorpusAuditError,
    validate_corpus_audit_artifact,
)
from archex.benchmark.cross_tool import NaiveBaselineModel, run_cross_tool
from archex.benchmark.delta_runner import run_all_delta
from archex.benchmark.determinism_economics import (
    DeterminismEconomicsError,
    build_fixture,
    load_artifact,
    load_fixture,
    require_openrouter_api_key,
    run_measurement,
)
from archex.benchmark.evidence import (
    BenchmarkEvidenceError,
    build_evidence_manifest,
    load_evidence_reports,
    prepare_evidence_directory,
    source_revision,
    validate_baseline_coverage,
    validate_evidence_directory,
    write_evidence_manifest,
)
from archex.benchmark.external_corpus import (
    SealedCorpusAccessError,
    enforce_sealed_corpus_access,
)
from archex.benchmark.gate import (
    DeltaQualityThresholds,
    LatencyViolation,
    LatencyWarning,
    QualityThresholds,
    check_delta_gate,
    check_fixed_agent_non_regression,
    check_gate,
    check_language_family_non_regression,
    check_latency_violations,
    check_latency_warnings,
    check_recall_regressions,
    check_strategy_non_regressions,
    check_warm_p95_latency,
    check_zero_recall_non_regression,
    non_token_quality_warnings,
    token_efficiency_violations,
)
from archex.benchmark.graphify import GraphifyAdapterError
from archex.benchmark.headroom import HeadroomAdapterError
from archex.benchmark.headtohead import (
    HeadToHeadManifestError,
    format_headtohead_markdown,
    load_headtohead_manifest,
    load_headtohead_results,
    reports_with_graphify_lanes,
    run_headtohead,
)
from archex.benchmark.loader import load_arch_tasks, load_delta_tasks, load_tasks
from archex.benchmark.models import (
    ArchitectureBenchmarkResult,
    BenchmarkReport,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    BundleOnlyEvaluatorCommand,
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
from archex.benchmark.replication import (
    ReplicationEvidenceError,
    validate_replication_artifact,
)
from archex.benchmark.reporter import (
    format_baseline_comparison,
    format_bucketed_summary,
    format_chunker_frontier_table,
    format_cross_tool_comparison,
    format_delta_summary,
    format_json,
    format_localization_summary,
    format_markdown,
    format_summary,
)
from archex.benchmark.runner import DEFAULT_STRATEGIES, load_selected_tasks, run_all
from archex.benchmark.scorecard import (
    build_m3_scorecard_artifact,
    format_m3_scorecard_markdown,
    save_m3_scorecard_artifact,
)
from archex.benchmark.triage import (
    format_triage_json,
    format_triage_markdown,
    load_benchmark_reports,
    load_benchmark_tasks,
    triage_failures,
)
from archex.exceptions import ArchexError

if TYPE_CHECKING:
    from archex.models import ChunkerName

DEFAULT_BENCHMARK_RESULTS_DIR = ".archex/benchmark-results"
DEFAULT_DELTA_RESULTS_DIR = ".archex/delta-results"


def _select_strategy_reports(
    reports: list[BenchmarkReport],
    strategy: Strategy,
) -> list[BenchmarkReport]:
    """Return one result per report for a gate's named subject strategy."""
    selected_reports: list[BenchmarkReport] = []
    for report in reports:
        selected = [result for result in report.results if result.strategy is strategy]
        if len(selected) != 1:
            raise click.ClickException(
                f"Expected exactly one {strategy.value} result for task {report.task_id!r}"
            )
        selected_reports.append(report.model_copy(update={"results": selected}))
    return selected_reports


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
    "--scout",
    is_flag=True,
    default=False,
    help="Include the archex_scout_fetch two-call scout plus exact fetch strategy.",
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
    "--allow-remote-code",
    is_flag=True,
    default=False,
    help="Allow explicitly selected pinned model paths that require Hugging Face remote code.",
)
@click.option(
    "--embedder",
    default="jina-v2",
    show_default=True,
    help="Embedder to pin across every vector-backed benchmark strategy.",
)
@click.option(
    "--chunker",
    default="default",
    show_default=True,
    type=click.Choice(["default", "cast"]),
    help="Chunker to pin across every archex benchmark strategy.",
)
@click.option(
    "--bm25-chunker",
    default=None,
    type=click.Choice(["default", "cast"]),
    help="Override chunker for BM25-backed strategies only.",
)
@click.option(
    "--vector-chunker",
    default=None,
    type=click.Choice(["default", "cast"]),
    help="Override chunker for vector-backed strategies only.",
)
@click.option(
    "--rerank-model",
    default=None,
    help="Cross-encoder model to use for the fusion+rerank benchmark strategy.",
)
@click.option(
    "--rerank-candidate-limit",
    default=None,
    type=int,
    help="Cap candidates scored by the cross-encoder reranker.",
)
@click.option(
    "--warm-cache",
    is_flag=True,
    default=False,
    help="Discard a cache-populating run per indexed strategy before timing results.",
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
@click.option(
    "--allow-sealed-corpus",
    is_flag=True,
    default=False,
    help=(
        "Required to target the sealed chronological holdout corpus "
        "(benchmarks/sealed_tasks); refused otherwise."
    ),
)
def run_cmd(
    output_dir: str,
    task_id: str | None,
    strategy_names: tuple[str, ...],
    scout: bool,
    tasks_dir: str,
    query_fusion: bool,
    cross_layer_fusion: bool,
    rerank: bool,
    splade: bool,
    module_prefilter: bool,
    allow_remote_code: bool,
    embedder: str,
    chunker: ChunkerName,
    bm25_chunker: ChunkerName | None,
    vector_chunker: ChunkerName | None,
    rerank_model: str | None,
    rerank_candidate_limit: int | None,
    warm_cache: bool,
    self_only: bool,
    no_progress: bool,
    allow_sealed_corpus: bool,
) -> None:
    """Run benchmarks across strategies."""
    try:
        enforce_sealed_corpus_access(Path(tasks_dir), allow_sealed=allow_sealed_corpus)
    except SealedCorpusAccessError as exc:
        raise click.ClickException(str(exc)) from exc
    strategies: list[Strategy] = list(DEFAULT_STRATEGIES)
    for name in strategy_names:
        strategy = Strategy(name)
        if strategy not in strategies:
            strategies.append(strategy)
    if scout and Strategy.ARCHEX_SCOUT_FETCH not in strategies:
        strategies.append(Strategy.ARCHEX_SCOUT_FETCH)
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
        allow_remote_code=allow_remote_code,
        rerank_model=rerank_model,
        chunker=chunker,
        bm25_chunker=bm25_chunker,
        vector_chunker=vector_chunker,
        rerank_candidate_limit=rerank_candidate_limit,
        warm_cache=warm_cache,
    )
    try:
        warmed_models = warm_benchmark_models(strategies, retrieval_options)
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc
    if warmed_models:
        click.echo(
            f"Benchmark model preflight loaded {len(warmed_models)} model(s).",
            err=True,
        )

    tasks_path = Path(tasks_dir)
    tasks = load_selected_tasks(tasks_path, task_filter=task_id, self_only=self_only)
    if tasks:
        output_path = Path(output_dir)
        try:
            prepare_evidence_directory(output_path)
        except BenchmarkEvidenceError as exc:
            raise click.ClickException(str(exc)) from exc

    try:
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
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    if reports:
        try:
            source_sha = source_revision(Path.cwd())
        except BenchmarkEvidenceError as exc:
            raise click.ClickException(str(exc)) from exc
        try:
            manifest = build_evidence_manifest(
                reports,
                tasks,
                strategies,
                retrieval_options,
                source_sha=source_sha,
                tasks_dir=tasks_path,
            )
            manifest_path = write_evidence_manifest(Path(output_dir), manifest)
            validate_evidence_directory(
                Path(output_dir),
                tasks_path,
                expected_source_sha=source_sha,
            )
        except BenchmarkEvidenceError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Recorded benchmark evidence manifest at {manifest_path}", err=True)

    click.echo(f"\nCompleted {len(reports)} benchmark(s).", err=True)


@benchmark_cmd.command("cross-tool")
@click.option(
    "--output",
    "output_dir",
    default=".archex/cross-tool-efficiency",
    type=click.Path(),
    help="Directory for the cross-tool comparison artifact.",
)
@click.option("--task", "task_id", default=None, help="Run a single task by task_id.")
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(exists=True),
    help="Directory containing task YAML files.",
)
@click.option(
    "--target-recall",
    default=1.0,
    show_default=True,
    type=click.FloatRange(0.0, 1.0, min_open=True),
    help="Required-file recall both paths must reach before tokens are compared.",
)
@click.option(
    "--context-window",
    default=5,
    show_default=True,
    type=click.IntRange(min=0),
    help="Context lines around each grep hit for the grep_window naive model.",
)
@click.option(
    "--naive-model",
    "naive_models",
    multiple=True,
    type=click.Choice([model.value for model in NaiveBaselineModel]),
    help="Naive baseline model(s) to compare (repeatable; default: all).",
)
@click.option(
    "--self-only",
    is_flag=True,
    default=False,
    help='Run only benchmark tasks whose repo is ".".',
)
def cross_tool_cmd(
    output_dir: str,
    task_id: str | None,
    tasks_dir: str,
    target_recall: float,
    context_window: int,
    naive_models: tuple[str, ...],
    self_only: bool,
) -> None:
    """Offline cross-tool tokens-at-fixed-recall comparison: archex vs naive grep/read.

    Benchmark-only: it never touches the query hot path, the in-process metrics
    ledger, retrieval ranking, or any product default.
    """
    models = (
        tuple(NaiveBaselineModel(name) for name in naive_models)
        if naive_models
        else (NaiveBaselineModel.FULL_FILE, NaiveBaselineModel.GREP_WINDOW)
    )
    tasks = load_selected_tasks(Path(tasks_dir), task_filter=task_id, self_only=self_only)

    def _log(index: int, task: BenchmarkTask) -> None:
        click.echo(f"[{index}/{len(tasks)}] {task.task_id} ({task.repo})", err=True)

    try:
        report = run_cross_tool(
            tasks,
            target_recall=target_recall,
            context_window=context_window,
            models=models,
            on_task=_log,
        )
    except ArchexError as exc:
        raise click.ClickException(str(exc)) from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact = output_path / "cross-tool-comparison.json"
    artifact.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    click.echo(format_cross_tool_comparison(report))
    click.echo(f"\nWrote cross-tool artifact to {artifact}", err=True)


@benchmark_cmd.command("bundle-eval")
@click.option(
    "--output",
    "output_dir",
    default=".archex/bundle-eval-results",
    type=click.Path(),
    help="Directory for bundle-only eval result JSON files.",
)
@click.option("--task", "task_id", default=None, help="Run a single task by task_id.")
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(exists=True),
    help="Directory containing task YAML files.",
)
@click.option(
    "--evaluator-command",
    required=True,
    help="Local evaluator executable. The command receives one JSON object on stdin.",
)
@click.option(
    "--evaluator-arg",
    "evaluator_args",
    multiple=True,
    help="Argument passed to the local evaluator command; repeat for multiple args.",
)
@click.option(
    "--timeout-seconds",
    default=600.0,
    show_default=True,
    type=float,
    help="Per-task evaluator command timeout.",
)
@click.option(
    "--bundle-format",
    default="markdown",
    show_default=True,
    type=click.Choice(["markdown", "json", "xml"]),
    help="Rendered bundle format passed to the evaluator command.",
)
@click.option(
    "--self-only",
    is_flag=True,
    default=False,
    help='Run only benchmark tasks whose repo is ".".',
)
def bundle_eval_cmd(
    output_dir: str,
    task_id: str | None,
    tasks_dir: str,
    evaluator_command: str,
    evaluator_args: tuple[str, ...],
    timeout_seconds: float,
    bundle_format: str,
    self_only: bool,
) -> None:
    """Run the explicit opt-in local bundle-only evaluator lane."""
    tasks_path = Path(tasks_dir)
    tasks = load_selected_tasks(tasks_path, task_filter=task_id, self_only=self_only)
    command = BundleOnlyEvaluatorCommand(
        command=evaluator_command,
        args=list(evaluator_args),
        timeout_seconds=timeout_seconds,
    )
    try:
        reports = run_bundle_only_eval_all(
            tasks,
            Path(output_dir),
            command=command,
            bundle_format=bundle_format,
        )
    except (ArchexError, BundleOnlyEvaluatorError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Completed {len(reports)} bundle-only eval(s).", err=True)


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
@click.option(
    "--baseline",
    "baseline_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Optional baseline result directory for strategy delta reporting.",
)
def report_cmd(output_format: str, input_dir: str, baseline_dir: str | None) -> None:
    """Generate formatted reports from benchmark results."""
    input_path = Path(input_dir)
    reports: list[BenchmarkReport] = []

    for json_file in sorted(input_path.glob("*.json")):
        if json_file.name == "manifest.json":
            continue
        data = json.loads(json_file.read_text(encoding="utf-8"))
        reports.append(BenchmarkReport.model_validate(data))
    baseline_reports: list[BenchmarkReport] = []
    if baseline_dir is not None:
        for json_file in sorted(Path(baseline_dir).glob("*.json")):
            if json_file.name == "manifest.json":
                continue
            data = json.loads(json_file.read_text(encoding="utf-8"))
            baseline_reports.append(BenchmarkReport.model_validate(data))
        if not baseline_reports:
            raise click.ClickException(f"No baseline result files found in {baseline_dir}")

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
        localization_summary = format_localization_summary(reports)
        if localization_summary:
            click.echo(localization_summary)
        if baseline_reports:
            click.echo(
                format_baseline_comparison(
                    reports,
                    baseline_reports,
                    candidate_strategy=Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT.value,
                    baseline_strategy=Strategy.ARCHEX_QUERY_HYBRID.value,
                )
            )


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


@benchmark_cmd.command("scorecard")
@click.option(
    "--input",
    "input_dir",
    default=DEFAULT_BENCHMARK_RESULTS_DIR,
    type=click.Path(exists=True),
    help="Directory containing manifest-backed result JSON files.",
)
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the task manifest bound to evidence.",
)
@click.option(
    "--strategy",
    "strategy_name",
    default=Strategy.ARCHEX_QUERY.value,
    type=click.Choice([s.value for s in Strategy]),
    help="Strategy to score.",
)
@click.option(
    "--format",
    "output_format",
    default="markdown",
    type=click.Choice(["markdown", "json"]),
    help="Output format.",
)
@click.option(
    "--output",
    "artifact_path",
    default=None,
    type=click.Path(),
    help="Optional path to also write the raw M3 scorecard artifact JSON.",
)
@click.option(
    "--allow-sealed-corpus",
    is_flag=True,
    default=False,
    help=(
        "Required to target the sealed chronological holdout corpus "
        "(benchmarks/sealed_tasks); refused otherwise."
    ),
)
def scorecard_cmd(
    input_dir: str,
    tasks_dir: str,
    strategy_name: str,
    output_format: str,
    artifact_path: str | None,
    allow_sealed_corpus: bool,
) -> None:
    """Report M3 language/repo-size/intent/family scorecards with raw provenance."""
    tasks_path = Path(tasks_dir)
    try:
        enforce_sealed_corpus_access(tasks_path, allow_sealed=allow_sealed_corpus)
    except SealedCorpusAccessError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        manifest, reports = load_evidence_reports(Path(input_dir), tasks_path)
    except BenchmarkEvidenceError as exc:
        raise click.ClickException(str(exc)) from exc

    tasks_by_id = load_benchmark_tasks(tasks_path)
    artifact = build_m3_scorecard_artifact(
        reports,
        tasks_by_id,
        manifest,
        strategy=Strategy(strategy_name),
    )
    if artifact_path is not None:
        save_m3_scorecard_artifact(Path(artifact_path), artifact)
    if output_format == "json":
        click.echo(artifact.model_dump_json(indent=2))
    else:
        click.echo(format_m3_scorecard_markdown(artifact))


_S7_R6_1_TASK_IDS = [
    "archex_query_pipeline",
    "celery_task_dispatch",
    "click_decorators",
    "django_middleware",
    "fastapi_dependency_injection",
    "loc_flask_blueprint_register",
    "gin_routing",
    "httpx_pooling",
    "mini_redis_async",
    "pydantic_validators",
    "pytest_fixtures",
    "react_hooks",
]


@benchmark_cmd.command("freeze-determinism-economics")
@click.option(
    "--output",
    default="benchmarks/determinism_economics_r6_1/sessions.json",
    type=click.Path(dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "--source-revision",
    required=True,
    help="Merged pre-registration SHA used to freeze the self-repository source.",
)
def freeze_determinism_economics_cmd(output: Path, source_revision: str) -> None:
    """Freeze selected context and recorded arm permutations before provider use."""
    if output.exists():
        raise click.ClickException(f"refusing to overwrite existing frozen fixture: {output}")
    fixture = build_fixture(
        task_ids=_S7_R6_1_TASK_IDS,
        repository_root=Path.cwd(),
        source_revision=source_revision,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fixture.model_dump_json(indent=2) + "\n")
    click.echo(f"Frozen {len(fixture.sessions)} S7 sessions at {output}.")


@benchmark_cmd.command("determinism-economics")
@click.option(
    "--sessions",
    default="benchmarks/determinism_economics_r6_1/sessions.json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "--output",
    default="benchmarks/evidence/s7-determinism-economics-r6.1.json",
    type=click.Path(dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option("--preregistration-commit", required=True)
def determinism_economics_cmd(sessions: Path, output: Path, preregistration_commit: str) -> None:
    """Measure provider-observed input cost across frozen ordering arms."""
    try:
        artifact = run_measurement(
            fixture=load_fixture(sessions),
            preregistration_commit=preregistration_commit,
            api_key=require_openrouter_api_key(),
        )
    except DeterminismEconomicsError as exc:
        raise click.ClickException(str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.model_dump_json(indent=2) + "\n")
    click.echo(f"Wrote S7 provider evidence to {output}.")


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
    "--input",
    "input_path",
    default=None,
    type=click.Path(file_okay=True, dir_okay=True),
    help=(
        "Evidence directory to validate with --kind evidence, "
        "or replication artifact file to validate with --kind replication."
    ),
)
@click.option(
    "--kind",
    default="tasks",
    type=click.Choice(
        [
            "tasks",
            "arch",
            "delta",
            "all",
            "evidence",
            "replication",
            "corpus-audit",
            "determinism-economics-r6-1",
        ]
    ),
    show_default=True,
    help="Task definition or evidence family to validate.",
)
def validate_cmd(
    tasks_dir: str,
    arch_tasks_dir: str,
    delta_tasks_dir: str,
    input_path: str | None,
    kind: str,
) -> None:
    """Validate benchmark task definitions."""
    repo_root = Path.cwd()
    target: Path | None = None
    if kind in {"evidence", "replication", "corpus-audit", "determinism-economics-r6-1"}:
        if input_path is None:
            raise click.ClickException(f"--input is required when --kind {kind} is selected")
        target = Path(input_path)
    if kind == "determinism-economics-r6-1" and target is not None:
        try:
            artifact = load_artifact(target)
        except DeterminismEconomicsError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(
            f"Valid S7 determinism evidence: {len(artifact.sessions)} session(s), "
            f"{len(artifact.measurement_receipts)} measured receipt(s)."
        )
        return

    if kind == "corpus-audit" and target is not None:
        try:
            audit = validate_corpus_audit_artifact(target)
        except CorpusAuditError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(
            f"Valid corpus audit: {audit.total_tasks} task(s), "
            f"milestone {audit.milestone}, verdict recorded."
        )
        return

    if kind == "replication" and target is not None:
        try:
            artifact = validate_replication_artifact(target)
        except ReplicationEvidenceError as exc:
            raise click.ClickException(str(exc)) from exc
        verdicts = ", ".join(f"{arm.arm_id}={arm.verdict.value}" for arm in artifact.arms)
        click.echo(f"Valid replication evidence: {len(artifact.arms)} arm(s) [{verdicts}].")
        return

    if kind == "evidence" and target is not None:
        try:
            manifest = validate_evidence_directory(
                target,
                Path(tasks_dir),
                expected_source_sha=source_revision(repo_root),
            )
        except BenchmarkEvidenceError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(
            f"Valid benchmark evidence: {len(manifest.task_ids)} task(s), "
            f"{len(manifest.strategies)} strategy/strategies."
        )
        return

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
@click.option(
    "--ranking-source",
    "ranking_source",
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help=(
        "Optional repo path to index for a PageRank/symbol_count ranking snapshot, "
        "attached to the saved baseline for ranking-stability gating."
    ),
)
def baseline_save_cmd(input_dir: str, output_path: str, ranking_source: str | None) -> None:
    """Save current benchmark results as a golden baseline."""
    input_path = Path(input_dir)
    reports: list[BenchmarkReport] = []
    for json_file in sorted(input_path.glob("*.json")):
        if json_file.name == "manifest.json":
            continue
        data = json.loads(json_file.read_text(encoding="utf-8"))
        reports.append(BenchmarkReport.model_validate(data))

    if not reports:
        raise click.ClickException(f"No result files found in {input_dir}")

    baseline = save_baseline(reports)
    if ranking_source is not None:
        ranking = build_ranking_snapshot(Path(ranking_source))
        baseline = baseline.model_copy(update={"ranking": ranking})
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    click.echo(f"Saved baseline with {len(baseline.entries)} entries to {output_path}")
    if ranking_source is not None:
        click.echo(f"Ranking snapshot:   {len(baseline.ranking)} files")


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
        if json_file.name == "manifest.json":
            continue
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
@click.option(
    "--tasks-dir",
    default="benchmarks/tasks",
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing the task manifest bound to evidence.",
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
@click.option(
    "--max-latency-ms",
    default=None,
    type=float,
    help=(
        "Hard-fail (non-zero exit) if mean task latency exceeds this value in ms. "
        "Distinct from --warn-latency-ms, which only prints a warning. Disabled by default."
    ),
)
@click.option(
    "--promotion-strategy",
    default=None,
    type=click.Choice([strategy.value for strategy in Strategy]),
    help="Candidate strategy subject to strict all-row promotion checks.",
)
@click.option(
    "--control-strategy",
    default=None,
    type=click.Choice([strategy.value for strategy in Strategy]),
    help="Same-run control strategy for required-file, region, and line non-regression.",
)
@click.option(
    "--min-token-efficiency-with-completion",
    default=None,
    type=float,
    help="Required completion-adjusted token-efficiency floor for a promotion gate.",
)
@click.option(
    "--max-p95-warm-latency-ms",
    default=None,
    type=float,
    help="Hard maximum aggregate p95 for measured warm candidate latency in ms.",
)
@click.option(
    "--allow-sealed-corpus",
    is_flag=True,
    default=False,
    help=(
        "Required to target the sealed chronological holdout corpus "
        "(benchmarks/sealed_tasks); refused otherwise."
    ),
)
def gate_cmd(
    input_dir: str,
    tasks_dir: str,
    min_recall: float,
    min_precision: float,
    min_f1: float,
    min_mrr: float,
    baseline_dir: str | None,
    warn_latency_ms: float,
    max_latency_ms: float | None,
    promotion_strategy: str | None,
    control_strategy: str | None,
    min_token_efficiency_with_completion: float | None,
    max_p95_warm_latency_ms: float | None,
    allow_sealed_corpus: bool,
) -> None:
    """Check benchmark results against quality thresholds."""
    try:
        enforce_sealed_corpus_access(Path(tasks_dir), allow_sealed=allow_sealed_corpus)
    except SealedCorpusAccessError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        current_manifest, reports = load_evidence_reports(
            Path(input_dir),
            Path(tasks_dir),
        )
    except BenchmarkEvidenceError as exc:
        raise click.ClickException(str(exc)) from exc

    thresholds = QualityThresholds(
        min_recall=min_recall,
        min_precision=min_precision,
        min_f1=min_f1,
        min_mrr=min_mrr,
        min_token_efficiency_with_completion=(
            0.0
            if min_token_efficiency_with_completion is None
            else min_token_efficiency_with_completion
        ),
        product_default_strategy=(
            promotion_strategy
            if promotion_strategy is not None
            else QualityThresholds().product_default_strategy
        ),
        warn_latency_ms=warn_latency_ms,
        max_latency_ms=max_latency_ms,
    )
    if promotion_strategy is not None:
        if control_strategy is None:
            raise click.ClickException("--promotion-strategy requires --control-strategy")
        if baseline_dir is not None:
            raise click.ClickException("--promotion-strategy cannot be combined with --baseline")
        if min_token_efficiency_with_completion is None:
            raise click.ClickException(
                "--promotion-strategy requires --min-token-efficiency-with-completion"
            )
        if max_p95_warm_latency_ms is None:
            raise click.ClickException("--promotion-strategy requires --max-p95-warm-latency-ms")
        promotion = Strategy(promotion_strategy)
        if promotion.value in thresholds.gate_exempt_strategies:
            raise click.ClickException(
                f"--promotion-strategy {promotion.value!r} is gate-exempt and cannot be promoted"
            )
        control = Strategy(control_strategy)
        if promotion is control:
            raise click.ClickException("--promotion-strategy and --control-strategy must differ")
        candidate_reports = _select_strategy_reports(reports, promotion)
        absolute_violations = check_gate(candidate_reports, thresholds)
        warm_latency_violations = check_warm_p95_latency(
            reports,
            strategy=promotion,
            max_p95_warm_latency_ms=max_p95_warm_latency_ms,
        )
        protected_evidence_regressions = check_strategy_non_regressions(
            reports,
            candidate_strategy=promotion,
            control_strategy=control,
        )
        zero_recall_regressions = check_zero_recall_non_regression(
            reports,
            candidate_strategy=promotion,
            control_strategy=control,
        )
        language_family_regressions = check_language_family_non_regression(
            reports,
            load_benchmark_tasks(Path(tasks_dir)),
            candidate_strategy=promotion,
            control_strategy=control,
        )
        fixed_agent_regressions = check_fixed_agent_non_regression(
            reports,
            candidate_strategy=promotion,
            control_strategy=control,
        )
        protected_evidence_regressions = [
            *protected_evidence_regressions,
            *zero_recall_regressions,
            *language_family_regressions,
            *fixed_agent_regressions,
        ]
        violations = [*absolute_violations, *warm_latency_violations]
        if violations or protected_evidence_regressions:
            click.echo(
                "PROMOTION GATE FAILED: "
                f"{len(violations) + len(protected_evidence_regressions)} violation(s)"
            )
            for violation in violations:
                click.echo(
                    f"  {violation.task_id}/{violation.strategy} {violation.metric}: "
                    f"{violation.actual:.3f} < {violation.threshold:.3f}"
                )
            for regression in protected_evidence_regressions:
                click.echo(
                    f"  {regression.task_id}/{regression.strategy} {regression.metric}: "
                    f"{regression.actual:.3f} < control {regression.baseline:.3f}"
                )
            raise SystemExit(1)
        click.echo("Quality gate passed.")
        return

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

    latency_violations: list[LatencyViolation] = check_latency_violations(reports, thresholds)
    if latency_violations:
        click.echo(
            f"LATENCY GATE FAILED: {len(latency_violations)} task(s) exceeded "
            f"{max_latency_ms:.0f}ms"
        )
        for v in latency_violations:
            click.echo(
                f"  {v.task_id}/{v.strategy}: {v.actual_ms:.0f}ms"
                f" (threshold: {v.threshold_ms:.0f}ms)"
            )
        raise SystemExit(1)

    absolute_violations = check_gate(reports, thresholds)
    token_violations = token_efficiency_violations(absolute_violations)
    advisory_warnings = non_token_quality_warnings(absolute_violations)

    if baseline_dir is not None:
        try:
            baseline_manifest, baseline_reports = load_evidence_reports(
                Path(baseline_dir),
                Path(tasks_dir),
            )
            validate_baseline_coverage(current_manifest, baseline_manifest)
        except BenchmarkEvidenceError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(format_chunker_frontier_table(reports, baseline_reports))

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


@headtohead_cmd.command("competitive")
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
def headtohead_competitive_cmd(input_dir: str, output_format: str) -> None:
    """Render the competitive comparison report (per repo/task family plus aggregate)."""
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
        augmented_reports = reports_with_graphify_lanes(manifest, reports)
        compression_results = load_compression_results(
            manifest, [report.task_id for report in augmented_reports]
        )
        click.echo(format_competitive_markdown(manifest, augmented_reports, compression_results))
    except (GraphifyAdapterError, HeadToHeadManifestError, HeadroomAdapterError) as exc:
        raise click.ClickException(str(exc)) from exc


# ---------------------------------------------------------------------------
# Delta benchmark subcommands
# ---------------------------------------------------------------------------


@benchmark_cmd.group("delta", invoke_without_command=True)
@click.pass_context
def delta_cmd(ctx: click.Context) -> None:
    """Delta indexing benchmarks: measure speedup and correctness.

    Bare invocation (no subcommand) runs the full CI-runnable pipeline: every
    task under ``benchmarks/delta_tasks`` followed by the quality gate,
    exiting non-zero on any correctness or speedup violation.
    """
    if ctx.invoked_subcommand is not None:
        return
    ctx.invoke(delta_run_cmd)
    ctx.invoke(delta_gate_cmd)


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
