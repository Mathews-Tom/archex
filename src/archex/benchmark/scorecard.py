"""M3 scorecard: language/repo-size/intent/family slice aggregation and raw provenance.

The evaluation program requires results reported "by language, repository
size, query intent, and task family" rather than collapsed into one
cross-family aggregate (which would hide a language- or family-specific
regression behind an improved mean). This module builds those four
scorecards from already-persisted ``BenchmarkReport``/``BenchmarkTask`` data
and packages them, alongside their source ``BenchmarkEvidenceManifest``, into
one raw ``M3ScorecardArtifact`` with full slice provenance (which task IDs
fed each row) so a reviewer can audit coverage rather than trust an opaque
mean.

Dimension mapping, chosen from fields the benchmark model already carries
rather than inventing a new taxonomy:

- **language** -- ``BenchmarkTask.languages`` (a multi-language task
  contributes to every one of its declared languages);
- **repository size** -- the new ``BenchmarkResult.repo_size_class``,
  classified once per query from the checked-out task repository
  (see ``classify_repo_size``);
- **query intent** -- ``BenchmarkTask.category`` (``self`` /
  ``external-framework`` / ``external-large`` / ``architecture-broad`` /
  ``framework-semantic``), which already encodes what kind of retrieval
  scope/intent a task represents;
- **task family** -- ``BenchmarkTask.family`` (``comprehension`` /
  ``localization``), the existing orthogonal task-shape axis.

"Cold/warm/process-reuse" query-runtime latency splits collapse to a
cold/warm pair here: this codebase's actual cache model (``BenchmarkResult
.cached`` / ``.cache_state``) is binary -- a "warm" result already means the
query hit the process-lifetime ``QueryRuntime`` cache introduced in M2, so
there is no third, independently observable "process-reuse without cache
hit" state to split out.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from archex.benchmark.evidence import (
    BenchmarkEvidenceManifest,  # noqa: TC001 — Pydantic needs at runtime
)
from archex.benchmark.models import (  # noqa: TC001 — Pydantic needs at runtime
    BenchmarkReport,
    BenchmarkResult,
    RepoSizeClass,
    Strategy,
    TaskCompletionResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from archex.benchmark.models import BenchmarkTask

#: Source extensions counted toward a repo's size classification -- a coarse
#: scale proxy, not a language census.
_SOURCE_EXTENSIONS = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".rb",
        ".php",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cc",
        ".cs",
    }
)
_EXCLUDED_DIR_NAMES = frozenset(
    {".git", "node_modules", "vendor", "dist", "build", "__pycache__", ".venv", "venv"}
)
_SMALL_MAX_LINES = 10_000
_MEDIUM_MAX_LINES = 100_000

#: Repo-size classifications are pure functions of an immutable checked-out
#: path's on-disk contents; memoizing by resolved path avoids re-walking the
#: same repository once per (task, strategy) pair within a benchmark run.
_repo_size_cache: dict[str, RepoSizeClass] = {}


def classify_repo_size(repo_path: Path) -> RepoSizeClass:
    """Classify a checked-out repository's size from a deterministic, offline line count."""
    cache_key = str(repo_path.resolve())
    cached = _repo_size_cache.get(cache_key)
    if cached is not None:
        return cached

    total_lines = 0
    for source_path in repo_path.rglob("*"):
        if not source_path.is_file() or source_path.suffix not in _SOURCE_EXTENSIONS:
            continue
        relative_parts = source_path.relative_to(repo_path).parts[:-1]
        if _EXCLUDED_DIR_NAMES.intersection(relative_parts):
            continue
        try:
            text = source_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        total_lines += sum(1 for line in text.splitlines() if line.strip())

    if total_lines < _SMALL_MAX_LINES:
        size_class = RepoSizeClass.SMALL
    elif total_lines < _MEDIUM_MAX_LINES:
        size_class = RepoSizeClass.MEDIUM
    else:
        size_class = RepoSizeClass.LARGE
    _repo_size_cache[cache_key] = size_class
    return size_class


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _completion_outcomes(results: list[BenchmarkResult]) -> list[float]:
    outcomes: list[float] = []
    for result in results:
        outcome = (
            result.bundle_only_success
            if result.bundle_only_success is not None
            else result.task_completion_result
        )
        if outcome is TaskCompletionResult.PASS:
            outcomes.append(1.0)
        elif outcome is TaskCompletionResult.FAIL:
            outcomes.append(0.0)
    return outcomes


class ScorecardRow(BaseModel):
    """Aggregated metrics for one scorecard slice value, one dimension, one strategy."""

    dimension: str
    value: str
    strategy: Strategy
    task_count: int
    task_ids: list[str]
    mean_recall: float
    mean_precision: float
    mean_f1_score: float
    mean_mrr: float
    mean_ndcg: float
    zero_recall_count: int
    zero_recall_rate: float
    mean_duplicate_rate: float
    mean_token_efficiency: float
    mean_relevance_per_1k_tokens: float | None
    warm_p50_latency_ms: float | None
    warm_p95_latency_ms: float | None
    warm_p99_latency_ms: float | None
    cold_p50_latency_ms: float | None
    cold_p95_latency_ms: float | None
    mean_tool_calls: float
    mean_post_bundle_search_turns: float | None
    mean_receipt_accuracy: float | None
    required_file_completeness_rate: float | None
    """Fraction of tasks whose required files were all present in the returned bundle.

    On every default path this is a function of required-file recall and nothing else:
    each task contributes ``1.0`` when no required file was missing and ``0.0``
    otherwise (``completion_result_from_missing``). **No model is in the loop** —
    archex never calls one to decide whether a task was solved, so this measures
    retrieval completeness, never downstream task success.

    The one exception is the opt-in ``benchmark bundle-eval`` lane. When a result
    carries a non-null ``bundle_only_success``, ``_completion_outcomes`` prefers it over
    required-file completeness, and that value is answer correctness rather than file
    completeness — a task with perfect required-file recall contributes ``0.0`` if the
    bundle-only answer was wrong. It has two possible producers: the operator's
    evaluator command may set ``bundle_only_success`` directly, or, when it omits the
    field, archex derives it by exact string comparison of the evaluator's ``answer``
    against the task's ``expected_answer``
    (``archex.benchmark.bundle_eval._with_expected_answer_success``). archex ships no
    evaluator and makes no hosted or network call for that lane; it invokes only the
    local command the operator supplies.

    ``None`` when every task in the slice reported ``UNKNOWN``.
    """


def _build_row(
    dimension: str,
    value: str,
    strategy: Strategy,
    pairs: list[tuple[str, BenchmarkResult]],
) -> ScorecardRow:
    results = [result for _, result in pairs]
    warm_samples = [
        result.warm_latency_ms
        for result in results
        if result.cache_state == "warm" and result.warm_latency_ms and result.warm_latency_ms > 0
    ]
    cold_samples = [
        result.wall_time_ms
        for result in results
        if result.cache_state == "cold" and result.wall_time_ms is not None
    ]
    region_results = [
        result.relevance_per_1k_tokens
        for result in results
        if result.relevance_per_1k_tokens is not None
    ]
    search_turns = [
        float(result.post_bundle_search_turns)
        for result in results
        if result.post_bundle_search_turns is not None
    ]
    receipt_scores = [
        1.0 if result.receipt_accuracy else 0.0
        for result in results
        if result.receipt_accuracy is not None
    ]
    completion_outcomes = _completion_outcomes(results)
    return ScorecardRow(
        dimension=dimension,
        value=value,
        strategy=strategy,
        task_count=len(results),
        task_ids=sorted(task_id for task_id, _ in pairs),
        mean_recall=_mean([result.recall for result in results]),
        mean_precision=_mean([result.precision for result in results]),
        mean_f1_score=_mean([result.f1_score for result in results]),
        mean_mrr=_mean([result.mrr for result in results]),
        mean_ndcg=_mean([result.ndcg for result in results]),
        zero_recall_count=sum(1 for result in results if result.recall <= 0.0),
        zero_recall_rate=_mean([1.0 if result.recall <= 0.0 else 0.0 for result in results]),
        mean_duplicate_rate=_mean([result.duplicate_rate for result in results]),
        mean_token_efficiency=_mean([result.token_efficiency for result in results]),
        mean_relevance_per_1k_tokens=_mean_optional(region_results),
        warm_p50_latency_ms=_percentile(warm_samples, 0.50),
        warm_p95_latency_ms=_percentile(warm_samples, 0.95),
        warm_p99_latency_ms=_percentile(warm_samples, 0.99),
        cold_p50_latency_ms=_percentile(cold_samples, 0.50),
        cold_p95_latency_ms=_percentile(cold_samples, 0.95),
        mean_tool_calls=_mean([float(result.tool_calls) for result in results]),
        mean_post_bundle_search_turns=_mean_optional(search_turns),
        mean_receipt_accuracy=_mean_optional(receipt_scores),
        required_file_completeness_rate=_mean_optional(completion_outcomes),
    )


def build_scorecard(
    reports: Iterable[BenchmarkReport],
    tasks_by_id: dict[str, BenchmarkTask],
    *,
    strategy: Strategy,
    dimension: str,
    key_fn: Callable[[BenchmarkTask, BenchmarkResult], list[str]],
) -> list[ScorecardRow]:
    """Group one strategy's results by ``key_fn`` and aggregate each bucket into a row."""
    buckets: dict[str, list[tuple[str, BenchmarkResult]]] = defaultdict(list)
    for report in reports:
        task = tasks_by_id.get(report.task_id)
        if task is None:
            continue
        result = next((r for r in report.results if r.strategy is strategy), None)
        if result is None:
            continue
        for key in key_fn(task, result):
            buckets[key].append((report.task_id, result))
    return [_build_row(dimension, value, strategy, buckets[value]) for value in sorted(buckets)]


def build_language_scorecard(
    reports: Iterable[BenchmarkReport],
    tasks_by_id: dict[str, BenchmarkTask],
    *,
    strategy: Strategy,
) -> list[ScorecardRow]:
    return build_scorecard(
        reports,
        tasks_by_id,
        strategy=strategy,
        dimension="language",
        key_fn=lambda task, _result: list(task.languages) if task.languages else ["unspecified"],
    )


def build_repo_size_scorecard(
    reports: Iterable[BenchmarkReport],
    tasks_by_id: dict[str, BenchmarkTask],
    *,
    strategy: Strategy,
) -> list[ScorecardRow]:
    return build_scorecard(
        reports,
        tasks_by_id,
        strategy=strategy,
        dimension="repo_size",
        key_fn=lambda _task, result: [
            result.repo_size_class.value if result.repo_size_class is not None else "unmeasured"
        ],
    )


def build_intent_scorecard(
    reports: Iterable[BenchmarkReport],
    tasks_by_id: dict[str, BenchmarkTask],
    *,
    strategy: Strategy,
) -> list[ScorecardRow]:
    return build_scorecard(
        reports,
        tasks_by_id,
        strategy=strategy,
        dimension="intent",
        key_fn=lambda task, _result: [
            task.category.value if task.category is not None else "unspecified"
        ],
    )


def build_family_scorecard(
    reports: Iterable[BenchmarkReport],
    tasks_by_id: dict[str, BenchmarkTask],
    *,
    strategy: Strategy,
) -> list[ScorecardRow]:
    return build_scorecard(
        reports,
        tasks_by_id,
        strategy=strategy,
        dimension="family",
        key_fn=lambda task, _result: [task.family.value],
    )


class M3ScorecardArtifact(BaseModel):
    """Raw M3 slice-provenance artifact: one strategy's scorecards across every dimension."""

    artifact_version: Literal[1] = 1
    manifest: BenchmarkEvidenceManifest
    strategy: Strategy
    language_scorecard: list[ScorecardRow]
    repo_size_scorecard: list[ScorecardRow]
    intent_scorecard: list[ScorecardRow]
    family_scorecard: list[ScorecardRow]


def build_m3_scorecard_artifact(
    reports: list[BenchmarkReport],
    tasks_by_id: dict[str, BenchmarkTask],
    manifest: BenchmarkEvidenceManifest,
    *,
    strategy: Strategy,
) -> M3ScorecardArtifact:
    """Build the raw M3 scorecard artifact for one strategy from persisted evidence."""
    return M3ScorecardArtifact(
        manifest=manifest,
        strategy=strategy,
        language_scorecard=build_language_scorecard(reports, tasks_by_id, strategy=strategy),
        repo_size_scorecard=build_repo_size_scorecard(reports, tasks_by_id, strategy=strategy),
        intent_scorecard=build_intent_scorecard(reports, tasks_by_id, strategy=strategy),
        family_scorecard=build_family_scorecard(reports, tasks_by_id, strategy=strategy),
    )


def save_m3_scorecard_artifact(path: Path, artifact: M3ScorecardArtifact) -> None:
    """Write the artifact as pretty-printed, stable-schema JSON."""
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def load_m3_scorecard_artifact(path: Path) -> M3ScorecardArtifact:
    """Load a previously saved M3 scorecard artifact."""
    return M3ScorecardArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def _format_optional(value: float | None, *, precision: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{precision}f}"


def _format_rows_table(rows: list[ScorecardRow]) -> list[str]:
    if not rows:
        return ["_No tasks in this dimension._"]
    lines = [
        "| Value | Tasks | Recall | F1 | MRR | Zero-Recall | Dup. Rate "
        "| Tok. Eff. | Warm p50 | Warm p95 | Cold p50 | Required-File Completeness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.value} | {row.task_count} | {row.mean_recall:.3f} "
            f"| {row.mean_f1_score:.3f} | {row.mean_mrr:.3f} "
            f"| {row.zero_recall_count} ({row.zero_recall_rate:.1%}) "
            f"| {row.mean_duplicate_rate:.3f} | {row.mean_token_efficiency:.3f} "
            f"| {_format_optional(row.warm_p50_latency_ms, precision=0)} "
            f"| {_format_optional(row.warm_p95_latency_ms, precision=0)} "
            f"| {_format_optional(row.cold_p50_latency_ms, precision=0)} "
            f"| {_format_optional(row.required_file_completeness_rate)} |"
        )
    return lines


def format_m3_scorecard_markdown(artifact: M3ScorecardArtifact) -> str:
    """Render every dimension's scorecard as Markdown, one table per dimension."""
    lines = [
        "# M3 External Quality Frontier Scorecard",
        "",
        f"Strategy: `{artifact.strategy.value}`",
        f"Source revision: `{artifact.manifest.source_revision}`",
        f"Task manifest digest: `{artifact.manifest.task_manifest_digest}`",
        "",
    ]
    for title, rows in (
        ("Language", artifact.language_scorecard),
        ("Repository Size", artifact.repo_size_scorecard),
        ("Query Intent", artifact.intent_scorecard),
        ("Task Family", artifact.family_scorecard),
    ):
        lines.append(f"## By {title}")
        lines.append("")
        lines.extend(_format_rows_table(rows))
        lines.append("")
    return "\n".join(lines)
