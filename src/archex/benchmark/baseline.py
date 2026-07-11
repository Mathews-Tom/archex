"""Baseline snapshot: save, load, and compare benchmark baselines for regression detection."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from archex.benchmark.models import BenchmarkReport  # noqa: TCH001 — Pydantic needs at runtime

BASELINE_MANIFEST_FILENAME = "manifest.json"
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class BaselineContractError(ValueError):
    """Raised when a benchmark baseline cannot prove its identity or coverage."""


class BaselineCoverageError(BaselineContractError):
    """Raised when baseline and current task/strategy keys differ."""


class BaselineManifest(BaseModel):
    """Identity and coverage contract for a directory of raw benchmark reports."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: int = Field(ge=1)
    archex_version: str = Field(min_length=1)
    source_revision: str
    task_manifest_digest: str
    task_ids: list[str] = Field(min_length=1)
    strategies: list[str] = Field(min_length=1)
    retrieval_options: dict[str, str] = Field(min_length=1)
    generated_at: str


class BaselineDirectory(BaseModel):
    """A validated canonical baseline and the reports it governs."""

    manifest: BaselineManifest
    reports: list[BenchmarkReport]


class BaselineEntry(BaseModel):
    task_id: str
    strategy: str
    recall: float
    precision: float
    f1_score: float
    mrr: float
    ndcg: float = 0.0
    map_score: float = 0.0
    token_efficiency: float = 0.0


class RankingSnapshotEntry(BaseModel):
    """Per-file PageRank centrality and symbol count for ranking-stability gating."""

    file_path: str
    centrality: float
    symbol_count: int


class Baseline(BaseModel):
    entries: list[BaselineEntry] = []
    ranking: list[RankingSnapshotEntry] = []
    created_at: str = ""
    archex_version: str = ""


class BaselineComparison(BaseModel):
    task_id: str
    strategy: str
    metric: str
    baseline_value: float
    current_value: float
    delta: float
    regression: bool


def save_baseline(
    reports: list[BenchmarkReport],
    archex_version: str = "",
) -> Baseline:
    """Extract metrics from reports into a Baseline snapshot."""
    entries: list[BaselineEntry] = []
    for report in reports:
        for r in report.results:
            entries.append(
                BaselineEntry(
                    task_id=r.task_id,
                    strategy=r.strategy.value,
                    recall=r.recall,
                    precision=r.precision,
                    f1_score=r.f1_score,
                    mrr=r.mrr,
                    ndcg=r.ndcg,
                    map_score=r.map_score,
                    token_efficiency=r.token_efficiency,
                )
            )
    return Baseline(
        entries=entries,
        created_at=datetime.now(tz=UTC).isoformat(),
        archex_version=archex_version,
    )


def load_baseline(data: dict[str, object]) -> Baseline:
    """Validate and load a baseline from parsed JSON data."""
    return Baseline.model_validate(data)


def task_manifest_digest(tasks_dir: Path) -> str:
    """Return the stable digest for the exact benchmark task manifest."""
    task_files = sorted(tasks_dir.glob("*.yaml"))
    if not task_files:
        raise BaselineContractError(f"No benchmark task files found in {tasks_dir}")

    digest = sha256()
    for task_file in task_files:
        digest.update(task_file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(task_file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _report_paths(directory: Path) -> list[Path]:
    return [
        path for path in sorted(directory.glob("*.json")) if path.name != BASELINE_MANIFEST_FILENAME
    ]


def load_benchmark_reports(directory: Path) -> list[BenchmarkReport]:
    """Load raw ``BenchmarkReport`` JSON files from *directory*."""
    reports = [
        BenchmarkReport.model_validate_json(path.read_text(encoding="utf-8"))
        for path in _report_paths(directory)
    ]
    if not reports:
        raise BaselineContractError(f"No benchmark result files found in {directory}")
    return reports


def _report_keys(reports: Sequence[BenchmarkReport]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for report in reports:
        for result in report.results:
            if result.task_id != report.task_id:
                raise BaselineContractError(
                    "Benchmark report task/result identity mismatch: "
                    f"{report.task_id!r} contains {result.task_id!r}"
                )
            key = (report.task_id, result.strategy.value)
            if key in keys:
                raise BaselineContractError(
                    f"Duplicate benchmark report result for {key[0]}/{key[1]}"
                )
            keys.add(key)
    return keys


def _validate_manifest_identity(manifest: BaselineManifest) -> None:
    if not manifest.archex_version.strip():
        raise BaselineContractError("Baseline manifest archex_version must not be empty")
    if not _GIT_REVISION_RE.fullmatch(manifest.source_revision):
        raise BaselineContractError(
            "Baseline manifest source_revision must be a resolved 40-character Git SHA"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.task_manifest_digest):
        raise BaselineContractError(
            "Baseline manifest task_manifest_digest must be a SHA-256 digest"
        )
    if len(manifest.task_ids) != len(set(manifest.task_ids)):
        raise BaselineContractError("Baseline manifest contains duplicate task IDs")
    if len(manifest.strategies) != len(set(manifest.strategies)):
        raise BaselineContractError("Baseline manifest contains duplicate strategies")
    if any(not value.strip() for value in manifest.task_ids):
        raise BaselineContractError("Baseline manifest contains an empty task ID")
    if any(not value.strip() for value in manifest.strategies):
        raise BaselineContractError("Baseline manifest contains an empty strategy")
    if any(
        not key.strip() or not value.strip() for key, value in manifest.retrieval_options.items()
    ):
        raise BaselineContractError(
            "Baseline manifest retrieval_options must contain non-empty keys and values"
        )


def _coverage_error(
    *,
    missing: set[tuple[str, str]],
    unexpected: set[tuple[str, str]],
) -> BaselineCoverageError:
    details: list[str] = []
    if missing:
        details.append(
            "missing=" + ", ".join(f"{task_id}/{strategy}" for task_id, strategy in sorted(missing))
        )
    if unexpected:
        details.append(
            "unexpected="
            + ", ".join(f"{task_id}/{strategy}" for task_id, strategy in sorted(unexpected))
        )
    return BaselineCoverageError("Baseline task/strategy coverage mismatch: " + "; ".join(details))


def validate_baseline_directory(
    manifest: BaselineManifest,
    reports: Sequence[BenchmarkReport],
    *,
    tasks_dir: Path,
) -> None:
    """Validate immutable identity and exact current task/strategy coverage."""
    _validate_manifest_identity(manifest)
    current_digest = task_manifest_digest(tasks_dir)
    if manifest.task_manifest_digest != current_digest:
        raise BaselineContractError(
            "Baseline task manifest digest does not match "
            f"{tasks_dir}: expected {current_digest}, got {manifest.task_manifest_digest}"
        )

    from archex.benchmark.loader import load_tasks

    current_task_ids = {task.task_id for task in load_tasks(tasks_dir)}
    manifest_task_ids = set(manifest.task_ids)
    if current_task_ids != manifest_task_ids:
        missing_tasks = current_task_ids - manifest_task_ids
        unexpected_tasks = manifest_task_ids - current_task_ids
        details: list[str] = []
        if missing_tasks:
            details.append("missing tasks=" + ", ".join(sorted(missing_tasks)))
        if unexpected_tasks:
            details.append("unexpected tasks=" + ", ".join(sorted(unexpected_tasks)))
        raise BaselineCoverageError("Baseline task coverage mismatch: " + "; ".join(details))

    expected_keys = {
        (task_id, strategy) for task_id in manifest.task_ids for strategy in manifest.strategies
    }
    actual_keys = _report_keys(reports)
    if expected_keys != actual_keys:
        raise _coverage_error(
            missing=expected_keys - actual_keys,
            unexpected=actual_keys - expected_keys,
        )


def load_baseline_directory(directory: Path, *, tasks_dir: Path) -> BaselineDirectory:
    """Load and validate the canonical baseline directory consumed by ``benchmark gate``."""
    manifest_path = directory / BASELINE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise BaselineContractError(
            f"Baseline directory {directory} is missing {BASELINE_MANIFEST_FILENAME}"
        )
    try:
        manifest = BaselineManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BaselineContractError(f"Invalid baseline manifest at {manifest_path}: {exc}") from exc
    reports = load_benchmark_reports(directory)
    validate_baseline_directory(manifest, reports, tasks_dir=tasks_dir)
    return BaselineDirectory(manifest=manifest, reports=reports)


def _resolved_source_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise BaselineContractError("Unable to resolve the current Git source revision") from exc
    source_revision = result.stdout.strip()
    if not _GIT_REVISION_RE.fullmatch(source_revision):
        raise BaselineContractError(
            f"Resolved source revision is not a full immutable Git SHA: {source_revision!r}"
        )
    return source_revision


def create_baseline_directory(
    input_dir: Path,
    output_dir: Path,
    *,
    tasks_dir: Path,
    retrieval_options: Mapping[str, str],
) -> BaselineDirectory:
    """Create an identity-bearing, coverage-complete baseline directory."""
    reports = load_benchmark_reports(input_dir)
    from archex.benchmark.loader import load_tasks

    task_ids = sorted(task.task_id for task in load_tasks(tasks_dir))
    strategies = sorted({result.strategy.value for report in reports for result in report.results})
    manifest = BaselineManifest(
        manifest_version=1,
        archex_version=version("archex"),
        source_revision=_resolved_source_revision(),
        task_manifest_digest=task_manifest_digest(tasks_dir),
        task_ids=task_ids,
        strategies=strategies,
        retrieval_options=dict(sorted(retrieval_options.items())),
        generated_at=datetime.now(tz=UTC).isoformat(),
    )
    validate_baseline_directory(manifest, reports, tasks_dir=tasks_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        raise BaselineContractError(
            f"Refusing to overwrite non-empty baseline directory {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for report_path in _report_paths(input_dir):
        shutil.copy2(report_path, output_dir / report_path.name)
    (output_dir / BASELINE_MANIFEST_FILENAME).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return BaselineDirectory(manifest=manifest, reports=reports)


def build_ranking_snapshot(repo_root: Path) -> list[RankingSnapshotEntry]:
    """Index *repo_root* and snapshot per-file PageRank centrality and symbol count.

    Used to populate ``Baseline.ranking`` and, later, to compare a promotion
    candidate's index against that baseline for ranking-stability regressions
    (see ``archex.benchmark.gate.check_ranking_stability``). Files with no import
    edges get centrality ``0.0``; every indexed file gets its chunk-derived
    symbol count regardless of edges.
    """
    from archex.api import index_repository
    from archex.config import load_config
    from archex.index.graph import DependencyGraph
    from archex.models import RepoSource

    source = RepoSource(local_path=str(repo_root))
    config = load_config(source)
    store = index_repository(source, config=config)
    try:
        centrality = DependencyGraph.from_edges(store.get_edges()).structural_centrality()
        file_metadata = store.get_file_metadata()
    finally:
        store.close()

    return [
        RankingSnapshotEntry(
            file_path=str(item["file_path"]),
            centrality=centrality.get(str(item["file_path"]), 0.0),
            symbol_count=int(item["symbol_count"]),
        )
        for item in file_metadata
    ]


_METRICS = ("recall", "precision", "f1_score", "mrr", "ndcg", "map_score", "token_efficiency")
_DEFAULT_TOLERANCE = 0.05


def compare_baseline(
    reports: list[BenchmarkReport],
    baseline: Baseline,
    tolerance: float = _DEFAULT_TOLERANCE,
    excluded_strategies: Collection[str] = (),
) -> list[BaselineComparison]:
    """Compare current reports against a baseline with exact coverage."""
    excluded = set(excluded_strategies)
    baseline_lookup = {
        (entry.task_id, entry.strategy): entry
        for entry in baseline.entries
        if entry.strategy not in excluded
    }
    current_results = {
        (report.task_id, result.strategy.value): result
        for report in reports
        for result in report.results
        if result.strategy.value not in excluded
    }
    baseline_keys = set(baseline_lookup)
    current_keys = set(current_results)
    if baseline_keys != current_keys:
        raise _coverage_error(
            missing=baseline_keys - current_keys,
            unexpected=current_keys - baseline_keys,
        )

    comparisons: list[BaselineComparison] = []
    for (task_id, strategy), result in current_results.items():
        entry = baseline_lookup[(task_id, strategy)]
        for metric in _METRICS:
            baseline_val = getattr(entry, metric)
            current_val = getattr(result, metric)
            delta = current_val - baseline_val
            regression = current_val < baseline_val - tolerance
            comparisons.append(
                BaselineComparison(
                    task_id=task_id,
                    strategy=strategy,
                    metric=metric,
                    baseline_value=baseline_val,
                    current_value=current_val,
                    delta=delta,
                    regression=regression,
                )
            )
    return comparisons
