"""Graphify graph/memory-layer adapter for the competitive comparison.

The public comparison keeps Graphify separate from retrieval engines because its
workflow spans graph construction plus graph-backed query. This adapter supports
both modes the follow-up comparison needs:

* local command mode — execute a configured command that reads benchmark payload
  JSON on stdin and emits one Graphify lane artifact JSON on stdout.
* artifact mode — import an operator-produced ``{task_id}.json`` artifact with
  pinned Graphify provenance when local execution is not feasible.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from shutil import which

from pydantic import Field, ValidationError

from archex.benchmark.models import (
    BenchmarkResult,
    BenchmarkSpecModel,
    BenchmarkTask,
    ComparisonLayerType,
    GraphifyLaneConfig,
    GraphifyLaneName,
    Strategy,
    TaskCompletionResult,
)


class GraphifyUnavailableError(RuntimeError):
    """Raised when local Graphify execution is required but unavailable."""


class GraphifyAdapterError(RuntimeError):
    """Raised when a Graphify lane cannot produce a valid benchmark result."""


class GraphifyArtifact(BenchmarkSpecModel):
    """Schema for one operator-produced ``{task_id}.json`` Graphify lane artifact."""

    task_id: str
    lane: GraphifyLaneName
    graphify_package: str = "graphifyy"
    graphify_version: str
    command: str
    includes_build_cost: bool
    tokens_total: int = Field(ge=0)
    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    files_accessed: int = Field(ge=0)
    recall: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg: float = Field(ge=0.0, le=1.0)
    map_score: float = Field(ge=0.0, le=1.0)
    required_file_recall: float = Field(ge=0.0, le=1.0)
    missed_required_file_rate: float = Field(ge=0.0, le=1.0)
    missed_required_task_rate: float = Field(ge=0.0, le=1.0)
    all_required_files_present: bool
    required_files_present: list[str] = []
    required_files_missing: list[str] = []
    result_files: list[str] = []
    task_completion_result: TaskCompletionResult = TaskCompletionResult.UNKNOWN
    bundle_completion_tokens: int = Field(default=0, ge=0)
    bundle_completion_files: list[str] = []
    token_efficiency: float = Field(ge=0.0, le=1.0)
    token_efficiency_with_completion: float = Field(ge=0.0, le=1.0)
    cold_start_ms: float = Field(ge=0.0)
    warm_latency_ms: float = Field(ge=0.0)
    wall_time_ms: float = Field(ge=0.0)
    cache_state: str = "cold"
    cached: bool = False
    receipt_accuracy: bool | None = None
    freshness_latency_ms: float = Field(default=0.0, ge=0.0)
    freshness_measured: bool = False
    freshness_correct: bool = False
    region_recall: float | None = None
    line_recall: float | None = None
    context_noise_ratio: float | None = None
    bundle_compression_ratio: float | None = None
    operational_notes: str | None = None
    local_offline_posture: str | None = None
    backend: str | None = None
    timestamp: str | None = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_artifact_matches_config(
    config: GraphifyLaneConfig,
    artifact: GraphifyArtifact,
    *,
    source: str,
) -> None:
    if artifact.graphify_package != config.package_name:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} {source} package "
            f"{artifact.graphify_package!r} does not match pinned package "
            f"{config.package_name!r}"
        )
    if artifact.graphify_version != config.version:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} {source} version "
            f"{artifact.graphify_version!r} does not match pinned version {config.version!r}"
        )
    if artifact.lane is not config.name:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} {source} lane {artifact.lane.value!r} "
            "does not match manifest lane"
        )
    if artifact.includes_build_cost is not config.includes_build_cost:
        verb = (
            "must include build cost"
            if config.includes_build_cost
            else "must not include build cost"
        )
        raise GraphifyAdapterError(f"graphify lane {config.name.value!r} {source} {verb}")
    if artifact.cache_state not in {"cold", "warm"}:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} {source} cache_state "
            f"{artifact.cache_state!r} must be 'cold' or 'warm'"
        )
    if config.includes_build_cost:
        if artifact.cold_start_ms <= 0.0:
            raise GraphifyAdapterError(
                f"graphify lane {config.name.value!r} {source} must report cold_start_ms > 0"
            )
        if artifact.cache_state != "cold" or artifact.cached:
            raise GraphifyAdapterError(
                f"graphify lane {config.name.value!r} {source} must report a cold, uncached run"
            )
    else:
        if artifact.cold_start_ms != 0.0:
            raise GraphifyAdapterError(
                f"graphify lane {config.name.value!r} {source} must report cold_start_ms == 0"
            )
        if artifact.cache_state != "warm" or not artifact.cached:
            raise GraphifyAdapterError(
                f"graphify lane {config.name.value!r} {source} must report a warm, cached run"
            )
    if artifact.wall_time_ms < (artifact.cold_start_ms + artifact.warm_latency_ms):
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} {source} wall_time_ms "
            "must cover cold_start_ms + warm_latency_ms"
        )


def _result_from_artifact(
    config: GraphifyLaneConfig,
    artifact: GraphifyArtifact,
    *,
    run_mode: str,
    artifact_path: Path | None = None,
    digest: str | None = None,
) -> BenchmarkResult:
    provenance = {
        "external_tool": config.name.value,
        "external_tool_version": config.version,
        "external_tool_embedder": "n/a",
        "external_tool_command": artifact.command,
        "external_tool_token_mode": "reported",
        "external_tool_result_count": str(len(artifact.result_files)),
        "external_tool_bootstrap_count": "1" if config.includes_build_cost else "0",
        "graphify_package": artifact.graphify_package,
        "graphify_command": artifact.command,
        "graphify_run_mode": run_mode,
        "graphify_layer_type": ComparisonLayerType.GRAPH_MEMORY.value,
        "graphify_includes_build_cost": str(artifact.includes_build_cost).lower(),
    }
    if artifact.backend:
        provenance["graphify_backend"] = artifact.backend
    if artifact.local_offline_posture:
        provenance["graphify_local_offline_posture"] = artifact.local_offline_posture
    if artifact.operational_notes:
        provenance["graphify_operational_notes"] = artifact.operational_notes
    if artifact_path is not None:
        provenance["graphify_artifact_path"] = str(artifact_path)
    if digest is not None:
        provenance["graphify_artifact_sha256"] = digest

    return BenchmarkResult(
        task_id=artifact.task_id,
        strategy=Strategy.EXTERNAL_MCP,
        strategy_label=config.name.value,
        tokens_total=artifact.tokens_total,
        tool_calls=artifact.tool_calls,
        files_accessed=artifact.files_accessed,
        recall=artifact.recall,
        precision=artifact.precision,
        f1_score=artifact.f1_score,
        mrr=artifact.mrr,
        ndcg=artifact.ndcg,
        map_score=artifact.map_score,
        tokens_input=artifact.tokens_input,
        tokens_output=artifact.tokens_output,
        token_efficiency=artifact.token_efficiency,
        tokens_raw_baseline=0,
        savings_vs_raw=0.0,
        wall_time_ms=artifact.wall_time_ms,
        cached=artifact.cached,
        timestamp=artifact.timestamp or _now_iso(),
        cache_state=artifact.cache_state,
        result_files=artifact.result_files,
        required_file_recall=artifact.required_file_recall,
        missed_required_file_rate=artifact.missed_required_file_rate,
        missed_required_task_rate=artifact.missed_required_task_rate,
        all_required_files_present=artifact.all_required_files_present,
        required_files_present=artifact.required_files_present,
        required_files_missing=artifact.required_files_missing,
        bundle_completion_tokens=artifact.bundle_completion_tokens,
        bundle_completion_files=artifact.bundle_completion_files,
        task_completion_result=artifact.task_completion_result,
        token_efficiency_with_completion=artifact.token_efficiency_with_completion,
        cold_start_ms=artifact.cold_start_ms,
        warm_latency_ms=artifact.warm_latency_ms,
        provenance=provenance,
        receipt_accuracy=artifact.receipt_accuracy,
        freshness_latency_ms=artifact.freshness_latency_ms,
        freshness_measured=artifact.freshness_measured,
        freshness_correct=artifact.freshness_correct,
        region_recall=artifact.region_recall,
        line_recall=artifact.line_recall,
        context_noise_ratio=artifact.context_noise_ratio,
        bundle_compression_ratio=artifact.bundle_compression_ratio,
    )


def _parse_artifact(
    config: GraphifyLaneConfig,
    artifact_path: Path,
) -> tuple[GraphifyArtifact, str]:
    raw = artifact_path.read_bytes()
    try:
        artifact = GraphifyArtifact.model_validate_json(raw)
    except ValidationError as exc:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} artifact {artifact_path} is invalid: {exc}"
        ) from exc
    _validate_artifact_matches_config(config, artifact, source=f"artifact {artifact_path}")
    return artifact, hashlib.sha256(raw).hexdigest()


def load_graphify_artifact(
    config: GraphifyLaneConfig,
    *,
    task_id: str,
    artifact_dir: Path,
) -> BenchmarkResult:
    """Import one operator-produced Graphify lane artifact for ``task_id``."""
    artifact_path = artifact_dir / f"{task_id}.json"
    if not artifact_path.is_file():
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} artifact not found: {artifact_path}"
        )
    artifact, digest = _parse_artifact(config, artifact_path)
    if artifact.task_id != task_id:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} artifact {artifact_path} has task_id "
            f"{artifact.task_id!r}, expected {task_id!r}"
        )
    return _result_from_artifact(
        config,
        artifact,
        run_mode="artifact",
        artifact_path=artifact_path,
        digest=digest,
    )


def load_graphify_results(
    lanes: list[GraphifyLaneConfig], task_ids: list[str]
) -> list[BenchmarkResult]:
    """Import every available artifact-backed Graphify result for ``task_ids``."""
    results: list[BenchmarkResult] = []
    for config in lanes:
        if config.artifact_dir is None:
            continue
        artifact_dir = Path(config.artifact_dir)
        for task_id in task_ids:
            artifact_path = artifact_dir / f"{task_id}.json"
            if not artifact_path.is_file():
                continue
            results.append(
                load_graphify_artifact(config, task_id=task_id, artifact_dir=artifact_dir)
            )
    return results


def run_graphify_lane(
    config: GraphifyLaneConfig,
    *,
    task: BenchmarkTask,
    repo_path: Path | None,
) -> BenchmarkResult:
    """Produce one Graphify lane result via artifact import or local command mode."""
    if config.artifact_dir is not None:
        return load_graphify_artifact(
            config,
            task_id=task.task_id,
            artifact_dir=Path(config.artifact_dir),
        )
    if which(config.command) is None:
        raise GraphifyUnavailableError(
            f"graphify lane {config.name.value!r} command {config.command!r} not found; "
            "set graphify_lanes[].artifact_dir to import operator artifacts instead"
        )
    if repo_path is None:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} local execution requires repo_path"
        )

    payload = {
        "task": task.model_dump(mode="json"),
        "repo_path": str(repo_path),
        "lane": config.name.value,
        "graphify": {
            "package_name": config.package_name,
            "version": config.version,
            "includes_build_cost": config.includes_build_cost,
        },
    }
    command = [config.command, *config.args]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            env={**os.environ, **config.env} if config.env else None,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} timed out after {config.timeout_seconds}s"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} failed with exit {completed.returncode}: {detail}"
        )
    try:
        artifact = GraphifyArtifact.model_validate_json(completed.stdout)
    except ValidationError as exc:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} local output is invalid: {exc}"
        ) from exc
    if artifact.task_id != task.task_id:
        raise GraphifyAdapterError(
            f"graphify lane {config.name.value!r} local output task_id {artifact.task_id!r} "
            f"does not match benchmark task {task.task_id!r}"
        )
    _validate_artifact_matches_config(config, artifact, source="local output")
    return _result_from_artifact(config, artifact, run_mode="local")
