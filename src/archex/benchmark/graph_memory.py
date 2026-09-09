"""Tool-neutral graph-memory lane contract for the competitive comparison.

A graph-memory lane is any tool whose product is a persistent code graph that is
queried after a build step. The public comparison therefore models two lanes per
tool — ``<tool>_build_plus_query`` and ``<tool>_query_warm`` — so graph-build cost
is never collapsed into warm query latency.

This module owns what every graph-memory tool records identically: the shared
artifact fields, the cold/warm timing contract, and the normalization into a
:class:`BenchmarkResult`. Tool-specific provenance (registry package identity,
distribution integrity, source revision, extraction tier, ...) stays in the
per-tool adapter, so a new tool cannot silently inherit a field it never measured
and existing artifacts do not gain requirements retroactively.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from archex.benchmark.models import (
    BenchmarkResult,
    BenchmarkSpecModel,
    ComparisonLayerType,
    GraphMemoryLaneConfig,
    Strategy,
    TaskCompletionResult,
)


class GraphMemoryUnavailableError(RuntimeError):
    """Raised when local graph-memory execution is required but unavailable."""


class GraphMemoryAdapterError(RuntimeError):
    """Raised when a graph-memory lane cannot produce a valid benchmark result."""


class GraphMemoryArtifact(BenchmarkSpecModel):
    """Fields every operator-produced graph-memory lane artifact records.

    Per-tool artifact schemas subclass this and add their own pinned-identity
    fields. Nothing here is specific to one tool, and nothing here is optional
    for a valid cell.
    """

    task_id: str
    lane: str
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


def now_iso() -> str:
    """UTC timestamp for artifacts produced in-session."""
    return datetime.now(UTC).isoformat()


def artifact_digest(raw: bytes) -> str:
    """SHA-256 of an artifact's exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def validate_graph_memory_artifact(
    config: GraphMemoryLaneConfig,
    artifact: GraphMemoryArtifact,
    *,
    source: str,
    package: str,
    version: str,
) -> None:
    """Fail closed unless ``artifact`` matches ``config``'s pinned lane contract.

    ``package``/``version`` come from the tool's own provenance fields so each
    adapter keeps its registry naming instead of sharing one field name.
    """
    lane = config.name
    if package != config.package_name:
        raise GraphMemoryAdapterError(
            f"graph-memory lane {lane!r} {source} package {package!r} does not match "
            f"pinned package {config.package_name!r}"
        )
    if version != config.version:
        raise GraphMemoryAdapterError(
            f"graph-memory lane {lane!r} {source} version {version!r} does not match "
            f"pinned version {config.version!r}"
        )
    if artifact.lane != lane:
        raise GraphMemoryAdapterError(
            f"graph-memory lane {lane!r} {source} lane {artifact.lane!r} "
            "does not match manifest lane"
        )
    if artifact.includes_build_cost is not config.includes_build_cost:
        verb = (
            "must include build cost"
            if config.includes_build_cost
            else "must not include build cost"
        )
        raise GraphMemoryAdapterError(f"graph-memory lane {lane!r} {source} {verb}")
    if artifact.cache_state not in {"cold", "warm"}:
        raise GraphMemoryAdapterError(
            f"graph-memory lane {lane!r} {source} cache_state "
            f"{artifact.cache_state!r} must be 'cold' or 'warm'"
        )
    if config.includes_build_cost:
        if artifact.cold_start_ms <= 0.0:
            raise GraphMemoryAdapterError(
                f"graph-memory lane {lane!r} {source} must report cold_start_ms > 0"
            )
        if artifact.cache_state != "cold" or artifact.cached:
            raise GraphMemoryAdapterError(
                f"graph-memory lane {lane!r} {source} must report a cold, uncached run"
            )
    else:
        if artifact.cold_start_ms != 0.0:
            raise GraphMemoryAdapterError(
                f"graph-memory lane {lane!r} {source} must report cold_start_ms == 0"
            )
        if artifact.cache_state != "warm" or not artifact.cached:
            raise GraphMemoryAdapterError(
                f"graph-memory lane {lane!r} {source} must report a warm, cached run"
            )
    if artifact.wall_time_ms < (artifact.cold_start_ms + artifact.warm_latency_ms):
        raise GraphMemoryAdapterError(
            f"graph-memory lane {lane!r} {source} wall_time_ms "
            "must cover cold_start_ms + warm_latency_ms"
        )


def result_from_graph_memory_artifact(
    config: GraphMemoryLaneConfig,
    artifact: GraphMemoryArtifact,
    *,
    run_mode: str,
    package: str,
    version: str,
    extra_provenance: dict[str, str] | None = None,
    artifact_path: Path | None = None,
    digest: str | None = None,
) -> BenchmarkResult:
    """Normalize one validated graph-memory artifact into a comparison result."""
    provenance = {
        "external_tool": config.name,
        "external_tool_version": version,
        "external_tool_embedder": "n/a",
        "external_tool_command": artifact.command,
        "external_tool_token_mode": "reported",
        "external_tool_result_count": str(len(artifact.result_files)),
        "external_tool_bootstrap_count": "1" if artifact.includes_build_cost else "0",
        "graph_memory_tool": config.tool.value,
        "graph_memory_mode": config.mode.value,
        "graph_memory_package": package,
        "graph_memory_command": artifact.command,
        "graph_memory_run_mode": run_mode,
        "graph_memory_layer_type": ComparisonLayerType.GRAPH_MEMORY.value,
        "graph_memory_includes_build_cost": str(artifact.includes_build_cost).lower(),
    }
    if artifact.backend:
        provenance["graph_memory_backend"] = artifact.backend
    if artifact.local_offline_posture:
        provenance["graph_memory_local_offline_posture"] = artifact.local_offline_posture
    if artifact.operational_notes:
        provenance["graph_memory_operational_notes"] = artifact.operational_notes
    if artifact_path is not None:
        provenance["graph_memory_artifact_path"] = str(artifact_path)
    if digest is not None:
        provenance["graph_memory_artifact_sha256"] = digest
    if extra_provenance:
        provenance.update(extra_provenance)

    return BenchmarkResult(
        task_id=artifact.task_id,
        strategy=Strategy.EXTERNAL_MCP,
        strategy_label=config.name,
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
        timestamp=artifact.timestamp or now_iso(),
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
