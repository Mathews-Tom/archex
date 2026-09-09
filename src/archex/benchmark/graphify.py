"""Graphify adapter for the graph-memory comparison lanes.

The public comparison keeps Graphify separate from retrieval engines because its
workflow spans graph construction plus graph-backed query. The shared cold/warm
contract lives in :mod:`archex.benchmark.graph_memory`; this module adds only
Graphify's pinned package identity and its two execution modes:

* local command mode — execute a configured command that reads benchmark payload
  JSON on stdin and emits one Graphify lane artifact JSON on stdout.
* artifact mode — import an operator-produced ``{task_id}.json`` artifact with
  pinned Graphify provenance when local execution is not feasible.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING

from pydantic import ValidationError

from archex.benchmark.graph_memory import (
    GraphMemoryAdapterError,
    GraphMemoryArtifact,
    GraphMemoryUnavailableError,
    artifact_digest,
    result_from_graph_memory_artifact,
    validate_graph_memory_artifact,
)

if TYPE_CHECKING:
    from archex.benchmark.models import (
        BenchmarkResult,
        BenchmarkTask,
        GraphMemoryLaneConfig,
    )


class GraphifyArtifact(GraphMemoryArtifact):
    """Schema for one operator-produced ``{task_id}.json`` Graphify lane artifact."""

    graphify_package: str = "graphifyy"
    graphify_version: str


def _validate_artifact(
    config: GraphMemoryLaneConfig,
    artifact: GraphifyArtifact,
    *,
    source: str,
) -> None:
    validate_graph_memory_artifact(
        config,
        artifact,
        source=source,
        package=artifact.graphify_package,
        version=artifact.graphify_version,
    )


def _result_from_artifact(
    config: GraphMemoryLaneConfig,
    artifact: GraphifyArtifact,
    *,
    run_mode: str,
    artifact_path: Path | None = None,
    digest: str | None = None,
) -> BenchmarkResult:
    return result_from_graph_memory_artifact(
        config,
        artifact,
        run_mode=run_mode,
        package=artifact.graphify_package,
        version=artifact.graphify_version,
        artifact_path=artifact_path,
        digest=digest,
    )


def _parse_artifact(
    config: GraphMemoryLaneConfig,
    artifact_path: Path,
) -> tuple[GraphifyArtifact, str]:
    raw = artifact_path.read_bytes()
    try:
        artifact = GraphifyArtifact.model_validate_json(raw)
    except ValidationError as exc:
        raise GraphMemoryAdapterError(
            f"graph-memory lane {config.name!r} artifact {artifact_path} is invalid: {exc}"
        ) from exc
    _validate_artifact(config, artifact, source=f"artifact {artifact_path}")
    return artifact, artifact_digest(raw)


def load_graphify_artifact(
    config: GraphMemoryLaneConfig,
    *,
    task_id: str,
    artifact_dir: Path,
) -> BenchmarkResult:
    """Import one operator-produced Graphify lane artifact for ``task_id``."""
    artifact_path = artifact_dir / f"{task_id}.json"
    if not artifact_path.is_file():
        raise GraphMemoryAdapterError(
            f"graph-memory lane {config.name!r} artifact not found: {artifact_path}"
        )
    artifact, digest = _parse_artifact(config, artifact_path)
    if artifact.task_id != task_id:
        raise GraphMemoryAdapterError(
            f"graph-memory lane {config.name!r} artifact {artifact_path} has task_id "
            f"{artifact.task_id!r}, expected {task_id!r}"
        )
    return _result_from_artifact(
        config,
        artifact,
        run_mode="artifact",
        artifact_path=artifact_path,
        digest=digest,
    )


def run_graphify_lane(
    config: GraphMemoryLaneConfig,
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
        raise GraphMemoryUnavailableError(
            f"graph-memory lane {config.name!r} command {config.command!r} not found; "
            "set graph_memory_lanes[].artifact_dir to import operator artifacts instead"
        )
    if repo_path is None:
        raise GraphMemoryAdapterError(
            f"graph-memory lane {config.name!r} local execution requires repo_path"
        )

    payload = {
        "task": task.model_dump(mode="json"),
        "repo_path": str(repo_path),
        "lane": config.name,
        "tool": config.tool.value,
        "mode": config.mode.value,
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
        raise GraphMemoryAdapterError(
            f"graph-memory lane {config.name!r} timed out after {config.timeout_seconds}s"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise GraphMemoryAdapterError(
            f"graph-memory lane {config.name!r} failed with exit {completed.returncode}: {detail}"
        )
    try:
        artifact = GraphifyArtifact.model_validate_json(completed.stdout)
    except ValidationError as exc:
        raise GraphMemoryAdapterError(
            f"graph-memory lane {config.name!r} local output is invalid: {exc}"
        ) from exc
    if artifact.task_id != task.task_id:
        raise GraphMemoryAdapterError(
            f"graph-memory lane {config.name!r} local output task_id {artifact.task_id!r} "
            f"does not match benchmark task {task.task_id!r}"
        )
    _validate_artifact(config, artifact, source="local output")
    return _result_from_artifact(config, artifact, run_mode="local")
