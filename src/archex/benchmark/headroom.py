"""Headroom-style compression-layer adapter for the public comparison.

Headroom is a context-compression / context-management layer, not a retrieval
engine. This adapter applies a configured compression command to already-gathered
context — either an archex-selected bundle (``archex_plus_headroom``) or raw/broad
context (``headroom_only_on_raw_context``) — and records compression provenance.
It never reframes Headroom as a retrieval system; ``CompressionLayerResult``
fixes ``layer_type`` to ``compression`` and records which lane produced the source.

Two run modes:

* local — run the pinned binary on the source text and measure the result. Fails
  with :class:`HeadroomUnavailableError` when the binary is absent, so an unavailable
  tool is never silently skipped.
* artifact — import operator-produced compression outputs (JSON) with pinned
  provenance. Selected by setting ``compression_layers[].artifact_dir``. This is
  the only mode that runs without the binary present.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from shutil import which

from pydantic import Field, ValidationError

from archex.benchmark.models import (
    BenchmarkSpecModel,
    ComparisonLayerType,
    CompressionLayerConfig,
    CompressionLayerMode,
    CompressionLayerResult,
)
from archex.reporting import count_tokens


class HeadroomUnavailableError(RuntimeError):
    """Raised when a local compression-layer binary is required but not found."""


class HeadroomAdapterError(RuntimeError):
    """Raised when a compression-layer run or artifact cannot produce a result."""


class HeadroomArtifactMode(BenchmarkSpecModel):
    """One mode entry inside an operator-produced compression artifact."""

    source_lane: str
    source_passthrough: bool
    bundle_tokens_uncompressed: int = Field(ge=0)
    bundle_tokens_compressed: int = Field(ge=0)
    command: str
    compression_settings: dict[str, str] = {}


class HeadroomArtifact(BenchmarkSpecModel):
    """Schema for an operator-produced ``{task_id}.json`` compression artifact."""

    task_id: str
    headroom_version: str
    modes: dict[CompressionLayerMode, HeadroomArtifactMode]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _render_settings(settings: dict[str, str]) -> str:
    return ";".join(f"{key}={settings[key]}" for key in sorted(settings))


def _compression_ratio(uncompressed: int, compressed: int) -> float:
    return (compressed / uncompressed) if uncompressed else 1.0


def apply_headroom_layer_local(
    config: CompressionLayerConfig,
    mode: CompressionLayerMode,
    *,
    task_id: str,
    source_lane: str,
    source_text: str,
) -> CompressionLayerResult:
    """Run the pinned compression binary on ``source_text`` and measure the result.

    Source text is fed on stdin; compressed text is read from stdout. Raises
    :class:`HeadroomUnavailableError` when the binary is not on PATH so an unavailable
    tool is reported, never silently skipped.
    """
    if which(config.command) is None:
        raise HeadroomUnavailableError(
            f"compression layer {config.name!r} command {config.command!r} not found; "
            "set compression_layers[].artifact_dir to import operator artifacts instead"
        )

    command = [config.command, *config.args]
    try:
        completed = subprocess.run(
            command,
            input=source_text,
            env={**os.environ, **config.env} if config.env else None,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HeadroomAdapterError(
            f"compression layer {config.name!r} timed out after {config.timeout_seconds}s"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise HeadroomAdapterError(
            f"compression layer {config.name!r} failed with exit {completed.returncode}: {detail}"
        )

    uncompressed = count_tokens(source_text)
    compressed = count_tokens(completed.stdout)
    return CompressionLayerResult(
        task_id=task_id,
        lane_label=mode.value,
        layer_type=ComparisonLayerType.COMPRESSION,
        mode=mode,
        source_lane=source_lane,
        source_passthrough=compressed >= uncompressed,
        bundle_tokens_uncompressed=uncompressed,
        bundle_tokens_compressed=compressed,
        bundle_compression_ratio=_compression_ratio(uncompressed, compressed),
        provenance={
            "layer": config.name,
            "version": config.version,
            "command": " ".join(command),
            "mode": mode.value,
            "source_lane": source_lane,
            "compression_settings": _render_settings(config.compression_settings),
            "run_mode": "local",
        },
        timestamp=_now_iso(),
    )


def _parse_artifact(
    config: CompressionLayerConfig, artifact_path: Path
) -> tuple[HeadroomArtifact, str]:
    raw = artifact_path.read_bytes()
    try:
        artifact = HeadroomArtifact.model_validate_json(raw)
    except ValidationError as exc:
        raise HeadroomAdapterError(
            f"compression layer {config.name!r} artifact {artifact_path} is invalid: {exc}"
        ) from exc
    if artifact.headroom_version != config.version:
        raise HeadroomAdapterError(
            f"compression layer {config.name!r} artifact {artifact_path} version "
            f"{artifact.headroom_version!r} does not match pinned version {config.version!r}"
        )
    return artifact, hashlib.sha256(raw).hexdigest()


def _result_from_entry(
    config: CompressionLayerConfig,
    mode: CompressionLayerMode,
    entry: HeadroomArtifactMode,
    *,
    task_id: str,
    artifact_path: Path,
    digest: str,
) -> CompressionLayerResult:
    return CompressionLayerResult(
        task_id=task_id,
        lane_label=mode.value,
        layer_type=ComparisonLayerType.COMPRESSION,
        mode=mode,
        source_lane=entry.source_lane,
        source_passthrough=entry.source_passthrough,
        bundle_tokens_uncompressed=entry.bundle_tokens_uncompressed,
        bundle_tokens_compressed=entry.bundle_tokens_compressed,
        bundle_compression_ratio=_compression_ratio(
            entry.bundle_tokens_uncompressed, entry.bundle_tokens_compressed
        ),
        provenance={
            "layer": config.name,
            "version": config.version,
            "command": entry.command,
            "mode": mode.value,
            "source_lane": entry.source_lane,
            "compression_settings": _render_settings(entry.compression_settings),
            "run_mode": "artifact",
            "artifact_path": str(artifact_path),
            "artifact_sha256": digest,
        },
        timestamp=_now_iso(),
    )


def load_headroom_artifact(
    config: CompressionLayerConfig,
    mode: CompressionLayerMode,
    *,
    task_id: str,
    artifact_dir: Path,
) -> CompressionLayerResult:
    """Import an operator-produced compression artifact for one task/mode.

    The artifact's ``headroom_version`` must match the manifest-pinned version so
    imported numbers carry pinned provenance. Records the artifact path and its
    SHA-256 so the source of every imported number is traceable.
    """
    artifact_path = artifact_dir / f"{task_id}.json"
    if not artifact_path.is_file():
        raise HeadroomAdapterError(
            f"compression layer {config.name!r} artifact not found: {artifact_path}"
        )
    artifact, digest = _parse_artifact(config, artifact_path)
    if mode not in artifact.modes:
        raise HeadroomAdapterError(
            f"compression layer {config.name!r} artifact {artifact_path} is missing mode "
            f"{mode.value!r}"
        )
    return _result_from_entry(
        config,
        mode,
        artifact.modes[mode],
        task_id=task_id,
        artifact_path=artifact_path,
        digest=digest,
    )


def load_headroom_results(
    config: CompressionLayerConfig, task_ids: list[str]
) -> list[CompressionLayerResult]:
    """Import every available compression result for ``task_ids`` (tolerant batch).

    Requires artifact mode (``config.artifact_dir`` set). Tasks with no artifact
    file are skipped; for present files every mode the artifact declares that is
    also enabled on the layer is imported. Corrupt or version-mismatched artifacts
    still raise so a bad import is never silently dropped.
    """
    if config.artifact_dir is None:
        raise HeadroomAdapterError(
            f"compression layer {config.name!r} requires artifact_dir for batch import"
        )
    artifact_dir = Path(config.artifact_dir)
    enabled = set(config.modes)
    results: list[CompressionLayerResult] = []
    for task_id in task_ids:
        artifact_path = artifact_dir / f"{task_id}.json"
        if not artifact_path.is_file():
            continue
        artifact, digest = _parse_artifact(config, artifact_path)
        for mode, entry in artifact.modes.items():
            if mode in enabled:
                results.append(
                    _result_from_entry(
                        config,
                        mode,
                        entry,
                        task_id=task_id,
                        artifact_path=artifact_path,
                        digest=digest,
                    )
                )
    return results


def run_compression_layer(
    config: CompressionLayerConfig,
    mode: CompressionLayerMode,
    *,
    task_id: str,
    source_lane: str,
    source_text: str,
) -> CompressionLayerResult:
    """Produce a compression-layer result, choosing artifact or local mode.

    Artifact mode is used when ``config.artifact_dir`` is set; otherwise the local
    binary is required. ``source_text`` is ignored in artifact mode.
    """
    if config.artifact_dir is not None:
        return load_headroom_artifact(
            config, mode, task_id=task_id, artifact_dir=Path(config.artifact_dir)
        )
    return apply_headroom_layer_local(
        config, mode, task_id=task_id, source_lane=source_lane, source_text=source_text
    )
