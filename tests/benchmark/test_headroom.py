"""Tests for the Headroom-style compression-layer adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from archex.benchmark.headroom import (
    HeadroomAdapterError,
    HeadroomUnavailableError,
    apply_headroom_layer_local,
    load_headroom_artifact,
    run_compression_layer,
)
from archex.benchmark.models import (
    ComparisonLayerType,
    CompressionLayerConfig,
    CompressionLayerMode,
)

_COMPRESS_SCRIPT = """\
import sys

text = sys.stdin.read()
lines = [line for line in text.splitlines() if line.strip()]
kept = lines[: max(1, len(lines) // 2)]
sys.stdout.write("\\n".join(kept))
"""

_PASSTHROUGH_SCRIPT = "import sys; sys.stdout.write(sys.stdin.read())"

_SOURCE_TEXT = "\n".join(f"line {n} with content tokens here" for n in range(40))


def _local_config(
    script_path: Path, *, settings: dict[str, str] | None = None
) -> CompressionLayerConfig:
    return CompressionLayerConfig(
        name="headroom",
        version="0.4.1",
        command=sys.executable,
        args=[str(script_path)],
        compression_settings=settings or {"profile": "balanced"},
    )


def _write_artifact(artifact_dir: Path, *, version: str = "0.4.1") -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": "task_a",
        "headroom_version": version,
        "modes": {
            "archex_plus_headroom": {
                "source_lane": "archex",
                "source_passthrough": True,
                "bundle_tokens_uncompressed": 1000,
                "bundle_tokens_compressed": 1000,
                "command": "headroom compress --profile balanced",
                "compression_settings": {"profile": "balanced"},
            },
            "headroom_only_on_raw_context": {
                "source_lane": "raw_files",
                "source_passthrough": False,
                "bundle_tokens_uncompressed": 5000,
                "bundle_tokens_compressed": 2000,
                "command": "headroom compress --profile aggressive",
                "compression_settings": {"profile": "aggressive"},
            },
        },
    }
    path = artifact_dir / "task_a.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_apply_headroom_layer_local_records_provenance_and_metadata(tmp_path: Path) -> None:
    script = tmp_path / "compress.py"
    script.write_text(_COMPRESS_SCRIPT, encoding="utf-8")
    config = _local_config(script)

    result = apply_headroom_layer_local(
        config,
        CompressionLayerMode.HEADROOM_ONLY_ON_RAW_CONTEXT,
        task_id="task_a",
        source_lane="raw_files",
        source_text=_SOURCE_TEXT,
    )

    assert result.lane_label == "headroom_only_on_raw_context"
    assert result.layer_type is ComparisonLayerType.COMPRESSION
    assert result.source_lane == "raw_files"
    assert result.bundle_tokens_compressed < result.bundle_tokens_uncompressed
    assert result.bundle_compression_ratio < 1.0
    assert result.source_passthrough is False
    assert result.provenance["version"] == "0.4.1"
    assert result.provenance["run_mode"] == "local"
    assert result.provenance["compression_settings"] == "profile=balanced"
    assert script.name in result.provenance["command"]


def test_apply_headroom_layer_local_marks_passthrough_when_unchanged(tmp_path: Path) -> None:
    script = tmp_path / "passthrough.py"
    script.write_text(_PASSTHROUGH_SCRIPT, encoding="utf-8")
    config = _local_config(script)

    result = apply_headroom_layer_local(
        config,
        CompressionLayerMode.ARCHEX_PLUS_HEADROOM,
        task_id="task_a",
        source_lane="archex",
        source_text=_SOURCE_TEXT,
    )

    assert result.source_passthrough is True
    assert result.bundle_compression_ratio == 1.0
    assert result.bundle_tokens_compressed == result.bundle_tokens_uncompressed


def test_apply_headroom_layer_local_raises_when_unavailable(tmp_path: Path) -> None:
    config = CompressionLayerConfig(
        name="headroom",
        version="0.4.1",
        command="archex-headroom-missing-binary",
    )

    with pytest.raises(HeadroomUnavailableError, match="not found"):
        apply_headroom_layer_local(
            config,
            CompressionLayerMode.ARCHEX_PLUS_HEADROOM,
            task_id="task_a",
            source_lane="archex",
            source_text=_SOURCE_TEXT,
        )


def test_run_compression_layer_uses_artifact_mode_without_binary(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "headroom-artifacts"
    _write_artifact(artifact_dir)
    config = CompressionLayerConfig(
        name="headroom",
        version="0.4.1",
        command="archex-headroom-missing-binary",
        artifact_dir=str(artifact_dir),
    )

    result = run_compression_layer(
        config,
        CompressionLayerMode.HEADROOM_ONLY_ON_RAW_CONTEXT,
        task_id="task_a",
        source_lane="ignored-in-artifact-mode",
        source_text="ignored",
    )

    assert result.source_lane == "raw_files"
    assert result.bundle_tokens_uncompressed == 5000
    assert result.bundle_tokens_compressed == 2000
    assert result.bundle_compression_ratio == pytest.approx(0.4)  # pyright: ignore[reportUnknownMemberType]
    assert result.source_passthrough is False
    assert result.provenance["run_mode"] == "artifact"
    assert len(result.provenance["artifact_sha256"]) == 64
    assert result.provenance["command"] == "headroom compress --profile aggressive"


def test_load_headroom_artifact_rejects_version_mismatch(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "headroom-artifacts"
    _write_artifact(artifact_dir, version="9.9.9")
    config = CompressionLayerConfig(name="headroom", version="0.4.1", command="headroom")

    with pytest.raises(HeadroomAdapterError, match="does not match pinned version"):
        load_headroom_artifact(
            config,
            CompressionLayerMode.ARCHEX_PLUS_HEADROOM,
            task_id="task_a",
            artifact_dir=artifact_dir,
        )


def test_load_headroom_artifact_rejects_missing_mode(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "headroom-artifacts"
    path = _write_artifact(artifact_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["modes"]["archex_plus_headroom"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = CompressionLayerConfig(name="headroom", version="0.4.1", command="headroom")

    with pytest.raises(HeadroomAdapterError, match="missing mode 'archex_plus_headroom'"):
        load_headroom_artifact(
            config,
            CompressionLayerMode.ARCHEX_PLUS_HEADROOM,
            task_id="task_a",
            artifact_dir=artifact_dir,
        )


def test_load_headroom_artifact_missing_file(tmp_path: Path) -> None:
    config = CompressionLayerConfig(name="headroom", version="0.4.1", command="headroom")

    with pytest.raises(HeadroomAdapterError, match="artifact not found"):
        load_headroom_artifact(
            config,
            CompressionLayerMode.ARCHEX_PLUS_HEADROOM,
            task_id="task_a",
            artifact_dir=tmp_path / "missing",
        )
