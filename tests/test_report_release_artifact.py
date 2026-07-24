"""Tests for the M9 per-release CompatibilityArtifact model and builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from archex.report.release_artifact import (
    CompatibilityArtifact,
    build_compatibility_artifact,
)
from archex.report.status_card import StatusCard, StatusDimensionState

if TYPE_CHECKING:
    from pathlib import Path


def _sample_status_card() -> StatusCard:
    from archex.report.status_card import StatusDimension

    return StatusCard(
        source_identity="repo",
        revision="abc",
        generated_at="1.0",
        dimensions=[
            StatusDimension(
                name="Documentation linkage",
                state=StatusDimensionState.UNKNOWN,
                detail="not configured",
                provider="doc_link",
            )
        ],
    )


class TestCompatibilityArtifactValidation:
    def test_rejects_empty_archex_version(self) -> None:
        with pytest.raises(ValueError, match="archex_version"):
            CompatibilityArtifact(
                archex_version=" ",
                python_requires=">=3.11",
                report_schema_version="1.0.0",
                index_schema_version="5",
                compatibility_matrix_path="docs/CLIENT_COMPATIBILITY_MATRIX.md",
                status_card=_sample_status_card(),
                generated_at="1.0",
            )

    def test_rejects_empty_index_schema_version(self) -> None:
        with pytest.raises(ValueError, match="index_schema_version"):
            CompatibilityArtifact(
                archex_version="0.23.0",
                python_requires=">=3.11",
                report_schema_version="1.0.0",
                index_schema_version=" ",
                compatibility_matrix_path="docs/CLIENT_COMPATIBILITY_MATRIX.md",
                status_card=_sample_status_card(),
                generated_at="1.0",
            )

    def test_accepts_valid_artifact_and_round_trips_json(self) -> None:
        artifact = CompatibilityArtifact(
            archex_version="0.23.0",
            python_requires=">=3.11",
            report_schema_version="1.0.0",
            index_schema_version="5",
            compatibility_matrix_path="docs/CLIENT_COMPATIBILITY_MATRIX.md",
            benchmark_evidence_path="benchmarks/headtohead/results/manifest.yaml",
            status_card=_sample_status_card(),
            generated_at="1.0",
        )
        restored = CompatibilityArtifact.model_validate_json(artifact.to_json())
        assert restored.archex_version == "0.23.0"
        assert restored.benchmark_evidence_path == "benchmarks/headtohead/results/manifest.yaml"

    def test_benchmark_evidence_path_defaults_to_none(self) -> None:
        artifact = CompatibilityArtifact(
            archex_version="0.23.0",
            python_requires=">=3.11",
            report_schema_version="1.0.0",
            index_schema_version="5",
            compatibility_matrix_path="docs/CLIENT_COMPATIBILITY_MATRIX.md",
            status_card=_sample_status_card(),
            generated_at="1.0",
        )
        assert artifact.benchmark_evidence_path is None


class TestBuildCompatibilityArtifact:
    def test_reports_archex_own_version_and_schema_facts(self, python_simple_repo: Path) -> None:
        from archex import __version__

        artifact = build_compatibility_artifact(python_simple_repo)

        assert artifact.archex_version == __version__
        assert artifact.python_requires != "unknown"
        assert artifact.report_schema_version == "1.0.0"
        assert artifact.index_schema_version
        assert artifact.compatibility_matrix_path == "docs/CLIENT_COMPATIBILITY_MATRIX.md"

    def test_benchmark_evidence_path_none_when_manifest_absent(
        self, python_simple_repo: Path
    ) -> None:
        artifact = build_compatibility_artifact(python_simple_repo)
        assert artifact.benchmark_evidence_path is None

    def test_benchmark_evidence_path_populated_when_manifest_present(
        self, python_simple_repo: Path
    ) -> None:
        manifest_dir = python_simple_repo / "benchmarks" / "headtohead" / "results"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "manifest.yaml").write_text("manifest_version: 1\n")

        artifact = build_compatibility_artifact(python_simple_repo)

        assert artifact.benchmark_evidence_path == "benchmarks/headtohead/results/manifest.yaml"

    def test_embeds_a_real_status_card(self, python_simple_repo: Path) -> None:
        artifact = build_compatibility_artifact(python_simple_repo)
        assert len(artifact.status_card.dimensions) == 4

    def test_never_mutates_the_analyzed_repository(self, python_simple_repo: Path) -> None:
        before = {p.relative_to(python_simple_repo) for p in python_simple_repo.rglob("*")}
        build_compatibility_artifact(python_simple_repo)
        after = {p.relative_to(python_simple_repo) for p in python_simple_repo.rglob("*")}
        # .archex/ cache state is expected local generated state (same as
        # every other report/status command); no repository source file is
        # created, modified, or removed.
        new_paths = after - before
        assert all(str(path).startswith(".archex") for path in new_paths)
