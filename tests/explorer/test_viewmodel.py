"""Tests for pure explorer view-model builders."""

from __future__ import annotations

from pathlib import Path

from archex.explorer.loader import ExplorerData
from archex.explorer.viewmodel import MAX_DIFF_FILE_ROWS, build_diff_view, build_manifest_view
from archex.report.artifact import (
    AnalysisArtifactV1,
    DiffAnalysis,
    DiffFileChange,
    ReportSchemaVersion,
    build_analysis_artifact,
)


def _edit_hub(repo: Path) -> None:
    hub = repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))


def test_build_manifest_view_projects_provenance_and_receipt_fields(
    impact_diff_repo: Path,
) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    data = ExplorerData(artifact=artifact, graph=None)

    manifest = build_manifest_view(data)

    assert manifest.source_identity == artifact.source_identity
    assert manifest.freshness == artifact.freshness.value
    assert manifest.completeness == artifact.completeness.value
    assert manifest.confidence == artifact.confidence.value
    assert manifest.redaction_mode == artifact.redaction_mode.value
    assert manifest.has_graph is False
    assert manifest.evidence_count == len(artifact.evidence_locations)


def test_build_diff_view_projects_changed_files_and_symbol_candidates(
    impact_diff_repo: Path,
) -> None:
    _edit_hub(impact_diff_repo)
    artifact = build_analysis_artifact(impact_diff_repo, base_ref="HEAD")
    data = ExplorerData(artifact=artifact, graph=None)

    view = build_diff_view(data)

    assert view.base_ref == "HEAD"
    assert view.changed_files_total == artifact.diff.changed_files_total
    assert [row.path for row in view.changed_files] == [
        change.path for change in artifact.diff.changed_files
    ]
    assert view.symbol_candidates_total == artifact.diff.symbol_candidates_total
    assert view.risk_level == artifact.diff.risk_level.value


def test_build_diff_view_bounds_changed_files_and_reports_total() -> None:

    changed = [
        DiffFileChange(path=f"file_{i}.py", status="M", handle=f"file:file_{i}.py")
        for i in range(MAX_DIFF_FILE_ROWS + 5)
    ]
    diff = DiffAnalysis(
        base_ref="main",
        changed_files=changed,
        changed_files_total=len(changed),
    )
    artifact = AnalysisArtifactV1(
        schema_version=ReportSchemaVersion(),
        generated_at="2026-07-24T00:00:00Z",
        source_identity="acme/widget",
        source_root="/repo",
        source_revision="deadbeef",
        working_tree_fingerprint="fp",
        index_generation="gen1",
        index_schema_version="1",
        chunker_revision="c1",
        config_fingerprint="cfg1",
        diff=diff,
    )
    data = ExplorerData(artifact=artifact, graph=None)

    view = build_diff_view(data)

    assert len(view.changed_files) == MAX_DIFF_FILE_ROWS
    assert view.changed_files_total == len(changed)
