"""Tests for pure explorer view-model builders."""

from __future__ import annotations

from pathlib import Path

from archex.explorer.loader import ExplorerData
from archex.explorer.viewmodel import (
    MAX_DIFF_FILE_ROWS,
    build_diff_view,
    build_health_view,
    build_manifest_view,
    build_module_map_view,
    build_neighborhood_view,
    build_receipt_view,
)
from archex.graph_artifact import (
    ArchGraph,
    GraphEdge,
    GraphEdgeType,
    GraphExportMetadata,
    GraphNode,
    GraphNodeType,
    GraphProject,
)
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


def _minimal_artifact() -> AnalysisArtifactV1:
    return AnalysisArtifactV1(
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
        parser_versions={"python": "tree-sitter-python"},
        excluded_counts={"unmapped": 2},
        unknown_counts={"symbol_kind": 1},
        diff=DiffAnalysis(base_ref="main"),
    )


def _small_graph() -> ArchGraph:
    return ArchGraph(
        project=GraphProject(name="widget", total_files=2),
        metadata=GraphExportMetadata(archex_version="0.22.0"),
        nodes=[
            GraphNode(id="file:a.py", type=GraphNodeType.FILE, label="a.py", module="pkg"),
            GraphNode(id="file:b.py", type=GraphNodeType.FILE, label="b.py", module="pkg"),
            GraphNode(
                id="symbol:a.py::f#function",
                type=GraphNodeType.SYMBOL,
                label="f",
                module="pkg",
            ),
        ],
        edges=[
            GraphEdge(source="file:a.py", target="file:b.py", type=GraphEdgeType.IMPORTS),
        ],
    )


def test_build_module_map_view_without_graph_is_unavailable() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=None)

    view = build_module_map_view(data)

    assert view.available is False
    assert view.modules == []


def test_build_module_map_view_aggregates_nodes_by_module() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_small_graph())

    view = build_module_map_view(data)

    assert view.available is True
    assert view.modules_total == 1
    row = view.modules[0]
    assert row.module == "pkg"
    assert row.node_count == 3
    assert row.file_count == 2
    assert row.symbol_count == 1


def test_build_neighborhood_view_without_graph_is_unavailable() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=None)

    view = build_neighborhood_view(data, "file:a.py")

    assert view.available is False
    assert view.error is not None


def test_build_neighborhood_view_without_query_is_empty_but_available() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_small_graph())

    view = build_neighborhood_view(data, None)

    assert view.available is True
    assert view.seed is None
    assert view.error is None


def test_build_neighborhood_view_finds_bounded_neighbors() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_small_graph())

    view = build_neighborhood_view(data, "file:a.py", depth=1, limit=25)

    assert view.error is None
    assert view.seed is not None
    assert view.seed.id == "file:a.py"
    assert {node.id for node in view.nodes} == {"file:a.py", "file:b.py"}


def test_build_neighborhood_view_reports_unresolvable_query_as_error() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_small_graph())

    view = build_neighborhood_view(data, "does-not-exist")

    assert view.available is True
    assert view.error is not None
    assert view.seed is None


def test_build_neighborhood_view_never_constructs_new_edges() -> None:
    """The view must only ever project `GraphQuery.neighbors`'s own bounded result."""
    graph = _small_graph()
    original_edge_count = len(graph.edges)
    data = ExplorerData(artifact=_minimal_artifact(), graph=graph)

    build_neighborhood_view(data, "file:a.py")

    assert len(graph.edges) == original_edge_count


def test_build_receipt_view_projects_evidence_and_counts() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=None)

    view = build_receipt_view(data)

    assert view.freshness == data.artifact.freshness.value
    assert view.excluded_counts == {"unmapped": 2}
    assert view.unknown_counts == {"symbol_kind": 1}


def test_build_health_view_projects_identity_fields() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=None)

    view = build_health_view(data)

    assert view.index_generation == "gen1"
    assert view.chunker_revision == "c1"
    assert view.parser_versions == {"python": "tree-sitter-python"}
