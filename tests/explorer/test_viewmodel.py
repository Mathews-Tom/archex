"""Tests for pure explorer view-model builders."""

from __future__ import annotations

from pathlib import Path

from archex.explorer.loader import ExplorerData
from archex.explorer.viewmodel import (
    MAX_DIFF_FILE_ROWS,
    MAX_NODE_SEARCH_ROWS,
    NeighborhoodView,
    build_diff_view,
    build_health_view,
    build_manifest_view,
    build_module_map_view,
    build_neighborhood_view,
    build_node_search_view,
    build_receipt_view,
    normalize_edge_types,
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


_SYMBOL_ID = "symbol:a.py::f#function"


def _chain_graph() -> ArchGraph:
    """`z -> a -> b -> c` plus `a -> f` and `b -> f`, for orientation/filter coverage."""
    return ArchGraph(
        project=GraphProject(name="widget", total_files=4),
        metadata=GraphExportMetadata(archex_version="0.22.0"),
        nodes=[
            GraphNode(id="file:a.py", type=GraphNodeType.FILE, label="a.py", module="pkg"),
            GraphNode(id="file:b.py", type=GraphNodeType.FILE, label="b.py", module="pkg"),
            GraphNode(id="file:c.py", type=GraphNodeType.FILE, label="c.py", module="pkg"),
            GraphNode(id="file:z.py", type=GraphNodeType.FILE, label="z.py", module="pkg"),
            GraphNode(id=_SYMBOL_ID, type=GraphNodeType.SYMBOL, label="f", module="pkg"),
        ],
        edges=[
            GraphEdge(source="file:z.py", target="file:a.py", type=GraphEdgeType.IMPORTS),
            GraphEdge(source="file:a.py", target="file:b.py", type=GraphEdgeType.IMPORTS),
            GraphEdge(source="file:b.py", target="file:c.py", type=GraphEdgeType.IMPORTS),
            GraphEdge(source="file:a.py", target=_SYMBOL_ID, type=GraphEdgeType.CONTAINS),
            GraphEdge(source="file:b.py", target=_SYMBOL_ID, type=GraphEdgeType.EXPOSES),
        ],
    )


def _orientations(view: NeighborhoodView) -> dict[tuple[str, str], str]:
    return {(edge.source_id, edge.target_id): edge.orientation for edge in view.edges}


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


def test_neighborhood_edges_are_oriented_relative_to_the_seed() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_chain_graph())

    view = build_neighborhood_view(data, "file:a.py", depth=2, limit=25)

    orientations = _orientations(view)
    # `z` imports the seed, so the seed is the dependency: inbound.
    assert orientations[("file:z.py", "file:a.py")] == "in"
    # The seed imports `b`, and `b` imports `c` further out: both outbound.
    assert orientations[("file:a.py", "file:b.py")] == "out"
    assert orientations[("file:b.py", "file:c.py")] == "out"
    # `b` and `f` are both one hop from the seed, so neither is closer to it.
    assert orientations[("file:b.py", _SYMBOL_ID)] == "lateral"


def test_neighborhood_orientation_follows_the_requested_direction() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_chain_graph())

    inbound = build_neighborhood_view(data, "file:a.py", direction="in", depth=1)
    outbound = build_neighborhood_view(data, "file:a.py", direction="out", depth=1)

    assert {edge.orientation for edge in inbound.edges} == {"in"}
    assert {edge.orientation for edge in outbound.edges} == {"out"}


def test_neighborhood_reports_edge_type_facets_over_the_unfiltered_traversal() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_chain_graph())

    view = build_neighborhood_view(data, "file:a.py", depth=2, limit=25)

    assert {facet.type: facet.count for facet in view.edge_type_facets} == {
        "imports": 3,
        "contains": 1,
        "exposes": 1,
    }
    assert all(facet.selected is False for facet in view.edge_type_facets)


def test_neighborhood_edge_type_filter_hides_other_types_and_narrows_nodes() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_chain_graph())

    view = build_neighborhood_view(data, "file:a.py", depth=2, limit=25, edge_types=["contains"])

    assert {edge.type for edge in view.edges} == {"contains"}
    assert view.filtered_edges == 4
    assert view.selected_edge_types == ["contains"]
    # A node the hidden edges were the only route to must not stay in the table.
    assert {node.id for node in view.nodes} == {"file:a.py", _SYMBOL_ID}
    # Facets still describe the whole neighborhood, so the filter can be widened again.
    assert {facet.type for facet in view.edge_type_facets} == {"imports", "contains", "exposes"}
    assert [facet.type for facet in view.edge_type_facets if facet.selected] == ["contains"]


def test_neighborhood_unrecognized_edge_type_is_ignored_rather_than_hiding_everything() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_chain_graph())

    view = build_neighborhood_view(
        data, "file:a.py", depth=2, limit=25, edge_types=["not-an-edge-type"]
    )

    assert view.selected_edge_types == []
    assert view.filtered_edges == 0
    assert len(view.edges) == 5


def test_normalize_edge_types_keeps_only_known_types_sorted() -> None:
    assert normalize_edge_types(["exposes", "imports", "bogus", "imports"]) == [
        "exposes",
        "imports",
    ]
    assert normalize_edge_types(None) == []


def test_build_node_search_view_without_graph_is_unavailable() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=None)

    view = build_node_search_view(data, "a.py")

    assert view.available is False
    assert view.matches == []


def test_build_node_search_view_without_query_returns_no_matches() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_chain_graph())

    view = build_node_search_view(data, None)

    assert view.available is True
    assert view.matches == []
    assert view.match_kind is None


def test_build_node_search_view_resolves_exact_and_fuzzy_matches() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_chain_graph())

    exact = build_node_search_view(data, "file:a.py")
    fuzzy = build_node_search_view(data, "a.p")

    assert [match.id for match in exact.matches] == ["file:a.py"]
    assert exact.match_kind == "id"
    assert [match.id for match in fuzzy.matches] == ["file:a.py", _SYMBOL_ID]
    assert fuzzy.match_kind == "fuzzy"


def test_build_node_search_view_reports_truncation_at_its_limit() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_chain_graph())

    view = build_node_search_view(data, ".py", limit=2)

    assert len(view.matches) == 2
    assert view.truncated is True
    assert view.omitted == 3


def test_build_node_search_view_caps_an_oversized_limit() -> None:
    data = ExplorerData(artifact=_minimal_artifact(), graph=_chain_graph())

    view = build_node_search_view(data, ".py", limit=MAX_NODE_SEARCH_ROWS * 10)

    assert view.limit == MAX_NODE_SEARCH_ROWS


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
