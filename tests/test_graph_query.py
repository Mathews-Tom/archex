from __future__ import annotations

from pathlib import Path

import pytest

from archex.graph_artifact import (
    ArchGraph,
    GraphEdge,
    GraphEdgeType,
    GraphExportMetadata,
    GraphNode,
    GraphNodeType,
    GraphProject,
    file_node_id,
    symbol_node_id,
)
from archex.graph_query import GraphQuery, GraphQueryError
from archex.index.store import IndexStore
from archex.models import CodeChunk, Edge, EdgeConfidence, EdgeKind, SymbolKind


def _graph() -> ArchGraph:
    app = file_node_id("pkg/app.py")
    models = file_node_id("pkg/models.py")
    db = file_node_id("pkg/db.py")
    util = file_node_id("pkg/util.py")
    hub = file_node_id("pkg/hub.py")
    app_symbol = symbol_node_id("pkg/app.py", "run", "function")
    return ArchGraph(
        project=GraphProject(name="query", languages={"python": 5}, total_files=5),
        metadata=GraphExportMetadata(archex_version="0.8.0"),
        nodes=[
            GraphNode(id=app, type=GraphNodeType.FILE, label="app.py", path="pkg/app.py"),
            GraphNode(id=models, type=GraphNodeType.FILE, label="models.py", path="pkg/models.py"),
            GraphNode(id=db, type=GraphNodeType.FILE, label="db.py", path="pkg/db.py"),
            GraphNode(id=util, type=GraphNodeType.FILE, label="util.py", path="pkg/util.py"),
            GraphNode(id=hub, type=GraphNodeType.FILE, label="hub.py", path="pkg/hub.py"),
            GraphNode(
                id=app_symbol,
                type=GraphNodeType.SYMBOL,
                label="run",
                path="pkg/app.py",
                symbol_kind="function",
                line_start=1,
                line_end=3,
            ),
        ],
        edges=[
            GraphEdge(
                source=app,
                target=models,
                type=GraphEdgeType.IMPORTS,
                location="pkg/app.py:1",
                confidence=EdgeConfidence.HEURISTIC,
                confidence_score=0.75,
                evidence=["fallback resolution"],
            ),
            GraphEdge(source=models, target=db, type=GraphEdgeType.IMPORTS),
            GraphEdge(source=app, target=app_symbol, type=GraphEdgeType.CONTAINS),
            GraphEdge(source=hub, target=app, type=GraphEdgeType.IMPORTS),
            GraphEdge(source=hub, target=models, type=GraphEdgeType.IMPORTS),
            GraphEdge(source=hub, target=db, type=GraphEdgeType.IMPORTS),
            GraphEdge(source=hub, target=util, type=GraphEdgeType.IMPORTS),
        ],
    )


def test_exact_path_lookup_wins_over_fuzzy_label_matches() -> None:
    query = GraphQuery(_graph())

    result = query.lookup("pkg/app.py")

    assert result.match_kind == "path"
    assert [node.id for node in result.matches] == [file_node_id("pkg/app.py")]


def test_duplicate_node_ids_are_deduped_deterministically() -> None:
    node_id = file_node_id("pkg/app.py")
    graph = ArchGraph(
        project=GraphProject(name="duplicates", total_files=1),
        metadata=GraphExportMetadata(archex_version="0.8.0"),
        nodes=[
            GraphNode(
                id=node_id,
                type=GraphNodeType.FILE,
                label="app.py",
                path="pkg/app.py",
                line_start=10,
            ),
            GraphNode(
                id=node_id,
                type=GraphNodeType.FILE,
                label="app.py",
                path="pkg/app.py",
                line_start=1,
            ),
        ],
    )

    query = GraphQuery(graph)

    result = query.lookup("pkg/app.py")
    assert result.matches[0].line_start == 1


def test_neighbors_include_confidence_and_evidence() -> None:
    query = GraphQuery(_graph())

    result = query.neighbors("pkg/app.py", direction="out", limit=1)

    assert result.truncated is True
    assert result.omitted_edges == 1
    assert result.edges[0].type == "imports"
    assert result.edges[0].confidence == "heuristic"
    assert result.edges[0].confidence_score == 0.75
    assert result.edges[0].evidence == ["fallback resolution"]
    assert result.edges[0].source.path == "pkg/app.py"
    assert result.edges[0].target.path == "pkg/models.py"


def test_shortest_path_avoids_non_seed_hubs() -> None:
    query = GraphQuery(_graph(), hub_degree=4)

    result = query.shortest_path("pkg/app.py", "pkg/db.py", direction="both")

    assert result.found is True
    assert [node.path for node in result.nodes] == ["pkg/app.py", "pkg/models.py", "pkg/db.py"]
    assert all(edge.source.path != "pkg/hub.py" for edge in result.edges)


def test_hubs_are_stable_and_truncated() -> None:
    query = GraphQuery(_graph(), hub_degree=2)

    result = query.hubs(limit=2)

    assert result.truncated is True
    assert result.omitted == 2
    assert [hub.path for hub in result.hubs] == ["pkg/hub.py", "pkg/app.py"]


def test_ambiguous_lookup_fails_loudly() -> None:
    query = GraphQuery(_graph())

    with pytest.raises(GraphQueryError, match="ambiguous"):
        query.neighbors("pkg", limit=1)


def test_loads_from_exported_artifact_once(tmp_path: Path) -> None:
    artifact = tmp_path / "archgraph.json"
    artifact.write_text(_graph().to_json(), encoding="utf-8")

    query = GraphQuery.from_artifact(artifact)

    assert query.lookup("pkg/app.py").matches[0].path == "pkg/app.py"


def test_loads_from_persisted_index(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with IndexStore(db_path) as store:
        store.insert_chunks(
            [
                CodeChunk(
                    id="pkg/app.py::run#function",
                    content="def run():\n    pass",
                    file_path="pkg/app.py",
                    start_line=1,
                    end_line=2,
                    symbol_name="run",
                    symbol_kind=SymbolKind.FUNCTION,
                    language="python",
                    symbol_id="pkg/app.py::run#function",
                    qualified_name="run",
                ),
                CodeChunk(
                    id="pkg/models.py::Model#class",
                    content="class Model: pass",
                    file_path="pkg/models.py",
                    start_line=1,
                    end_line=1,
                    symbol_name="Model",
                    symbol_kind=SymbolKind.CLASS,
                    language="python",
                    symbol_id="pkg/models.py::Model#class",
                    qualified_name="Model",
                ),
            ]
        )
        store.insert_edges(
            [
                Edge(
                    source="pkg/app.py",
                    target="pkg/models.py",
                    kind=EdgeKind.IMPORTS,
                    location="pkg/app.py:1",
                    confidence=EdgeConfidence.EXTRACTED,
                    evidence=["import pkg.models"],
                )
            ]
        )

    query = GraphQuery.from_index(db_path)

    result = query.neighbors("pkg/app.py", direction="out")
    assert result.edges[0].target.path == "pkg/models.py"
    assert result.edges[0].evidence == ["import pkg.models"]
