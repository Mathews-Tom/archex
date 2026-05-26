from __future__ import annotations

import json

import pytest

from archex.graph_artifact import (
    ArchGraph,
    GraphArtifactError,
    GraphEdge,
    GraphEdgeType,
    GraphExportMetadata,
    GraphLayer,
    GraphNode,
    GraphNodeType,
    GraphProject,
    GraphSchemaVersion,
    assert_supported_schema_version,
    file_node_id,
    interface_node_id,
    module_node_id,
    symbol_node_id,
)


def _metadata() -> GraphExportMetadata:
    return GraphExportMetadata(archex_version="0.6.2")


def test_empty_graph_serializes_with_schema_version() -> None:
    graph = ArchGraph(
        project=GraphProject(name="empty", total_files=0),
        metadata=_metadata(),
    )

    data = json.loads(graph.to_json())

    assert data["schema_version"]["value"] == "1.0.0"
    assert data["nodes"] == []
    assert data["edges"] == []
    assert data["project"]["name"] == "empty"


def test_graph_orders_nodes_edges_and_layer_members_deterministically() -> None:
    graph = ArchGraph(
        project=GraphProject(name="small"),
        metadata=_metadata(),
        nodes=[
            GraphNode(
                id=symbol_node_id("pkg/app.py", "run", "function"),
                type=GraphNodeType.SYMBOL,
                label="run",
            ),
            GraphNode(
                id=file_node_id("./pkg/app.py"),
                type=GraphNodeType.FILE,
                label="pkg/app.py",
            ),
            GraphNode(
                id=module_node_id("pkg"),
                type=GraphNodeType.MODULE,
                label="pkg",
            ),
        ],
        edges=[
            GraphEdge(
                source=module_node_id("pkg"),
                target=file_node_id("pkg/app.py"),
                type=GraphEdgeType.BELONGS_TO_MODULE,
            ),
            GraphEdge(
                source=file_node_id("pkg/app.py"),
                target=symbol_node_id("pkg/app.py", "run", "function"),
                type=GraphEdgeType.CONTAINS,
            ),
        ],
        layers=[
            GraphLayer(
                id="layer:pkg",
                name="pkg",
                node_ids=[
                    file_node_id("pkg/app.py"),
                    module_node_id("pkg"),
                    file_node_id("pkg/app.py"),
                ],
            )
        ],
    )

    data = json.loads(graph.to_json())

    assert [node["id"] for node in data["nodes"]] == [
        "file:pkg/app.py",
        "module:pkg",
        "symbol:pkg/app.py::run#function",
    ]
    assert [(edge["source"], edge["target"], edge["type"]) for edge in data["edges"]] == [
        ("file:pkg/app.py", "symbol:pkg/app.py::run#function", "contains"),
        ("module:pkg", "file:pkg/app.py", "belongs_to_module"),
    ]
    assert data["layers"][0]["node_ids"] == ["file:pkg/app.py", "module:pkg"]


def test_symbol_rich_ids_preserve_structural_characters() -> None:
    graph = ArchGraph(
        project=GraphProject(name="symbols"),
        metadata=_metadata(),
        nodes=[
            GraphNode(
                id=symbol_node_id("src/pkg/operators.py", "Vector[Index[str]].__call__", "method"),
                type=GraphNodeType.SYMBOL,
                label="Vector[Index[str]].__call__",
                path="src/pkg/operators.py",
                symbol_kind="method",
                line_start=12,
                line_end=24,
            ),
            GraphNode(
                id=interface_node_id("src/pkg/operators.py", "Vector[Index[str]].__call__"),
                type=GraphNodeType.INTERFACE,
                label="Vector[Index[str]].__call__",
                path="src/pkg/operators.py",
            ),
        ],
    )

    data = json.loads(graph.to_json())

    assert data["nodes"][0]["id"] == "interface:src/pkg/operators.py::Vector[Index[str]].__call__"
    assert data["nodes"][1]["id"] == (
        "symbol:src/pkg/operators.py::Vector[Index[str]].__call__#method"
    )


def test_node_id_must_match_node_type_prefix() -> None:
    with pytest.raises(ValueError, match="file node id must start"):
        GraphNode(id="symbol:pkg/app.py::run#function", type=GraphNodeType.FILE, label="app.py")


def test_rejects_unknown_major_schema_version() -> None:
    with pytest.raises(GraphArtifactError, match="Unsupported graph schema major version 2"):
        assert_supported_schema_version(GraphSchemaVersion(value="2.0.0"))
