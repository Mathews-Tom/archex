"""Tests for artifact-only explorer data loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archex.explorer import loader
from archex.explorer.loader import ExplorerDataError, load_explorer_data
from archex.graph_artifact import (
    ArchGraph,
    GraphEdge,
    GraphEdgeType,
    GraphExportMetadata,
    GraphNode,
    GraphNodeType,
    GraphProject,
)


def _artifact_json(source_revision: str = "deadbeef") -> dict[str, object]:
    return {
        "schema_version": {"value": "1.0.0"},
        "archex_version": "0.22.0",
        "generated_at": "2026-07-24T00:00:00Z",
        "source_identity": "acme/widget",
        "source_root": "/repo",
        "source_revision": source_revision,
        "working_tree_fingerprint": "fp",
        "index_generation": "gen1",
        "index_schema_version": "1",
        "chunker_revision": "c1",
        "config_fingerprint": "cfg1",
        "diff": {"base_ref": "main"},
    }


def test_load_explorer_data_reads_artifact_only(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(_artifact_json()))

    data = load_explorer_data(artifact_path)

    assert data.artifact.source_identity == "acme/widget"
    assert data.graph is None


def test_load_explorer_data_accepts_optional_graph(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(_artifact_json()))
    graph = ArchGraph(
        project=GraphProject(name="widget", total_files=1),
        metadata=GraphExportMetadata(archex_version="0.22.0"),
    )
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(graph.to_json())

    data = load_explorer_data(artifact_path, graph_path)

    assert data.graph is not None
    assert data.graph.project.name == "widget"


def test_load_explorer_data_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(ExplorerDataError):
        load_explorer_data(tmp_path / "missing.json")


def test_load_explorer_data_rejects_malformed_artifact_json(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{not valid json")

    with pytest.raises(ExplorerDataError):
        load_explorer_data(artifact_path)


def test_load_explorer_data_rejects_artifact_missing_required_fields(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps({"schema_version": {"value": "1.0.0"}}))

    with pytest.raises(ExplorerDataError):
        load_explorer_data(artifact_path)


def test_load_explorer_data_rejects_unsupported_schema_major(tmp_path: Path) -> None:
    payload = _artifact_json()
    payload["schema_version"] = {"value": "2.0.0"}
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(payload))

    with pytest.raises(ExplorerDataError):
        load_explorer_data(artifact_path)


def test_load_explorer_data_rejects_malformed_graph(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(_artifact_json()))
    graph_path = tmp_path / "graph.json"
    graph_path.write_text("{not valid json")

    with pytest.raises(ExplorerDataError):
        load_explorer_data(artifact_path, graph_path)


def test_oversized_report_artifact_is_rejected_before_it_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(_artifact_json()))
    monkeypatch.setattr(loader, "MAX_ARTIFACT_BYTES", 10)

    def _fail(_path: Path) -> object:
        raise AssertionError("an oversized artifact must not be parsed")

    monkeypatch.setattr(loader, "load_analysis_artifact", _fail)

    with pytest.raises(ExplorerDataError, match="above the explorer's 10-byte limit"):
        load_explorer_data(artifact_path)


def test_oversized_graph_artifact_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(_artifact_json()))
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        ArchGraph(
            project=GraphProject(name="widget"),
            metadata=GraphExportMetadata(archex_version="0.22.0"),
        ).model_dump_json()
    )
    monkeypatch.setattr(loader, "MAX_GRAPH_BYTES", 5)

    with pytest.raises(ExplorerDataError, match="above the explorer's 5-byte limit"):
        load_explorer_data(artifact_path, graph_path)


def test_graph_above_the_node_cap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(_artifact_json()))
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        ArchGraph(
            project=GraphProject(name="widget"),
            metadata=GraphExportMetadata(archex_version="0.22.0"),
            nodes=[
                GraphNode(id=f"file:f{index}.py", type=GraphNodeType.FILE, label=f"f{index}.py")
                for index in range(3)
            ],
        ).model_dump_json()
    )
    monkeypatch.setattr(loader, "MAX_GRAPH_NODES", 2)

    with pytest.raises(ExplorerDataError, match="above the explorer's 2-node limit"):
        load_explorer_data(artifact_path, graph_path)


def test_graph_above_the_edge_cap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(_artifact_json()))
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        ArchGraph(
            project=GraphProject(name="widget"),
            metadata=GraphExportMetadata(archex_version="0.22.0"),
            nodes=[
                GraphNode(id=f"file:f{index}.py", type=GraphNodeType.FILE, label=f"f{index}.py")
                for index in range(3)
            ],
            edges=[
                GraphEdge(
                    source="file:f0.py", target=f"file:f{index}.py", type=GraphEdgeType.IMPORTS
                )
                for index in (1, 2)
            ],
        ).model_dump_json()
    )
    monkeypatch.setattr(loader, "MAX_GRAPH_EDGES", 1)

    with pytest.raises(ExplorerDataError, match="above the explorer's 1-edge limit"):
        load_explorer_data(artifact_path, graph_path)


def test_missing_artifact_reports_a_read_failure_not_a_size_failure(tmp_path: Path) -> None:
    with pytest.raises(ExplorerDataError, match="Failed to read report artifact"):
        load_explorer_data(tmp_path / "absent.json")
