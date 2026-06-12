from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from archex.cli.main import cli


def test_graph_export_writes_default_json_artifact(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["graph", "export", str(python_simple_repo)])

    assert result.exit_code == 0, result.output
    output_path = python_simple_repo / ".archex" / "archgraph.json"
    assert result.output.strip() == str(output_path)
    data = json.loads(output_path.read_text(encoding="utf-8"))
    node_ids = [node["id"] for node in data["nodes"]]
    edge_types = [edge["type"] for edge in data["edges"]]
    assert data["schema_version"]["value"] == "1.1.0"
    assert data["project"]["total_files"] >= 4
    assert node_ids == sorted(node_ids, key=lambda node_id: (node_id.split(":", 1)[0], node_id))
    assert "file:main.py" in node_ids
    assert "config:pyproject.toml" in node_ids
    assert "symbol:main.py::run#function" in node_ids
    assert "contains" in edge_types
    assert "imports" in edge_types


def test_graph_export_json_is_deterministic(python_simple_repo: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = runner.invoke(
        cli,
        ["graph", "export", str(python_simple_repo), "--output", str(first_path)],
    )
    second = runner.invoke(
        cli,
        ["graph", "export", str(python_simple_repo), "--output", str(second_path)],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")


def test_graph_export_markdown_can_write_explicit_output(
    python_simple_repo: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    output_path = tmp_path / "archgraph.md"

    result = runner.invoke(
        cli,
        [
            "graph",
            "export",
            str(python_simple_repo),
            "--format",
            "markdown",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    rendered = output_path.read_text(encoding="utf-8")
    assert rendered.startswith("# Architecture Graph:")
    assert "| Files |" in rendered
    assert "`file:main.py`" in rendered


def _export_graph_artifact(repo: Path, output: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["graph", "export", str(repo), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output


def test_graph_neighbors_reads_artifact_without_reindexing(
    python_simple_repo: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "archgraph.json"
    _export_graph_artifact(python_simple_repo, artifact)
    runner = CliRunner()

    with patch("archex.cli.graph_cmd.index_repository", side_effect=AssertionError("reindexed")):
        result = runner.invoke(
            cli,
            [
                "graph",
                "neighbors",
                "main.py",
                "--graph",
                str(artifact),
                "--format",
                "markdown",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "# Graph Neighbors:" in result.output
    assert "`main.py`" in result.output
    assert "imports" in result.output
    assert "extracted (1.00)" in result.output
    assert "parser chunk span main.py:8-15" in result.output


def test_graph_path_outputs_edge_confidence_json(python_simple_repo: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "archgraph.json"
    _export_graph_artifact(python_simple_repo, artifact)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "graph",
            "path",
            "main.py",
            "models.py",
            "--graph",
            str(artifact),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["found"] is True
    assert data["edges"][0]["type"] == "imports"
    assert data["edges"][0]["confidence"] == "extracted"
    assert data["edges"][0]["source"]["path"] == "main.py"
    assert data["edges"][0]["target"]["path"] == "models.py"
    assert data["edges"][0]["evidence"]


def test_graph_stats_and_hubs_render_markdown(python_simple_repo: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "archgraph.json"
    _export_graph_artifact(python_simple_repo, artifact)
    runner = CliRunner()

    stats = runner.invoke(
        cli,
        ["graph", "stats", "--graph", str(artifact), "--format", "markdown", "--hub-degree", "1"],
    )
    hubs = runner.invoke(
        cli,
        ["graph", "hubs", "--graph", str(artifact), "--format", "markdown", "--threshold", "1"],
    )

    assert stats.exit_code == 0, stats.output
    assert "# Graph Stats:" in stats.output
    assert "## Edge Types" in stats.output
    assert hubs.exit_code == 0, hubs.output
    assert "# Graph Hubs" in hubs.output
    assert "| Path | ID | Type | Degree |" in hubs.output
