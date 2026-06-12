from __future__ import annotations

import json
from pathlib import Path

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
