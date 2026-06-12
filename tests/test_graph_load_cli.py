from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli


def _export_graph(runner: CliRunner, repo: Path, output_path: Path) -> None:
    result = runner.invoke(
        cli,
        ["graph", "export", str(repo), "--output", str(output_path)],
    )
    assert result.exit_code == 0, result.output


def test_graph_inspect_reads_exported_artifact(python_simple_repo: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    graph_path = tmp_path / "archgraph.json"
    _export_graph(runner, python_simple_repo, graph_path)

    result = runner.invoke(cli, ["graph", "inspect", str(graph_path), "--format", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["schema_version"] == "1.1.0"
    assert data["nodes"] > 0
    assert data["edges"] > 0


def test_explain_reads_graph_without_source_index(python_simple_repo: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    graph_path = tmp_path / "archgraph.json"
    _export_graph(runner, python_simple_repo, graph_path)

    result = runner.invoke(
        cli,
        ["explain", "--graph", str(graph_path), "main.py", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["target_type"] == "file"
    assert data["files"] == ["main.py"]
    assert "utils.py" in data["imports"]


def test_onboard_reads_graph_without_source_index(python_simple_repo: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    graph_path = tmp_path / "archgraph.json"
    _export_graph(runner, python_simple_repo, graph_path)

    result = runner.invoke(cli, ["onboard", "--graph", str(graph_path), "--max-files", "3"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("# Onboarding:")
    assert "## Configuration Surface" in result.output


def test_graph_load_rejects_unknown_major_version(tmp_path: Path) -> None:
    runner = CliRunner()
    graph_path = tmp_path / "archgraph.json"
    graph_path.write_text(
        json.dumps(
            {
                "schema_version": {"value": "2.0.0"},
                "project": {"name": "bad"},
                "metadata": {"archex_version": "0.6.2"},
                "nodes": [],
                "edges": [],
                "layers": [],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(cli, ["graph", "inspect", str(graph_path)])

    assert result.exit_code != 0
    assert "Unsupported graph schema major version 2" in result.output


def test_graph_load_rejects_malformed_artifact(tmp_path: Path) -> None:
    runner = CliRunner()
    graph_path = tmp_path / "archgraph.json"
    graph_path.write_text('{"schema_version": {"value": "1.0.0"}}', encoding="utf-8")

    result = runner.invoke(cli, ["graph", "inspect", str(graph_path)])

    assert result.exit_code != 0
    assert "Malformed graph artifact" in result.output
