from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from archex.cli.main import cli

if TYPE_CHECKING:
    from pathlib import Path


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "archex, version 0.2.0" in result.output


def test_help_contains_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    output = result.output
    assert "analyze" in output
    assert "query" in output
    assert "compare" in output
    assert "cache" in output


def test_analyze_local_json(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", str(python_simple_repo), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "repo" in data
    assert "stats" in data
    assert "interface_surface" in data


def test_analyze_local_markdown(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", str(python_simple_repo), "--format", "markdown"])
    assert result.exit_code == 0, result.output
    output = result.output
    assert "# Architecture Profile" in output
    assert "## Stats" in output


def test_analyze_error_handling() -> None:
    from unittest.mock import patch

    from archex.exceptions import ArchexError

    runner = CliRunner()
    with patch("archex.cli.analyze_cmd.analyze", side_effect=ArchexError("Test error")):
        result = runner.invoke(cli, ["analyze", "/fake/repo"])
    assert result.exit_code != 0
    assert "Test error" in result.output


def test_query_error_handling(python_simple_repo: Path) -> None:
    from unittest.mock import patch

    from archex.exceptions import ArchexError

    runner = CliRunner()
    with patch("archex.cli.query_cmd.query", side_effect=ArchexError("Query failed")):
        result = runner.invoke(cli, ["query", str(python_simple_repo), "test question"])
    assert result.exit_code != 0
    assert "Query failed" in result.output


def test_compare_error_handling() -> None:
    from unittest.mock import patch

    from archex.exceptions import ArchexError

    runner = CliRunner()
    with patch("archex.cli.compare_cmd.analyze", side_effect=ArchexError("Analyze failed")):
        result = runner.invoke(cli, ["compare", "/fake/a", "/fake/b"])
    assert result.exit_code != 0
    assert "Analyze failed" in result.output


def test_compare_type_check_raises_type_error() -> None:
    from archex.cli.compare_cmd import render_comparison_markdown

    # Test that non-ComparisonResult raises TypeError
    with pytest.raises(TypeError, match="Expected ComparisonResult"):
        render_comparison_markdown({"not": "a_comparison_result"})
