from __future__ import annotations

import json
from typing import TYPE_CHECKING

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
