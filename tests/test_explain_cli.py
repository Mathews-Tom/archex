from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli


def test_explain_file_markdown(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["explain", str(python_simple_repo), "main.py"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("# Explain: main.py")
    assert "## Public Surface" in result.output
    assert "`run`" in result.output
    assert "`utils.py`" in result.output


def test_explain_symbol_json(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "explain",
            str(python_simple_repo),
            "main.py::run#function",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["target_type"] == "symbol"
    assert data["target"] == "main.py::run#function"
    assert data["files"] == ["main.py"]
    assert data["public_interfaces"][0]["qualified_name"] == "run"


def test_explain_module_json(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["explain", str(python_simple_repo), "--module", "services", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["target_type"] == "module"
    assert data["target"] == "services"
    assert data["files"] == ["services/__init__.py", "services/auth.py"]
    assert "utils.py" in data["imports"]


def test_explain_missing_target_fails(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["explain", str(python_simple_repo), "missing.py"])

    assert result.exit_code != 0
    assert "Target file does not exist in index: missing.py" in result.output
