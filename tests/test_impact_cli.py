from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli


def test_impact_explicit_changed_file_reports_dependents(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "impact",
            str(python_simple_repo),
            "--changed-file",
            "utils.py",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["changed_files"] == [{"old_path": None, "path": "utils.py", "status": "M"}]
    assert "utils.py" in data["affected_files"]
    assert "main.py" in data["affected_files"]
    assert "services/auth.py" in data["affected_files"]
    assert "public_interface_changed" in data["risk"]["reasons"]


def test_impact_explicit_unmapped_file_reports_risk(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "impact",
            str(python_simple_repo),
            "--changed-file",
            "README.md",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["affected_files"] == []
    assert data["unmapped_files"] == ["README.md"]
    assert data["risk"]["level"] == "moderate"
    assert data["risk"]["reasons"] == ["unmapped_file"]


def test_impact_git_diff_mode_handles_no_changes(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["impact", str(python_simple_repo), "--base", "HEAD", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["changed_files"] == []
    assert data["risk"]["level"] == "low"


def test_impact_git_error_is_not_hidden(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["impact", str(python_simple_repo), "--base", "missing-ref"],
    )

    assert result.exit_code != 0
    assert "git diff failed for base missing-ref" in result.output
