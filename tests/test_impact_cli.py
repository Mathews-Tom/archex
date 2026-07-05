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


def test_impact_diff_classifies_hub_edit_as_high(impact_diff_repo: Path) -> None:
    hub = impact_diff_repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["impact", str(impact_diff_repo), "--diff", "HEAD", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["diff_ref"] == "HEAD"
    assert [
        (symbol["file_path"], symbol["symbol_name"], symbol["level"])
        for symbol in data["affected_symbols"]
    ] == [("hub.py", "shared_helper", "high")]


def test_impact_diff_classifies_leaf_edit_as_low(impact_diff_repo: Path) -> None:
    leaf = impact_diff_repo / "leaf.py"
    old_text = leaf.read_text()
    leaf.write_text(old_text.replace("shared_helper(value) - 1", "shared_helper(value) - 2"))
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["impact", str(impact_diff_repo), "--diff", "HEAD", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [
        (symbol["file_path"], symbol["symbol_name"], symbol["level"])
        for symbol in data["affected_symbols"]
    ] == [("leaf.py", "isolated", "low")]


def test_impact_diff_bare_flag_defaults_to_head(impact_diff_repo: Path) -> None:
    hub = impact_diff_repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["impact", str(impact_diff_repo), "--diff", "--format", "json"],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["diff_ref"] == "HEAD"
    assert data["affected_symbols"] != []


def test_impact_diff_rejects_changed_file_combination(impact_diff_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["impact", str(impact_diff_repo), "--diff", "HEAD", "--changed-file", "hub.py"],
    )

    assert result.exit_code != 0
    assert "--diff cannot be combined with --changed-file" in result.output


def test_impact_diff_rejects_explicit_base_combination(impact_diff_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["impact", str(impact_diff_repo), "--diff", "HEAD", "--base", "main"],
    )

    assert result.exit_code != 0
    assert "--diff cannot be combined with --base" in result.output


def test_impact_without_diff_flag_output_has_no_diff_fields(python_simple_repo: Path) -> None:
    """Backward compatibility: omitting --diff must not add diff_ref/affected_symbols."""
    runner = CliRunner()

    changed_file_result = runner.invoke(
        cli,
        ["impact", str(python_simple_repo), "--changed-file", "utils.py", "--format", "json"],
    )
    assert changed_file_result.exit_code == 0, changed_file_result.output
    changed_file_data = json.loads(changed_file_result.output)
    assert "diff_ref" not in changed_file_data
    assert "affected_symbols" not in changed_file_data

    base_result = runner.invoke(
        cli,
        ["impact", str(python_simple_repo), "--base", "HEAD", "--format", "json"],
    )
    assert base_result.exit_code == 0, base_result.output
    base_data = json.loads(base_result.output)
    assert "diff_ref" not in base_data
    assert "affected_symbols" not in base_data
