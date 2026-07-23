"""CLI tests for `archex report diff`."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli


def _edit_hub(repo: Path) -> None:
    hub = repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))


def test_report_diff_json_matches_artifact_schema(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    runner = CliRunner()

    result = runner.invoke(
        cli, ["report", "diff", str(impact_diff_repo), "--base", "HEAD", "--format", "json"]
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["schema_version"]["value"] == "1.0.0"
    assert data["diff"]["changed_files_total"] == 1
    assert data["diff"]["changed_files"][0]["path"] == "hub.py"


def test_report_diff_markdown_is_default_format(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    runner = CliRunner()

    result = runner.invoke(cli, ["report", "diff", str(impact_diff_repo), "--base", "HEAD"])

    assert result.exit_code == 0, result.output
    assert result.output.startswith("# Diff Review:")
    assert "```mermaid" in result.output


def test_report_diff_html_format_is_offline_self_contained(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    runner = CliRunner()

    result = runner.invoke(
        cli, ["report", "diff", str(impact_diff_repo), "--base", "HEAD", "--format", "html"]
    )

    assert result.exit_code == 0, result.output
    assert result.output.startswith("<!DOCTYPE html>")
    assert "<script" not in result.output.lower()
    assert "https://" not in result.output


def test_report_diff_rejects_unknown_format(impact_diff_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli, ["report", "diff", str(impact_diff_repo), "--base", "HEAD", "--format", "yaml"]
    )

    assert result.exit_code != 0
    assert "yaml" in result.output.lower()


def test_report_diff_invalid_base_ref_is_a_clean_cli_error(impact_diff_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli, ["report", "diff", str(impact_diff_repo), "--base", "not-a-real-ref"]
    )

    assert result.exit_code != 0
    assert "not-a-real-ref" in result.output


def test_report_diff_no_network_access_required(impact_diff_repo: Path) -> None:
    """Smoke check that report diff against a local-only fixture repo succeeds offline.

    `impact_diff_repo` has no remote configured; a passing run without
    network mocking demonstrates the command performs no remote fetch.
    """
    _edit_hub(impact_diff_repo)
    runner = CliRunner()

    result = runner.invoke(
        cli, ["report", "diff", str(impact_diff_repo), "--base", "HEAD", "--format", "json"]
    )

    assert result.exit_code == 0, result.output
