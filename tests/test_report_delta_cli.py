"""CLI tests for `archex report delta`."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli


def _edit_hub(repo: Path) -> None:
    hub = repo / "hub.py"
    hub.write_text(hub.read_text().replace("value * 2", "value * 3"))


def test_report_delta_json_is_default_format(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    runner = CliRunner()

    result = runner.invoke(cli, ["report", "delta", str(impact_diff_repo), "--base", "HEAD"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["changed_files_total"] == 1
    assert data["base_ref"] == "HEAD"


def test_report_delta_markdown_format(impact_diff_repo: Path) -> None:
    _edit_hub(impact_diff_repo)
    runner = CliRunner()

    result = runner.invoke(
        cli, ["report", "delta", str(impact_diff_repo), "--base", "HEAD", "--format", "markdown"]
    )

    assert result.exit_code == 0, result.output
    assert result.output.startswith("## Diff Review Delta")


def test_report_delta_invalid_base_ref_is_a_clean_cli_error(impact_diff_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli, ["report", "delta", str(impact_diff_repo), "--base", "not-a-real-ref"]
    )

    assert result.exit_code != 0
    assert "not-a-real-ref" in result.output
