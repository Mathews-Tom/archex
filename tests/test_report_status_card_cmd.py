"""Tests for the `archex report status-card` CLI command (M9)."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli


class TestReportStatusCardCmd:
    def test_default_format_is_markdown(self, python_simple_repo: Path) -> None:
        runner = CliRunner()

        result = runner.invoke(cli, ["report", "status-card", str(python_simple_repo)])

        assert result.exit_code == 0, result.output
        assert result.output.startswith("# Documentation & Release Status:")
        assert "### Documentation linkage" in result.output

    def test_json_format_round_trips(self, python_simple_repo: Path) -> None:
        runner = CliRunner()

        result = runner.invoke(
            cli, ["report", "status-card", str(python_simple_repo), "--format", "json"]
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["schema_version"] == "1.0.0"
        assert len(payload["dimensions"]) == 4

    def test_json_output_has_no_composite_score_field(self, python_simple_repo: Path) -> None:
        runner = CliRunner()

        result = runner.invoke(
            cli, ["report", "status-card", str(python_simple_repo), "--format", "json"]
        )

        payload = json.loads(result.output)
        top_level_keys = {key.lower() for key in payload}
        assert not top_level_keys & {"score", "grade", "health", "rating"}
        for dimension in payload["dimensions"]:
            dimension_keys = {key.lower() for key in dimension}
            assert not dimension_keys & {"score", "grade", "health", "rating"}
