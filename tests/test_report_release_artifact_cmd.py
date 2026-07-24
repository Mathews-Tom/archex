"""Tests for the `archex report release-artifact` CLI command (M9)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.cli.main import cli

if TYPE_CHECKING:
    from pathlib import Path


class TestReportReleaseArtifactCmd:
    def test_outputs_valid_json(self, python_simple_repo: Path) -> None:
        runner = CliRunner()

        result = runner.invoke(cli, ["report", "release-artifact", str(python_simple_repo)])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["schema_version"] == "1.0.0"
        assert payload["archex_version"]
        assert payload["report_schema_version"] == "1.0.0"
        assert payload["index_schema_version"]
        assert "status_card" in payload

    def test_defaults_source_to_current_directory(self) -> None:
        runner = CliRunner()

        result = runner.invoke(cli, ["report", "release-artifact"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["compatibility_matrix_path"] == "docs/CLIENT_COMPATIBILITY_MATRIX.md"
