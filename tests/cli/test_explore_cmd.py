"""Tests for the `archex explore` CLI command."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from archex.cli.main import cli


def _artifact_json(path: Path) -> Path:
    payload = {
        "schema_version": {"value": "1.0.0"},
        "archex_version": "0.22.0",
        "generated_at": "2026-07-24T00:00:00Z",
        "source_identity": "acme/widget",
        "source_root": "/repo",
        "source_revision": "deadbeef",
        "working_tree_fingerprint": "fp",
        "index_generation": "gen1",
        "index_schema_version": "1",
        "chunker_revision": "c1",
        "config_fingerprint": "cfg1",
        "diff": {"base_ref": "main"},
    }
    artifact_path = path / "artifact.json"
    artifact_path.write_text(json.dumps(payload))
    return artifact_path


def test_explore_reports_a_clean_error_for_malformed_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{not valid json")
    runner = CliRunner()

    result = runner.invoke(cli, ["explore", str(artifact_path)])

    assert result.exit_code != 0
    assert "Malformed" in result.output or "malformed" in result.output.lower()


def test_explore_reports_a_clean_error_for_missing_artifact() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["explore", "/does/not/exist.json"])

    assert result.exit_code != 0


def test_explore_prints_the_loopback_url_and_stops_on_interrupt(tmp_path: Path) -> None:
    artifact_path = _artifact_json(tmp_path)
    runner = CliRunner()
    before = threading.active_count()

    def _serve_forever_then_interrupt(_self: object) -> None:
        raise KeyboardInterrupt

    with patch(
        "archex.explorer.server.ExplorerServer.serve_forever",
        _serve_forever_then_interrupt,
    ):
        result = runner.invoke(cli, ["explore", str(artifact_path), "--port", "0"])

    assert result.exit_code == 0, result.output
    assert "archex explorer listening at http://127.0.0.1:" in result.output
    assert "?token=" in result.output
    assert threading.active_count() == before
