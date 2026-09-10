"""Tests for the `archex explore` CLI command."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from click.testing import CliRunner

from archex.cli.main import cli
from archex.explorer import loader

if TYPE_CHECKING:
    import pytest


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


def test_explore_export_writes_offline_html_and_starts_no_server(tmp_path: Path) -> None:
    artifact_path = _artifact_json(tmp_path)
    destination = tmp_path / "site"
    runner = CliRunner()

    def _must_not_serve(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("--export must not start a server")

    with patch("archex.cli.explore_cmd.create_server", _must_not_serve):
        result = runner.invoke(cli, ["explore", str(artifact_path), "--export", str(destination)])

    assert result.exit_code == 0, result.output
    assert "wrote" in result.output
    index = destination / "index.html"
    assert index.is_file()
    html = index.read_text(encoding="utf-8")
    assert "<script" not in html
    assert "token=" not in html


def test_explore_export_reports_a_clean_error_for_an_unusable_destination(tmp_path: Path) -> None:
    artifact_path = _artifact_json(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.write_text("x")
    runner = CliRunner()

    result = runner.invoke(cli, ["explore", str(artifact_path), "--export", str(occupied)])

    assert result.exit_code != 0
    assert "Invalid value for '--export'" in result.output


def test_explore_reports_a_clean_error_for_an_oversized_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_path = _artifact_json(tmp_path)
    monkeypatch.setattr(loader, "MAX_ARTIFACT_BYTES", 4)
    runner = CliRunner()

    result = runner.invoke(cli, ["explore", str(artifact_path)])

    assert result.exit_code != 0
    assert "above the explorer's 4-byte limit" in result.output
