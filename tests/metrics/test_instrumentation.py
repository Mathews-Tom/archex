from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from click.testing import CliRunner

from archex.cli.main import cli
from archex.integrations.mcp import handle_query_repo
from archex.metrics.health import read_metrics_health
from archex.metrics.policy import METRICS_ENV
from archex.metrics.storage import MetricsStore

if TYPE_CHECKING:
    import pytest


class FakeBundle:
    query = "where is auth"
    token_count = 10
    chunks: list[object] = []
    receipt = None
    retrieval_metadata = SimpleNamespace(seed_file_paths=[], expanded_file_paths=[])

    def to_prompt(self, format: str = "xml") -> str:
        return "FAKE CONTEXT"

def _fake_scout_result() -> SimpleNamespace:
    return SimpleNamespace(
        query="where is auth",
        ranked_files=[SimpleNamespace(path="app.py")],
        modules=[],
        symbols=[],
        graph=[],
        receipt=None,
        fetch_plan=SimpleNamespace(handles=[]),
        budget=SimpleNamespace(
            omitted_files=0,
            omitted_symbols=0,
            omitted_modules=0,
            omitted_graph_edges=0,
        ),
    )


def _enable_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(METRICS_ENV, "on")

def test_cli_query_records_counter_without_changing_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_metrics(monkeypatch, tmp_path)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = CliRunner()

    with (
        patch("archex.cli.query_cmd.query", return_value=FakeBundle()),
        patch("archex.cli.query_cmd.get_files_token_count", return_value=100),
        patch("archex.cli.query_cmd.get_repo_total_tokens", return_value=1000),
    ):
        result = runner.invoke(cli, ["query", str(repo_root), "where", "is", "auth"])
    assert result.exit_code == 0, result.output
    assert result.output == "FAKE CONTEXT\n"
    with MetricsStore(tmp_path / ".archex" / "usage.sqlite").connect() as conn:
        event = conn.execute(
            "SELECT surface, tool_name, tokens_saved, whole_repo_tokens_avoided FROM usage_events"
        ).fetchone()
    assert event["surface"] == "cli"
    assert event["tool_name"] == "query"
    assert event["tokens_saved"] == 90
    assert event["whole_repo_tokens_avoided"] == 990


def test_cli_query_default_off_avoids_metric_work_and_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = CliRunner()

    with (
        patch("archex.cli.query_cmd.query", return_value=FakeBundle()),
        patch("archex.cli.query_cmd.get_files_token_count", side_effect=AssertionError("raw scan")),
    ):
        result = runner.invoke(cli, ["query", str(repo_root), "where", "is", "auth"])

    assert result.exit_code == 0, result.output
    assert result.output == "FAKE CONTEXT\n"
    assert not (tmp_path / ".archex" / "usage.sqlite").exists()


def test_cli_query_timing_savings_stderr_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_metrics(monkeypatch, tmp_path)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = CliRunner()
    with (
        patch("archex.cli.query_cmd.query", return_value=FakeBundle()),
        patch("archex.cli.query_cmd.get_files_token_count", return_value=100),
        patch("archex.cli.query_cmd.get_repo_total_tokens", return_value=1000),
    ):
        result = runner.invoke(cli, ["query", "--timing", str(repo_root), "where is auth"])

    assert result.exit_code == 0, result.output
    assert result.stdout == "FAKE CONTEXT\n"
    assert "[savings] Raw equivalent: 100 tokens across 0 files" in result.stderr


def test_cli_query_recording_failure_is_non_fatal_and_visible_in_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = CliRunner()

    with (
        patch("archex.cli.query_cmd.query", return_value=FakeBundle()),
        patch("archex.cli.query_cmd.get_files_token_count", return_value=100),
        patch("archex.cli.query_cmd.get_repo_total_tokens", return_value=1000),
        patch("archex.cli.query_cmd.record_query_usage", side_effect=RuntimeError("boom")),
    ):
        result = runner.invoke(cli, ["query", str(repo_root), "where is auth"])

    assert result.exit_code == 0, result.output
    assert result.output == "FAKE CONTEXT\n"
    health = read_metrics_health(db_path=tmp_path / ".archex" / "usage.sqlite")
    assert health.status == "warning"
    assert health.last_failure_operation == "record"
    assert health.last_failure_message == "boom"


def test_cli_scout_records_counter_without_changing_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = CliRunner()

    with (
        patch("archex.cli.scout_cmd.scout", return_value=_fake_scout_result()),
        patch("archex.cli.scout_cmd.render_scout", return_value="SCOUT MAP\n"),
        patch("archex.cli.scout_cmd.get_files_token_count", return_value=120),
        patch("archex.cli.scout_cmd.get_repo_total_tokens", return_value=1000),
    ):
        result = runner.invoke(cli, ["scout", str(repo_root), "where is auth"])

    assert result.exit_code == 0, result.output
    assert result.output == "SCOUT MAP\n"
    with MetricsStore(tmp_path / ".archex" / "usage.sqlite").connect() as conn:
        event = conn.execute(
            "SELECT surface, tool_name, tokens_raw_equivalent, whole_repo_tokens FROM usage_events"
        ).fetchone()
    assert event["surface"] == "cli"
    assert event["tool_name"] == "scout"
    assert event["tokens_raw_equivalent"] == 120
    assert event["whole_repo_tokens"] == 1000


def test_mcp_query_records_counter_and_preserves_response_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    with (
        patch("archex.integrations.mcp.query", return_value=FakeBundle()),
        patch("archex.integrations.mcp.render_xml", return_value="<context />"),
        patch("archex.integrations.mcp.render_xml_envelope", return_value="<envelope />"),
        patch("archex.integrations.mcp.get_files_token_count", return_value=100),
        patch("archex.integrations.mcp.get_repo_total_tokens", return_value=1000),
    ):
        payload = json.loads(handle_query_repo(str(repo_root), "where is auth"))

    assert sorted(payload) == ["_meta", "content", "receipt"]
    assert payload["content"] == "<context />"
    with MetricsStore(tmp_path / ".archex" / "usage.sqlite").connect() as conn:
        event = conn.execute("SELECT surface, tool_name, tokens_saved FROM usage_events").fetchone()
    assert event["surface"] == "mcp"
    assert event["tool_name"] == "query_repo"
    assert event["tokens_saved"] == 90

