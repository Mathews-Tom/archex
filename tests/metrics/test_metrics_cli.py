from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.cli.main import cli
from archex.metrics.health import read_metrics_health, record_metrics_failure
from archex.metrics.policy import set_metrics_enabled
from archex.metrics.recorder import MetricsRecorder, TraceDetails, UsageEvent
from archex.metrics.storage import metrics_db_path

if TYPE_CHECKING:
    import pytest


def test_bare_metrics_prints_current_repo_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        repo_root = Path(cwd).resolve()
        _record(repo_root, tmp_path)
        result = runner.invoke(cli, ["metrics"])

    assert result.exit_code == 0, result.output
    assert "Saved tokens:           90 vs returned full files" in result.output
    assert "Whole-repo avoided:     990 upper-bound/context only" in result.output
    assert "Recording:              on" in result.output
    assert "Trace:                  off" in result.output


def test_metrics_summary_repos_inspect_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo_root = tmp_path / "workspace" / "repo-a"
    repo_root.mkdir(parents=True)
    _record(repo_root, tmp_path)
    runner = CliRunner()

    summary = runner.invoke(
        cli,
        ["metrics", "summary", str(repo_root), "--format", "json"],
    )
    repos = runner.invoke(cli, ["metrics", "repos", "--format", "json"])
    inspect = runner.invoke(cli, ["metrics", "inspect", str(repo_root), "--format", "json"])
    workspace = runner.invoke(
        cli,
        ["metrics", "summary", "--workspace", str(tmp_path / "workspace"), "--format", "json"],
    )

    assert summary.exit_code == 0, summary.output
    assert repos.exit_code == 0, repos.output
    assert inspect.exit_code == 0, inspect.output
    assert workspace.exit_code == 0, workspace.output
    summary_payload = json.loads(summary.output)
    repos_payload = json.loads(repos.output)
    inspect_payload = json.loads(inspect.output)
    workspace_payload = json.loads(workspace.output)
    assert summary_payload["totals"]["tokens_saved"] == 90
    assert workspace_payload["totals"]["tokens_saved"] == 90
    assert repos_payload["repos"][0]["display_name"] == "repo-a"
    assert inspect_payload["events"][0]["tool_name"] == "query"


def test_metrics_inspect_json_includes_opt_in_trace_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = CliRunner()
    trace_enable = runner.invoke(cli, ["metrics", "trace", "enable"])
    _record(repo_root, tmp_path)

    inspect = runner.invoke(cli, ["metrics", "inspect", str(repo_root), "--format", "json"])

    assert trace_enable.exit_code == 0, trace_enable.output
    assert inspect.exit_code == 0, inspect.output
    event = json.loads(inspect.output)["events"][0]
    assert event["trace"]["query_text"] == "where is auth"
    assert event["trace"]["returned_file_paths"] == []


def test_metrics_export_redacts_paths_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo_root = tmp_path / "secret-org" / "secret-repo"
    repo_root.mkdir(parents=True)
    _record(repo_root, tmp_path)
    runner = CliRunner()
    redacted = tmp_path / "redacted.json"
    pathful = tmp_path / "pathful.json"

    default_result = runner.invoke(cli, ["metrics", "export", "--output", str(redacted)])
    included_result = runner.invoke(
        cli,
        ["metrics", "export", "--output", str(pathful), "--include-local-paths"],
    )

    assert default_result.exit_code == 0, default_result.output
    assert included_result.exit_code == 0, included_result.output
    redacted_payload = json.loads(redacted.read_text(encoding="utf-8"))
    pathful_payload = json.loads(pathful.read_text(encoding="utf-8"))
    assert "repo_root" not in redacted_payload["repos"][0]
    assert pathful_payload["repos"][0]["repo_root"] == str(repo_root.resolve())


def test_metrics_controls_enable_disable_trace_and_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = CliRunner()
    default_summary = runner.invoke(cli, ["metrics", "summary", "--format", "json"])

    disable = runner.invoke(cli, ["metrics", "disable"])
    disabled_summary = runner.invoke(cli, ["metrics", "summary", "--format", "json"])
    enable = runner.invoke(cli, ["metrics", "enable"])
    trace_enable = runner.invoke(cli, ["metrics", "trace", "enable"])
    traced_summary = runner.invoke(cli, ["metrics", "summary", "--format", "json"])
    trace_disable = runner.invoke(cli, ["metrics", "trace", "disable"])
    delete = runner.invoke(cli, ["metrics", "delete", "--all"])

    assert json.loads(default_summary.output)["recording_enabled"] is False
    assert disable.exit_code == 0, disable.output
    assert json.loads(disabled_summary.output)["recording_enabled"] is False
    assert enable.exit_code == 0, enable.output
    traced_payload = json.loads(traced_summary.output)
    assert traced_payload["recording_enabled"] is True
    assert traced_payload["trace_enabled"] is True
    assert trace_enable.exit_code == 0, trace_enable.output
    assert trace_disable.exit_code == 0, trace_disable.output
    assert delete.exit_code == 0, delete.output
    assert not metrics_db_path(home=tmp_path).exists()


def test_metrics_repair_clears_stale_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    db_path = metrics_db_path(home=tmp_path)
    record_metrics_failure("record", "Path does not exist: /repo", db_path=db_path)
    runner = CliRunner()

    cleared = runner.invoke(cli, ["metrics", "repair"])
    repeated = runner.invoke(cli, ["metrics", "repair"])

    assert cleared.exit_code == 0, cleared.output
    assert "Cleared metrics health warning" in cleared.output
    assert read_metrics_health(db_path=db_path).status == "ok"
    assert "nothing to repair" in repeated.output


def test_metrics_summary_surface_mix_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo_root = tmp_path / "workspace" / "repo-mix"
    repo_root.mkdir(parents=True)
    _record(repo_root, tmp_path)
    MetricsRecorder(metrics_db_path(home=tmp_path)).record(
        UsageEvent(
            repo_root=repo_root,
            surface="mcp",
            tool_name="query_repo",
            category="context_retrieval",
            tokens_returned=20,
            tokens_raw_equivalent=120,
            whole_repo_tokens=1000,
            occurred_at=datetime.now(UTC),
            file_count=2,
            freshness="clean",
            index_revision="rev",
        )
    )
    runner = CliRunner()

    summary_json = runner.invoke(cli, ["metrics", "summary", str(repo_root), "--format", "json"])
    summary_text = runner.invoke(cli, ["metrics", "summary", str(repo_root)])

    assert summary_json.exit_code == 0, summary_json.output
    assert summary_text.exit_code == 0, summary_text.output
    by_surface = json.loads(summary_json.output)["totals"]["by_surface"]
    assert by_surface["cli"]["event_count"] == 1
    assert by_surface["mcp"]["event_count"] == 1
    assert by_surface["python_api"]["event_count"] == 0
    assert by_surface["cli"]["tokens_saved"] == 90
    assert by_surface["mcp"]["tokens_saved"] == 100
    assert by_surface["python_api"]["tokens_saved"] == 0
    assert "Surface mix:            cli 1, mcp 1, python_api 0" in summary_text.output


def _record(repo_root: Path, tmp_path: Path) -> None:
    set_metrics_enabled(True, db_path=metrics_db_path(home=tmp_path))
    MetricsRecorder(metrics_db_path(home=tmp_path)).record(
        UsageEvent(
            repo_root=repo_root,
            surface="cli",
            tool_name="query",
            category="context_retrieval",
            tokens_returned=10,
            tokens_raw_equivalent=100,
            whole_repo_tokens=1000,
            occurred_at=datetime.now(UTC),
            file_count=1,
            freshness="clean",
            index_revision="rev",
            trace=TraceDetails(query_text="where is auth"),
        )
    )
