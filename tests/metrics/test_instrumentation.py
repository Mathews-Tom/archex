from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from click.testing import CliRunner

from archex.api import query, record_usage_event
from archex.cli.main import cli
from archex.integrations.mcp import handle_get_file_tree, handle_query_repo, handle_scout_repo
from archex.metrics.categories import category_for_tool
from archex.metrics.health import read_metrics_health
from archex.metrics.policy import METRICS_ENV, resolve_metrics_policy
from archex.metrics.recorder import UsageEvent
from archex.metrics.storage import MetricsStore
from archex.models import Config, RepoSource

if TYPE_CHECKING:
    import pytest


class FakeBundle:
    query = "where is auth"
    token_count = 10
    chunks: list[object] = []
    receipt = None
    retrieval_metadata = SimpleNamespace(seed_file_paths=[], expanded_file_paths=[])

    def to_prompt(self, format: str = "xml", *, full: bool = False) -> str:
        return "FAKE CONTEXT"


class FakeTree:
    root = "repo"
    entries: list[object] = []

    def model_dump_json(self, indent: int | None = None) -> str:
        return '{\n  "root": "repo",\n  "entries": []\n}'


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
    _enable_metrics(monkeypatch, tmp_path)
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
    _enable_metrics(monkeypatch, tmp_path)
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
    _enable_metrics(monkeypatch, tmp_path)
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
        event = conn.execute(
            "SELECT surface, tool_name, tokens_saved, whole_repo_tokens_avoided FROM usage_events"
        ).fetchone()
    assert event["surface"] == "mcp"
    assert event["tool_name"] == "query_repo"
    assert event["tokens_saved"] == 90
    assert event["whole_repo_tokens_avoided"] == 990
    assert payload["_meta"]["tokens_raw_equivalent"] == 100
    assert "upload" not in payload["_meta"]
    policy = resolve_metrics_policy(db_path=tmp_path / ".archex" / "usage.sqlite")
    assert not policy.hosted_upload_enabled


def test_mcp_query_default_off_prevents_writes(
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
        patch(
            "archex.integrations.mcp.get_repo_total_tokens",
            side_effect=AssertionError("repo scan"),
        ),
    ):
        payload = json.loads(handle_query_repo(str(repo_root), "where is auth"))

    assert payload["content"] == "<context />"
    assert not (tmp_path / ".archex" / "usage.sqlite").exists()


def test_mcp_query_recording_failure_preserves_success_and_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_metrics(monkeypatch, tmp_path)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with (
        patch("archex.integrations.mcp.query", return_value=FakeBundle()),
        patch("archex.integrations.mcp.render_xml", return_value="<context />"),
        patch("archex.integrations.mcp.render_xml_envelope", return_value="<envelope />"),
        patch("archex.integrations.mcp.get_files_token_count", return_value=100),
        patch("archex.integrations.mcp.get_repo_total_tokens", return_value=1000),
        patch("archex.integrations.mcp.record_query_usage", side_effect=RuntimeError("boom")),
    ):
        payload = json.loads(handle_query_repo(str(repo_root), "where is auth"))

    assert payload["content"] == "<context />"
    health = read_metrics_health(db_path=tmp_path / ".archex" / "usage.sqlite")
    assert health.status == "warning"
    assert health.last_failure_operation == "record"
    assert health.last_failure_message == "boom"


def test_mcp_scout_records_counter_and_preserves_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_metrics(monkeypatch, tmp_path)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with (
        patch("archex.integrations.mcp.scout", return_value=_fake_scout_result()),
        patch("archex.integrations.mcp.render_scout", return_value='{"ok": true}'),
        patch("archex.integrations.mcp.get_repo_total_tokens", return_value=1000),
        patch("archex.integrations.mcp.get_files_token_count", return_value=120),
    ):
        payload = json.loads(handle_scout_repo(str(repo_root), "where is auth"))

    assert sorted(payload) == ["_meta", "content", "receipt"]
    assert payload["content"] == {"ok": True}
    assert payload["_meta"]["tokens_raw_equivalent"] == 1000
    with MetricsStore(tmp_path / ".archex" / "usage.sqlite").connect() as conn:
        event = conn.execute(
            "SELECT surface, tool_name, tokens_raw_equivalent, whole_repo_tokens FROM usage_events"
        ).fetchone()
    assert event["surface"] == "mcp"
    assert event["tool_name"] == "scout_repo"
    assert event["tokens_raw_equivalent"] == 120
    assert event["whole_repo_tokens"] == 1000


def test_metrics_category_mapping_is_centralized() -> None:
    assert category_for_tool("query") == "context_retrieval"
    assert category_for_tool("scout_repo") == "context_retrieval"
    assert category_for_tool("tree") == "structural_tools"
    assert category_for_tool("get_file_tree") == "structural_tools"


def test_cli_tree_records_structural_tool_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_metrics(monkeypatch, tmp_path)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    runner = CliRunner()

    with (
        patch("archex.cli.tree_cmd.file_tree", return_value=FakeTree()),
        patch("archex.cli.tree_cmd.get_repo_total_tokens", return_value=500),
    ):
        result = runner.invoke(cli, ["tree", str(repo_root)])

    assert result.exit_code == 0, result.output
    assert result.output == "repo\n"
    with MetricsStore(tmp_path / ".archex" / "usage.sqlite").connect() as conn:
        event = conn.execute(
            "SELECT surface, tool_name, category, tokens_raw_equivalent FROM usage_events"
        ).fetchone()
    assert event["surface"] == "cli"
    assert event["tool_name"] == "tree"
    assert event["category"] == "structural_tools"
    assert event["tokens_raw_equivalent"] == 500


def test_mcp_file_tree_records_structural_tool_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_metrics(monkeypatch, tmp_path)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with (
        patch("archex.integrations.mcp.file_tree", return_value=FakeTree()),
        patch("archex.integrations.mcp.get_repo_total_tokens", return_value=500),
    ):
        payload = json.loads(handle_get_file_tree(str(repo_root)))

    assert payload["content"] == {"root": "repo", "entries": []}
    assert payload["_meta"]["tool_name"] == "get_file_tree"
    with MetricsStore(tmp_path / ".archex" / "usage.sqlite").connect() as conn:
        event = conn.execute(
            "SELECT surface, tool_name, category, tokens_raw_equivalent FROM usage_events"
        ).fetchone()
    assert event["surface"] == "mcp"
    assert event["tool_name"] == "get_file_tree"
    assert event["category"] == "structural_tools"
    assert event["tokens_raw_equivalent"] == 500


def test_python_api_query_does_not_write_metrics_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    python_simple_repo: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    source = RepoSource(local_path=str(python_simple_repo))

    bundle = query(source, "authentication", config=Config(cache=False, languages=["python"]))

    assert bundle.chunks
    assert not (tmp_path / ".archex" / "usage.sqlite").exists()


def test_python_api_explicit_usage_event_records_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    python_simple_repo: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    source = RepoSource(local_path=str(python_simple_repo))

    record_usage_event(
        UsageEvent(
            repo_root=python_simple_repo,
            surface="python_api",
            tool_name="query",
            category="context_retrieval",
            tokens_returned=25,
            tokens_raw_equivalent=100,
            whole_repo_tokens=1000,
            file_count=2,
        )
    )

    with MetricsStore(tmp_path / ".archex" / "usage.sqlite").connect() as conn:
        event = conn.execute(
            "SELECT surface, tool_name, tokens_saved, whole_repo_tokens_avoided FROM usage_events"
        ).fetchone()
    assert source.local_path == str(python_simple_repo)
    assert event["surface"] == "python_api"
    assert event["tool_name"] == "query"
    assert event["tokens_saved"] == 75
    assert event["whole_repo_tokens_avoided"] == 975


def test_python_api_explicit_usage_event_respects_env_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    python_simple_repo: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(METRICS_ENV, "off")

    record_usage_event(
        UsageEvent(
            repo_root=python_simple_repo,
            surface="python_api",
            tool_name="query",
            category="context_retrieval",
            tokens_returned=25,
            tokens_raw_equivalent=100,
            whole_repo_tokens=1000,
            file_count=2,
        )
    )

    assert not (tmp_path / ".archex" / "usage.sqlite").exists()


def test_targeted_read_is_index_only_and_within_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from archex.metrics.capture import record_query_usage
    from archex.metrics.policy import set_metrics_enabled
    from archex.models import CodeChunk, ContextBundle, RankedChunk

    monkeypatch.setenv("HOME", str(tmp_path))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    db_path = tmp_path / ".archex" / "usage.sqlite"
    set_metrics_enabled(True, db_path=db_path)

    chunk = CodeChunk(
        id="m.py:f:3",
        content="def f():\n    return compute()\n    # trailing body line",
        file_path="m.py",
        start_line=3,
        end_line=5,
        language="python",
        token_count=12,
    )
    bundle = ContextBundle(
        query="where is f",
        chunks=[RankedChunk(chunk=chunk)],
        token_count=12,
    )
    source = RepoSource(local_path=str(repo_root))

    def _no_fs_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("metrics path read file contents")

    with (
        patch("pathlib.Path.read_bytes", _no_fs_read),
        patch("pathlib.Path.read_text", _no_fs_read),
    ):
        record_query_usage(
            source,
            bundle,
            surface="cli",
            tokens_raw_equivalent=100,
            whole_repo_tokens=1000,
            db_path=db_path,
        )

    with MetricsStore(db_path).connect() as conn:
        row = conn.execute(
            "SELECT tokens_returned, tokens_targeted_read, tokens_raw_equivalent FROM usage_events"
        ).fetchone()
    assert row["tokens_targeted_read"] is not None
    assert row["tokens_returned"] <= row["tokens_targeted_read"] <= row["tokens_raw_equivalent"]


def test_targeted_read_helper_is_deterministic_and_bounded() -> None:
    from archex.metrics.capture import _targeted_read_tokens  # pyright: ignore[reportPrivateUsage]
    from archex.models import CodeChunk, RankedChunk

    chunks = [
        RankedChunk(
            chunk=CodeChunk(
                id="m.py:a:1",
                content="def a():\n    return 1",
                file_path="m.py",
                start_line=1,
                end_line=2,
                language="python",
            )
        ),
        RankedChunk(
            chunk=CodeChunk(
                id="m.py:b:40",
                content="def b():\n    return 2",
                file_path="m.py",
                start_line=40,
                end_line=41,
                language="python",
            )
        ),
    ]
    first = _targeted_read_tokens(chunks, full_file_tokens=500, returned_tokens=8)
    second = _targeted_read_tokens(chunks, full_file_tokens=500, returned_tokens=8)
    assert first == second
    assert first is not None
    assert 8 <= first <= 500
    # No chunks -> no spans -> baseline omitted.
    assert _targeted_read_tokens([], full_file_tokens=500, returned_tokens=8) is None
    # Degenerate returned > full_file: targeted is capped at full_file, never above it.
    assert _targeted_read_tokens(chunks, full_file_tokens=3, returned_tokens=50) == 3
