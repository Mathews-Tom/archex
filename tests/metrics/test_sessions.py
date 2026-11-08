from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from archex.cli.main import cli
from archex.metrics.policy import set_metrics_enabled
from archex.metrics.recorder import MetricsRecorder, UsageEvent
from archex.metrics.reporter import MetricsReporter
from archex.metrics.sessions import import_omp_session, parse_omp_session
from archex.metrics.storage import MetricsStore, metrics_db_path

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def session_file(tmp_path: Path) -> Callable[..., Path]:
    def create(*, usages: list[dict[str, object] | None] | None = None) -> Path:
        now = datetime.now(UTC) - timedelta(minutes=1)
        entries: list[dict[str, object]] = [
            {"type": "title", "title": "private title"},
            {
                "type": "session",
                "version": 3,
                "id": "private-host-session",
                "timestamp": now.isoformat(),
                "cwd": "/private/unrelated/path",
            },
            {
                "type": "message",
                "id": "user",
                "timestamp": now.isoformat(),
                "message": {"role": "user", "content": "private prompt"},
            },
        ]
        if usages is None:
            usages = [
                {
                    "input": 100,
                    "output": 20,
                    "cacheRead": 50,
                    "cacheWrite": 10,
                    "totalTokens": 180,
                    "cost": {"total": 0.003},
                },
                {
                    "input": 200,
                    "output": 30,
                    "cacheRead": 70,
                    "cacheWrite": 0,
                    "totalTokens": 300,
                    "cost": {"total": 0.004},
                },
            ]
        for index, usage in enumerate(usages):
            entries.append(
                {
                    "type": "message",
                    "id": f"assistant-{index}",
                    "timestamp": (now + timedelta(seconds=index + 1)).isoformat(),
                    "message": {
                        "role": "assistant",
                        "content": "private response",
                        "usage": usage,
                        "duration": 12.5 + index,
                    },
                }
            )
        path = tmp_path / "private-transcript.jsonl"
        path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
        return path

    return create


def test_import_is_idempotent_private_and_separate_from_retrieval_savings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, session_file: Callable[..., Path]
) -> None:
    monkeypatch.delenv("ARCHEX_USAGE_METRICS", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    db = tmp_path / "usage.sqlite"
    set_metrics_enabled(True, db_path=db)
    path = session_file()
    first = import_omp_session(path, repo, db_path=db)
    import_omp_session(path, repo, db_path=db)
    reporter = MetricsReporter(db)
    rows = reporter.sessions(source=repo)["sessions"]
    assert len(rows) == 1
    row = rows[0]
    assert row["session_key"] == first["session_key"]
    assert row["request_count"] == row["receipt_count"] == 2
    assert row["input_tokens"] == 300
    assert row["output_tokens"] == 50
    assert row["cache_read_tokens"] == 120
    assert row["cache_write_tokens"] == 10
    assert row["total_tokens"] == 480
    assert math.isclose(row["host_cost_estimate_usd"], 0.007)
    assert row["request_duration_ms"] == 26
    assert row["cost_basis"] == "host_estimate_not_billing_receipt"
    assert reporter.summary(source=repo)["event_count"] == 0
    exported = json.dumps(reporter.export())
    with MetricsStore(db).connect() as conn:
        stored = "\n".join(conn.iterdump())
    for sensitive in (
        "private title",
        "private-host-session",
        "/private/unrelated/path",
        "private prompt",
        "private response",
        "private-transcript.jsonl",
    ):
        assert sensitive not in exported
        assert sensitive not in stored


def test_missing_fields_do_not_become_zero_or_partial_totals(
    session_file: Callable[..., Path],
) -> None:
    receipt = parse_omp_session(
        session_file(
            usages=[
                {"input": 10, "output": 0, "cost": {"total": 0}},
                {"output": 0},
            ]
        )
    )
    assert receipt.totals["input_tokens"] is None
    assert receipt.totals["output_tokens"] == 0
    assert receipt.totals["cache_read_tokens"] is None
    assert receipt.totals["host_cost_estimate_usd"] is None
    assert receipt.totals["request_duration_ms"] == 26
    missing = parse_omp_session(session_file(usages=[None]))
    assert missing.request_count == 1
    assert missing.receipt_count == 0
    assert missing.totals["input_tokens"] is None
    empty = parse_omp_session(session_file(usages=[]))
    assert empty.request_count == 0
    assert all(value is None for value in empty.totals.values())


@pytest.mark.parametrize("value", [-1, True, 1.5, "10", float("nan"), float("inf"), 2**64])
def test_rejects_invalid_counters_without_partial_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_file: Callable[..., Path],
    value: object,
) -> None:
    monkeypatch.delenv("ARCHEX_USAGE_METRICS", raising=False)
    db = tmp_path / "usage.sqlite"
    set_metrics_enabled(True, db_path=db)
    with pytest.raises(ValueError):
        import_omp_session(session_file(usages=[{"input": value}]), tmp_path, db_path=db)
    with MetricsStore(db).connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM host_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM repos").fetchone()[0] == 0


def test_off_override_prevents_writes_even_after_enablement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_file: Callable[..., Path],
) -> None:
    db = tmp_path / "usage.sqlite"
    path = session_file()
    monkeypatch.setenv("ARCHEX_USAGE_METRICS", "off")
    with pytest.raises(ValueError, match="disabled"):
        import_omp_session(path, tmp_path, db_path=db)
    assert not db.exists()
    set_metrics_enabled(True, db_path=db)
    with pytest.raises(ValueError, match="disabled"):
        import_omp_session(path, tmp_path, db_path=db)
    with MetricsStore(db).connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM host_sessions").fetchone()[0] == 0


def test_older_snapshot_cannot_replace_a_more_complete_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_file: Callable[..., Path],
) -> None:
    monkeypatch.setenv("ARCHEX_USAGE_METRICS", "on")
    db = tmp_path / "usage.sqlite"
    path = session_file()
    full = path.read_text()
    import_omp_session(path, tmp_path, db_path=db)
    entries = full.splitlines()
    path.write_text("\n".join(entries[:-1]) + "\n")
    result = import_omp_session(path, tmp_path, db_path=db)
    assert result["updated"] is False
    row = MetricsReporter(db).sessions(source=tmp_path)["sessions"][0]
    assert row["request_count"] == 2
    assert row["input_tokens"] == 300


def test_activity_association_is_scoped_to_repository_and_time_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_file: Callable[..., Path],
) -> None:
    monkeypatch.setenv("ARCHEX_USAGE_METRICS", "on")
    db = tmp_path / "usage.sqlite"
    other = tmp_path / "other"
    other.mkdir()
    path = session_file()
    receipt = parse_omp_session(path)
    import_omp_session(path, tmp_path, db_path=db)
    recorder = MetricsRecorder(db)
    for root, when, tokens in [
        (tmp_path, receipt.started_at + timedelta(seconds=1), 5),
        (tmp_path, receipt.started_at - timedelta(seconds=1), 100),
        (other, receipt.started_at + timedelta(seconds=1), 200),
    ]:
        recorder.record(
            UsageEvent(
                repo_root=root,
                surface="cli",
                tool_name="query",
                category="context_retrieval",
                tokens_returned=tokens,
                tokens_raw_equivalent=1000,
                occurred_at=when,
            )
        )
    report = MetricsReporter(db)
    row = report.sessions(source=tmp_path)["sessions"][0]
    assert row["archex_activity"] == {
        "association": "same_repo_time_window_not_session_attribution",
        "event_count": 1,
        "tokens_returned": 5,
    }
    assert report.sessions(source=other)["sessions"] == []
    assert report.export()["host_sessions"][0]["session_key"] == row["session_key"]


def test_v2_upgrade_preserves_existing_data_and_prunes_only_expired_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_file: Callable[..., Path],
) -> None:
    monkeypatch.setenv("ARCHEX_USAGE_METRICS", "on")
    db = tmp_path / "usage.sqlite"
    recorder = MetricsRecorder(db)
    recorder.record(
        UsageEvent(
            repo_root=tmp_path,
            surface="cli",
            tool_name="query",
            category="context_retrieval",
            tokens_returned=10,
            tokens_raw_equivalent=100,
        )
    )
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE host_sessions")
        conn.execute("PRAGMA user_version = 2")
    import_omp_session(session_file(), tmp_path, db_path=db)
    reporter = MetricsReporter(db)
    assert reporter.summary(source=tmp_path)["event_count"] == 1
    assert reporter.sessions(source=tmp_path)["session_count"] == 1
    with MetricsStore(db).connect() as conn, conn:
        conn.execute(
            "UPDATE host_sessions SET imported_at = ?",
            ((datetime.now(UTC) - timedelta(days=91)).isoformat(),),
        )
    assert reporter.sessions(source=tmp_path)["sessions"] == []
    assert reporter.summary(source=tmp_path)["event_count"] == 1


def test_cli_import_report_and_export_work_without_retaining_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_file: Callable[..., Path],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ARCHEX_USAGE_METRICS", "on")
    runner = CliRunner()
    imported = runner.invoke(cli, ["metrics", "import-session", str(session_file()), str(tmp_path)])
    assert imported.exit_code == 0, imported.output
    reported = runner.invoke(cli, ["metrics", "sessions", str(tmp_path), "--format", "json"])
    assert reported.exit_code == 0, reported.output
    row = json.loads(reported.output)["sessions"][0]
    assert row["total_tokens"] == 480
    assert math.isclose(row["host_cost_estimate_usd"], 0.007)
    assert metrics_db_path().exists()
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("private invalid body\n")
    failure = runner.invoke(cli, ["metrics", "import-session", str(malformed), str(tmp_path)])
    assert failure.exit_code != 0
    assert "private invalid body" not in failure.output
    assert "line 1" in failure.output
    other = tmp_path / "other"
    other.mkdir()
    imported_unknown = runner.invoke(
        cli, ["metrics", "import-session", str(session_file(usages=[None])), str(other)]
    )
    assert imported_unknown.exit_code == 0, imported_unknown.output
    unknown_report = runner.invoke(cli, ["metrics", "sessions", str(other), "--format", "json"])
    assert unknown_report.exit_code == 0, unknown_report.output
    assert json.loads(unknown_report.output)["sessions"][0]["input_tokens"] is None


def test_auxiliary_receipts_count_but_missing_duration_stays_unknown(
    session_file: Callable[..., Path],
) -> None:
    path = session_file()
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    auxiliary = {
        "type": "model_usage",
        "id": "auxiliary",
        "timestamp": entries[-1]["timestamp"],
        "purpose": "judgment",
        "usage": {
            "input": 7,
            "output": 3,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 10,
            "cost": {"total": 0.001},
        },
    }
    with path.open("a") as stream:
        stream.write(json.dumps(auxiliary) + "\n")
    receipt = parse_omp_session(path)
    assert receipt.request_count == receipt.receipt_count == 3
    assert receipt.totals["input_tokens"] == 307
    assert receipt.totals["output_tokens"] == 53
    assert receipt.totals["total_tokens"] == 490
    assert receipt.totals["request_duration_ms"] is None


def test_fork_import_excludes_inherited_requests_without_double_charging_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_file: Callable[..., Path],
) -> None:
    monkeypatch.setenv("ARCHEX_USAGE_METRICS", "on")
    db = tmp_path / "usage.sqlite"
    path = session_file()
    import_omp_session(path, tmp_path, db_path=db)
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    fork_time = datetime.fromisoformat(entries[-1]["timestamp"]) + timedelta(seconds=1)
    entries[1].update(
        id="fork", parentSession="/private/parent.jsonl", timestamp=fork_time.isoformat()
    )
    native = dict(entries[-1], id="native", timestamp=fork_time.isoformat())
    entries.append(native)
    fork = tmp_path / "fork.jsonl"
    fork.write_text("".join(json.dumps(e) + "\n" for e in entries))
    import_omp_session(fork, tmp_path, db_path=db)
    rows = MetricsReporter(db).sessions(source=tmp_path)["sessions"]
    assert sum(row["request_count"] for row in rows) == 3
    assert sum(row["input_tokens"] for row in rows) == 500


def test_import_bounds_reject_oversized_entries_files_and_non_regular_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_file: Callable[..., Path],
) -> None:
    import os

    monkeypatch.setattr("archex.metrics.sessions.MAX_ENTRY_BYTES", 64)
    path = session_file()
    with pytest.raises(ValueError, match="entry"):
        parse_omp_session(path)
    monkeypatch.setattr("archex.metrics.sessions.MAX_SESSION_BYTES", 1)
    with pytest.raises(ValueError, match="import limit"):
        parse_omp_session(path)
    if os.name == "posix":
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        with pytest.raises(ValueError, match="regular file"):
            parse_omp_session(fifo)
