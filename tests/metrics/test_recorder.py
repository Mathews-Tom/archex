from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from archex.metrics.health import read_metrics_health, record_metrics_failure
from archex.metrics.policy import METRICS_ENV, TRACE_ENV, set_metrics_enabled
from archex.metrics.recorder import MetricsRecorder, TraceDetails, UsageEvent
from archex.metrics.registry import RepoRegistry
from archex.metrics.storage import MetricsStore


def test_record_writes_anonymous_event_and_daily_aggregate(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"
    repo_root = _repo(tmp_path)
    set_metrics_enabled(True, db_path=db_path)

    MetricsRecorder(db_path).record(_event(repo_root))

    with MetricsStore(db_path).connect() as conn:
        event = conn.execute("SELECT * FROM usage_events").fetchone()
        daily = conn.execute("SELECT * FROM daily_usage").fetchone()
        traces = conn.execute("SELECT COUNT(*) FROM usage_traces").fetchone()[0]
        repo = conn.execute("SELECT repo_id, repo_root FROM repos").fetchone()

    assert event["repo_id"] == repo["repo_id"]
    assert event["tokens_returned"] == 10
    assert event["tokens_raw_equivalent"] == 100
    assert event["tokens_saved"] == 90
    assert event["savings_pct"] == 90.0
    assert event["whole_repo_tokens_avoided"] == 990
    assert event["trace_id"] is None
    assert daily["tokens_returned"] == 10
    assert daily["tokens_raw_equivalent"] == 100
    assert daily["tokens_saved"] == 90
    assert daily["event_count"] == 1
    assert traces == 0


def test_metrics_env_off_prevents_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "usage.sqlite"
    monkeypatch.setenv(METRICS_ENV, "off")

    MetricsRecorder(db_path).record(_event(_repo(tmp_path)))

    assert not db_path.exists()


def test_trace_rows_exist_only_after_explicit_trace_enable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "usage.sqlite"
    set_metrics_enabled(True, db_path=db_path)
    repo_root = _repo(tmp_path)
    event = _event(
        repo_root,
        trace=TraceDetails(
            query_text="find auth flow",
            returned_file_paths=["src/auth.py"],
            symbols=["login"],
            handles=["file:src/auth.py"],
            skipped_counts={"below_threshold": 2},
        ),
    )

    MetricsRecorder(db_path).record(event)
    monkeypatch.setenv(TRACE_ENV, "on")
    MetricsRecorder(db_path).record(event)

    with MetricsStore(db_path).connect() as conn:
        traces = conn.execute("SELECT * FROM usage_traces").fetchall()

    assert len(traces) == 1
    assert traces[0]["query_text"] == "find auth flow"
    assert "src/auth.py" in traces[0]["returned_file_paths"]
    assert "login" in traces[0]["symbols"]
    assert "file:src/auth.py" in traces[0]["handles"]
    assert "below_threshold" in traces[0]["skipped_counts"]


def test_trace_filter_rejects_code_output_and_prompt_fields() -> None:
    for key in ("source_code", "source_snippet", "rendered_output", "prompt_body"):
        with pytest.raises(ValueError, match="disallowed fields"):
            TraceDetails.from_mapping({key: "secret"})


def test_retention_prunes_raw_events_and_traces_but_keeps_daily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "usage.sqlite"
    set_metrics_enabled(True, db_path=db_path)
    repo_root = _repo(tmp_path)
    recorder = MetricsRecorder(db_path)
    monkeypatch.setenv(TRACE_ENV, "on")
    old = datetime.now(UTC) - timedelta(days=91)
    event = _event(repo_root, occurred_at=old, trace=TraceDetails(query_text="old query"))

    recorder.record(event)
    recorder.prune()

    with MetricsStore(db_path).connect() as conn:
        event_count = conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
        trace_count = conn.execute("SELECT COUNT(*) FROM usage_traces").fetchone()[0]
        daily_count = conn.execute("SELECT COUNT(*) FROM daily_usage").fetchone()[0]

    assert event_count == 0
    assert trace_count == 0
    assert daily_count == 1


def test_recorder_failures_are_non_fatal_and_record_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "usage.sqlite"
    set_metrics_enabled(True, db_path=db_path)

    def fail_get_or_create(self: RepoRegistry, repo_root: str | Path) -> object:
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(RepoRegistry, "get_or_create", fail_get_or_create)

    MetricsRecorder(db_path).record(_event(_repo(tmp_path)))

    health = read_metrics_health(db_path=db_path)
    assert health.status == "warning"
    assert health.last_failure_operation == "record"
    assert health.last_failure_message == "registry unavailable"


def test_successful_record_clears_prior_health_warning(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"
    set_metrics_enabled(True, db_path=db_path)
    record_metrics_failure("record", "Path does not exist: /repo", db_path=db_path)
    assert read_metrics_health(db_path=db_path).status == "warning"

    MetricsRecorder(db_path).record(_event(_repo(tmp_path)))

    health = read_metrics_health(db_path=db_path)
    assert health.status == "ok"
    assert health.last_failure_message is None


def _event(
    repo_root: Path,
    *,
    occurred_at: datetime | None = None,
    trace: TraceDetails | None = None,
) -> UsageEvent:
    return UsageEvent(
        repo_root=repo_root,
        surface="cli",
        tool_name="query",
        category="context_retrieval",
        tokens_returned=10,
        tokens_raw_equivalent=100,
        whole_repo_tokens=1000,
        occurred_at=occurred_at,
        file_count=2,
        freshness="clean",
        index_revision="rev",
        trace=trace,
    )


def _repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    return repo_root
