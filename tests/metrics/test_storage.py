from __future__ import annotations

import sqlite3
from pathlib import Path

from archex.metrics.storage import (
    DEFAULT_RAW_EVENT_RETENTION_DAYS,
    DEFAULT_TRACE_RETENTION_DAYS,
    SCHEMA_VERSION,
    MetricsStore,
    metrics_db_path,
)


def test_metrics_db_path_uses_machine_global_home(tmp_path: Path) -> None:
    assert metrics_db_path(home=tmp_path) == tmp_path / ".archex" / "usage.sqlite"


def test_bootstrap_creates_schema_and_default_settings(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "usage.sqlite"
    store = MetricsStore(db_path)

    with store.connect() as conn:
        table_names = _table_names(conn)
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        settings = {
            str(row["key"]): str(row["value"])
            for row in conn.execute("SELECT key, value FROM settings")
        }
        health = conn.execute("SELECT status FROM metrics_health WHERE id = 1").fetchone()

    assert user_version == SCHEMA_VERSION
    assert {
        "repos",
        "usage_events",
        "daily_usage",
        "usage_traces",
        "settings",
        "metrics_health",
    }.issubset(table_names)
    assert settings["metrics_enabled"] == "false"
    assert settings["trace_enabled"] == "false"
    assert settings["raw_event_retention_days"] == str(DEFAULT_RAW_EVENT_RETENTION_DAYS)
    assert settings["trace_retention_days"] == str(DEFAULT_TRACE_RETENTION_DAYS)
    assert settings["hosted_upload_enabled"] == "false"
    assert health["status"] == "ok"


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"
    store = MetricsStore(db_path)

    with store.connect() as conn, conn:
        conn.execute("UPDATE settings SET value = 'false' WHERE key = 'metrics_enabled'")

    with store.connect() as conn:
        metrics_enabled = conn.execute(
            "SELECT value FROM settings WHERE key = 'metrics_enabled'"
        ).fetchone()[0]
        repo_columns = [row["name"] for row in conn.execute("PRAGMA table_info(repos)")]

    assert metrics_enabled == "false"
    assert repo_columns == ["repo_id", "repo_root", "display_name", "first_seen_at", "last_seen_at"]


def test_usage_events_schema_has_no_privileged_payload_columns(tmp_path: Path) -> None:
    forbidden = {
        "query",
        "query_text",
        "file_path",
        "file_paths",
        "path_hash",
        "symbol",
        "symbols",
        "handle",
        "handles",
        "source_snippet",
        "rendered_output",
        "prompt_body",
        "remote_url",
        "org_name",
        "repo_name",
    }

    with MetricsStore(tmp_path / "usage.sqlite").connect() as conn:
        event_columns = {row["name"] for row in conn.execute("PRAGMA table_info(usage_events)")}

    assert event_columns.isdisjoint(forbidden)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
