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


def test_v1_ledger_migrates_to_targeted_read_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "usage.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE repos (
            repo_id TEXT PRIMARY KEY, repo_root TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL
        );
        CREATE TABLE usage_events (
            event_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL, repo_id TEXT NOT NULL,
            surface TEXT NOT NULL, tool_name TEXT NOT NULL, category TEXT NOT NULL,
            tokens_returned INTEGER NOT NULL, tokens_raw_equivalent INTEGER NOT NULL,
            tokens_saved INTEGER NOT NULL, savings_pct REAL NOT NULL,
            whole_repo_tokens INTEGER, whole_repo_tokens_avoided INTEGER,
            baseline_type TEXT NOT NULL, file_count INTEGER NOT NULL DEFAULT 0,
            freshness TEXT, index_revision TEXT, trace_id TEXT
        );
        CREATE TABLE daily_usage (
            day TEXT NOT NULL, repo_id TEXT NOT NULL, surface TEXT NOT NULL,
            tool_name TEXT NOT NULL, category TEXT NOT NULL,
            tokens_returned INTEGER NOT NULL DEFAULT 0,
            tokens_raw_equivalent INTEGER NOT NULL DEFAULT 0,
            tokens_saved INTEGER NOT NULL DEFAULT 0,
            whole_repo_tokens_avoided INTEGER NOT NULL DEFAULT 0,
            event_count INTEGER NOT NULL DEFAULT 0,
            first_event_at TEXT NOT NULL, last_event_at TEXT NOT NULL,
            PRIMARY KEY (day, repo_id, surface, tool_name, category)
        );
        CREATE TABLE usage_traces (
            trace_id TEXT PRIMARY KEY, event_id TEXT NOT NULL, expires_at TEXT NOT NULL,
            query_text TEXT, returned_file_paths TEXT NOT NULL DEFAULT '[]',
            symbols TEXT NOT NULL DEFAULT '[]', handles TEXT NOT NULL DEFAULT '[]',
            skipped_counts TEXT NOT NULL DEFAULT '{}',
            tokens_returned INTEGER NOT NULL, tokens_raw_equivalent INTEGER NOT NULL,
            tokens_saved INTEGER NOT NULL, savings_pct REAL NOT NULL,
            whole_repo_tokens INTEGER, whole_repo_tokens_avoided INTEGER,
            repo_id TEXT NOT NULL, index_revision TEXT
        );
        INSERT INTO repos VALUES ('r1', '/repo', 'repo', '2026-01-01', '2026-01-01');
        INSERT INTO usage_events(
            event_id, occurred_at, repo_id, surface, tool_name, category,
            tokens_returned, tokens_raw_equivalent, tokens_saved, savings_pct,
            whole_repo_tokens, whole_repo_tokens_avoided, baseline_type, file_count
        ) VALUES ('e1', '2026-01-01T00:00:00+00:00', 'r1', 'cli', 'query',
            'context_retrieval', 10, 100, 90, 90.0, 1000, 990, 'returned_full_files', 2);
        """
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    # Opening through MetricsStore runs the forward migration in place.
    with MetricsStore(db_path).connect() as conn:
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
        event_cols = {row["name"] for row in conn.execute("PRAGMA table_info(usage_events)")}
        daily_cols = {row["name"] for row in conn.execute("PRAGMA table_info(daily_usage)")}
        trace_cols = {row["name"] for row in conn.execute("PRAGMA table_info(usage_traces)")}
        row = conn.execute("SELECT * FROM usage_events WHERE event_id = 'e1'").fetchone()

    assert user_version == SCHEMA_VERSION
    targeted = {
        "tokens_targeted_read",
        "tokens_saved_vs_targeted_read",
        "savings_pct_vs_targeted_read",
    }
    assert targeted <= event_cols
    assert {"tokens_targeted_read", "tokens_saved_vs_targeted_read"} <= daily_cols
    assert targeted <= trace_cols
    # Pre-existing data survives; the new columns default to NULL on the legacy row.
    assert row["tokens_returned"] == 10
    assert row["tokens_raw_equivalent"] == 100
    assert row["tokens_targeted_read"] is None


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
