"""SQLite storage bootstrap for the machine-local usage metrics ledger."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_RAW_EVENT_RETENTION_DAYS = 90
DEFAULT_TRACE_RETENTION_DAYS = 14


def metrics_db_path(*, home: Path | None = None) -> Path:
    """Return the default machine-local metrics ledger path."""
    root = home if home is not None else Path.home()
    return root.expanduser() / ".archex" / "usage.sqlite"


class MetricsStore:
    """Owns SQLite connection setup and schema management for metrics storage."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path).expanduser() if db_path is not None else metrics_db_path()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            _configure_connection(conn)
            self.bootstrap(conn)
            yield conn
        finally:
            conn.close()

    def bootstrap(self, conn: sqlite3.Connection) -> None:
        """Create or migrate metrics tables in a single local SQLite ledger."""
        with conn:
            conn.execute("PRAGMA user_version")
            _create_schema(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            _seed_settings(conn)


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS repos (
            repo_id TEXT PRIMARY KEY,
            repo_root TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS usage_events (
            event_id TEXT PRIMARY KEY,
            occurred_at TEXT NOT NULL,
            repo_id TEXT NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE,
            surface TEXT NOT NULL CHECK (surface IN ('cli', 'mcp', 'python_api')),
            tool_name TEXT NOT NULL,
            category TEXT NOT NULL CHECK (category IN ('context_retrieval', 'structural_tools')),
            tokens_returned INTEGER NOT NULL CHECK (tokens_returned >= 0),
            tokens_raw_equivalent INTEGER NOT NULL CHECK (tokens_raw_equivalent >= 0),
            tokens_saved INTEGER NOT NULL CHECK (tokens_saved >= 0),
            savings_pct REAL NOT NULL CHECK (savings_pct >= 0),
            whole_repo_tokens INTEGER CHECK (whole_repo_tokens IS NULL OR whole_repo_tokens >= 0),
            whole_repo_tokens_avoided INTEGER CHECK (
                whole_repo_tokens_avoided IS NULL OR whole_repo_tokens_avoided >= 0
            ),
            baseline_type TEXT NOT NULL,
            file_count INTEGER NOT NULL DEFAULT 0 CHECK (file_count >= 0),
            freshness TEXT,
            index_revision TEXT,
            trace_id TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_usage_events_repo_time
            ON usage_events(repo_id, occurred_at);
        CREATE INDEX IF NOT EXISTS idx_usage_events_time
            ON usage_events(occurred_at);

        CREATE TABLE IF NOT EXISTS daily_usage (
            day TEXT NOT NULL,
            repo_id TEXT NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE,
            surface TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            category TEXT NOT NULL,
            tokens_returned INTEGER NOT NULL DEFAULT 0,
            tokens_raw_equivalent INTEGER NOT NULL DEFAULT 0,
            tokens_saved INTEGER NOT NULL DEFAULT 0,
            whole_repo_tokens_avoided INTEGER NOT NULL DEFAULT 0,
            event_count INTEGER NOT NULL DEFAULT 0,
            first_event_at TEXT NOT NULL,
            last_event_at TEXT NOT NULL,
            PRIMARY KEY (day, repo_id, surface, tool_name, category)
        );

        CREATE INDEX IF NOT EXISTS idx_daily_usage_repo_day
            ON daily_usage(repo_id, day);
        CREATE INDEX IF NOT EXISTS idx_daily_usage_day
            ON daily_usage(day);

        CREATE TABLE IF NOT EXISTS usage_traces (
            trace_id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL REFERENCES usage_events(event_id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            query_text TEXT,
            returned_file_paths TEXT NOT NULL DEFAULT '[]',
            symbols TEXT NOT NULL DEFAULT '[]',
            handles TEXT NOT NULL DEFAULT '[]',
            skipped_counts TEXT NOT NULL DEFAULT '{}',
            tokens_returned INTEGER NOT NULL CHECK (tokens_returned >= 0),
            tokens_raw_equivalent INTEGER NOT NULL CHECK (tokens_raw_equivalent >= 0),
            tokens_saved INTEGER NOT NULL CHECK (tokens_saved >= 0),
            savings_pct REAL NOT NULL CHECK (savings_pct >= 0),
            whole_repo_tokens INTEGER CHECK (whole_repo_tokens IS NULL OR whole_repo_tokens >= 0),
            whole_repo_tokens_avoided INTEGER CHECK (
                whole_repo_tokens_avoided IS NULL OR whole_repo_tokens_avoided >= 0
            ),
            repo_id TEXT NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE,
            index_revision TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_usage_traces_expires_at
            ON usage_traces(expires_at);

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS metrics_health (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            status TEXT NOT NULL CHECK (status IN ('ok', 'warning')),
            last_failure_at TEXT,
            last_failure_operation TEXT,
            last_failure_message TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


def _seed_settings(conn: sqlite3.Connection) -> None:
    settings = {
        "metrics_enabled": "true",
        "trace_enabled": "false",
        "raw_event_retention_days": str(DEFAULT_RAW_EVENT_RETENTION_DAYS),
        "trace_retention_days": str(DEFAULT_TRACE_RETENTION_DAYS),
        "cost_per_million_tokens": "",
        "hosted_upload_enabled": "false",
    }
    conn.executemany(
        "INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)",
        settings.items(),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO metrics_health(
            id, status, last_failure_at, last_failure_operation, last_failure_message
        ) VALUES (1, 'ok', NULL, NULL, NULL)
        """
    )
