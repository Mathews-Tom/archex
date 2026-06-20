"""Non-fatal metrics recorder for anonymous counters and opt-in traces."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from archex.metrics.health import record_metrics_failure
from archex.metrics.math import TokenSavings, compute_token_savings
from archex.metrics.policy import METRICS_ENV, MetricsPolicy, resolve_metrics_policy
from archex.metrics.registry import RepoRegistry
from archex.metrics.storage import MetricsStore

logger = logging.getLogger(__name__)

Surface = Literal["cli", "mcp", "python_api"]
Category = Literal["context_retrieval", "structural_tools"]
_ALLOWED_TRACE_KEYS = frozenset(
    {
        "query_text",
        "returned_file_paths",
        "symbols",
        "handles",
        "skipped_counts",
    }
)


@dataclass(frozen=True)
class TraceDetails:
    query_text: str | None = None
    returned_file_paths: list[str] = field(default_factory=lambda: [])
    symbols: list[str] = field(default_factory=lambda: [])
    handles: list[str] = field(default_factory=lambda: [])
    skipped_counts: dict[str, int] = field(default_factory=lambda: {})

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> TraceDetails:
        extra = set(value) - _ALLOWED_TRACE_KEYS
        if extra:
            names = ", ".join(sorted(extra))
            raise ValueError(f"trace details contain disallowed fields: {names}")
        return cls(
            query_text=_optional_str(value.get("query_text")),
            returned_file_paths=_string_list(value.get("returned_file_paths")),
            symbols=_string_list(value.get("symbols")),
            handles=_string_list(value.get("handles")),
            skipped_counts=_int_mapping(value.get("skipped_counts")),
        )


@dataclass(frozen=True)
class UsageEvent:
    repo_root: Path
    surface: Surface
    tool_name: str
    category: Category
    tokens_returned: int
    tokens_raw_equivalent: int
    whole_repo_tokens: int | None = None
    occurred_at: datetime | None = None
    file_count: int = 0
    freshness: str | None = None
    index_revision: str | None = None
    trace: TraceDetails | None = None


class MetricsRecorder:
    """Single writer for metrics event rows, aggregates, traces, pruning, and health."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._store = MetricsStore(db_path)
        self._registry = RepoRegistry(db_path)

    def record(self, event: UsageEvent, *, force: bool = False) -> None:
        """Record an event without raising into query/scout/MCP call paths."""
        try:
            if os.environ.get(METRICS_ENV, "").casefold() == "off":
                return
            policy = resolve_metrics_policy(db_path=self._db_path)
            if not policy.metrics_enabled:
                if not force:
                    return
                policy = replace(policy, metrics_enabled=True)
            self._record(event, policy)
        except Exception as exc:  # pragma: no cover - defensive non-fatal boundary
            logger.debug("metrics recording failed", exc_info=True)
            self._record_failure("record", exc)

    def prune(self) -> None:
        """Prune expired raw events and traces without raising into readers."""
        try:
            policy = resolve_metrics_policy(db_path=self._db_path)
            if not policy.metrics_enabled:
                return
            with self._store.connect() as conn, conn:
                self._prune(conn, policy)
        except Exception as exc:  # pragma: no cover - defensive non-fatal boundary
            logger.debug("metrics pruning failed", exc_info=True)
            self._record_failure("prune", exc)

    def _record(self, event: UsageEvent, policy: MetricsPolicy) -> None:
        registered = self._registry.get_or_create(event.repo_root)
        savings = compute_token_savings(
            tokens_returned=event.tokens_returned,
            tokens_raw_equivalent=event.tokens_raw_equivalent,
            whole_repo_tokens=event.whole_repo_tokens,
        )
        occurred_at = event.occurred_at or datetime.now(UTC)
        event_id = str(uuid4())
        trace_id = str(uuid4()) if policy.trace_enabled and event.trace is not None else None
        with self._store.connect() as conn, conn:
            self._insert_event(
                conn,
                event,
                savings,
                registered.repo_id,
                event_id,
                trace_id,
                occurred_at,
            )
            self._upsert_daily_usage(conn, event, savings, registered.repo_id, occurred_at)
            if trace_id is not None and event.trace is not None:
                self._insert_trace(
                    conn,
                    event,
                    event.trace,
                    savings,
                    registered.repo_id,
                    event_id,
                    trace_id,
                    occurred_at + timedelta(days=policy.trace_retention_days),
                )
            self._prune(conn, policy, now=occurred_at)
            self._clear_health(conn)

    def _record_failure(self, operation: str, exc: Exception) -> None:
        try:
            record_metrics_failure(operation, str(exc), db_path=self._db_path)
        except Exception:
            logger.debug("metrics health recording failed", exc_info=True)

    @staticmethod
    def _clear_health(conn: sqlite3.Connection) -> None:
        """Reset a prior warning once recording succeeds so health reflects current state."""
        conn.execute(
            """
            UPDATE metrics_health
            SET status = 'ok',
                last_failure_at = NULL,
                last_failure_operation = NULL,
                last_failure_message = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1 AND status != 'ok'
            """
        )

    @staticmethod
    def _insert_event(
        conn: sqlite3.Connection,
        event: UsageEvent,
        savings: TokenSavings,
        repo_id: str,
        event_id: str,
        trace_id: str | None,
        occurred_at: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO usage_events(
                event_id, occurred_at, repo_id, surface, tool_name, category,
                tokens_returned, tokens_raw_equivalent, tokens_saved, savings_pct,
                whole_repo_tokens, whole_repo_tokens_avoided, baseline_type, file_count,
                freshness, index_revision, trace_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                _format_dt(occurred_at),
                repo_id,
                event.surface,
                event.tool_name,
                event.category,
                savings.tokens_returned,
                savings.tokens_raw_equivalent,
                savings.tokens_saved,
                savings.savings_pct,
                savings.whole_repo_tokens,
                savings.whole_repo_tokens_avoided,
                savings.baseline_type,
                event.file_count,
                event.freshness,
                event.index_revision,
                trace_id,
            ),
        )

    @staticmethod
    def _upsert_daily_usage(
        conn: sqlite3.Connection,
        event: UsageEvent,
        savings: TokenSavings,
        repo_id: str,
        occurred_at: datetime,
    ) -> None:
        occurred = _format_dt(occurred_at)
        conn.execute(
            """
            INSERT INTO daily_usage(
                day, repo_id, surface, tool_name, category,
                tokens_returned, tokens_raw_equivalent, tokens_saved,
                whole_repo_tokens_avoided, event_count, first_event_at, last_event_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(day, repo_id, surface, tool_name, category) DO UPDATE SET
                tokens_returned = tokens_returned + excluded.tokens_returned,
                tokens_raw_equivalent = tokens_raw_equivalent + excluded.tokens_raw_equivalent,
                tokens_saved = tokens_saved + excluded.tokens_saved,
                whole_repo_tokens_avoided = (
                    whole_repo_tokens_avoided + excluded.whole_repo_tokens_avoided
                ),
                event_count = event_count + 1,
                first_event_at = min(first_event_at, excluded.first_event_at),
                last_event_at = max(last_event_at, excluded.last_event_at)
            """,
            (
                occurred_at.date().isoformat(),
                repo_id,
                event.surface,
                event.tool_name,
                event.category,
                savings.tokens_returned,
                savings.tokens_raw_equivalent,
                savings.tokens_saved,
                savings.whole_repo_tokens_avoided or 0,
                occurred,
                occurred,
            ),
        )

    @staticmethod
    def _insert_trace(
        conn: sqlite3.Connection,
        event: UsageEvent,
        trace: TraceDetails,
        savings: TokenSavings,
        repo_id: str,
        event_id: str,
        trace_id: str,
        expires_at: datetime,
    ) -> None:
        conn.execute(
            """
            INSERT INTO usage_traces(
                trace_id, event_id, expires_at, query_text, returned_file_paths, symbols,
                handles, skipped_counts, tokens_returned, tokens_raw_equivalent,
                tokens_saved, savings_pct, whole_repo_tokens, whole_repo_tokens_avoided,
                repo_id, index_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace_id,
                event_id,
                _format_dt(expires_at),
                trace.query_text,
                json.dumps(trace.returned_file_paths, sort_keys=True),
                json.dumps(trace.symbols, sort_keys=True),
                json.dumps(trace.handles, sort_keys=True),
                json.dumps(trace.skipped_counts, sort_keys=True),
                savings.tokens_returned,
                savings.tokens_raw_equivalent,
                savings.tokens_saved,
                savings.savings_pct,
                savings.whole_repo_tokens,
                savings.whole_repo_tokens_avoided,
                repo_id,
                event.index_revision,
            ),
        )

    @staticmethod
    def _prune(
        conn: sqlite3.Connection,
        policy: MetricsPolicy,
        *,
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(UTC)
        raw_cutoff = _format_dt(current - timedelta(days=policy.raw_event_retention_days))
        conn.execute("DELETE FROM usage_traces WHERE expires_at < ?", (_format_dt(current),))
        conn.execute("DELETE FROM usage_events WHERE occurred_at < ?", (raw_cutoff,))


def _format_dt(value: datetime) -> str:
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return normalized.astimezone(UTC).isoformat(timespec="seconds")


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("trace list fields must be lists")
    return [str(item) for item in cast("list[object]", value)]


def _int_mapping(value: object) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("skipped_counts must be a mapping")
    result: dict[str, int] = {}
    for key, item in cast("dict[object, object]", value).items():
        if not isinstance(item, int):
            raise ValueError("skipped_counts values must be integers")
        result[str(key)] = item
    return result
