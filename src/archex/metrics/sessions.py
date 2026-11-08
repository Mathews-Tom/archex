"""Explicit OMP session-receipt import; never persists transcript content."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from archex.metrics.policy import resolve_metrics_policy
from archex.metrics.registry import RepoRegistry
from archex.metrics.storage import MetricsStore

TOKEN_FIELDS = {
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_read_tokens": "cacheRead",
    "cache_write_tokens": "cacheWrite",
    "total_tokens": "totalTokens",
}
RECEIPT_FIELDS = (*TOKEN_FIELDS, "host_cost_estimate_usd", "request_duration_ms")
ASSOCIATION = "same_repo_time_window_not_session_attribution"
MAX_SESSION_BYTES = 256 * 1024 * 1024
MAX_ENTRY_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class OmpSessionReceipt:
    session_id: str
    started_at: datetime
    last_event_at: datetime
    request_count: int
    receipt_count: int
    totals: dict[str, int | float | None]


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("session entry timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("invalid session entry timestamp") from None
    if parsed.tzinfo is None:
        raise ValueError("session entry timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _number(value: object, *, integer: bool = False) -> int | float | None:
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError("receipt counters must be non-negative finite numbers or null")
    number = cast("int | float", value)
    if type(number) is int and number > 2**63 - 1:
        raise ValueError("receipt counter exceeds SQLite integer range")
    if number < 0 or not math.isfinite(number) or (integer and type(number) is not int):
        raise ValueError("receipt counters must be non-negative finite numbers or null")
    return number


def parse_omp_session(path: Path) -> OmpSessionReceipt:
    """Read numeric receipts from native assistant and off-transcript model calls.

    Forks exclude copied requests predating their new header. Subagent usage in
    tool results is outside this projection, so this is not whole-session spend.
    Missing counters stay unknown even when other requests have that counter.
    """
    session_id: str | None = None
    started_at: datetime | None = None
    last_event_at: datetime | None = None
    request_count = receipt_count = 0
    forked = False
    seen: set[str] = set()
    totals: dict[str, int | float | None] = dict.fromkeys(RECEIPT_FIELDS, 0)
    flags = os.O_RDONLY
    if os.name == "posix":
        flags |= os.O_NONBLOCK
    with os.fdopen(os.open(path, flags), "rb") as stream:
        file_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("session input must be a regular file")
        if file_stat.st_size > MAX_SESSION_BYTES:
            raise ValueError("session input exceeds the 256 MiB import limit")
        bytes_read = 0
        lines = iter(lambda: stream.readline(MAX_ENTRY_BYTES + 1), b"")
        for line_number, line in enumerate(lines, 1):
            bytes_read += len(line)
            if len(line) > MAX_ENTRY_BYTES or bytes_read > MAX_SESSION_BYTES:
                raise ValueError("session input exceeds the 16 MiB entry or 256 MiB file limit")
            if not line.strip():
                continue
            try:
                raw: object = json.loads(line)
            except (ValueError, RecursionError):
                raise ValueError(f"invalid session JSON at line {line_number}") from None
            if not isinstance(raw, dict):
                raise ValueError(f"session entry at line {line_number} is not an object")
            entry = cast("dict[str, object]", raw)
            if entry.get("type") == "session":
                if session_id is not None or entry.get("version") != 3:
                    raise ValueError("expected exactly one OMP v3 session header")
                identifier = entry.get("id")
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError("OMP session header requires an id")
                session_id = identifier
                started_at = last_event_at = _timestamp(entry.get("timestamp"))
                parent = entry.get("parentSession")
                if parent is not None and not isinstance(parent, str):
                    raise ValueError("OMP parentSession must be a string or null")
                forked = bool(parent)
                continue
            if entry.get("type") == "model_usage":
                message = entry
            elif entry.get("type") == "message":
                message_raw = entry.get("message")
                if not isinstance(message_raw, dict):
                    raise ValueError("session message must be an object")
                message = cast("dict[str, object]", message_raw)
                if message.get("role") != "assistant":
                    continue
            else:
                continue
            if started_at is None or last_event_at is None:
                raise ValueError("assistant receipt precedes the OMP session header")
            identifier = entry.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("assistant entries require distinct entry ids")
            seen.add(identifier)
            timestamp = _timestamp(entry.get("timestamp"))
            if timestamp < started_at:
                if forked:
                    continue
                raise ValueError("assistant receipt predates its session")
            last_event_at = max(last_event_at, timestamp)
            request_count += 1
            usage_raw = message.get("usage")
            if usage_raw is not None and not isinstance(usage_raw, dict):
                raise ValueError("assistant usage must be an object or null")
            usage = cast("dict[str, object]", usage_raw) if isinstance(usage_raw, dict) else {}
            if usage:
                receipt_count += 1
            cost_raw = usage.get("cost")
            if cost_raw is not None and not isinstance(cost_raw, dict):
                raise ValueError("assistant usage cost must be an object or null")
            cost = cast("dict[str, object]", cost_raw) if isinstance(cost_raw, dict) else {}
            values = {
                name: _number(usage.get(key), integer=True) for name, key in TOKEN_FIELDS.items()
            }
            values["host_cost_estimate_usd"] = _number(cost.get("total"))
            values["request_duration_ms"] = _number(message.get("duration"))
            for name, value in values.items():
                previous = totals[name]
                totals[name] = None if previous is None or value is None else previous + value
                _number(totals[name], integer=name in TOKEN_FIELDS)
    if session_id is None or started_at is None or last_event_at is None:
        raise ValueError("missing OMP v3 session header")
    if request_count == 0:
        # No observed request is not evidence of zero usage or zero cost.
        totals = dict.fromkeys(RECEIPT_FIELDS)
    return OmpSessionReceipt(
        session_id, started_at, last_event_at, request_count, receipt_count, totals
    )


def import_omp_session(
    path: Path, repo_root: Path, *, db_path: Path | None = None
) -> dict[str, object]:
    """Import an explicitly supplied saved session, idempotently and only when enabled."""
    policy = resolve_metrics_policy(db_path=db_path)
    if not policy.metrics_enabled:
        raise ValueError("metrics recording is disabled; run `archex metrics enable` first")
    if not repo_root.is_dir():
        raise ValueError("source must be an existing repository directory")
    receipt = parse_omp_session(path)
    repo = RepoRegistry(db_path).get_or_create(repo_root)
    # Random machine-local repo UUID salts the host identifier; neither the
    # transcript path nor the host's raw session id is persisted or exported.
    session_key = hashlib.sha256(
        json.dumps(["omp", repo.repo_id, receipt.session_id]).encode("utf-8")
    ).hexdigest()
    columns = (
        "session_key",
        "repo_id",
        "host",
        "started_at",
        "last_event_at",
        "imported_at",
        "request_count",
        "receipt_count",
        *RECEIPT_FIELDS,
    )
    values = (
        session_key,
        repo.repo_id,
        "omp",
        receipt.started_at.isoformat(),
        receipt.last_event_at.isoformat(),
        datetime.now(UTC).isoformat(),
        receipt.request_count,
        receipt.receipt_count,
        *(receipt.totals[name] for name in RECEIPT_FIELDS),
    )
    updates = ", ".join(f"{name} = excluded.{name}" for name in columns[4:])
    with MetricsStore(db_path).connect() as conn, conn:
        changed = conn.execute(
            f"""
            INSERT INTO host_sessions({", ".join(columns)})
            VALUES ({", ".join("?" for _ in columns)})
            ON CONFLICT(session_key) DO UPDATE SET {updates}
            WHERE excluded.last_event_at >= host_sessions.last_event_at
                AND excluded.request_count >= host_sessions.request_count
            """,
            values,
        ).rowcount
    return {"session_key": session_key, "updated": bool(changed), "host": "omp"}


def session_payload(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, object]:
    """Keep host expenditure and nearby retrieval counters separate, never subtract them."""
    activity = conn.execute(
        """
        SELECT COUNT(*) AS event_count, SUM(tokens_returned) AS tokens_returned
        FROM usage_events
        WHERE repo_id = ? AND julianday(occurred_at) >= julianday(?)
            AND julianday(occurred_at) <= julianday(?)
        """,
        (row["repo_id"], row["started_at"], row["last_event_at"]),
    ).fetchone()
    payload: dict[str, object] = dict(row)
    payload["token_basis"] = "omp_normalized_usage"
    payload["usage_scope"] = "saved_main_session_requests_excludes_subagents"
    payload["cost_basis"] = "host_estimate_not_billing_receipt"
    payload["duration_basis"] = "sum_of_host_request_durations_not_session_wall_time"
    payload["archex_activity"] = {
        "association": ASSOCIATION,
        "event_count": activity["event_count"],
        "tokens_returned": activity["tokens_returned"],
    }
    return payload


def render_sessions_text(payload: dict[str, object]) -> str:
    rows = cast("list[dict[str, object]]", payload["sessions"])
    lines = ["archex metrics sessions", f"Window: {payload['window']}"]
    lines.append("Coverage: saved main-session requests only; subagent usage is excluded.")
    for row in rows:
        lines.append(f"\n{row['session_key']} ({row['host']})")
        lines.append(f"Requests: {row['request_count']}; usage receipts: {row['receipt_count']}")
        for name in RECEIPT_FIELDS:
            value = row[name]
            lines.append(f"{name}: {'unknown' if value is None else value}")
        activity = cast("dict[str, object]", row["archex_activity"])
        lines.append(f"Same-repo/window archex events: {activity['event_count']} (not attribution)")
    lines.append("Cost is a host estimate, not an invoice; no causal savings are computed.")
    return "\n".join(lines) + "\n"
