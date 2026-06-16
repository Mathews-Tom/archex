"""Metrics health state persisted in the local usage ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from archex.metrics.storage import MetricsStore


@dataclass(frozen=True)
class MetricsHealth:
    status: str
    last_failure_at: str | None
    last_failure_operation: str | None
    last_failure_message: str | None

    @property
    def warning(self) -> str:
        if self.status == "ok" or self.last_failure_message is None:
            return ""
        operation = self.last_failure_operation or "metrics"
        return f"{operation}: {self.last_failure_message}"


def read_metrics_health(*, db_path: Path | None = None) -> MetricsHealth:
    with MetricsStore(db_path).connect() as conn:
        row = conn.execute(
            """
            SELECT status, last_failure_at, last_failure_operation, last_failure_message
            FROM metrics_health WHERE id = 1
            """
        ).fetchone()
    if row is None:
        return MetricsHealth("ok", None, None, None)
    return MetricsHealth(
        status=str(row["status"]),
        last_failure_at=_optional_str(row["last_failure_at"]),
        last_failure_operation=_optional_str(row["last_failure_operation"]),
        last_failure_message=_optional_str(row["last_failure_message"]),
    )


def record_metrics_failure(
    operation: str,
    message: str,
    *,
    db_path: Path | None = None,
) -> None:
    with MetricsStore(db_path).connect() as conn, conn:
        conn.execute(
            """
            INSERT INTO metrics_health(
                id, status, last_failure_at, last_failure_operation,
                last_failure_message, updated_at
            ) VALUES (1, 'warning', ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                status = 'warning',
                last_failure_at = excluded.last_failure_at,
                last_failure_operation = excluded.last_failure_operation,
                last_failure_message = excluded.last_failure_message,
                updated_at = CURRENT_TIMESTAMP
            """,
            (_utc_now(), operation, message[:500]),
        )


def clear_metrics_health(*, db_path: Path | None = None) -> None:
    with MetricsStore(db_path).connect() as conn, conn:
        conn.execute(
            """
            UPDATE metrics_health
            SET status = 'ok',
                last_failure_at = NULL,
                last_failure_operation = NULL,
                last_failure_message = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """
        )


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
