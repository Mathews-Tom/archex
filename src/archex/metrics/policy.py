"""Policy resolution for local metrics recording and detailed traces."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from archex.metrics.storage import (
    DEFAULT_RAW_EVENT_RETENTION_DAYS,
    DEFAULT_TRACE_RETENTION_DAYS,
    MetricsStore,
)

METRICS_ENV = "ARCHEX_USAGE_METRICS"
TRACE_ENV = "ARCHEX_USAGE_TRACE"


@dataclass(frozen=True)
class MetricsPolicy:
    metrics_enabled: bool
    trace_enabled: bool
    raw_event_retention_days: int
    trace_retention_days: int
    hosted_upload_enabled: bool = False


def resolve_metrics_policy(
    *,
    db_path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> MetricsPolicy:
    """Resolve persisted settings with environment variable overrides."""
    values = _settings(db_path)
    env_values = os.environ if env is None else env

    metrics_enabled = _bool_setting(values.get("metrics_enabled"), default=True)
    trace_enabled = _bool_setting(values.get("trace_enabled"), default=False)

    if env_values.get(METRICS_ENV, "").casefold() == "off":
        metrics_enabled = False
    trace_override = env_values.get(TRACE_ENV, "").casefold()
    if trace_override == "on":
        trace_enabled = True
    elif trace_override == "off":
        trace_enabled = False

    return MetricsPolicy(
        metrics_enabled=metrics_enabled,
        trace_enabled=trace_enabled,
        raw_event_retention_days=_int_setting(
            values.get("raw_event_retention_days"),
            default=DEFAULT_RAW_EVENT_RETENTION_DAYS,
        ),
        trace_retention_days=_int_setting(
            values.get("trace_retention_days"),
            default=DEFAULT_TRACE_RETENTION_DAYS,
        ),
        hosted_upload_enabled=_bool_setting(values.get("hosted_upload_enabled"), default=False),
    )


def set_metrics_enabled(enabled: bool, *, db_path: Path | None = None) -> None:
    _set_setting("metrics_enabled", _bool_value(enabled), db_path=db_path)


def set_trace_enabled(enabled: bool, *, db_path: Path | None = None) -> None:
    _set_setting("trace_enabled", _bool_value(enabled), db_path=db_path)


def _settings(db_path: Path | None) -> dict[str, str]:
    with MetricsStore(db_path).connect() as conn:
        rows = conn.execute("SELECT key, value FROM settings")
        return {str(row["key"]): str(row["value"]) for row in rows}


def _set_setting(key: str, value: str, *, db_path: Path | None) -> None:
    with MetricsStore(db_path).connect() as conn, conn:
        conn.execute(
            """
            INSERT INTO settings(key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, value),
        )


def _bool_setting(value: str | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _int_setting(value: str | None, *, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _bool_value(value: bool) -> str:
    return "true" if value else "false"
