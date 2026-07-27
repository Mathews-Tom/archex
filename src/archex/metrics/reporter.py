"""Read-side reporting for the local usage metrics ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from archex.metrics.health import MetricsHealth, read_metrics_health
from archex.metrics.policy import MetricsPolicy, resolve_metrics_policy
from archex.metrics.recorder import MetricsRecorder
from archex.metrics.storage import MetricsStore


@dataclass(frozen=True)
class TimeWindow:
    label: str
    cutoff: str


_SURFACES: tuple[str, ...] = ("cli", "mcp", "python_api")


class MetricsReporter:
    """Builds summaries, repo lists, inspect rows, and export payloads."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._store = MetricsStore(db_path)

    def summary(
        self,
        *,
        source: str | Path | None = None,
        global_scope: bool = False,
        workspace: str | Path | None = None,
        since: str = "7d",
    ) -> dict[str, Any]:
        window = _window(since)
        MetricsRecorder(self._db_path).prune()
        policy = resolve_metrics_policy(db_path=self._db_path)
        health = read_metrics_health(db_path=self._db_path)
        with self._store.connect() as conn:
            repo_ids = _repo_ids(
                conn,
                source=source,
                global_scope=global_scope,
                workspace=workspace,
            )
            totals = _totals(conn, window, repo_ids)
        return _summary_payload(totals, policy=policy, health=health, window=window)

    def repos(self, *, since: str = "30d") -> dict[str, Any]:
        window = _window(since)
        MetricsRecorder(self._db_path).prune()
        policy = resolve_metrics_policy(db_path=self._db_path)
        health = read_metrics_health(db_path=self._db_path)
        with self._store.connect() as conn:
            rows = conn.execute(
                """
                SELECT r.repo_id, r.display_name, r.repo_root,
                    COUNT(e.event_id) AS event_count,
                    COALESCE(SUM(e.tokens_returned), 0) AS tokens_returned,
                    COALESCE(SUM(e.tokens_raw_equivalent), 0) AS tokens_raw_equivalent,
                    COALESCE(SUM(e.tokens_saved), 0) AS tokens_saved,
                    COALESCE(SUM(e.whole_repo_tokens_avoided), 0) AS whole_repo_tokens_avoided
                FROM repos r
                LEFT JOIN usage_events e
                    ON e.repo_id = r.repo_id AND e.occurred_at >= ?
                GROUP BY r.repo_id
                ORDER BY tokens_saved DESC, r.display_name ASC, r.repo_id ASC
                """,
                (window.cutoff,),
            ).fetchall()
        repos = [_repo_payload(row) for row in rows]
        return {
            "recording_enabled": policy.metrics_enabled,
            "trace_enabled": policy.trace_enabled,
            "window": window.label,
            "repo_count": len(repos),
            "repos": repos,
            "health_warning": health.warning,
        }

    def inspect(
        self,
        *,
        source: str | Path | None = None,
        since: str = "24h",
    ) -> dict[str, Any]:
        window = _window(since)
        MetricsRecorder(self._db_path).prune()
        policy = resolve_metrics_policy(db_path=self._db_path)
        health = read_metrics_health(db_path=self._db_path)
        with self._store.connect() as conn:
            repo_ids = _repo_ids(conn, source=source, global_scope=source is None, workspace=None)
            clauses = ["e.occurred_at >= ?"]
            params: list[object] = [window.cutoff]
            if repo_ids is not None:
                clauses.append(f"e.repo_id IN ({_placeholders(repo_ids)})")
                params.extend(repo_ids)
            rows = conn.execute(
                f"""
                SELECT e.*, t.query_text, t.returned_file_paths, t.symbols, t.handles,
                    t.skipped_counts
                FROM usage_events e
                LEFT JOIN usage_traces t ON t.event_id = e.event_id
                WHERE {" AND ".join(clauses)}
                ORDER BY e.occurred_at DESC, e.event_id ASC
                """,
                params,
            ).fetchall()
        return {
            "recording_enabled": policy.metrics_enabled,
            "trace_enabled": policy.trace_enabled,
            "window": window.label,
            "event_count": len(rows),
            "events": [_event_payload(row) for row in rows],
            "health_warning": health.warning,
        }

    def export(
        self,
        *,
        since: str = "90d",
        include_local_paths: bool = False,
    ) -> dict[str, Any]:
        window = _window(since)
        MetricsRecorder(self._db_path).prune()
        policy = resolve_metrics_policy(db_path=self._db_path)
        health = read_metrics_health(db_path=self._db_path)
        with self._store.connect() as conn:
            repos = [
                _export_repo(row, include_local_paths=include_local_paths)
                for row in conn.execute("SELECT * FROM repos ORDER BY display_name, repo_id")
            ]
            events = [
                _event_payload(row)
                for row in conn.execute(
                    """
                    SELECT * FROM usage_events
                    WHERE occurred_at >= ?
                    ORDER BY occurred_at, event_id
                    """,
                    (window.cutoff,),
                )
            ]
            daily = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM daily_usage WHERE last_event_at >= ? ORDER BY day, repo_id",
                    (window.cutoff,),
                )
            ]
            traces = [
                dict(row)
                for row in conn.execute("SELECT * FROM usage_traces ORDER BY expires_at, trace_id")
            ]
        return {
            "recording_enabled": policy.metrics_enabled,
            "trace_enabled": policy.trace_enabled,
            "window": window.label,
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "local_paths_included": include_local_paths,
            "repos": repos,
            "usage_events": events,
            "daily_usage": daily,
            "usage_traces": traces,
            "health_warning": health.warning,
        }


def render_summary_text(payload: dict[str, Any]) -> str:
    totals = payload["totals"]
    surface_mix = ", ".join(
        f"{surface} {totals['by_surface'][surface]['event_count']}" for surface in _SURFACES
    )
    lines = [
        "archex metrics",
        f"Window:                 {payload['window']}",
        f"Recording:              {_on_off(payload['recording_enabled'])}",
        f"Trace:                  {_on_off(payload['trace_enabled'])}",
        f"Events:                 {payload['event_count']}",
        f"Surface mix:            {surface_mix}",
        f"Returned tokens:        {totals['tokens_returned']}",
        f"Full-file tokens:       {totals['tokens_raw_equivalent']}",
        f"Targeted-read tokens:   {totals['tokens_targeted_read']}",
        f"Savings vs targeted:    "
        f"{totals['savings_pct_vs_targeted_read']:.1f}% (headline, vs realistic targeted read)",
        f"Savings vs full-file:   {totals['savings_pct']:.1f}% (compression, vs full-file paste)",
        f"Whole-repo avoided:     {totals['whole_repo_tokens_avoided']} (upper bound, not savings)",
    ]
    if payload["health_warning"]:
        lines.append(f"Metrics warning:        {payload['health_warning']}")
    return "\n".join(lines).rstrip() + "\n"


def render_repos_text(payload: dict[str, Any]) -> str:
    lines = [
        "archex metrics repos",
        f"Window: {payload['window']}",
        f"Recording: {_on_off(payload['recording_enabled'])}",
        f"Trace: {_on_off(payload['trace_enabled'])}",
    ]
    for repo in payload["repos"]:
        lines.append(
            f"- {repo['display_name']}: {repo['tokens_saved']} saved tokens, "
            f"{repo['event_count']} events"
        )
    if payload["health_warning"]:
        lines.append(f"Metrics warning: {payload['health_warning']}")
    return "\n".join(lines).rstrip() + "\n"


def render_inspect_text(payload: dict[str, Any]) -> str:
    lines = [
        "archex metrics inspect",
        f"Window: {payload['window']}",
        f"Events: {payload['event_count']}",
        f"Recording: {_on_off(payload['recording_enabled'])}",
        f"Trace: {_on_off(payload['trace_enabled'])}",
    ]
    for event in payload["events"]:
        lines.append(
            f"- {event['occurred_at']} {event['surface']}/{event['tool_name']}: "
            f"{event['tokens_saved']} saved tokens"
        )
    if payload["health_warning"]:
        lines.append(f"Metrics warning: {payload['health_warning']}")
    return "\n".join(lines).rstrip() + "\n"


def _summary_payload(
    totals: dict[str, Any],
    *,
    policy: MetricsPolicy,
    health: MetricsHealth,
    window: TimeWindow,
) -> dict[str, Any]:
    return {
        "recording_enabled": policy.metrics_enabled,
        "trace_enabled": policy.trace_enabled,
        "window": window.label,
        "event_count": totals["event_count"],
        "totals": totals,
        "health_warning": health.warning,
    }


def _repo_ids(
    conn: Any,
    *,
    source: str | Path | None,
    global_scope: bool,
    workspace: str | Path | None,
) -> list[str] | None:
    if global_scope:
        return None
    if workspace is not None:
        root = Path(workspace).expanduser().resolve()
        rows = conn.execute("SELECT repo_id, repo_root FROM repos").fetchall()
        return [
            str(row["repo_id"]) for row in rows if _is_relative_to(Path(row["repo_root"]), root)
        ]
    repo_root = Path("." if source is None else source).expanduser().resolve()
    row = conn.execute(
        "SELECT repo_id FROM repos WHERE repo_root = ?",
        (str(repo_root),),
    ).fetchone()
    return [] if row is None else [str(row["repo_id"])]


def _totals(conn: Any, window: TimeWindow, repo_ids: list[str] | None) -> dict[str, Any]:
    clauses = ["occurred_at >= ?"]
    params: list[object] = [window.cutoff]
    if repo_ids is not None:
        if not repo_ids:
            return _empty_totals()
        clauses.append(f"repo_id IN ({_placeholders(repo_ids)})")
        params.extend(repo_ids)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS event_count,
            COALESCE(SUM(tokens_returned), 0) AS tokens_returned,
            COALESCE(SUM(tokens_raw_equivalent), 0) AS tokens_raw_equivalent,
            COALESCE(SUM(tokens_saved), 0) AS tokens_saved,
            COALESCE(SUM(whole_repo_tokens_avoided), 0) AS whole_repo_tokens_avoided,
            COALESCE(SUM(tokens_targeted_read), 0) AS tokens_targeted_read,
            COALESCE(SUM(tokens_saved_vs_targeted_read), 0) AS tokens_saved_vs_targeted_read
        FROM usage_events
        WHERE {" AND ".join(clauses)}
        """,
        params,
    ).fetchone()
    totals: dict[str, Any] = {
        "event_count": int(row["event_count"]),
        "tokens_returned": int(row["tokens_returned"]),
        "tokens_raw_equivalent": int(row["tokens_raw_equivalent"]),
        "tokens_saved": int(row["tokens_saved"]),
        "whole_repo_tokens_avoided": int(row["whole_repo_tokens_avoided"]),
        "tokens_targeted_read": int(row["tokens_targeted_read"]),
    }
    totals["savings_pct"] = _savings_pct(totals["tokens_saved"], totals["tokens_raw_equivalent"])
    totals["savings_pct_vs_targeted_read"] = _savings_pct(
        int(row["tokens_saved_vs_targeted_read"]), totals["tokens_targeted_read"]
    )
    totals["by_surface"] = _surface_mix(conn, clauses, params)
    return totals


def _empty_totals() -> dict[str, Any]:
    return {
        "event_count": 0,
        "tokens_returned": 0,
        "tokens_raw_equivalent": 0,
        "tokens_saved": 0,
        "whole_repo_tokens_avoided": 0,
        "tokens_targeted_read": 0,
        "savings_pct": 0.0,
        "savings_pct_vs_targeted_read": 0.0,
        "by_surface": _empty_surface_mix(),
    }


def _surface_mix(conn: Any, clauses: list[str], params: list[object]) -> dict[str, dict[str, int]]:
    rows = conn.execute(
        f"""
        SELECT surface, COUNT(*) AS event_count,
            COALESCE(SUM(tokens_saved), 0) AS tokens_saved
        FROM usage_events
        WHERE {" AND ".join(clauses)}
        GROUP BY surface
        """,
        params,
    ).fetchall()
    counts = {
        str(row["surface"]): {
            "event_count": int(row["event_count"]),
            "tokens_saved": int(row["tokens_saved"]),
        }
        for row in rows
    }
    return {
        surface: counts.get(surface, {"event_count": 0, "tokens_saved": 0}) for surface in _SURFACES
    }


def _empty_surface_mix() -> dict[str, dict[str, int]]:
    return {surface: {"event_count": 0, "tokens_saved": 0} for surface in _SURFACES}


def _repo_payload(row: Any) -> dict[str, Any]:
    raw = int(row["tokens_raw_equivalent"])
    saved = int(row["tokens_saved"])
    return {
        "repo_id": str(row["repo_id"]),
        "display_name": str(row["display_name"]),
        "event_count": int(row["event_count"]),
        "tokens_returned": int(row["tokens_returned"]),
        "tokens_raw_equivalent": raw,
        "tokens_saved": saved,
        "savings_pct": _savings_pct(saved, raw),
        "whole_repo_tokens_avoided": int(row["whole_repo_tokens_avoided"]),
    }


def _event_payload(row: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event_id": str(row["event_id"]),
        "occurred_at": str(row["occurred_at"]),
        "repo_id": str(row["repo_id"]),
        "surface": str(row["surface"]),
        "tool_name": str(row["tool_name"]),
        "category": str(row["category"]),
        "tokens_returned": int(row["tokens_returned"]),
        "tokens_raw_equivalent": int(row["tokens_raw_equivalent"]),
        "tokens_saved": int(row["tokens_saved"]),
        "savings_pct": float(row["savings_pct"]),
        "whole_repo_tokens": _optional_int(row["whole_repo_tokens"]),
        "whole_repo_tokens_avoided": _optional_int(row["whole_repo_tokens_avoided"]),
        "baseline_type": str(row["baseline_type"]),
        "file_count": int(row["file_count"]),
        "freshness": _optional_str(row["freshness"]),
        "index_revision": _optional_str(row["index_revision"]),
        "trace_id": _optional_str(row["trace_id"]),
    }
    if _row_has_column(row, "query_text") and row["query_text"] is not None:
        payload["trace"] = {
            "query_text": row["query_text"],
            "returned_file_paths": json.loads(row["returned_file_paths"]),
            "symbols": json.loads(row["symbols"]),
            "handles": json.loads(row["handles"]),
            "skipped_counts": json.loads(row["skipped_counts"]),
        }
    return payload


def _export_repo(row: Any, *, include_local_paths: bool) -> dict[str, Any]:
    payload = {
        "repo_id": str(row["repo_id"]),
        "display_name": str(row["display_name"]),
        "first_seen_at": str(row["first_seen_at"]),
        "last_seen_at": str(row["last_seen_at"]),
    }
    if include_local_paths:
        payload["repo_root"] = str(row["repo_root"])
    return payload


def _window(value: str) -> TimeWindow:
    now = datetime.now(UTC)
    amount = int(value[:-1]) if len(value) > 1 else 0
    unit = value[-1:].lower()
    if amount < 0:
        raise ValueError("since must be non-negative")
    if unit == "d":
        delta = timedelta(days=amount)
    elif unit == "h":
        delta = timedelta(hours=amount)
    else:
        raise ValueError("since must use h or d suffix")
    return TimeWindow(value, (now - delta).isoformat(timespec="seconds"))


def _savings_pct(saved: int, raw: int) -> float:
    return (saved / raw * 100.0) if raw > 0 else 0.0


def _placeholders(values: list[str]) -> str:
    return ",".join("?" for _ in values)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _row_has_column(row: Any, key: str) -> bool:
    return key in tuple(row.keys())


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    return int(cast("str | int | float", value)) if value is not None else None


def _on_off(value: object) -> str:
    return "on" if value else "off"
