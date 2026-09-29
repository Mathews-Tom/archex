"""Diagnostics log and latency budget shared by every archex client hook.

Standard library only: hook entry points import this before they know whether
a call is theirs to handle, so it must not pull in the index, the tokenizer,
or pydantic.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_HOOK_TIMEOUT_SECONDS = 0.5
TIMEOUT_ENV_VAR = "ARCHEX_HOOK_TIMEOUT_SECONDS"

DEFAULT_DIAGNOSTICS_LOG_PATH = Path.home() / ".archex" / "hook-diagnostics.log"
DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG"


def hook_timeout_seconds() -> float:
    """The hook's wall-clock budget: `ARCHEX_HOOK_TIMEOUT_SECONDS` when positive."""
    raw = os.environ.get(TIMEOUT_ENV_VAR)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return DEFAULT_HOOK_TIMEOUT_SECONDS


def diagnostics_log_path() -> Path:
    raw = os.environ.get(DIAGNOSTICS_LOG_ENV_VAR)
    return Path(raw).expanduser() if raw else DEFAULT_DIAGNOSTICS_LOG_PATH


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_diagnostic(kind: str, *, detail: str, cwd: str | None = None) -> None:
    """Append one JSON line to the diagnostics log; never raises."""
    try:
        path = diagnostics_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, str] = {
            "timestamp": utc_now_iso(),
            "kind": kind,
            "detail": detail,
        }
        if cwd is not None:
            entry["cwd"] = cwd
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # diagnostics logging must never raise into a hook's exit path
