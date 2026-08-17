"""Claude Code SessionStart hook for bounded Archex project-session context.

This opt-in adapter accepts only Claude Code's ``startup`` and ``resume``
SessionStart sources. It translates the event's ``cwd`` into the shared
session-primer renderer and emits its receipt-bearing content as
``additionalContext``. It never captures records, indexes a repository, or
blocks a session. Any malformed payload, stale/missing index, renderer error,
or timeout is logged locally and produces no hook output.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

from archex.integrations.hook import hook_timeout_seconds, log_diagnostic
from archex.session import render_session_primer

if TYPE_CHECKING:
    from archex.session.models import SessionPrimer

SESSION_START_EVENT_NAME = "SessionStart"
SUPPORTED_SESSION_SOURCES: frozenset[str] = frozenset({"startup", "resume"})
SESSION_START_MATCHER = "|".join(sorted(SUPPORTED_SESSION_SOURCES))


def main() -> None:
    """Read one Claude Code hook payload and exit successfully on every path."""
    try:
        _main_impl()
    except BaseException as exc:  # noqa: BLE001 - session startup must never be blocked
        log_diagnostic("session_hook_unhandled_exception", detail=repr(exc))
    finally:
        sys.stdout.flush()
        os._exit(0)


def _main_impl() -> None:
    payload = _parse_payload(sys.stdin.read())
    if payload is None:
        return
    result = handle_session_start(payload)
    if result is not None:
        sys.stdout.write(json.dumps(result))


def _parse_payload(raw: str) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        log_diagnostic("session_hook_malformed_payload", detail=f"invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        log_diagnostic("session_hook_malformed_payload", detail="payload is not a JSON object")
        return None
    return cast("dict[str, Any]", payload)


def handle_session_start(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Render a primer for a supported SessionStart payload or return no output."""
    source = payload.get("source")
    if not isinstance(source, str) or source not in SUPPORTED_SESSION_SOURCES:
        return None
    cwd = payload.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        log_diagnostic("session_hook_malformed_payload", detail="cwd is missing or not a string")
        return None

    primer = _render_with_timeout(cwd)
    if primer is None or not primer.ready or not primer.content:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": SESSION_START_EVENT_NAME,
            "additionalContext": primer.content,
        }
    }


def _render_with_timeout(cwd: str) -> SessionPrimer | None:
    timeout = hook_timeout_seconds()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="archex-session-hook")
    future = executor.submit(render_session_primer, cwd)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        log_diagnostic("session_hook_timeout", detail=f"render exceeded {timeout}s", cwd=cwd)
        return None
    except Exception as exc:  # noqa: BLE001 - session startup must never be blocked
        log_diagnostic("session_hook_render_error", detail=repr(exc), cwd=cwd)
        return None
    finally:
        # ``main`` uses ``os._exit`` so an over-budget renderer cannot delay startup.
        executor.shutdown(wait=False, cancel_futures=False)


if __name__ == "__main__":
    main()
