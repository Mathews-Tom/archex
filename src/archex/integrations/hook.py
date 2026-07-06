"""Claude Code PreToolUse hook: augments Grep/Glob with archex search results.

Contract (M19 — non-blocking client hook integration):

- Only `Grep` and `Glob` tool calls are inspected (`AUGMENTED_TOOLS`). Every other
  tool, including `Read`, is never touched — this hook must never interfere with
  read-before-edit semantics.
- Every code path exits 0. A missing/stale index, a timeout, a malformed payload,
  or an internal error all degrade to a silent no-op from the agent's point of
  view; the failure is instead written to a local diagnostics log
  (`_diagnostics_log_path`). Failures are loud in diagnostics, silent to the
  agent flow — this is the one place "fail fast" is the wrong default, because
  blocking or erroring the agent over a context-augmentation failure would be
  worse than returning no extra context.
- The archex lookup itself runs under a hard wall-clock timeout
  (`_hook_timeout_seconds`, ~500ms by default). A lookup that is still running
  past the budget is abandoned in place (its thread is not joined) and the
  process exits immediately via `os._exit` so a stuck lookup can never block the
  agent loop.
- Every injected `additionalContext` block is stamped with a freshness/receipt
  marker (the index revision and a UTC generation timestamp) so a downstream
  agent can tell how current the injected context is, mirroring the receipt
  contract used by `query`/`scout`.

Invoked as a subprocess: `python -m archex.integrations.hook`, reading the
PreToolUse JSON payload from stdin and writing the hook JSON output to stdout.
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from archex.index.store import IndexStore
from archex.receipt import index_revision_from_store
from archex.status import inspect_project_status

if TYPE_CHECKING:
    from archex.models import CodeChunk

HOOK_EVENT_NAME = "PreToolUse"

#: Tools this hook augments. Read is deliberately excluded — see module docstring.
AUGMENTED_TOOLS: frozenset[str] = frozenset({"Grep", "Glob"})

#: Claude Code hook `matcher` value that selects exactly `AUGMENTED_TOOLS`. The
#: installer (`archex.client_setup`) reuses this constant so the installed
#: config and this module's own runtime filter can never drift apart.
HOOK_MATCHER = "|".join(sorted(AUGMENTED_TOOLS))

DEFAULT_HOOK_TIMEOUT_SECONDS = 0.5
_TIMEOUT_ENV_VAR = "ARCHEX_HOOK_TIMEOUT_SECONDS"

DEFAULT_DIAGNOSTICS_LOG_PATH = Path.home() / ".archex" / "hook-diagnostics.log"
_DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG"

MAX_RESULTS = 5

IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def main() -> None:
    """Entry point for `python -m archex.integrations.hook`.

    Guarantees exit 0 on every path, including an exception this module did not
    anticipate — see the module docstring's non-blocking contract. `os._exit` is
    used instead of a normal return so a lookup thread still running past the
    timeout can never delay process exit.
    """
    try:
        _main_impl()
    except BaseException as exc:  # noqa: BLE001 - the non-blocking contract requires this
        log_diagnostic("unhandled_exception", detail=repr(exc))
    os._exit(0)


def _main_impl() -> None:
    raw = sys.stdin.read()
    payload = _parse_payload(raw)
    if payload is None:
        return
    result = handle_pre_tool_use(payload)
    if result is None:
        return
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


def _parse_payload(raw: str) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        log_diagnostic("malformed_payload", detail=f"invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        log_diagnostic("malformed_payload", detail="payload is not a JSON object")
        return None
    return cast("dict[str, Any]", payload)


def handle_pre_tool_use(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Core handler: a parsed PreToolUse payload in, hook JSON output or `None` out.

    `None` means "no decision" — Claude Code proceeds exactly as if this hook
    were not installed. Never raises: any failure degrades to `None` plus a
    diagnostics log line.
    """
    try:
        tool_name = payload.get("tool_name")
        if tool_name not in AUGMENTED_TOOLS:
            return None
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            log_diagnostic("malformed_payload", detail="tool_input is not a JSON object")
            return None
        query = _extract_query(tool_name, cast("dict[str, Any]", tool_input))
        if not query:
            return None
        cwd_raw = payload.get("cwd")
        cwd = cwd_raw if isinstance(cwd_raw, str) and cwd_raw else os.getcwd()
        context = lookup_with_timeout(cwd, query)
        if context is None:
            return None
        return _build_output(context)
    except Exception as exc:  # noqa: BLE001 - degrade to no-op, never raise
        log_diagnostic("internal_error", detail=repr(exc))
        return None


def _extract_query(tool_name: str, tool_input: dict[str, Any]) -> str:
    pattern = tool_input.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        return ""
    if tool_name == "Grep":
        return pattern.strip()
    # Glob patterns are filesystem globs, not search terms — pull identifier-like
    # tokens out of them so "**/*_service.py" becomes a useful symbol query.
    return " ".join(IDENTIFIER_TOKEN_RE.findall(pattern))


def lookup_with_timeout(cwd: str, query: str) -> str | None:
    timeout = _hook_timeout_seconds()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="archex-hook")
    future = executor.submit(_lookup, cwd, query)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        log_diagnostic("timeout", detail=f"lookup exceeded {timeout}s", cwd=cwd)
        return None
    except Exception as exc:  # noqa: BLE001 - degrade to no-op, never raise
        log_diagnostic("lookup_error", detail=repr(exc), cwd=cwd)
        return None
    finally:
        # Don't wait for a timed-out lookup thread — main()'s os._exit reclaims it.
        executor.shutdown(wait=False, cancel_futures=False)


def _lookup(cwd: str, query: str) -> str | None:
    try:
        status = inspect_project_status(cwd)
    except ValueError as exc:
        log_diagnostic("status_error", detail=str(exc), cwd=cwd)
        return None
    if status.state != "fresh":
        log_diagnostic("index_not_fresh", detail=f"state={status.state}", cwd=cwd)
        return None

    store = IndexStore(status.index_path)
    try:
        chunks = store.search_symbols(query, limit=MAX_RESULTS)
        if not chunks:
            return None
        revision = index_revision_from_store(store)
    finally:
        store.close()
    return _render_context(query, chunks, revision)


def _render_context(query: str, chunks: list[CodeChunk], revision: str) -> str:
    lines = [
        f"[archex receipt] index_revision={revision[:12]} generated_at={_utc_now_iso()}",
        f"archex symbol matches for grep/glob pattern {query!r}:",
    ]
    for chunk in chunks:
        label = chunk.symbol_name or Path(chunk.file_path).name
        lines.append(f"- {label} — {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
    return "\n".join(lines)


def _build_output(context: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": HOOK_EVENT_NAME,
            "additionalContext": context,
        }
    }


def _hook_timeout_seconds() -> float:
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    return DEFAULT_HOOK_TIMEOUT_SECONDS


def _diagnostics_log_path() -> Path:
    raw = os.environ.get(_DIAGNOSTICS_LOG_ENV_VAR)
    return Path(raw).expanduser() if raw else DEFAULT_DIAGNOSTICS_LOG_PATH


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_diagnostic(kind: str, *, detail: str, cwd: str | None = None) -> None:
    try:
        path = _diagnostics_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, str] = {
            "timestamp": _utc_now_iso(),
            "kind": kind,
            "detail": detail,
        }
        if cwd is not None:
            entry["cwd"] = cwd
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # diagnostics logging must never raise into the hook's exit path


if __name__ == "__main__":
    main()
