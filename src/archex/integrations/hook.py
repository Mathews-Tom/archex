"""Symbol-search lookup engine used by the Cursor diagnostics hook.

This module has no hook entry point. Claude Code, Codex, OpenCode, and
oh-my-pi/Pi annotate search results through `archex.annotate` (see
`archex.integrations.annotate_hook`), and the old pattern-search
`PreToolUse` process that lived here is gone. What still uses this module:

- `archex.integrations.cursor_hook`, which imports `lookup_with_timeout` and
  `IDENTIFIER_TOKEN_RE`;
- `_lookup`, which the index-provenance tests call directly.

Contract of `lookup_with_timeout` (M19 — non-blocking client hook integration):

- It never raises. A missing/stale index, a timeout, or an internal error all
  degrade to `None`; the failure is instead written to the local diagnostics
  log (`archex.integrations.diagnostics.log_diagnostic`). Failures are loud in
  diagnostics, silent to the agent flow — this is the one place "fail fast" is
  the wrong default, because blocking or erroring the agent over a
  context-augmentation failure would be worse than returning no extra context.
- The lookup runs under a hard wall-clock timeout (`hook_timeout_seconds`,
  500ms by default). A lookup still running past the budget is abandoned in
  place (its thread is not joined); the calling hook exits through `os._exit`,
  so a stuck lookup can never block the agent loop.
- Every context block is stamped with a freshness/receipt marker (the index
  revision and a UTC generation timestamp) so a downstream agent can tell how
  current it is, mirroring the receipt contract used by `query`/`scout`.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from archex.index.store import IndexStore
from archex.integrations.diagnostics import hook_timeout_seconds, log_diagnostic, utc_now_iso
from archex.receipt import index_revision_from_store
from archex.status import inspect_project_status

if TYPE_CHECKING:
    from archex.models import CodeChunk

MAX_RESULTS = 5

IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def lookup_with_timeout(cwd: str, query: str) -> str | None:
    timeout = hook_timeout_seconds()
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
        # Don't wait for a timed-out lookup thread — the hook's os._exit reclaims it.
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
        f"[archex receipt] index_revision={revision[:12]} generated_at={utc_now_iso()}",
        f"archex symbol matches for grep/glob pattern {query!r}:",
    ]
    for chunk in chunks:
        label = chunk.symbol_name or Path(chunk.file_path).name
        lines.append(f"- {label} — {chunk.file_path}:{chunk.start_line}-{chunk.end_line}")
    return "\n".join(lines)
