"""Bounded, atomically written post-edit state under the project cache dir.

The state file (`.archex/post-edit-state.json`) is the one piece of shared
mutable state R21 introduces. Several client hook processes can run
concurrently -- an agent that edits three files in one turn produces three
independent subprocesses -- so every mutation here is a read-modify-write
under an exclusive lock, published with a temp-file rename.

Failure discipline mirrors the existing hook contract: nothing in this module
raises into a client's edit path. A missing file yields a default state; an
unreadable, malformed, or unknown-version file yields a default state plus a
diagnostics line; a lock that cannot be acquired inside the deadline abandons
the write rather than delaying the agent. Callers therefore never need a
try/except around these functions to keep a hook non-blocking.

POSIX only: locking uses ``fcntl.flock``. archex publishes no Windows wheel
classifier and its CI matrix is Linux plus macOS.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from archex.integrations.hook import log_diagnostic
from archex.post_edit.models import (
    MAX_EVENT_PATHS,
    MAX_PATH_LENGTH,
    MAX_PENDING_PATHS,
    POST_EDIT_STATE_VERSION,
    PostEditEvent,
    PostEditState,
    PostEditStatus,
)
from archex.project import ProjectState

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

STATE_FILENAME = "post-edit-state.json"
LOCK_FILENAME = "post-edit-state.lock"

#: Wall-clock budget for acquiring the state lock. Deliberately short: the
#: critical section is a small read, merge, and rename, so a wait longer than
#: this means a stuck peer, and delaying the agent is worse than skipping one
#: state update the working-tree delta would re-derive anyway.
LOCK_TIMEOUT_SECONDS = 2.0
_LOCK_POLL_SECONDS = 0.01


def post_edit_state_path(repo_root: Path) -> Path:
    """Path of the post-edit state document for ``repo_root``."""
    return ProjectState(repo_root=repo_root).post_edit_state_path


def _lock_path(repo_root: Path) -> Path:
    return ProjectState(repo_root=repo_root).project_dir / LOCK_FILENAME


def utc_now_iso() -> str:
    """UTC timestamp in the same second-resolution shape the hooks log."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_paths(repo_root: Path, raw_paths: Iterable[object]) -> tuple[list[str], list[str]]:
    """Split client-reported paths into accepted repo-relative and rejected.

    Accepts absolute or relative paths. Both the repository root and the
    candidate are fully resolved before containment is checked, so a symlink
    pointing outside the repository is rejected rather than recorded.
    Returned accepted paths are sorted, deduplicated, POSIX-separated, and
    relative to ``repo_root``; rejected entries are returned as their
    original text for diagnostics.

    Work is bounded at :data:`MAX_EVENT_PATHS`: once the cap is reached the
    remaining entries are counted without being resolved or stringified, so
    an oversized payload cannot spend the hook's deadline here.
    """
    try:
        resolved_root = repo_root.resolve()
    except OSError:
        return [], [str(item) for item in raw_paths]

    accepted: set[str] = set()
    rejected: list[str] = []
    remaining = iter(raw_paths)
    for index, item in enumerate(remaining):
        if index >= MAX_EVENT_PATHS:
            over_cap = 1 + sum(1 for _ in remaining)
            rejected.append(f"<{over_cap} path(s) beyond the {MAX_EVENT_PATHS}-path event cap>")
            break
        relative = _normalize_one(resolved_root, item)
        if relative is None:
            rejected.append(str(item))
            continue
        accepted.add(relative)
    return sorted(accepted), rejected


def _normalize_one(resolved_root: Path, item: object) -> str | None:
    if not isinstance(item, str):
        return None
    text = item.strip()
    if not text or len(text) > MAX_PATH_LENGTH or "\x00" in text:
        return None
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        return None
    return resolved.relative_to(resolved_root).as_posix()


def build_event(
    *,
    client: str,
    tool_name: str,
    repo_root: Path,
    raw_paths: Iterable[object],
    observed_at: str | None = None,
) -> tuple[PostEditEvent, list[str]]:
    """Normalize a client payload into an event plus its rejected paths."""
    accepted, rejected = normalize_paths(repo_root, raw_paths)
    event = PostEditEvent(
        client=client,
        tool_name=tool_name,
        paths=accepted,
        observed_at=observed_at or utc_now_iso(),
    )
    return event, rejected


def read_state(repo_root: Path) -> PostEditState:
    """Read the persisted state, degrading to a default on any problem.

    Decoding is deliberately lossy rather than strict. A torn write, a disk
    fault, or external tampering can leave bytes that are not valid UTF-8,
    and ``UnicodeDecodeError`` is a ``ValueError`` -- it would escape an
    ``OSError``-only guard and reach the client's edit path, which this
    module promises never to do.
    """
    path = post_edit_state_path(repo_root)
    try:
        raw = path.read_bytes().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return PostEditState()
    except OSError as exc:
        log_diagnostic("post_edit_state_read_error", detail=repr(exc), cwd=str(repo_root))
        return PostEditState()
    return _parse_state(raw, repo_root)


def _parse_state(raw: str, repo_root: Path) -> PostEditState:
    """Validate a state document, degrading to a default on any problem.

    The guards are broad on purpose. ``json.loads`` raises
    ``json.JSONDecodeError`` on malformed text but ``RecursionError`` on a
    deeply nested one, and pydantic can raise beyond ``ValidationError`` for
    an adversarial document. None of those may reach a client, so the whole
    parse degrades rather than enumerating failure types.
    """
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        log_diagnostic("post_edit_state_corrupt", detail=repr(exc), cwd=str(repo_root))
        return PostEditState()
    if not isinstance(payload, dict):
        log_diagnostic(
            "post_edit_state_corrupt",
            detail=f"expected object, got {type(payload).__name__}",
            cwd=str(repo_root),
        )
        return PostEditState()
    document = cast("dict[str, Any]", payload)
    version = document.get("version")
    if version != POST_EDIT_STATE_VERSION:
        log_diagnostic(
            "post_edit_state_version_mismatch",
            detail=f"found {version!r}, expected {POST_EDIT_STATE_VERSION}",
            cwd=str(repo_root),
        )
        return PostEditState()
    try:
        return PostEditState.model_validate(document)
    except (ValidationError, ValueError, RecursionError) as exc:
        log_diagnostic("post_edit_state_corrupt", detail=repr(exc), cwd=str(repo_root))
        return PostEditState()


def record_edit(repo_root: Path, event: PostEditEvent) -> PostEditState | None:
    """Merge one edit event into the state and mark it dirty.

    Returns the persisted state, or ``None`` when the update was abandoned
    (lock contention or an unwritable cache directory). ``None`` is a normal
    outcome, not an error: the working-tree delta re-derives changed files
    independently, so a skipped record costs scoping precision, not
    correctness.
    """
    if not event.paths:
        return None

    def merge(current: PostEditState) -> PostEditState:
        combined = sorted(set(current.pending_paths) | set(event.paths))
        retained = combined[:MAX_PENDING_PATHS]
        dropped = current.dropped_path_count + max(0, len(combined) - MAX_PENDING_PATHS)
        return current.model_copy(
            update={
                "version": POST_EDIT_STATE_VERSION,
                "status": PostEditStatus.DIRTY,
                "pending_paths": retained,
                "dropped_path_count": dropped,
                "last_event_at": event.observed_at,
                "last_client": event.client,
                "last_tool_name": event.tool_name,
            }
        )

    return _mutate(repo_root, merge)


def mark_synchronized(
    repo_root: Path,
    *,
    generation_id: str,
    synchronized_paths: Iterable[str],
    synchronized_at: str | None = None,
) -> PostEditState | None:
    """Retire the paths a completed, freshness-validated refresh covered.

    Only the exact snapshot the caller consumed is retired. A path recorded by
    another hook process while the refresh was running stays pending and the
    state stays dirty, so a concurrent edit can never be reported as covered
    by a generation that predates it.

    Retiring the whole snapshot also resets ``dropped_path_count``, because
    that counter describes the completeness of the snapshot being retired,
    not of the index. The refresh is working-tree-wide, so dropped files were
    synchronized even though they never appeared in a report; the next
    snapshot starts complete. Callers that need the drop count for a report
    must read it before calling this.
    """
    consumed = set(synchronized_paths)

    def retire(current: PostEditState) -> PostEditState:
        remaining = [path for path in current.pending_paths if path not in consumed]
        cleared = not remaining
        return current.model_copy(
            update={
                "version": POST_EDIT_STATE_VERSION,
                "status": PostEditStatus.CLEAN if cleared else PostEditStatus.DIRTY,
                "pending_paths": remaining,
                "dropped_path_count": 0 if cleared else current.dropped_path_count,
                "synchronized_generation": generation_id,
                "synchronized_at": synchronized_at or utc_now_iso(),
            }
        )

    return _mutate(repo_root, retire)


def clear_state(repo_root: Path) -> None:
    """Remove the state document; a missing file is already the clean state."""
    try:
        post_edit_state_path(repo_root).unlink(missing_ok=True)
    except OSError as exc:
        log_diagnostic("post_edit_state_write_error", detail=repr(exc), cwd=str(repo_root))


def _mutate(
    repo_root: Path, transform: Callable[[PostEditState], PostEditState]
) -> PostEditState | None:
    path = post_edit_state_path(repo_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log_diagnostic("post_edit_state_write_error", detail=repr(exc), cwd=str(repo_root))
        return None

    with _state_lock(repo_root) as acquired:
        if not acquired:
            return None
        updated = transform(read_state(repo_root))
        if not _write_atomic(path, updated, repo_root):
            return None
        return updated


def _write_atomic(path: Path, state: PostEditState, repo_root: Path) -> bool:
    payload = json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        log_diagnostic("post_edit_state_write_error", detail=repr(exc), cwd=str(repo_root))
        temp_path.unlink(missing_ok=True)
        return False
    return True


class _StateLock:
    """Context manager yielding whether the exclusive lock was acquired."""

    def __init__(self, repo_root: Path, timeout: float) -> None:
        self._repo_root = repo_root
        self._timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> bool:
        lock_path = _lock_path(self._repo_root)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError as exc:
            log_diagnostic("post_edit_lock_error", detail=repr(exc), cwd=str(self._repo_root))
            return False
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                if time.monotonic() >= deadline:
                    _close_quietly(fd, self._repo_root)
                    log_diagnostic(
                        "post_edit_lock_timeout",
                        detail=f"waited {self._timeout}s for {lock_path}",
                        cwd=str(self._repo_root),
                    )
                    return False
                time.sleep(_LOCK_POLL_SECONDS)
                continue
            self._fd = fd
            return True

    def __exit__(self, *_exc: object) -> None:
        """Release the lock without ever raising into the caller's edit path.

        Every syscall in this module is guarded; acquisition and release are
        the two places that would otherwise let an ``OSError`` (a closed or
        reused descriptor) escape the ``with`` block that owns it.
        """
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:
            log_diagnostic("post_edit_lock_error", detail=repr(exc), cwd=str(self._repo_root))
        _close_quietly(fd, self._repo_root)


def _close_quietly(fd: int, repo_root: Path) -> None:
    try:
        os.close(fd)
    except OSError as exc:
        log_diagnostic("post_edit_lock_error", detail=repr(exc), cwd=str(repo_root))


def _state_lock(repo_root: Path, timeout: float | None = None) -> _StateLock:
    return _StateLock(repo_root, LOCK_TIMEOUT_SECONDS if timeout is None else timeout)
