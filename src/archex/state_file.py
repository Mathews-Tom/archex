"""Shared primitives for repo-local, atomically published state documents.

Two subsystems keep a small mutable document under a repository's `.archex`
directory: R21's post-edit edit tracking (`post-edit-state.json`) and R23's
cached status snapshot (`status-snapshot.json`). Both are written by several
concurrent processes -- one hook subprocess per edited file, a watch thread,
and a foreground indexing run -- and neither may raise into the caller, which
is usually an agent's edit path.

The two rules both documents obey live here so there is one implementation of
each rather than a copy per subsystem:

- **Mutual exclusion.** A read-modify-write cycle runs under an exclusive
  `flock` on a sibling lock file. Waiting past a short deadline abandons the
  write rather than delaying the caller; both callers can re-derive their
  document from cheaper sources on the next event.
- **Atomic publication.** The new content is written to a temp file in the
  same directory and moved into place with `Path.replace`, so a reader either
  sees the whole previous document or the whole new one, never a torn write.

Diagnostic kind names stay caller-owned (`StateFileDiagnostics`) so each
subsystem's log lines remain greppable on their own terms.
"""

from __future__ import annotations

import fcntl
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: How often a blocked writer retries the lock while waiting for its deadline.
LOCK_POLL_SECONDS = 0.01


@dataclass(frozen=True)
class StateFileDiagnostics:
    """Diagnostic kind names one subsystem uses for its state document."""

    lock_error: str
    lock_timeout: str
    write_error: str


def _log(kind: str, detail: str, cwd: Path) -> None:
    """Log through the hook diagnostics log, importing it only when needed.

    The import is local so this module stays cheap to import: the status
    reader must not pull in the search hook (and through it the index store)
    merely to read a cached JSON document.
    """
    from archex.integrations.hook import log_diagnostic

    log_diagnostic(kind, detail=detail, cwd=str(cwd))


class ExclusiveLock:
    """Context manager yielding whether the exclusive lock was acquired."""

    def __init__(
        self,
        lock_path: Path,
        *,
        timeout: float,
        cwd: Path,
        diagnostics: StateFileDiagnostics,
    ) -> None:
        self._lock_path = lock_path
        self._timeout = timeout
        self._cwd = cwd
        self._diagnostics = diagnostics
        self._fd: int | None = None

    def __enter__(self) -> bool:
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        except OSError as exc:
            _log(self._diagnostics.lock_error, repr(exc), self._cwd)
            return False
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                if time.monotonic() >= deadline:
                    self._close_quietly(fd)
                    _log(
                        self._diagnostics.lock_timeout,
                        f"waited {self._timeout}s for {self._lock_path}",
                        self._cwd,
                    )
                    return False
                time.sleep(LOCK_POLL_SECONDS)
                continue
            self._fd = fd
            return True

    def __exit__(self, *_exc: object) -> None:
        """Release the lock without ever raising into the caller's edit path.

        Every syscall here is guarded; acquisition and release are the two
        places that would otherwise let an ``OSError`` (a closed or reused
        descriptor) escape the ``with`` block that owns it.
        """
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:
            _log(self._diagnostics.lock_error, repr(exc), self._cwd)
        self._close_quietly(fd)

    def _close_quietly(self, fd: int) -> None:
        try:
            os.close(fd)
        except OSError as exc:
            _log(self._diagnostics.lock_error, repr(exc), self._cwd)


def write_text_atomic(
    path: Path,
    text: str,
    *,
    cwd: Path,
    diagnostics: StateFileDiagnostics,
) -> bool:
    """Publish ``text`` at ``path`` by rename; return whether it landed."""
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    except OSError as exc:
        _log(diagnostics.write_error, repr(exc), cwd)
        temp_path.unlink(missing_ok=True)
        return False
    return True


def ensure_parent_dir(path: Path, *, cwd: Path, diagnostics: StateFileDiagnostics) -> bool:
    """Create ``path``'s parent directory; return whether it now exists."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _log(diagnostics.write_error, repr(exc), cwd)
        return False
    return True
