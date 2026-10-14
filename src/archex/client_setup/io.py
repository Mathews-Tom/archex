"""JSON/TOML merge primitives and atomic client-config writes."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import cast


def _current_umask() -> int:
    """Return the process umask.

    ``os.umask`` can only be read by setting it, so this sets and restores the
    mask. The mask is process-global: another thread creating a file in the
    window between the two calls would see it. The installer is a
    single-threaded CLI command, which is the only caller.
    """
    mask = os.umask(0)
    os.umask(mask)
    return mask


def write_config_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` so a reader sees the old or the new file, never a part.

    Client configs such as ``~/.claude.json`` hold state the client cannot
    rebuild, so a kill between ``Path.write_text``'s truncate and its write
    would lose it. This writes a sibling temp file, syncs it, and renames it
    over the target, keeping the two properties an in-place write has:

    - a symlink is followed and its referent replaced, so the link survives
      (dotfile managers such as stow point these paths at a repository);
    - an existing file keeps its permission bits, and a new file gets the
      ``0o666 & ~umask`` mode ``Path.write_text`` would give, not the
      ``0o600`` of a bare temp file.

    Unlike ``archex.state_file.write_text_atomic``, failure is never
    swallowed: an installer that could not write its config must say so. The
    temp file and its descriptor are released on every failure, including
    ``KeyboardInterrupt``, and the original is left untouched.

    The rename does not coordinate with a client that rewrites the same file
    concurrently: an update the client makes between this function's caller
    reading the config and this rename is lost.
    """
    target = path.resolve()
    try:
        mode = stat.S_IMODE(target.stat().st_mode)
    except FileNotFoundError:
        mode = 0o666 & ~_current_umask()
    fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        try:
            handle = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def marked_block_span(existing: str, start_marker: str, end_marker: str) -> tuple[int, int] | None:
    """Where one marker-delimited block sits, including its trailing newline."""
    start = existing.find(start_marker)
    if start == -1:
        return None
    end = existing.find(end_marker, start)
    if end == -1:
        return None  # malformed marker pair -- leave untouched rather than guess
    end += len(end_marker)
    if end < len(existing) and existing[end] == "\n":
        end += 1
    return start, end


def strip_marked_toml_block(existing: str, start_marker: str, end_marker: str) -> str:
    """Remove one marker-delimited block, restoring the bytes around it."""
    span = marked_block_span(existing, start_marker, end_marker)
    if span is None:
        return existing
    before, after = existing[: span[0]], existing[span[1] :]
    # The block renderers always separate a freshly appended block from prior
    # content with exactly one blank line -- strip that same separator back out
    # so a strip+re-add round-trips byte-for-byte (idempotent reinstall, and a
    # clean `remove`).
    if before.endswith("\n\n"):
        before = before[:-1]
    return before + after


def upsert_marked_toml_block(existing: str, block: str, start_marker: str, end_marker: str) -> str:
    """Replace a marker-delimited block where it sits, or append it after other content."""
    span = marked_block_span(existing, start_marker, end_marker)
    if span is not None:
        return existing[: span[0]] + block + existing[span[1] :]
    if not existing.strip():
        return block
    return existing.rstrip("\n") + "\n\n" + block


def read_json_object(path: Path) -> dict[str, object]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast("dict[str, object]", payload_obj)
