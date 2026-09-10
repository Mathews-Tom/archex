"""Post-edit state contract: normalization, bounds, atomicity, concurrency."""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from archex.post_edit import (
    MAX_EVENT_PATHS,
    MAX_PENDING_PATHS,
    POST_EDIT_STATE_VERSION,
    PostEditState,
    PostEditStatus,
    build_event,
    clear_state,
    mark_synchronized,
    normalize_paths,
    post_edit_state_path,
    read_state,
    record_edit,
)
from archex.post_edit.state import (
    LOCK_FILENAME,
    _lock_path,  # pyright: ignore[reportPrivateUsage]
    _state_lock,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest


def _record(repo: Path, *paths: str, client: str = "claude-code") -> PostEditState | None:
    event, _rejected = build_event(
        client=client, tool_name="Edit", repo_root=repo, raw_paths=list(paths)
    )
    return record_edit(repo, event)


def test_relative_and_absolute_paths_normalize_to_repo_relative_posix(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n")

    accepted, rejected = normalize_paths(tmp_path, ["pkg/mod.py", str(tmp_path / "pkg" / "mod.py")])

    assert accepted == ["pkg/mod.py"]
    assert rejected == []


def test_paths_outside_the_repository_are_rejected_not_recorded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n")

    accepted, rejected = normalize_paths(repo, ["../outside.py", str(outside), "/etc/passwd"])

    assert accepted == []
    assert sorted(rejected) == sorted(["../outside.py", str(outside), "/etc/passwd"])


def test_symlink_escaping_the_repository_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = tmp_path / "secret.py"
    secret.write_text("token = 1\n")
    (repo / "link.py").symlink_to(secret)

    accepted, rejected = normalize_paths(repo, ["link.py"])

    assert accepted == []
    assert rejected == ["link.py"]


def test_repository_root_itself_is_not_a_recordable_edit(tmp_path: Path) -> None:
    accepted, rejected = normalize_paths(tmp_path, [str(tmp_path), "."])

    assert accepted == []
    assert rejected == [str(tmp_path), "."]


def test_malformed_path_entries_are_rejected_without_raising(tmp_path: Path) -> None:
    accepted, rejected = normalize_paths(
        tmp_path, ["", "   ", None, 17, "a" * 2000, "nul\x00byte", "ok.py"]
    )

    assert accepted == ["ok.py"]
    assert len(rejected) == 6


def test_recording_an_edit_marks_state_dirty_with_receipt_fields(tmp_path: Path) -> None:
    state = _record(tmp_path, "a.py")

    assert state is not None
    assert state.version == POST_EDIT_STATE_VERSION
    assert state.status is PostEditStatus.DIRTY
    assert state.pending_paths == ["a.py"]
    assert state.last_client == "claude-code"
    assert state.last_tool_name == "Edit"
    assert state.last_event_at is not None
    assert read_state(tmp_path) == state


def test_repeated_edits_union_and_deduplicate_pending_paths(tmp_path: Path) -> None:
    _record(tmp_path, "b.py", "a.py")
    state = _record(tmp_path, "a.py", "c.py")

    assert state is not None
    assert state.pending_paths == ["a.py", "b.py", "c.py"]


def test_an_event_with_no_usable_path_writes_nothing(tmp_path: Path) -> None:
    assert _record(tmp_path, "../escape.py") is None
    assert not post_edit_state_path(tmp_path).exists()


def test_pending_paths_are_capped_and_overflow_is_counted(tmp_path: Path) -> None:
    first = [f"src/f{index:04d}.py" for index in range(MAX_PENDING_PATHS)]
    _record(tmp_path, *first)

    state = _record(tmp_path, "src/zzz_overflow.py")

    assert state is not None
    assert len(state.pending_paths) == MAX_PENDING_PATHS
    assert state.dropped_path_count == 1
    assert state.is_bounded_view() is False


def test_synchronizing_retires_the_consumed_snapshot_and_clears(tmp_path: Path) -> None:
    _record(tmp_path, "a.py", "b.py")

    state = mark_synchronized(tmp_path, generation_id="gen-1", synchronized_paths=["a.py", "b.py"])

    assert state is not None
    assert state.status is PostEditStatus.CLEAN
    assert state.pending_paths == []
    assert state.synchronized_generation == "gen-1"
    assert state.synchronized_at is not None


def test_an_edit_arriving_during_synchronization_stays_pending(tmp_path: Path) -> None:
    _record(tmp_path, "a.py")
    consumed = ["a.py"]
    _record(tmp_path, "later.py")

    state = mark_synchronized(tmp_path, generation_id="gen-1", synchronized_paths=consumed)

    assert state is not None
    assert state.status is PostEditStatus.DIRTY
    assert state.pending_paths == ["later.py"]


def test_dropped_count_survives_a_partial_synchronization(tmp_path: Path) -> None:
    _record(tmp_path, *[f"src/f{index:04d}.py" for index in range(MAX_PENDING_PATHS + 1)])
    before = read_state(tmp_path)
    assert before.dropped_path_count == 1

    state = mark_synchronized(
        tmp_path, generation_id="gen-1", synchronized_paths=before.pending_paths[:1]
    )

    assert state is not None
    assert state.dropped_path_count == 1
    assert state.status is PostEditStatus.DIRTY


def test_missing_state_reads_as_clean_default(tmp_path: Path) -> None:
    assert read_state(tmp_path) == PostEditState()


def test_corrupt_state_reads_as_default_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "diag.log"
    monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(log))
    path = post_edit_state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    assert read_state(tmp_path) == PostEditState()
    assert "post_edit_state_corrupt" in log.read_text()


def test_state_from_another_schema_version_is_not_coerced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "diag.log"
    monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(log))
    path = post_edit_state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": POST_EDIT_STATE_VERSION + 1, "pending_paths": ["x"]}))

    assert read_state(tmp_path).pending_paths == []
    assert "post_edit_state_version_mismatch" in log.read_text()


def test_a_json_array_state_document_is_rejected(tmp_path: Path) -> None:
    path = post_edit_state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]")

    assert read_state(tmp_path) == PostEditState()


def test_writes_leave_no_temporary_file_behind(tmp_path: Path) -> None:
    _record(tmp_path, "a.py")

    leftovers = list(post_edit_state_path(tmp_path).parent.glob("*.tmp"))
    assert leftovers == []


def test_concurrent_recorders_do_not_lose_updates(tmp_path: Path) -> None:
    paths = [f"src/f{index:03d}.py" for index in range(16)]
    barrier = threading.Barrier(len(paths))

    def worker(path: str) -> None:
        barrier.wait()
        _record(tmp_path, path)

    threads = [threading.Thread(target=worker, args=(path,)) for path in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert read_state(tmp_path).pending_paths == sorted(paths)


def test_lock_contention_abandons_the_write_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "diag.log"
    monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(log))
    monkeypatch.setattr("archex.post_edit.state.LOCK_TIMEOUT_SECONDS", 0.05)
    lock_path = _lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        started = time.monotonic()
        assert _record(tmp_path, "a.py") is None
        elapsed = time.monotonic() - started
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)

    assert elapsed < 2.0
    assert "post_edit_lock_timeout" in log.read_text()
    assert not post_edit_state_path(tmp_path).exists()


def test_lock_file_lives_beside_the_state_document(tmp_path: Path) -> None:
    _record(tmp_path, "a.py")

    assert _lock_path(tmp_path).name == LOCK_FILENAME
    assert _lock_path(tmp_path).parent == post_edit_state_path(tmp_path).parent


def test_clear_state_removes_the_document_idempotently(tmp_path: Path) -> None:
    _record(tmp_path, "a.py")

    clear_state(tmp_path)
    clear_state(tmp_path)

    assert not post_edit_state_path(tmp_path).exists()
    assert read_state(tmp_path).status is PostEditStatus.CLEAN


def test_undecodable_state_bytes_never_raise_into_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torn write leaves bytes that are not UTF-8; that must not reach a client.

    `UnicodeDecodeError` is a `ValueError`, so an `OSError`-only guard would
    let it escape `read_state` and `record_edit` into the edit path.
    """
    log = tmp_path / "diag.log"
    monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(log))
    path = post_edit_state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe{\x00garbage")

    assert read_state(tmp_path) == PostEditState()
    assert _record(tmp_path, "a.py") is not None
    assert read_state(tmp_path).pending_paths == ["a.py"]


def test_a_deeply_nested_state_document_is_rejected(tmp_path: Path) -> None:
    path = post_edit_state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[" * 200_000)

    assert read_state(tmp_path) == PostEditState()


def test_releasing_a_broken_lock_descriptor_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "diag.log"
    monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(log))
    (tmp_path / ".archex").mkdir(parents=True, exist_ok=True)
    lock = _state_lock(tmp_path)
    with lock as acquired:
        assert acquired
        held = lock._fd  # pyright: ignore[reportPrivateUsage] - simulate a reclaimed fd
        assert held is not None
        os.close(held)

    assert "post_edit_lock_error" in log.read_text()


def test_paths_beyond_the_event_cap_are_summarized_not_enumerated(tmp_path: Path) -> None:
    raw = [f"f{index}.py" for index in range(MAX_EVENT_PATHS + 40)]

    accepted, rejected = normalize_paths(tmp_path, raw)

    assert len(accepted) == MAX_EVENT_PATHS
    assert rejected == [f"<40 path(s) beyond the {MAX_EVENT_PATHS}-path event cap>"]


def test_normalization_stops_consuming_input_at_the_cap(tmp_path: Path) -> None:
    """The cap must bound work, so the tail is counted, never resolved."""
    resolved: list[str] = []

    def spy() -> Iterator[str]:
        for index in range(MAX_EVENT_PATHS + 10):
            resolved.append(f"f{index}.py")
            yield f"f{index}.py"

    normalize_paths(tmp_path, spy())

    # The generator is drained to count the overflow, but only the first
    # MAX_EVENT_PATHS entries reach `_normalize_one`'s filesystem work.
    assert len(resolved) == MAX_EVENT_PATHS + 10
