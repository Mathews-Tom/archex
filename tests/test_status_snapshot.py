"""Cached status snapshot contract: states, transitions, bounds, isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from archex.post_edit import build_event, mark_synchronized, record_edit
from archex.project import ProjectState, init_project, reset_project
from archex.status_snapshot import (
    MAX_SNAPSHOT_BYTES,
    STATUS_SNAPSHOT_VERSION,
    WATCH_OBSERVATION_TTL_SECONDS,
    ReceiptState,
    SnapshotState,
    StatusState,
    WatchState,
    _snapshot_lock,  # pyright: ignore[reportPrivateUsage]
    clear_snapshot,
    project_repo_root,
    publish_status,
    publish_status_watch_observation,
    read_status,
    refresh_edit_overlay,
    republish_measurement,
    stale_after_seconds,
    status_snapshot_path,
    utc_now_iso,
)

if TYPE_CHECKING:
    import pytest


_NOW = 1_800_000_000.0


def _repo(tmp_path: Path) -> Path:
    (tmp_path / ".archex").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _git_repo(tmp_path: Path) -> Path:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ("init",),
        ("config", "user.email", "test@archex.test"),
        ("config", "user.name", "archex-test"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("# repo\n", encoding="utf-8")
    return repo


def _publish(
    repo: Path,
    *,
    index_fresh: bool = True,
    now: float = _NOW,
    reindex_required: bool = False,
    working_tree_dirty: bool = False,
) -> None:
    publish_status(
        repo,
        index_fresh=index_fresh,
        index_revision="abc1234",
        generation_id="f" * 64,
        indexed_commit="c" * 40,
        current_commit="c" * 40,
        files_indexed=120,
        chunks_indexed=980,
        reindex_required=reindex_required,
        working_tree_dirty=working_tree_dirty,
        now=now,
    )


def _record_edit(repo: Path, path: str = "pkg/mod.py", *, at: float = _NOW + 1) -> None:
    """Record an edit at a controlled instant on the tests' fake clock.

    `pending` is a comparison between the last recorded edit and the last
    index measurement, so an edit stamped with the real wall clock would be
    compared against a fake-clock measurement and classify by accident.
    """
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text("x = 1\n")
    event, _rejected = build_event(
        client="claude-code",
        tool_name="Edit",
        repo_root=repo,
        raw_paths=[path],
        observed_at=utc_now_iso(at),
    )
    record_edit(repo, event)


def test_published_snapshot_reads_back_as_fresh_with_its_measurement(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    _publish(repo)
    view = read_status(repo, now=_NOW + 5)

    assert view.state is StatusState.FRESH
    assert view.age_seconds == 5
    assert view.snapshot is not None
    assert view.snapshot.version == STATUS_SNAPSHOT_VERSION
    assert view.snapshot.files_indexed == 120
    assert view.snapshot.receipt_state is ReceiptState.COMPLETE


def test_index_that_no_longer_describes_the_tree_reads_as_dirty(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    _publish(repo, index_fresh=False, reindex_required=True)
    view = read_status(repo, now=_NOW)

    assert view.state is StatusState.DIRTY
    assert view.snapshot is not None
    assert view.snapshot.reindex_required is True
    # A flagged store can attest nothing about the current tree.
    assert view.snapshot.receipt_state is ReceiptState.UNKNOWN


def test_recorded_edit_outranks_dirty_and_reports_pending_count(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo, index_fresh=False)

    _record_edit(repo)
    refresh_edit_overlay(repo, now=_NOW + 1)
    view = read_status(repo, now=_NOW + 1)

    assert view.state is StatusState.PENDING
    assert view.snapshot is not None
    assert view.snapshot.pending_delta_files == 1
    assert view.snapshot.last_edit_client == "claude-code"
    assert view.snapshot.last_edit_tool == "Edit"


def test_repeated_overlays_do_not_promote_a_pending_edit_to_covered(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    _record_edit(repo, at=_NOW + 1)

    refresh_edit_overlay(repo, now=_NOW + 2)
    refresh_edit_overlay(repo, now=_NOW + 3)
    view = read_status(repo, now=_NOW + 4)

    # Each overlay carries the index measurement instant forward instead of
    # stamping its own, so the edit stays newer than the measurement.
    assert view.state is StatusState.PENDING
    assert view.snapshot is not None
    assert view.snapshot.index_measured_at == utc_now_iso(_NOW)


def test_a_later_index_measurement_clears_an_unsynchronized_edit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    # A post-edit hook whose refresh timed out: the edit stays recorded in
    # R21 state forever, because only a successful sync retires it.
    _record_edit(repo, at=_NOW + 1)
    refresh_edit_overlay(repo, now=_NOW + 2)
    assert read_status(repo, now=_NOW + 2).state is StatusState.PENDING

    _publish(repo, now=_NOW + 60)
    view = read_status(repo, now=_NOW + 60)

    # An ordinary index run does cover those files, so the surface heals
    # rather than showing `pending` for the rest of the repository's life.
    assert view.state is StatusState.FRESH
    assert view.snapshot is not None
    assert view.snapshot.pending_delta_files == 0


def test_an_overlay_cannot_promote_a_dirty_measurement_to_fresh(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    # `archex status` publishes exactly this shape: the index is behind the
    # tree (a newer commit, or a changed file) without the store being
    # flagged for a rebuild.
    _publish(repo, index_fresh=False, reindex_required=False)
    assert read_status(repo, now=_NOW).state is StatusState.DIRTY

    _record_edit(repo, at=_NOW + 1)
    refresh_edit_overlay(repo, now=_NOW + 2)
    mark_synchronized(repo, generation_id="f" * 64, synchronized_paths=["pkg/mod.py"])
    refresh_edit_overlay(repo, now=_NOW + 3)
    view = read_status(repo, now=_NOW + 3)

    # An overlay learns nothing new about the index, so it must not decide the
    # index now describes the tree.
    assert view.state is StatusState.DIRTY
    assert view.snapshot is not None
    assert view.snapshot.index_fresh is False


def test_unchanged_generation_is_restamped_without_the_index_fields(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)

    restamped = republish_measurement(repo, generation_id="f" * 64, now=_NOW + 300)

    assert restamped is not None
    # The measurement instant advances, which is what keeps a long warm
    # session out of `stale`, while every index-derived field is carried.
    assert restamped.written_at_epoch == _NOW + 300
    assert restamped.index_revision == "abc1234"
    assert restamped.files_indexed == 120
    assert read_status(repo, now=_NOW + 300).state is StatusState.FRESH


def test_restamp_refuses_a_different_or_absent_generation(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    assert republish_measurement(repo, generation_id="f" * 64) is None

    _publish(repo)

    assert republish_measurement(repo, generation_id="a" * 64) is None
    assert republish_measurement(repo, generation_id="") is None
    # The caller must fall back to a full publication, so nothing was written.
    assert read_status(repo, now=_NOW).snapshot is not None


def test_restamp_still_reflects_an_edit_recorded_since_the_measurement(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    _record_edit(repo, at=_NOW + 1)

    republish_measurement(repo, generation_id="f" * 64, now=_NOW + 2)

    # Re-stamping advances the measurement instant, so an edit older than the
    # new instant is covered -- the same self-healing rule a full publication
    # applies, since a validated cache hit is evidence about the current tree.
    assert read_status(repo, now=_NOW + 2).state is StatusState.FRESH


def test_synchronizing_the_recorded_edit_returns_the_surface_to_fresh(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    _record_edit(repo)
    refresh_edit_overlay(repo, now=_NOW + 1)
    assert read_status(repo, now=_NOW + 1).state is StatusState.PENDING

    mark_synchronized(repo, generation_id="f" * 64, synchronized_paths=["pkg/mod.py"])
    refresh_edit_overlay(repo, now=_NOW + 2)
    view = read_status(repo, now=_NOW + 2)

    assert view.state is StatusState.FRESH
    assert view.snapshot is not None
    assert view.snapshot.pending_delta_files == 0
    assert view.snapshot.last_successful_sync != ""
    # Index-derived fields survive an index-free overlay.
    assert view.snapshot.files_indexed == 120
    assert view.snapshot.index_revision == "abc1234"


def test_dropped_edit_records_downgrade_the_receipt_to_partial(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    for index in range(205):
        _record_edit(repo, f"pkg/mod{index}.py")

    refresh_edit_overlay(repo, now=_NOW + 1)
    view = read_status(repo, now=_NOW + 1)

    assert view.state is StatusState.PENDING
    assert view.snapshot is not None
    assert view.snapshot.pending_view_complete is False
    assert view.snapshot.receipt_state is ReceiptState.PARTIAL


def test_absent_snapshot_reads_as_missing_not_as_fresh(tmp_path: Path) -> None:
    view = read_status(_repo(tmp_path))

    assert view.state is StatusState.MISSING
    assert view.snapshot is None
    assert view.detail != ""


def test_unparsable_snapshot_reads_as_corrupt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    status_snapshot_path(repo).write_bytes(b"\x00not json{")

    view = read_status(repo)

    assert view.state is StatusState.CORRUPT
    assert view.snapshot is None


def test_json_array_snapshot_reads_as_corrupt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    status_snapshot_path(repo).write_text("[]\n")

    assert read_status(repo).state is StatusState.CORRUPT


def test_schema_violating_snapshot_reads_as_corrupt_not_unsupported(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    status_snapshot_path(repo).write_text(
        json.dumps({"version": STATUS_SNAPSHOT_VERSION, "state": "sideways"})
    )

    assert read_status(repo).state is StatusState.CORRUPT


def test_a_document_without_a_version_reads_as_corrupt_not_unsupported(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    status_snapshot_path(repo).write_text(json.dumps({"state": "fresh"}))

    view = read_status(repo)

    # Different remedies: an unversioned document needs re-publishing, not an
    # archex upgrade. The shell and TypeScript renderers agree.
    assert view.state is StatusState.CORRUPT


def test_a_non_integer_version_reads_as_corrupt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    status_snapshot_path(repo).write_text(json.dumps({"version": "1", "state": "fresh"}))

    assert read_status(repo).state is StatusState.CORRUPT


def test_a_document_without_a_measurement_instant_makes_no_age_claim(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    status_snapshot_path(repo).write_text(
        json.dumps(
            {
                "version": STATUS_SNAPSHOT_VERSION,
                "state": "fresh",
                "written_at": "",
                "written_at_epoch": 0,
            }
        )
    )

    view = read_status(repo, now=_NOW)

    # A zero instant is unknown, not 1970: it must not masquerade as staleness.
    assert view.state is StatusState.FRESH
    assert view.age_seconds is None


def test_future_version_reads_as_unsupported_and_names_both_versions(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    status_snapshot_path(repo).write_text(
        json.dumps(
            {
                "version": STATUS_SNAPSHOT_VERSION + 1,
                "state": "fresh",
                "written_at": "2027-01-01T00:00:00Z",
                "written_at_epoch": 1_800_000_000,
            }
        )
    )

    view = read_status(repo)

    assert view.state is StatusState.UNSUPPORTED
    assert view.snapshot is None
    assert str(STATUS_SNAPSHOT_VERSION + 1) in view.detail
    assert str(STATUS_SNAPSHOT_VERSION) in view.detail


def test_old_measurement_reads_as_stale_but_keeps_its_last_known_fields(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)

    view = read_status(repo, now=_NOW + stale_after_seconds() + 1)

    assert view.state is StatusState.STALE
    assert view.snapshot is not None
    assert view.snapshot.state is SnapshotState.FRESH
    assert view.age_seconds == stale_after_seconds() + 1
    assert "freshness budget" in view.detail


def test_stale_budget_is_environment_overridable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    monkeypatch.setenv("ARCHEX_STATUS_STALE_AFTER_SECONDS", "5")

    assert read_status(repo, now=_NOW + 4).state is StatusState.FRESH
    assert read_status(repo, now=_NOW + 6).state is StatusState.STALE


def test_invalid_stale_budget_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHEX_STATUS_STALE_AFTER_SECONDS", "not-a-number")
    default = stale_after_seconds()
    monkeypatch.setenv("ARCHEX_STATUS_STALE_AFTER_SECONDS", "0")

    assert stale_after_seconds() == default


def test_watch_observation_is_reported_active_only_while_recent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    assert read_status(repo, now=_NOW).watch_state is WatchState.UNOBSERVED

    publish_status_watch_observation(repo, now=_NOW + 1)

    assert read_status(repo, now=_NOW + 2).watch_state is WatchState.ACTIVE
    late = _NOW + 2 + WATCH_OBSERVATION_TTL_SECONDS
    assert read_status(repo, now=late).watch_state is WatchState.UNOBSERVED


def test_full_publication_preserves_an_earlier_watch_observation(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    publish_status_watch_observation(repo, now=_NOW + 1)

    _publish(repo, now=_NOW + 2)

    assert read_status(repo, now=_NOW + 3).watch_state is WatchState.ACTIVE


def test_overlay_without_a_previous_measurement_publishes_nothing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    assert refresh_edit_overlay(repo) is None
    assert not status_snapshot_path(repo).exists()
    assert read_status(repo).state is StatusState.MISSING


def test_lock_contention_skips_the_write_and_preserves_the_previous_document(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    before = status_snapshot_path(repo).read_text(encoding="utf-8")

    with _snapshot_lock(repo) as acquired:
        assert acquired
        # Same process, so a second flock on a new descriptor is the only
        # faithful contention simulation available.
        skipped = publish_status(repo, index_fresh=False, now=_NOW + 10)

    assert skipped is None
    assert status_snapshot_path(repo).read_text(encoding="utf-8") == before


def test_worst_case_document_stays_inside_the_declared_size_bound(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    for index in range(205):
        _record_edit(repo, f"pkg/mod{index}.py")
    publish_status(
        repo,
        index_fresh=False,
        index_revision="r" * 200,
        generation_id="g" * 200,
        indexed_commit="i" * 200,
        current_commit="u" * 200,
        files_indexed=10**9,
        chunks_indexed=10**9,
        working_tree_dirty=True,
        reindex_required=True,
        now=_NOW,
    )
    publish_status_watch_observation(repo, now=_NOW)

    raw = status_snapshot_path(repo).read_bytes()

    assert len(raw) <= MAX_SNAPSHOT_BYTES
    document = json.loads(raw)
    assert all(not isinstance(value, (list, dict)) for value in document.values())
    assert len(document["generation_id"]) == 64


def test_client_reported_names_are_clipped_rather_than_stored_whole(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    (repo / "a.py").write_text("x = 1\n")
    event, _rejected = build_event(
        client="c" * 500, tool_name="t" * 500, repo_root=repo, raw_paths=["a.py"]
    )
    record_edit(repo, event)

    refresh_edit_overlay(repo, now=_NOW + 1)
    view = read_status(repo, now=_NOW + 1)

    assert view.snapshot is not None
    assert len(view.snapshot.last_edit_client) == 64
    assert len(view.snapshot.last_edit_tool) == 64


def test_reading_a_snapshot_opens_no_index_and_launches_no_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path)
    _publish(repo)
    import os
    import subprocess

    from archex.index.store import IndexStore

    def _forbidden(*_args: object, **_kwargs: object) -> object:
        message = "repaint must not open an index or launch a process"
        raise AssertionError(message)

    monkeypatch.setattr(IndexStore, "__init__", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(os, "fork", _forbidden, raising=False)
    monkeypatch.setattr(os, "posix_spawn", _forbidden, raising=False)

    view = read_status(repo, now=_NOW)

    assert view.state is StatusState.FRESH


def test_reading_a_snapshot_never_imports_the_index_store() -> None:
    import subprocess
    import sys

    probe = (
        "import sys, archex.status_snapshot as s;"
        "print('archex.index.store' in sys.modules, 'archex.post_edit' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False False"


def test_global_cache_directory_publishes_no_snapshot(tmp_path: Path) -> None:
    global_cache = tmp_path / "cache" / "abcdef"
    global_cache.mkdir(parents=True)

    assert project_repo_root(global_cache) is None


def test_uninitialized_project_directory_publishes_no_snapshot(tmp_path: Path) -> None:
    project_dir = tmp_path / ".archex"
    project_dir.mkdir()

    assert project_repo_root(project_dir) is None


def test_initialized_project_directory_resolves_to_its_repository_root(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    init_project(repo)

    assert project_repo_root(ProjectState(repo_root=repo).project_dir) == repo


def test_reset_removes_the_snapshot_so_the_surface_reports_missing(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path)
    init_project(repo)
    _publish(repo)
    assert read_status(repo, now=_NOW).state is StatusState.FRESH

    reset_project(repo, force=True)

    assert read_status(repo).state is StatusState.MISSING


def test_clearing_the_snapshot_is_idempotent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _publish(repo)

    clear_snapshot(repo)
    clear_snapshot(repo)

    assert read_status(repo).state is StatusState.MISSING
