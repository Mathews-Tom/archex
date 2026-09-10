"""Bounded, versioned cached status snapshot (R23).

Archex already knows precisely how fresh an index is -- but only inside a
process that has opened the index. A persistent client status surface repaints
several times a minute (Claude Code debounces its statusline at 300 ms and
cancels an in-flight script when a new update arrives), so it cannot afford an
index open, a parse, or a cold Python start per repaint. This module is the
cheap side of that split: lifecycle code that already holds the expensive
state *publishes* a small document, and every renderer only *reads* it.

Two invariants make that safe:

- **The writer decides what is true; the reader decides what is current.**
  A persisted document carries one of three states (:class:`SnapshotState`:
  ``fresh``, ``dirty``, ``pending``). Reading adds the four states only a
  reader can establish (:class:`StatusState`: ``missing``, ``corrupt``,
  ``unsupported``, ``stale``). ``unsupported`` is kept distinct from
  ``corrupt`` because they have different remediations: a version this build
  cannot read means upgrade, unparsable bytes mean re-publish.
- **Bounded by construction.** Every field is a scalar with a declared
  maximum length; the document holds no path list, no language map, and no
  token-savings figure. Its serialized size therefore has a provable ceiling
  (:data:`MAX_SNAPSHOT_BYTES`) that does not depend on repository size or on a
  truncation rule that could be forgotten.

Nothing here raises into a caller. Publication happens on an agent's edit path
and inside indexing; a status write that fails is a display defect, never a
correctness one, so every failure degrades to a diagnostics line.

One limit of a cached surface is worth stating: a reader never stats the
index, so an index deleted out of band (a manual ``rm``, a wiped cache) is
reflected only when the snapshot's freshness budget expires and it reads
``stale``. ``archex reset`` clears the snapshot with the index it describes,
and ``archex status`` clears it when there is nothing left to describe, so the
supported paths report ``missing`` immediately.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field, ValidationError

from archex.project import PROJECT_DIR_NAME, ProjectState
from archex.state_file import (
    ExclusiveLock,
    StateFileDiagnostics,
    ensure_parent_dir,
    write_text_atomic,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from archex.post_edit.models import PostEditState

SNAPSHOT_FILENAME = "status-snapshot.json"
LOCK_FILENAME = "status-snapshot.lock"

#: Schema version of the persisted snapshot. A document carrying any other
#: value reads as :attr:`StatusState.UNSUPPORTED` rather than being coerced --
#: another writer's field semantics are not guessable.
STATUS_SNAPSHOT_VERSION = 1
#: Wall-clock budget for acquiring the snapshot lock. Much shorter than R21's
#: state lock: a status write is pure display value and the critical section
#: is a small read, merge, and rename, so a contended writer should give up
#: almost immediately rather than add tail latency to the warm query path it
#: usually runs on. The next lifecycle event re-derives whatever was skipped.
LOCK_TIMEOUT_SECONDS = 0.25

#: Age past which a reader reports :attr:`StatusState.STALE` instead of the
#: persisted state. A snapshot is a cache of a measurement, and an old
#: measurement of a repository under active editing is not evidence of
#: freshness. Overridable per environment for hosts whose lifecycle events are
#: rarer than this default.
DEFAULT_STALE_AFTER_SECONDS = 900
_STALE_ENV_VAR = "ARCHEX_STATUS_STALE_AFTER_SECONDS"

#: Age past which an observed watch event stops implying a live watcher.
#: ``unobserved`` means "no watch refresh seen recently", never "no watcher is
#: running": an idle watcher publishes nothing because nothing changed.
WATCH_OBSERVATION_TTL_SECONDS = 300

#: Provable ceiling on the serialized document, asserted against a worst-case
#: snapshot in the tests rather than enforced by truncating a live one.
MAX_SNAPSHOT_BYTES = 1024

#: Field length caps. Identity strings are hex digests or short revisions;
#: client and tool names come from a client payload and are clipped rather
#: than trusted.
_MAX_IDENTITY_LENGTH = 64
_MAX_NAME_LENGTH = 64
_MAX_TIMESTAMP_LENGTH = 20

_DIAGNOSTICS = StateFileDiagnostics(
    lock_error="status_snapshot_lock_error",
    lock_timeout="status_snapshot_lock_timeout",
    write_error="status_snapshot_write_error",
)


class SnapshotState(StrEnum):
    """Index/edit state a writer can establish while holding real state."""

    #: The index describes the working tree and no edit is awaiting sync.
    FRESH = "fresh"
    #: The index no longer describes the working tree, or a reindex is flagged.
    DIRTY = "dirty"
    #: Edits are recorded and not yet synchronized into the index.
    PENDING = "pending"


class StatusState(StrEnum):
    """Every state a renderer can be asked to display."""

    FRESH = "fresh"
    DIRTY = "dirty"
    PENDING = "pending"
    #: A valid snapshot whose measurement is too old to assert freshness.
    STALE = "stale"
    #: No snapshot has been published (or it was cleared by a reset).
    MISSING = "missing"
    #: A snapshot exists but is unreadable, unparsable, or invalid.
    CORRUPT = "corrupt"
    #: A snapshot exists and parses but declares an unreadable version.
    UNSUPPORTED = "unsupported"


class WatchState(StrEnum):
    """Whether a watch-driven refresh was observed recently."""

    ACTIVE = "active"
    UNOBSERVED = "unobserved"


class ReceiptState(StrEnum):
    """Completeness of the freshness receipt behind the snapshot."""

    #: A generation identity exists and the edit view lists every edit.
    COMPLETE = "complete"
    #: A generation identity exists but recorded edits were dropped by a cap.
    PARTIAL = "partial"
    #: No generation identity is persisted, so nothing can be attested.
    UNKNOWN = "unknown"


class StatusSnapshot(BaseModel):
    """The persisted document. Scalars only, every string length-capped."""

    version: int = STATUS_SNAPSHOT_VERSION
    state: SnapshotState
    #: UTC ISO-8601 instant the snapshot was measured, in the same
    #: second-resolution shape the hooks log.
    written_at: str = Field(max_length=_MAX_TIMESTAMP_LENGTH)
    #: The same instant as Unix epoch seconds, so a renderer with a clock can
    #: judge staleness without parsing a timestamp.
    written_at_epoch: int
    #: When the index state described here was measured. Equal to
    #: ``written_at`` for a full publication; carried forward unchanged by
    #: index-free overlays, so "has anything been edited since the index last
    #: described the tree?" stays answerable after repeated overlays.
    index_measured_at: str = Field(default="", max_length=_MAX_TIMESTAMP_LENGTH)
    #: Index revision as the hook receipts report it. A renderer with limited
    #: width is expected to show a prefix; the document stores it whole.
    index_revision: str = Field(default="", max_length=_MAX_IDENTITY_LENGTH)
    #: Generation identity of the index the measurement describes.
    generation_id: str = Field(default="", max_length=_MAX_IDENTITY_LENGTH)
    indexed_commit: str = Field(default="", max_length=_MAX_IDENTITY_LENGTH)
    current_commit: str = Field(default="", max_length=_MAX_IDENTITY_LENGTH)
    files_indexed: int = 0
    chunks_indexed: int = 0
    #: Whether the working tree carried uncommitted changes at measurement
    #: time. Orthogonal to freshness: a synchronized index describes a dirty
    #: tree exactly as well as a clean one.
    working_tree_dirty: bool = False
    #: Whether the index described the working tree at measurement time.
    #: Persisted rather than re-derived: an index-free overlay must not be
    #: able to promote a writer's `dirty` measurement to `fresh`, and
    #: `reindex_required` alone does not carry that information -- an index
    #: can be behind the tree without being flagged for a rebuild.
    index_fresh: bool = True
    #: Whether the store is flagged for a full reindex.
    reindex_required: bool = False
    #: Recorded edits awaiting synchronization (R21 ``pending_paths``).
    pending_delta_files: int = 0
    #: Whether that count is the whole truth (R21 ``dropped_path_count == 0``).
    pending_view_complete: bool = True
    last_edit_at: str = Field(default="", max_length=_MAX_TIMESTAMP_LENGTH)
    last_edit_client: str = Field(default="", max_length=_MAX_NAME_LENGTH)
    last_edit_tool: str = Field(default="", max_length=_MAX_NAME_LENGTH)
    last_successful_sync: str = Field(default="", max_length=_MAX_TIMESTAMP_LENGTH)
    receipt_state: ReceiptState = ReceiptState.UNKNOWN
    #: When a watch-driven refresh was last observed, empty if never.
    watch_observed_at: str = Field(default="", max_length=_MAX_TIMESTAMP_LENGTH)
    watch_observed_epoch: int = 0


class StatusView(BaseModel):
    """What a renderer receives: a display state plus its evidence."""

    state: StatusState
    #: Present whenever a valid document was read, including when the reader
    #: overrode its state with ``stale`` -- a stale snapshot's fields are the
    #: last known measurement and are still worth showing, labelled as old.
    snapshot: StatusSnapshot | None = None
    #: Why the reader chose a state the writer did not persist.
    detail: str = ""
    #: Snapshot age in whole seconds, when both a snapshot and a clock exist.
    age_seconds: int | None = None
    watch_state: WatchState = WatchState.UNOBSERVED


def status_snapshot_path(repo_root: Path) -> Path:
    """Path of the cached status snapshot for ``repo_root``."""
    return ProjectState(repo_root=repo_root).status_snapshot_path


def _lock_path(repo_root: Path) -> Path:
    return ProjectState(repo_root=repo_root).project_dir / LOCK_FILENAME


def utc_now_iso(now: float | None = None) -> str:
    """UTC timestamp in the shape every archex state document uses."""
    moment = datetime.now(UTC) if now is None else datetime.fromtimestamp(now, tz=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def stale_after_seconds() -> int:
    """Snapshot age budget, overridable per environment."""
    raw = os.environ.get(_STALE_ENV_VAR)
    if raw is None:
        return DEFAULT_STALE_AFTER_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_STALE_AFTER_SECONDS
    return value if value > 0 else DEFAULT_STALE_AFTER_SECONDS


def project_repo_root(cache_dir: str | Path) -> Path | None:
    """Repository root a project-layout ``cache_dir`` belongs to, else None.

    Indexing also runs against the shared global cache and against remote
    checkouts, neither of which has a repo-local `.archex` directory to
    publish into. Deriving the root from the configured cache directory keeps
    the check exact and costs no subprocess -- a project cache directory is
    always ``<root>/.archex`` -- and it stays correct when the command was
    invoked from a subdirectory, unlike deriving it from the caller's path.
    """
    resolved = Path(cache_dir).expanduser()
    if resolved.name != PROJECT_DIR_NAME:
        return None
    root = resolved.parent
    return root if ProjectState(repo_root=root).initialized() else None


def pending_after_measurement(*, edit_state: PostEditState, index_measured_at: str) -> bool:
    """Whether recorded edits are newer than the last index measurement.

    This is what makes `pending` self-healing. R21's edit state is retired by
    a successful post-edit synchronization, so a hook whose refresh timed out
    leaves paths recorded forever -- and a later ordinary `archex index` or
    warm query, which does cover those files, would otherwise never clear the
    surface. Comparing the last recorded edit against the instant of the last
    index measurement answers the question a status surface is actually
    asked: has anything been edited since the index last described the tree?

    Both timestamps are second-resolution UTC ISO-8601, so lexicographic
    order is chronological order. The comparison is inclusive: an edit landing
    in the same second as a measurement is treated as not-yet-covered, which
    errs toward `pending` rather than toward claiming freshness.

    R21's enum is imported here rather than at module scope so reading a
    snapshot never pulls in `archex.post_edit`, which re-exports the impact
    renderer and through it the whole index pipeline. Only writers classify.
    """
    from archex.post_edit.models import PostEditStatus

    if edit_state.status is not PostEditStatus.DIRTY or not edit_state.pending_paths:
        return False
    return (edit_state.last_event_at or "") >= index_measured_at


def derive_state(*, index_fresh: bool, pending: bool) -> SnapshotState:
    """Classify index and edit state into one persistable state.

    ``pending`` outranks ``dirty`` because it is the strictly more precise
    statement: it names recorded edits the index has not covered yet, rather
    than merely observing that tree and index disagree.
    """
    if pending:
        return SnapshotState.PENDING
    return SnapshotState.FRESH if index_fresh else SnapshotState.DIRTY


def _derive_receipt_state(
    *, generation_id: str, pending_view_complete: bool, reindex_required: bool
) -> ReceiptState:
    if not generation_id or reindex_required:
        return ReceiptState.UNKNOWN
    return ReceiptState.COMPLETE if pending_view_complete else ReceiptState.PARTIAL


def _clip(value: str | None, limit: int) -> str:
    if not value:
        return ""
    return value[:limit]


def build_snapshot(
    *,
    index_fresh: bool,
    edit_state: PostEditState,
    index_measured_at: str = "",
    index_revision: str = "",
    generation_id: str | None = "",
    indexed_commit: str | None = "",
    current_commit: str | None = "",
    files_indexed: int = 0,
    chunks_indexed: int = 0,
    working_tree_dirty: bool = False,
    reindex_required: bool = False,
    now: float | None = None,
    watch_observed_at: str = "",
    watch_observed_epoch: int = 0,
) -> StatusSnapshot:
    """Assemble a snapshot from already-measured lifecycle state.

    ``index_measured_at`` is when the index state being described was
    measured. A full publication passes its own instant; an index-free
    overlay carries the earlier one forward, so repeated overlays cannot
    silently promote a pending edit to covered.
    """
    moment = datetime.now(UTC).timestamp() if now is None else now
    written_at = utc_now_iso(moment)
    measured_at = index_measured_at or written_at
    resolved_generation = _clip(generation_id, _MAX_IDENTITY_LENGTH)
    pending = pending_after_measurement(edit_state=edit_state, index_measured_at=measured_at)
    view_complete = edit_state.is_bounded_view() if pending else True
    return StatusSnapshot(
        state=derive_state(index_fresh=index_fresh, pending=pending),
        written_at=written_at,
        written_at_epoch=int(moment),
        index_measured_at=measured_at,
        index_revision=_clip(index_revision, _MAX_IDENTITY_LENGTH),
        generation_id=resolved_generation,
        indexed_commit=_clip(indexed_commit, _MAX_IDENTITY_LENGTH),
        current_commit=_clip(current_commit, _MAX_IDENTITY_LENGTH),
        files_indexed=files_indexed,
        chunks_indexed=chunks_indexed,
        working_tree_dirty=working_tree_dirty,
        index_fresh=index_fresh,
        reindex_required=reindex_required,
        pending_delta_files=len(edit_state.pending_paths) if pending else 0,
        pending_view_complete=view_complete,
        last_edit_at=_clip(edit_state.last_event_at, _MAX_TIMESTAMP_LENGTH),
        last_edit_client=_clip(edit_state.last_client, _MAX_NAME_LENGTH),
        last_edit_tool=_clip(edit_state.last_tool_name, _MAX_NAME_LENGTH),
        last_successful_sync=_clip(edit_state.synchronized_at, _MAX_TIMESTAMP_LENGTH),
        receipt_state=_derive_receipt_state(
            generation_id=resolved_generation,
            pending_view_complete=view_complete,
            reindex_required=reindex_required,
        ),
        watch_observed_at=_clip(watch_observed_at, _MAX_TIMESTAMP_LENGTH),
        watch_observed_epoch=watch_observed_epoch,
    )


def read_status(repo_root: Path, *, now: float | None = None) -> StatusView:
    """Read the cached snapshot and classify it for display.

    Opens no index, spawns nothing, and parses no source. Every failure mode
    resolves to a distinct display state instead of an exception.
    """
    path = status_snapshot_path(repo_root)
    try:
        raw = path.read_bytes().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return StatusView(state=StatusState.MISSING, detail="no status snapshot published")
    except OSError as exc:
        _log_diagnostic("status_snapshot_read_error", repr(exc), repo_root)
        return StatusView(state=StatusState.CORRUPT, detail=f"unreadable snapshot: {exc}")
    return _classify(raw, repo_root, now=now)


def _classify(raw: str, repo_root: Path, *, now: float | None) -> StatusView:
    """Validate a document and apply the reader-owned states.

    The guards are broad for the same reason R21's parser is: `json.loads`
    raises `JSONDecodeError` on malformed text but `RecursionError` on a
    deeply nested one, and pydantic can raise beyond `ValidationError` for an
    adversarial document. A renderer must get a state, never a traceback.
    """
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        _log_diagnostic("status_snapshot_corrupt", repr(exc), repo_root)
        return StatusView(state=StatusState.CORRUPT, detail="snapshot is not valid JSON")
    if not isinstance(payload, dict):
        _log_diagnostic(
            "status_snapshot_corrupt", f"expected object, got {type(payload).__name__}", repo_root
        )
        return StatusView(state=StatusState.CORRUPT, detail="snapshot is not a JSON object")

    document = cast("dict[str, Any]", payload)
    version = document.get("version")
    # A document with no version, or a version that is not an integer, is
    # malformed rather than from another build: it needs re-publishing, not an
    # upgrade. Only an integer this build does not read is `unsupported`. The
    # shell and TypeScript renderers classify the same way, so all three
    # surfaces give one remedy for one document.
    if not isinstance(version, int) or isinstance(version, bool):
        _log_diagnostic("status_snapshot_corrupt", f"version {version!r}", repo_root)
        return StatusView(
            state=StatusState.CORRUPT, detail="snapshot carries no usable schema version"
        )
    if version != STATUS_SNAPSHOT_VERSION:
        return StatusView(
            state=StatusState.UNSUPPORTED,
            detail=f"snapshot version {version!r}; this archex reads {STATUS_SNAPSHOT_VERSION}",
        )
    try:
        snapshot = StatusSnapshot.model_validate(document)
    except (ValidationError, ValueError, RecursionError) as exc:
        _log_diagnostic("status_snapshot_corrupt", repr(exc), repo_root)
        return StatusView(state=StatusState.CORRUPT, detail="snapshot failed validation")

    moment = datetime.now(UTC).timestamp() if now is None else now
    # A missing or nonsensical measurement instant yields no age rather than a
    # 57-year-old one, so it cannot masquerade as staleness. Same rule in the
    # shell and TypeScript renderers.
    age = max(0, int(moment) - snapshot.written_at_epoch) if snapshot.written_at_epoch > 0 else None
    watch_state = (
        WatchState.ACTIVE
        if snapshot.watch_observed_epoch > 0
        and int(moment) - snapshot.watch_observed_epoch <= WATCH_OBSERVATION_TTL_SECONDS
        else WatchState.UNOBSERVED
    )
    budget = stale_after_seconds()
    if age is not None and age > budget:
        return StatusView(
            state=StatusState.STALE,
            snapshot=snapshot,
            detail=f"measured {age}s ago, past the {budget}s freshness budget",
            age_seconds=age,
            watch_state=watch_state,
        )
    return StatusView(
        state=StatusState(snapshot.state.value),
        snapshot=snapshot,
        age_seconds=age,
        watch_state=watch_state,
    )


def write_snapshot(repo_root: Path, snapshot: StatusSnapshot) -> StatusSnapshot | None:
    """Publish ``snapshot`` atomically; return it, or None if the write was skipped."""
    return _mutate(repo_root, lambda _previous: snapshot)


def _mutate(
    repo_root: Path, transform: Callable[[StatusSnapshot | None], StatusSnapshot | None]
) -> StatusSnapshot | None:
    """Read-modify-write the document under the exclusive lock.

    ``transform`` receives the currently published snapshot (``None`` when
    there is no readable one) and returns the document to publish, or ``None``
    to publish nothing. Reading inside the lock is what keeps two concurrent
    overlays -- an edit event and a watch stamp, say -- from each building on
    the same stale document and dropping the other's contribution.
    """
    path = status_snapshot_path(repo_root)
    if not ensure_parent_dir(path, cwd=repo_root, diagnostics=_DIAGNOSTICS):
        return None
    with _snapshot_lock(repo_root) as acquired:
        if not acquired:
            return None
        updated = transform(read_status(repo_root).snapshot)
        if updated is None:
            return None
        payload = json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        if not write_text_atomic(path, payload, cwd=repo_root, diagnostics=_DIAGNOSTICS):
            return None
        return updated


def publish_status(
    repo_root: Path,
    *,
    index_fresh: bool,
    index_revision: str = "",
    generation_id: str | None = "",
    indexed_commit: str | None = "",
    current_commit: str | None = "",
    files_indexed: int = 0,
    chunks_indexed: int = 0,
    working_tree_dirty: bool = False,
    reindex_required: bool = False,
    watch_observed: bool = False,
    now: float | None = None,
) -> StatusSnapshot | None:
    """Publish a full measurement, preserving an earlier watch observation.

    Called from code that has already paid for the expensive state: the four
    index publication points and the `archex status` inspector. Never raises;
    a failed publish leaves the previous document in place.
    """

    def _build(previous: StatusSnapshot | None) -> StatusSnapshot:
        moment = datetime.now(UTC).timestamp() if now is None else now
        if watch_observed:
            observed_at, observed_epoch = utc_now_iso(moment), int(moment)
        elif previous is not None:
            observed_at, observed_epoch = (
                previous.watch_observed_at,
                previous.watch_observed_epoch,
            )
        else:
            observed_at, observed_epoch = "", 0
        return build_snapshot(
            index_fresh=index_fresh,
            edit_state=_read_edit_state(repo_root),
            index_revision=index_revision,
            generation_id=generation_id,
            indexed_commit=indexed_commit,
            current_commit=current_commit,
            files_indexed=files_indexed,
            chunks_indexed=chunks_indexed,
            working_tree_dirty=working_tree_dirty,
            reindex_required=reindex_required,
            now=moment,
            watch_observed_at=observed_at,
            watch_observed_epoch=observed_epoch,
        )

    try:
        return _mutate(repo_root, _build)
    except Exception as exc:  # noqa: BLE001 - status is display value, never fatal
        _log_diagnostic("status_snapshot_publish_error", repr(exc), repo_root)
        return None


def republish_measurement(
    repo_root: Path, *, generation_id: str, now: float | None = None
) -> StatusSnapshot | None:
    """Re-stamp an existing measurement whose generation is unchanged.

    The cheap half of the publication contract, for the hottest lifecycle
    event: a validated cache hit establishes that the index still describes
    the tree, but re-deriving the index revision costs an O(files) scan and
    hash, and the counts and revision cannot have changed if the generation
    identity has not. So the measurement instant is advanced -- which is what
    keeps the surface out of `stale` -- while every index-derived field is
    carried forward.

    Returns ``None`` when there is no published snapshot to re-stamp, or when
    its generation differs; the caller must then pay for a full publication.
    """

    def _restamp(previous: StatusSnapshot | None) -> StatusSnapshot | None:
        if previous is None or not generation_id or previous.generation_id != generation_id:
            return None
        moment = datetime.now(UTC).timestamp() if now is None else now
        return build_snapshot(
            index_fresh=previous.index_fresh,
            edit_state=_read_edit_state(repo_root),
            index_revision=previous.index_revision,
            generation_id=previous.generation_id,
            indexed_commit=previous.indexed_commit,
            current_commit=previous.current_commit,
            files_indexed=previous.files_indexed,
            chunks_indexed=previous.chunks_indexed,
            working_tree_dirty=previous.working_tree_dirty,
            reindex_required=previous.reindex_required,
            now=moment,
            watch_observed_at=previous.watch_observed_at,
            watch_observed_epoch=previous.watch_observed_epoch,
        )

    try:
        return _mutate(repo_root, _restamp)
    except Exception as exc:  # noqa: BLE001 - status is display value, never fatal
        _log_diagnostic("status_snapshot_publish_error", repr(exc), repo_root)
        return None


def refresh_edit_overlay(repo_root: Path, *, now: float | None = None) -> StatusSnapshot | None:
    """Recompute the cached state from R21 edit state without opening the index.

    An edit event changes what the status surface should say without changing
    what the index contains, so the pending/fresh transition must be
    publishable without paying for an index open.
    """
    return _overlay(repo_root, now=now, watch_observed=False)


def publish_status_watch_observation(
    repo_root: Path, *, now: float | None = None
) -> StatusSnapshot | None:
    """Stamp a watch-driven refresh onto the existing snapshot."""
    return _overlay(repo_root, now=now, watch_observed=True)


def _overlay(repo_root: Path, *, now: float | None, watch_observed: bool) -> StatusSnapshot | None:
    """Re-derive the snapshot from cheap state, carrying index fields over.

    Index-derived fields come from the previous measurement, including the
    instant it was taken and whether the index described the tree then: an
    overlay learns nothing new about the index and must not re-decide either.
    When there is no readable previous measurement there is nothing to
    overlay, and the next index publication creates the document. Runs on a
    client's edit path, so every failure degrades to a diagnostics line.
    """

    def _build(previous: StatusSnapshot | None) -> StatusSnapshot | None:
        if previous is None:
            return None
        moment = datetime.now(UTC).timestamp() if now is None else now
        observed_at = utc_now_iso(moment) if watch_observed else previous.watch_observed_at
        observed_epoch = int(moment) if watch_observed else previous.watch_observed_epoch
        return build_snapshot(
            index_fresh=previous.index_fresh,
            edit_state=_read_edit_state(repo_root),
            index_measured_at=previous.index_measured_at or previous.written_at,
            index_revision=previous.index_revision,
            generation_id=previous.generation_id,
            indexed_commit=previous.indexed_commit,
            current_commit=previous.current_commit,
            files_indexed=previous.files_indexed,
            chunks_indexed=previous.chunks_indexed,
            working_tree_dirty=previous.working_tree_dirty,
            reindex_required=previous.reindex_required,
            now=moment,
            watch_observed_at=observed_at,
            watch_observed_epoch=observed_epoch,
        )

    try:
        return _mutate(repo_root, _build)
    except Exception as exc:  # noqa: BLE001 - runs on a client's edit path
        _log_diagnostic("status_snapshot_publish_error", repr(exc), repo_root)
        return None


def clear_snapshot(repo_root: Path) -> None:
    """Remove the snapshot; a missing document reads as ``missing``.

    Used when the measured subject stops existing -- a deleted or unreadable
    index -- so the surface says "no measurement" instead of replaying a
    measurement of something that is gone.
    """
    try:
        status_snapshot_path(repo_root).unlink(missing_ok=True)
    except OSError as exc:
        _log_diagnostic("status_snapshot_write_error", repr(exc), repo_root)


def _read_edit_state(repo_root: Path) -> PostEditState:
    """Read R21's edit state, importing it only when a write is happening.

    Local import keeps `read_status` -- the path every renderer takes -- free
    of R21's state module, which transitively imports the index store.
    """
    from archex.post_edit.state import read_state

    return read_state(repo_root)


def _snapshot_lock(repo_root: Path, timeout: float | None = None) -> ExclusiveLock:
    return ExclusiveLock(
        _lock_path(repo_root),
        timeout=LOCK_TIMEOUT_SECONDS if timeout is None else timeout,
        cwd=repo_root,
        diagnostics=_DIAGNOSTICS,
    )


def _log_diagnostic(kind: str, detail: str, repo_root: Path) -> None:
    from archex.integrations.hook import log_diagnostic

    log_diagnostic(kind, detail=detail, cwd=str(repo_root))
