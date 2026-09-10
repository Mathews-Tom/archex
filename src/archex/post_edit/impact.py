"""Synchronize recorded edits and project a fresh, bounded impact summary.

This is the fail-closed half of R21. The client boundary fails open — an
edit is never blocked or failed — but the content this module produces is
allowed to reach an agent only when the index generation it was derived from
provably describes the current working tree. A stale, incomplete, or
unverifiable generation emits nothing rather than a confident-looking answer
about a tree that has already moved on.

The pipeline is deliberately thin, because every hard part already exists:

1. Read the bounded edit state (:mod:`archex.post_edit.state`).
2. Refresh through the ordinary public entry point
   (:func:`archex.api.index_repository`), which chooses content-hash delta
   synchronization or a full rebuild by the existing ``delta_threshold``
   rule. No second index path is introduced here.
3. Validate freshness against the refreshed store: a persisted generation id
   must exist, the store's recorded working-tree signature must still equal
   the signature recomputed from disk, and the store must not be flagged for
   reindex.
4. Project :func:`archex.impact.analyze_impact` — the existing deterministic
   file-level analysis — into a bounded text block.

The projection deliberately never claims symbol-call blast radius. archex's
dependency edges are file-level imports and references, so the rendered risk
line says exactly that, and every truncation, unindexed path, and dropped
edit is stated rather than implied away.
"""

from __future__ import annotations

import os
import threading
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from archex.api import index_repository
from archex.cache import CacheManager
from archex.config import load_config, load_index_config
from archex.impact import ImpactFileChange, ImpactReport, analyze_impact
from archex.index.delta import compute_working_tree_signature
from archex.integrations.hook import log_diagnostic
from archex.models import RepoSource
from archex.post_edit.models import PostEditState, PostEditStatus
from archex.post_edit.state import mark_synchronized, read_state, utc_now_iso
from archex.receipt import index_revision_from_store
from archex.serve.generation import compute_generation_id, read_generation_id

if TYPE_CHECKING:
    from pathlib import Path

    from archex.index.store import IndexStore
    from archex.models import Config, IndexConfig

#: Wall-clock budget for the whole synchronize-and-report cycle. Larger than
#: the search hook's 500 ms because a delta refresh reparses changed files;
#: still bounded, because the agent is waiting on the hook's exit.
DEFAULT_POST_EDIT_TIMEOUT_SECONDS = 8.0
_TIMEOUT_ENV_VAR = "ARCHEX_POST_EDIT_TIMEOUT_SECONDS"

#: Rendering caps. The point of the block is orientation, not a full report;
#: `archex report impact` remains the unbounded surface.
MAX_RENDERED_CHANGED_FILES = 15
MAX_RENDERED_AFFECTED_FILES = 15
MAX_RENDERED_AFFECTED_TESTS = 10


class PostEditOutcome(StrEnum):
    """Why a post-edit cycle did or did not produce agent-visible content."""

    #: A fresh generation produced a bounded impact block.
    EMITTED = "emitted"
    #: No recorded edit was awaiting synchronization.
    NO_PENDING_EDITS = "no_pending_edits"
    #: The refresh completed but the resulting generation is not provably
    #: current, so nothing is emitted and the state stays dirty.
    STALE = "stale"
    #: Synchronization could not run (no project, unreadable index, refresh
    #: failure). Nothing is emitted.
    UNAVAILABLE = "unavailable"
    #: The cycle exceeded its wall-clock budget and was abandoned.
    TIMED_OUT = "timed_out"


class PostEditFeedback(BaseModel):
    """Result of one synchronize-and-report cycle."""

    outcome: PostEditOutcome
    #: Rendered block for the client, present only for ``EMITTED``.
    text: str | None = None
    #: Generation the impact was derived from, present only for ``EMITTED``.
    generation_id: str | None = None
    #: Index revision receipt of that generation.
    index_revision: str | None = None
    #: Repo-relative paths this cycle consumed from the pending state.
    synchronized_paths: list[str] = Field(default_factory=list[str])
    #: Why a non-``EMITTED`` outcome happened, for diagnostics.
    detail: str | None = None

    def emitted(self) -> bool:
        return self.outcome is PostEditOutcome.EMITTED


def post_edit_timeout_seconds() -> float:
    """Wall-clock budget for one cycle, overridable per environment."""
    raw = os.environ.get(_TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_POST_EDIT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_POST_EDIT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_POST_EDIT_TIMEOUT_SECONDS


def synchronize_and_report_with_timeout(repo_root: Path, *, client: str) -> PostEditFeedback:
    """Run one cycle under a wall-clock budget, degrading to no output.

    A timed-out cycle leaves the state dirty on purpose so the next edit
    event retries. Two details make that promise hold rather than merely be
    stated. The worker runs on a daemon thread, so it can never block
    interpreter shutdown for a caller that returns normally instead of
    exiting the process. And the same deadline is handed to the worker,
    which re-checks it before retiring any pending path -- otherwise an
    abandoned worker could finish late and quietly retire the very edit this
    call just reported as timed out, losing it instead of retrying it.
    """
    timeout = post_edit_timeout_seconds()
    deadline = time.monotonic() + timeout
    result: list[PostEditFeedback] = []

    def run() -> None:
        try:
            result.append(synchronize_and_report(repo_root, client=client, deadline=deadline))
        except BaseException as exc:  # noqa: BLE001 - degrade to no output, never raise
            log_diagnostic("post_edit_internal_error", detail=repr(exc), cwd=str(repo_root))

    worker = threading.Thread(target=run, name="archex-post-edit", daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        log_diagnostic(
            "post_edit_timeout",
            detail=f"synchronize exceeded {timeout}s",
            cwd=str(repo_root),
        )
        return PostEditFeedback(outcome=PostEditOutcome.TIMED_OUT, detail=f"exceeded {timeout}s")
    if not result:
        return PostEditFeedback(
            outcome=PostEditOutcome.UNAVAILABLE, detail="synchronize produced no result"
        )
    return result[0]


def synchronize_and_report(
    repo_root: Path, *, client: str, deadline: float | None = None
) -> PostEditFeedback:
    """Refresh the index for recorded edits and project a fresh impact block.

    ``deadline`` is a :func:`time.monotonic` instant after which this cycle's
    caller has already given up. Passing it keeps a late-finishing cycle from
    retiring pending paths nobody will ever see reported.
    """
    state = read_state(repo_root)
    if state.status is not PostEditStatus.DIRTY or not state.pending_paths:
        return PostEditFeedback(outcome=PostEditOutcome.NO_PENDING_EDITS)

    consumed = list(state.pending_paths)
    source = RepoSource(local_path=str(repo_root))
    try:
        config = load_config(source)
        index_config = load_index_config(source)
        store = index_repository(source, config=config, index_config=index_config)
    except Exception as exc:  # noqa: BLE001 - a refresh failure is never fatal here
        log_diagnostic("post_edit_refresh_error", detail=repr(exc), cwd=str(repo_root))
        return PostEditFeedback(outcome=PostEditOutcome.UNAVAILABLE, detail=repr(exc))

    try:
        stale_reason, generation_id = _validate_generation(store, repo_root, config, index_config)
        if stale_reason is not None or generation_id is None:
            reason = stale_reason or "no persisted generation id"
            log_diagnostic("post_edit_stale_generation", detail=reason, cwd=str(repo_root))
            return PostEditFeedback(outcome=PostEditOutcome.STALE, detail=reason)
        index_revision = index_revision_from_store(store)
        report = analyze_impact(store, _changes_for(repo_root, consumed))
    except Exception as exc:  # noqa: BLE001 - degrade to no output, never raise
        log_diagnostic("post_edit_analysis_error", detail=repr(exc), cwd=str(repo_root))
        return PostEditFeedback(outcome=PostEditOutcome.UNAVAILABLE, detail=repr(exc))
    finally:
        store.close()

    if deadline is not None and time.monotonic() >= deadline:
        log_diagnostic(
            "post_edit_abandoned_after_deadline",
            detail="finished past the caller's budget; edits stay pending",
            cwd=str(repo_root),
        )
        return PostEditFeedback(
            outcome=PostEditOutcome.TIMED_OUT, detail="finished past the caller's budget"
        )

    mark_synchronized(repo_root, generation_id=generation_id, synchronized_paths=consumed)
    text = render_post_edit_block(
        report,
        client=client,
        generation_id=generation_id,
        index_revision=index_revision,
        dropped_path_count=state.dropped_path_count,
    )
    return PostEditFeedback(
        outcome=PostEditOutcome.EMITTED,
        text=text,
        generation_id=generation_id,
        index_revision=index_revision,
        synchronized_paths=consumed,
    )


def _validate_generation(
    store: IndexStore, repo_root: Path, config: Config, index_config: IndexConfig
) -> tuple[str | None, str | None]:
    """Return ``(stale_reason, generation_id)``; a reason means do not emit.

    The load-bearing check re-derives the generation identity from live disk
    state and requires it to equal the one the store published. That is
    strictly stronger than comparing the working-tree signature alone: for a
    clean tree the signature is the constant ``"clean"`` regardless of which
    commit is checked out, so a ``git checkout`` landing between the refresh
    and this check would slip through a signature-only comparison while the
    index still described the previous commit. Recomputing the identity
    folds in the current HEAD, the live chunk and file counts, the schema,
    and the retrieval configuration alongside the signature.

    Two cheaper guards run first so the common stale cases get a precise
    reason: a store that is mid-write or pre-identity has no generation at
    all, and a store flagged for reindex is untrustworthy regardless of what
    the identity says.
    """
    generation_id = read_generation_id(store)
    if generation_id is None:
        return "no persisted generation id", None
    if store.needs_reindex():
        return "store flagged for reindex", None
    recorded_signature = store.get_metadata("working_tree_signature")
    if recorded_signature is None:
        return "no recorded working-tree signature", None
    if compute_working_tree_signature(repo_root, config) != recorded_signature:
        return "working tree changed during synchronization", None

    live_identity = compute_generation_id(
        schema_version=store.get_metadata("schema_version") or "",
        commit_hash=CacheManager.git_head(str(repo_root)),
        working_tree_signature=recorded_signature,
        file_count=store.get_file_count(),
        chunk_count=store.get_chunk_count(),
        index_config=index_config,
    )
    if live_identity != generation_id:
        return "index generation does not describe the current checkout", None
    return None, generation_id


def _changes_for(repo_root: Path, paths: list[str]) -> list[ImpactFileChange]:
    """Map recorded paths onto the change shape `analyze_impact` consumes."""
    return [
        ImpactFileChange(path=path, status="M" if (repo_root / path).exists() else "D")
        for path in sorted(paths)
    ]


def render_post_edit_block(
    report: ImpactReport,
    *,
    client: str,
    generation_id: str,
    index_revision: str,
    dropped_path_count: int = 0,
) -> str:
    """Render a bounded, receipt-bearing, explicitly file-scoped summary."""
    changed = [change.path for change in report.changed_files]
    affected = [path for path in report.affected_files if path not in set(changed)]
    truncated = (
        len(changed) > MAX_RENDERED_CHANGED_FILES
        or len(affected) > MAX_RENDERED_AFFECTED_FILES
        or len(report.affected_tests) > MAX_RENDERED_AFFECTED_TESTS
    )
    complete = not truncated and not report.unmapped_files and dropped_path_count == 0

    lines = [
        f"[archex post-edit receipt] generation={generation_id[:12]} "
        f"index_revision={index_revision[:12]} generated_at={utc_now_iso()} "
        f"client={client} confidence={'complete' if complete else 'partial'}",
        "archex post-edit impact — file-scoped: derived from indexed import and "
        "reference edges between files, not from call-graph analysis.",
    ]
    lines.extend(_section("Edited files", changed, MAX_RENDERED_CHANGED_FILES))
    lines.extend(_section("Files that depend on them", affected, MAX_RENDERED_AFFECTED_FILES))
    lines.extend(
        _section("Tests in the affected set", report.affected_tests, MAX_RENDERED_AFFECTED_TESTS)
    )
    if report.unmapped_files:
        lines.append(
            f"- not indexed ({len(report.unmapped_files)}), so their dependents are "
            f"unknown: {', '.join(report.unmapped_files[:MAX_RENDERED_CHANGED_FILES])}"
        )
    if dropped_path_count:
        lines.append(
            f"- {dropped_path_count} further edited path(s) exceeded the recorded-edit "
            "cap and are not represented above."
        )
    return "\n".join(lines)


def _section(title: str, paths: list[str], cap: int) -> list[str]:
    if not paths:
        return [f"{title}: none."]
    shown = paths[:cap]
    header = f"{title} ({len(paths)}"
    header += f", showing {len(shown)}):" if len(paths) > cap else "):"
    return [header, *(f"- {path}" for path in shown)]


def current_state(repo_root: Path) -> PostEditState:
    """Read-only accessor other surfaces (for example status) can rely on."""
    return read_state(repo_root)
