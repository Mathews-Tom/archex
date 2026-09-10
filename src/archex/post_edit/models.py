"""Client-neutral post-edit event and state models.

R21 gives every supported client one shared vocabulary for "the agent just
finished editing these files". Client adapters translate their own upstream
payload into a :class:`PostEditEvent`; the shared state file persists the
accumulated, bounded result as a :class:`PostEditState` that both the impact
renderer and any later status surface read.

Two invariants the rest of the package depends on:

- Every path in a state or event is repository-relative POSIX text. Absolute
  client paths, ``..`` traversal, and symlink escapes are resolved and
  rejected at the boundary (:mod:`archex.post_edit.state`), never stored.
- The state is bounded. ``pending_paths`` is capped at
  :data:`MAX_PENDING_PATHS`; anything beyond the cap is counted in
  ``dropped_path_count`` rather than silently discarded, so a consumer can
  say "this list is incomplete" instead of implying completeness.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

#: Schema version of the persisted post-edit state document. A state file
#: carrying any other version is treated as unreadable (a fresh default state
#: is returned) rather than being coerced -- an older or newer writer's field
#: semantics are not guessable.
POST_EDIT_STATE_VERSION = 1

#: Maximum number of repo-relative paths retained in ``pending_paths``.
#: Beyond this the state records a count instead of the paths, so a runaway
#: codemod cannot grow the file without bound.
MAX_PENDING_PATHS = 200

#: Maximum accepted length of a single incoming path string, in characters.
#: Longer values are rejected at normalization rather than stored.
MAX_PATH_LENGTH = 1024

#: Maximum number of paths accepted from one client event, before the
#: ``MAX_PENDING_PATHS`` state cap is applied. Bounds the work a single
#: malformed or hostile payload can cause inside the hook's deadline.
MAX_EVENT_PATHS = 512


class PostEditStatus(StrEnum):
    """Whether recorded edits have been synchronized into the index yet."""

    #: No edit has been recorded since the last successful synchronization.
    CLEAN = "clean"
    #: At least one edit is recorded and not yet synchronized.
    DIRTY = "dirty"


class PostEditEvent(BaseModel):
    """One normalized post-edit notification from a client adapter."""

    #: Client identifier, matching ``archex.client_setup.ClientName``.
    client: str
    #: The client's own tool name for the edit (``Edit``, ``apply_patch``, ...),
    #: retained verbatim for diagnostics.
    tool_name: str
    #: Repo-relative POSIX paths the client reported as edited.
    paths: list[str] = Field(default_factory=list[str])
    #: UTC ISO-8601 timestamp of when the adapter observed the event.
    observed_at: str


class PostEditState(BaseModel):
    """Bounded, versioned record of edits awaiting index synchronization."""

    version: int = POST_EDIT_STATE_VERSION
    status: PostEditStatus = PostEditStatus.CLEAN
    #: Repo-relative POSIX paths recorded since the last synchronization,
    #: sorted and deduplicated, capped at :data:`MAX_PENDING_PATHS`.
    pending_paths: list[str] = Field(default_factory=list[str])
    #: How many path records the cap forced out of ``pending_paths`` for the
    #: current snapshot. A path dropped, edited again, and dropped again
    #: counts twice, so treat this as an at-least count rather than a
    #: distinct-path total. Any non-zero value means ``pending_paths`` is an
    #: incomplete view of the edits; it resets when the snapshot is retired.
    dropped_path_count: int = 0
    #: UTC ISO-8601 timestamp of the most recent recorded edit.
    last_event_at: str | None = None
    #: Client that produced the most recent recorded edit.
    last_client: str | None = None
    #: Tool name of the most recent recorded edit.
    last_tool_name: str | None = None
    #: Generation id the state was last synchronized against.
    synchronized_generation: str | None = None
    #: UTC ISO-8601 timestamp of the last successful synchronization.
    synchronized_at: str | None = None

    def is_bounded_view(self) -> bool:
        """Whether ``pending_paths`` lists every edit recorded since sync."""
        return self.dropped_path_count == 0
