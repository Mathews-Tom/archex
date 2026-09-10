"""Client-neutral post-edit edit tracking (R21).

Client hook adapters call :func:`build_event` and :func:`record_edit` to note
that an agent finished editing files; later stages read the bounded state to
decide whether the index needs synchronizing and whether impact output is
allowed to claim freshness.
"""

from __future__ import annotations

from archex.post_edit.models import (
    MAX_EVENT_PATHS,
    MAX_PATH_LENGTH,
    MAX_PENDING_PATHS,
    POST_EDIT_STATE_VERSION,
    PostEditEvent,
    PostEditState,
    PostEditStatus,
)
from archex.post_edit.state import (
    LOCK_TIMEOUT_SECONDS,
    build_event,
    clear_state,
    mark_synchronized,
    normalize_paths,
    post_edit_state_path,
    read_state,
    record_edit,
    utc_now_iso,
)

__all__ = [
    "LOCK_TIMEOUT_SECONDS",
    "MAX_EVENT_PATHS",
    "MAX_PATH_LENGTH",
    "MAX_PENDING_PATHS",
    "POST_EDIT_STATE_VERSION",
    "PostEditEvent",
    "PostEditState",
    "PostEditStatus",
    "build_event",
    "clear_state",
    "mark_synchronized",
    "normalize_paths",
    "post_edit_state_path",
    "read_state",
    "record_edit",
    "utc_now_iso",
]
