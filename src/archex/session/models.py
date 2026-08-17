"""Explicit, revision-aware project-session ledger models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class SessionRecordKind(StrEnum):
    """The explicit project-session facts Archex can retain."""

    ACTIVE_TASK = "active_task"
    DECISION = "decision"
    BLOCKER = "blocker"
    RATIONALE = "rationale"


class SessionRecordState(StrEnum):
    """Lifecycle state for a durable session record."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


class SessionRecord(BaseModel):
    """One explicit, local project-session record."""

    id: str
    kind: SessionRecordKind
    content: str = Field(min_length=1)
    repo_root: str
    worktree_path: str
    branch: str | None = None
    creator: str
    created_at: datetime
    index_revision: str | None = None
    file_path: str | None = None
    symbol_id: str | None = None
    state: SessionRecordState
    invalidated_at: datetime | None = None

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/")
        if normalized.startswith("/") or ".." in normalized.split("/"):
            raise ValueError("file_path must be a relative repository path")
        return normalized

    @model_validator(mode="after")
    def validate_anchor_revision(self) -> SessionRecord:
        anchored = self.file_path is not None or self.symbol_id is not None
        if anchored and self.index_revision is None:
            raise ValueError("anchored records require an index revision")
        return self


class SessionSkippedRecord(BaseModel):
    """A record omitted from a rendered primer with an explicit reason."""

    record_id: str
    reason: str


def _empty_skipped_records() -> list[SessionSkippedRecord]:
    return []


def _empty_session_records() -> list[SessionRecord]:
    return []


class SessionReceipt(BaseModel):
    """Provenance and bounded-rendering result for a session primer."""

    requested_budget: int = Field(gt=0)
    consumed_budget: int = Field(ge=0)
    index_revision: str | None = None
    index_state: str
    worktree_state: str
    changed_file_count: int = Field(ge=0)
    included_record_ids: list[str] = Field(default_factory=list)
    skipped_records: list[SessionSkippedRecord] = Field(default_factory=_empty_skipped_records)
    recommended_next_action: str


class SessionPrimer(BaseModel):
    """A bounded explicit session context bundle and its receipt."""

    ready: bool
    content: str
    records: list[SessionRecord] = Field(default_factory=_empty_session_records)
    receipt: SessionReceipt
