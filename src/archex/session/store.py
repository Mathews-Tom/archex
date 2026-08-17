"""SQLite persistence for explicit repo-local project-session records."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from archex.session.models import SessionRecord, SessionRecordKind, SessionRecordState

_CREATE_RECORDS = """
CREATE TABLE IF NOT EXISTS session_records (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    repo_root TEXT NOT NULL,
    worktree_path TEXT NOT NULL,
    branch TEXT,
    creator TEXT NOT NULL,
    created_at TEXT NOT NULL,
    index_revision TEXT,
    file_path TEXT,
    symbol_id TEXT,
    state TEXT NOT NULL,
    invalidated_at TEXT
);
"""
_CREATE_ACTIVE_SCOPE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_session_records_scope_state
ON session_records(worktree_path, branch, state, created_at);
"""
_RECORD_COLUMNS = """
id, kind, content, repo_root, worktree_path, branch, creator, created_at,
index_revision, file_path, symbol_id, state, invalidated_at
"""


class SessionLedger:
    """Owns the local, explicit-write-only project-session SQLite ledger."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """The repo-local SQLite sidecar path."""
        return self._path

    def capture(
        self,
        *,
        kind: SessionRecordKind,
        content: str,
        repo_root: Path,
        worktree_path: Path,
        branch: str | None,
        creator: str,
        index_revision: str | None,
        file_path: str | None = None,
        symbol_id: str | None = None,
    ) -> SessionRecord:
        """Persist one explicit record, superseding a prior scoped active task."""
        timestamp = _now()
        record = SessionRecord(
            id=str(uuid.uuid4()),
            kind=kind,
            content=content.strip(),
            repo_root=str(repo_root),
            worktree_path=str(worktree_path),
            branch=branch,
            creator=creator,
            created_at=timestamp,
            index_revision=index_revision,
            file_path=file_path,
            symbol_id=symbol_id,
            state=SessionRecordState.ACTIVE,
        )
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if kind is SessionRecordKind.ACTIVE_TASK:
                conn.execute(
                    """
                    UPDATE session_records
                    SET state = ?
                    WHERE kind = ? AND worktree_path = ? AND branch IS ? AND state = ?
                    """,
                    (
                        SessionRecordState.SUPERSEDED.value,
                        SessionRecordKind.ACTIVE_TASK.value,
                        str(worktree_path),
                        branch,
                        SessionRecordState.ACTIVE.value,
                    ),
                )
            conn.execute(
                f"INSERT INTO session_records ({_RECORD_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _record_values(record),
            )
        return record

    def list_records(
        self,
        *,
        worktree_path: Path,
        branch: str | None,
        include_inactive: bool = False,
    ) -> list[SessionRecord]:
        """List records for exactly one worktree and branch scope."""
        state_clause = "" if include_inactive else "AND state = ?"
        params: tuple[object, ...] = (str(worktree_path), branch)
        if not include_inactive:
            params += (SessionRecordState.ACTIVE.value,)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT {_RECORD_COLUMNS}
                FROM session_records
                WHERE worktree_path = ? AND branch IS ? {state_clause}
                ORDER BY created_at ASC, id ASC
                """,
                params,
            ).fetchall()
        return [_row_to_record(cast("tuple[object, ...]", row)) for row in rows]

    def invalidate(
        self,
        record_id: str,
        *,
        worktree_path: Path,
        branch: str | None,
    ) -> SessionRecord:
        """Invalidate an active record without erasing its audit trail."""
        timestamp = _now()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE session_records
                SET state = ?, invalidated_at = ?
                WHERE id = ? AND worktree_path = ? AND branch IS ? AND state = ?
                """,
                (
                    SessionRecordState.INVALIDATED.value,
                    timestamp.isoformat(),
                    record_id,
                    str(worktree_path),
                    branch,
                    SessionRecordState.ACTIVE.value,
                ),
            ).rowcount
            if updated != 1:
                raise KeyError(f"No active session record with id {record_id!r} in this scope")
            row = conn.execute(
                f"SELECT {_RECORD_COLUMNS} FROM session_records WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Session record {record_id!r} disappeared after invalidation")
        return _row_to_record(cast("tuple[object, ...]", row))

    def delete(
        self,
        record_id: str,
        *,
        worktree_path: Path,
        branch: str | None,
    ) -> None:
        """Permanently delete one explicitly identified record in the current scope."""
        with self._connect() as conn:
            deleted = conn.execute(
                """
                DELETE FROM session_records
                WHERE id = ? AND worktree_path = ? AND branch IS ?
                """,
                (record_id, str(worktree_path), branch),
            ).rowcount
        if deleted != 1:
            raise KeyError(f"No session record with id {record_id!r} in this scope")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(_CREATE_RECORDS)
            conn.execute(_CREATE_ACTIVE_SCOPE_INDEX)
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _record_values(record: SessionRecord) -> tuple[object, ...]:
    return (
        record.id,
        record.kind.value,
        record.content,
        record.repo_root,
        record.worktree_path,
        record.branch,
        record.creator,
        record.created_at.isoformat(),
        record.index_revision,
        record.file_path,
        record.symbol_id,
        record.state.value,
        record.invalidated_at.isoformat() if record.invalidated_at is not None else None,
    )


def _row_to_record(row: tuple[object, ...]) -> SessionRecord:
    invalidated_at = row[12]
    return SessionRecord(
        id=str(row[0]),
        kind=SessionRecordKind(str(row[1])),
        content=str(row[2]),
        repo_root=str(row[3]),
        worktree_path=str(row[4]),
        branch=str(row[5]) if row[5] is not None else None,
        creator=str(row[6]),
        created_at=datetime.fromisoformat(str(row[7])),
        index_revision=str(row[8]) if row[8] is not None else None,
        file_path=str(row[9]) if row[9] is not None else None,
        symbol_id=str(row[10]) if row[10] is not None else None,
        state=SessionRecordState(str(row[11])),
        invalidated_at=datetime.fromisoformat(str(invalidated_at)) if invalidated_at else None,
    )
