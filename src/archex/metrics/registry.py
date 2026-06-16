"""Local repository registry for anonymized metrics rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from archex.metrics.storage import MetricsStore


@dataclass(frozen=True)
class RegisteredRepo:
    repo_id: str
    repo_root: Path
    display_name: str


class RepoRegistry:
    """Maps local repo roots to random machine-local UUIDs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._store = MetricsStore(db_path)

    def get_or_create(self, repo_root: str | Path) -> RegisteredRepo:
        root = Path(repo_root).expanduser().resolve()
        now = _utc_now()
        with self._store.connect() as conn, conn:
            existing = conn.execute(
                "SELECT repo_id, repo_root, display_name FROM repos WHERE repo_root = ?",
                (str(root),),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    "UPDATE repos SET last_seen_at = ? WHERE repo_id = ?",
                    (now, str(existing["repo_id"])),
                )
                return RegisteredRepo(
                    repo_id=str(existing["repo_id"]),
                    repo_root=Path(str(existing["repo_root"])),
                    display_name=str(existing["display_name"]),
                )

            repo_id = str(uuid4())
            display_name = root.name or str(root)
            conn.execute(
                """
                INSERT INTO repos(repo_id, repo_root, display_name, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (repo_id, str(root), display_name, now, now),
            )
            return RegisteredRepo(repo_id=repo_id, repo_root=root, display_name=display_name)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
