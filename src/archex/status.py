"""Read-only repo-local project status inspection."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from archex.cache import CacheManager
from archex.config import load_config
from archex.index.delta import compute_working_tree_signature
from archex.index.store import IndexStore
from archex.metrics.storage import metrics_db_path
from archex.project import ProjectState


@dataclass(frozen=True)
class ProjectStatus:
    repo_root: Path
    initialized: bool
    state: str
    index_path: Path
    current_commit: str
    indexed_commit: str
    working_tree: str
    files_indexed: int
    chunks_indexed: int
    languages: dict[str, int]
    vector_index_available: bool
    dogfood_latest_path: Path | None
    error: str = ""
    metrics_savings: dict[str, int | float] | None = None


def inspect_project_status(source: str | Path) -> ProjectStatus:
    """Inspect repo-local lifecycle state without building an index."""
    project = ProjectState.resolve(source)
    current_commit = CacheManager.git_head(str(project.repo_root)) or ""
    index_path = project.index_path
    dogfood_latest = project.dogfood_dir / "latest.json"

    if not project.initialized():
        return ProjectStatus(
            repo_root=project.repo_root,
            initialized=False,
            state="uninitialized",
            index_path=index_path,
            current_commit=current_commit,
            indexed_commit="",
            working_tree="unknown",
            files_indexed=0,
            chunks_indexed=0,
            languages={},
            vector_index_available=False,
            dogfood_latest_path=dogfood_latest if dogfood_latest.exists() else None,
        )

    if not index_path.exists():
        return ProjectStatus(
            repo_root=project.repo_root,
            initialized=True,
            state="missing_index",
            index_path=index_path,
            current_commit=current_commit,
            indexed_commit="",
            working_tree="unknown",
            files_indexed=0,
            chunks_indexed=0,
            languages={},
            vector_index_available=False,
            dogfood_latest_path=dogfood_latest if dogfood_latest.exists() else None,
        )

    config = load_config(project.repo_root)
    current_signature = compute_working_tree_signature(project.repo_root, config)

    try:
        store = IndexStore(index_path)
    except Exception as exc:
        return ProjectStatus(
            repo_root=project.repo_root,
            initialized=True,
            state="corrupt",
            index_path=index_path,
            current_commit=current_commit,
            indexed_commit="",
            working_tree="unknown",
            files_indexed=0,
            chunks_indexed=0,
            languages={},
            vector_index_available=False,
            dogfood_latest_path=dogfood_latest if dogfood_latest.exists() else None,
            error=str(exc),
        )

    try:
        indexed_commit = store.get_metadata("commit_hash") or ""
        indexed_signature = store.get_metadata("working_tree_signature") or ""
        files_indexed = store.get_file_count()
        chunks_indexed = store.get_chunk_count()
        languages = _language_counts(store.get_file_metadata())
        needs_reindex = store.needs_reindex()
    except Exception as exc:
        return ProjectStatus(
            repo_root=project.repo_root,
            initialized=True,
            state="corrupt",
            index_path=index_path,
            current_commit=current_commit,
            indexed_commit="",
            working_tree="unknown",
            files_indexed=0,
            chunks_indexed=0,
            languages={},
            vector_index_available=False,
            dogfood_latest_path=dogfood_latest if dogfood_latest.exists() else None,
            error=str(exc),
        )
    finally:
        store.close()

    if needs_reindex:
        state = "needs_reindex"
    elif indexed_commit and current_commit and indexed_commit != current_commit:
        state = "stale"
    elif indexed_signature != current_signature:
        state = "dirty"
    else:
        state = "fresh"

    return ProjectStatus(
        repo_root=project.repo_root,
        initialized=True,
        state=state,
        index_path=index_path,
        current_commit=current_commit,
        indexed_commit=indexed_commit,
        working_tree="clean" if current_signature == "clean" else "dirty",
        files_indexed=files_indexed,
        chunks_indexed=chunks_indexed,
        languages=languages,
        vector_index_available=_vector_index_available(project),
        dogfood_latest_path=dogfood_latest if dogfood_latest.exists() else None,
        metrics_savings=_metrics_savings(project.repo_root),
    )


def _metrics_savings(repo_root: Path) -> dict[str, int | float] | None:
    db_path = metrics_db_path()
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            repo = conn.execute(
                "SELECT repo_id FROM repos WHERE repo_root = ?",
                (str(repo_root.resolve()),),
            ).fetchone()
            if repo is None:
                return None
            row = conn.execute(
                """
                SELECT COUNT(*) AS event_count,
                    COALESCE(SUM(tokens_saved), 0) AS tokens_saved,
                    COALESCE(SUM(tokens_returned), 0) AS tokens_returned,
                    COALESCE(SUM(tokens_raw_equivalent), 0) AS tokens_raw_equivalent,
                    COALESCE(SUM(tokens_targeted_read), 0) AS tokens_targeted_read,
                    COALESCE(SUM(tokens_saved_vs_targeted_read), 0) AS tokens_saved_vs_targeted_read
                FROM usage_events
                WHERE repo_id = ?
                """,
                (str(repo["repo_id"]),),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    raw = int(row["tokens_raw_equivalent"])
    saved = int(row["tokens_saved"])
    targeted = int(row["tokens_targeted_read"])
    saved_vs_targeted = int(row["tokens_saved_vs_targeted_read"])
    return {
        "event_count": int(row["event_count"]),
        "tokens_saved": saved,
        "tokens_returned": int(row["tokens_returned"]),
        "tokens_raw_equivalent": raw,
        "tokens_targeted_read": targeted,
        "savings_pct": (saved / raw * 100.0) if raw > 0 else 0.0,
        "savings_pct_vs_targeted_read": (
            (saved_vs_targeted / targeted * 100.0) if targeted > 0 else 0.0
        ),
    }


def _language_counts(file_metadata: list[dict[str, str | int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in file_metadata:
        language = str(item["language"])
        counts[language] = counts.get(language, 0) + 1
    return dict(sorted(counts.items()))


def _vector_index_available(project: ProjectState) -> bool:
    if project.vector_dir.exists() and any(project.vector_dir.glob("*.vectors.npz")):
        return True
    return any(project.project_dir.glob("*.vectors.npz"))
