"""Read-only repo-local project status inspection."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from archex.cache import CacheManager
from archex.config import load_config
from archex.index.adopt import project_index_is_provenanced
from archex.index.delta import compute_working_tree_signature
from archex.index.store import IndexStore
from archex.metrics.storage import MetricsStore, metrics_db_path
from archex.project import ProjectState
from archex.receipt import index_revision_from_store
from archex.serve.generation import read_generation_id


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
    chunks_fts: int = 0
    dogfood_latest_path: Path | None = None
    error: str = ""
    metrics_savings: dict[str, int | float] | None = None
    #: Persisted generation identity and index revision of the inspected
    #: store, empty when there is no readable index. Carried so a caller can
    #: refresh the cached status snapshot (R23) with the same receipt fields
    #: indexing publishes, instead of downgrading it to "unknown".
    generation_id: str = ""
    index_revision: str = ""


#: States in which the index is not read at all, so `ProjectStatus` carries no
#: aggregates for it.
_UNREAD_STATES = frozenset({"uninitialized", "missing_index", "unprovenanced", "corrupt"})


@dataclass(frozen=True)
class IndexFreshness:
    """The lifecycle state `inspect_project_status` reports, without aggregates.

    Latency-bound callers (the annotation hook) need only the state and the
    index path; the file, chunk, language, and metrics aggregates cost more
    than the rest of the check combined on a large index.
    """

    repo_root: Path
    state: str
    index_path: Path
    current_commit: str
    indexed_commit: str
    working_tree: str
    error: str = ""


def inspect_index_freshness(source: str | Path) -> IndexFreshness:
    """Classify the repo-local index as `fresh`, `stale`, `dirty`, or unusable."""
    project = ProjectState.resolve(source)
    current_commit = CacheManager.git_head(str(project.repo_root)) or ""
    index_path = project.index_path

    def unread(state: str, error: str = "") -> IndexFreshness:
        return IndexFreshness(
            repo_root=project.repo_root,
            state=state,
            index_path=index_path,
            current_commit=current_commit,
            indexed_commit="",
            working_tree="unknown",
            error=error,
        )

    if not project.initialized():
        return unread("uninitialized")
    if not index_path.exists():
        return unread("missing_index")
    if not project_index_is_provenanced(project.repo_root, index_path):
        # Every surface built on this inspection serves the store it names —
        # `archex status`, `doctor`, `setup`, the session primer, and the
        # tool-call hooks, which inject index rows straight into an agent's
        # context. `.archex/index.db` is an ordinary file a published
        # repository can commit, so an index without archex's own marker is
        # of unknown origin: it is reported as such rather than read.
        return unread("unprovenanced", "index carries no archex cache marker for this checkout")

    config = load_config(project.repo_root)
    current_signature = compute_working_tree_signature(project.repo_root, config)

    try:
        store = IndexStore(index_path)
    except Exception as exc:
        return unread("corrupt", str(exc))
    try:
        indexed_commit = store.get_metadata("commit_hash") or ""
        indexed_signature = store.get_metadata("working_tree_signature") or ""
        needs_reindex = store.needs_reindex()
    except Exception as exc:
        return unread("corrupt", str(exc))
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
    return IndexFreshness(
        repo_root=project.repo_root,
        state=state,
        index_path=index_path,
        current_commit=current_commit,
        indexed_commit=indexed_commit,
        working_tree="clean" if current_signature == "clean" else "dirty",
    )


def inspect_project_status(source: str | Path) -> ProjectStatus:
    """Inspect repo-local lifecycle state without building an index."""
    freshness = inspect_index_freshness(source)
    project = ProjectState(repo_root=freshness.repo_root)
    dogfood_latest = project.dogfood_dir / "latest.json"
    dogfood_latest_path = dogfood_latest if dogfood_latest.exists() else None

    def unread(state: str, error: str) -> ProjectStatus:
        return ProjectStatus(
            repo_root=freshness.repo_root,
            initialized=state != "uninitialized",
            state=state,
            index_path=freshness.index_path,
            current_commit=freshness.current_commit,
            indexed_commit="",
            working_tree="unknown",
            files_indexed=0,
            chunks_indexed=0,
            chunks_fts=0,
            languages={},
            vector_index_available=False,
            dogfood_latest_path=dogfood_latest_path,
            error=error,
        )

    if freshness.state in _UNREAD_STATES:
        return unread(freshness.state, freshness.error)

    try:
        store = IndexStore(freshness.index_path)
    except Exception as exc:
        return unread("corrupt", str(exc))
    try:
        files_indexed = store.get_file_count()
        chunks_indexed = store.get_chunk_count()
        languages = _language_counts(store.get_file_metadata())
        chunks_fts = store.get_fts_chunk_count()
        generation_id = read_generation_id(store) or ""
        index_revision = index_revision_from_store(store)
    except Exception as exc:
        return unread("corrupt", str(exc))
    finally:
        store.close()

    return ProjectStatus(
        repo_root=freshness.repo_root,
        initialized=True,
        state=freshness.state,
        index_path=freshness.index_path,
        current_commit=freshness.current_commit,
        indexed_commit=freshness.indexed_commit,
        working_tree=freshness.working_tree,
        files_indexed=files_indexed,
        chunks_indexed=chunks_indexed,
        chunks_fts=chunks_fts,
        languages=languages,
        vector_index_available=_vector_index_available(project),
        dogfood_latest_path=dogfood_latest_path,
        metrics_savings=_metrics_savings(freshness.repo_root),
        generation_id=generation_id,
        index_revision=index_revision,
    )


def _metrics_savings(repo_root: Path) -> dict[str, int | float] | None:
    db_path = metrics_db_path()
    if not db_path.exists():
        return None
    try:
        # Open through MetricsStore so the ledger is migrated before querying the
        # targeted-read columns; a raw connection on a pre-migration ledger would
        # raise and silently drop the savings line until another command migrates it.
        with MetricsStore(db_path).connect() as conn:
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
