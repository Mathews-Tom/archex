"""Making an index that came from elsewhere describe *this* checkout.

Two paths install an index this checkout did not build: importing a portable
artifact, and seeding from another worktree of the same repository. Both
inherit the producer's identity metadata, and both must overwrite it before
the store is usable, for two independent reasons.

**Reuse.** The destination's own cache lookup keys on this checkout's
resolved path and revision. A store still claiming the producer's identity
is discarded on the very next command, so the import or seed buys nothing.

**Provenance.** The cache marker (`.archex/index.meta`) records
`sha256("<resolved absolute path>@<commit>")`. Reusing a project-layout
index requires that marker, precisely because everything a database says
about itself lives inside the database — and `.archex/index.db` is an
ordinary file a published repository can commit. Writing the marker is
therefore an assertion this machine makes about a store it has just
validated and installed, not a property the store carries in from outside.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from archex.index.store import IndexStore
    from archex.models import Config, IndexConfig


def stamp_project_index_identity(
    store: IndexStore,
    *,
    repo_root: Path,
    source_identity: str,
    commit_hash: str,
    config: Config,
    index_config: IndexConfig,
) -> None:
    """Record `repo_root`'s identity on an open store, as a full index would.

    These are the same fields, in the same order, that the ordinary
    full-index and delta paths write when they publish a store: revision,
    source identity, measurement time, working-tree signature, and the
    derived generation id. The WAL is checkpointed last so the database file
    alone carries every write — callers publish by copying that file.
    """
    from archex.index.delta import compute_working_tree_signature
    from archex.serve.generation import finalize_generation_id

    store.set_metadata("commit_hash", commit_hash)
    store.set_metadata("source_identity", source_identity)
    store.set_metadata("indexed_at", str(time.time()))
    store.set_metadata("working_tree_signature", compute_working_tree_signature(repo_root, config))
    finalize_generation_id(store, index_config)
    store.conn.execute("PRAGMA wal_checkpoint(FULL)")


def project_index_is_provenanced(repo_root: Path, index_path: Path) -> bool:
    """Whether `index_path` carries archex's own marker for `repo_root`.

    Any surface that serves a repo-local index has to answer this, not just
    the cache-mediated ones: `.archex/index.db` is an ordinary file a
    published repository can commit, and an index database describes itself,
    so its own metadata cannot be the evidence for trusting it.

    The marker is keyed on the checkout's resolved path and the revision the
    index was published at, and authenticated with this machine's secret, so
    both the live HEAD and the revision the store records are tried — a
    legitimate index keeps its marker across commits until the next index
    run republishes it. Reading the store's recorded revision is safe
    because it only selects which key to verify; the secret is what decides
    whether the marker is genuine.
    """
    from archex.cache import CacheManager
    from archex.models import RepoSource

    cache = CacheManager(cache_dir=str(index_path.parent), project_layout=True)
    source = RepoSource(local_path=str(repo_root))
    commits = [CacheManager.git_head(str(repo_root)) or "", _recorded_commit(index_path)]
    return any(
        commit and cache.marker_matches(cache.cache_key(source, head_override=commit))
        for commit in commits
    )


def _recorded_commit(index_path: Path) -> str:
    """Read an index's recorded revision without opening it for writing."""
    import sqlite3
    from urllib.parse import quote

    try:
        conn = sqlite3.connect(f"file:{quote(str(index_path))}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return ""
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = 'commit_hash'").fetchone()
    except sqlite3.Error:
        return ""
    finally:
        conn.close()
    return "" if row is None else str(row[0])
