"""In-process warm serving cache for repeat query() calls against one repo.

``QueryRuntime`` caches, per repository ``cache_key``, the derived retrieval
structures a warm query needs — a BM25 index backed by its own persistent
store connection, the dependency graph, module summaries, and the hydrated
chunk list — keyed by the store's generation ID
(:mod:`archex.serve.generation`). A cache hit skips rebuilding these
entirely; only a generation change (a real content, config, or revision
change, or a store with no generation ID at all) triggers a rebuild.

Scope note (M2, first round): this caches the full per-generation chunk
list once rather than lazily hydrating only search candidates and their
graph neighbors. True candidate-only content hydration would require
making ``CodeChunk.content`` lazily fetched and would touch BM25's
boost-scanning and ``assemble_context``'s graph-expansion path — deferred
pending a real p95/RSS measurement on frozen fixtures, which is the actual
arbiter of whether generation-level caching alone clears the target.

Thread-safety: ``query()`` may be invoked from different threads across
calls (for example a threaded MCP dispatch loop). Each cached snapshot's
store connection opens with ``check_same_thread=False``, and callers must
hold ``snapshot.lock`` for the duration of any operation that touches
``snapshot.store`` (BM25/SPLADE search, boost lookups) — matching the
pre-existing single-connection constraint already documented at those
call sites for a fresh per-query store, just extended across calls that
now share one persistent connection instead of opening a new one each
time. Read-only in-memory attributes (``graph``, ``all_chunks``,
``surrogate_lookup``, ``modules``) need no lock once built.

Revision safety: every query() call re-derives current_generation_id from a
freshly opened, independently-refreshed store (via _ensure_index()) before
ever consulting the runtime, so a cached snapshot is used only when that
fresh check confirms its generation still matches — an in-place delta
applied since the snapshot was built always produces a different
generation ID and forces a rebuild. There is one narrow, pre-existing race
this does not close: if a delta commits to the same on-disk file between a
snapshot being handed out and that call's search executing, SQLite's WAL
snapshot-per-read semantics mean the live search could observe newer
content than the snapshot's frozen ``all_chunks``/``graph``/``modules``
for that one query. Concurrent delta application against the same
cache_key was already unguarded by any lock before this module existed
(_ensure_index() has no cross-call mutual exclusion); this module does not
widen that gap, only makes the common warm-reuse path fast.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from archex.index.bm25 import BM25Index
from archex.index.graph import DependencyGraph
from archex.index.store import IndexStore

if TYPE_CHECKING:
    from archex.models import ChunkSurrogate, CodeChunk, IndexConfig, Module


@dataclass
class WarmSnapshot:
    """Cached derived retrieval structures for one repository generation."""

    generation_id: str
    store: IndexStore
    bm25: BM25Index
    graph: DependencyGraph
    all_chunks: list[CodeChunk]
    surrogate_lookup: dict[str, ChunkSurrogate] | None
    modules: list[Module]
    lock: threading.Lock


def _build_snapshot(
    db_path: str,
    generation_id: str,
    index_config: IndexConfig,
) -> WarmSnapshot:
    """Build a fresh warm snapshot from a dedicated, persistent store connection.

    Mirrors query()'s pre-runtime cached-path hydration exactly (same
    BM25/graph/co-directory-edge/chunk/surrogate/module construction) so
    output stays byte-equivalent whether or not a runtime is used.
    """
    from archex.api import (
        _modules_or_raise,  # pyright: ignore[reportPrivateUsage]
        _surrogate_lookup,  # pyright: ignore[reportPrivateUsage]
    )

    store = IndexStore(db_path, check_same_thread=False)
    try:
        bm25 = BM25Index(
            store,
            identifier_fragment_tokenization=index_config.identifier_fragment_tokenization,
        )
        graph = DependencyGraph.from_edges(store.get_edges())
        if graph.file_edge_count == 0 and graph.file_count > 1:
            graph.add_co_directory_edges()
        all_chunks = store.get_chunks()
        surrogate_lookup = _surrogate_lookup(store, all_chunks, index_config)
        modules = _modules_or_raise(store, index_config)
    except BaseException:
        store.close()
        raise
    return WarmSnapshot(
        generation_id=generation_id,
        store=store,
        bm25=bm25,
        graph=graph,
        all_chunks=all_chunks,
        surrogate_lookup=surrogate_lookup,
        modules=modules,
        lock=threading.Lock(),
    )


class QueryRuntime:
    """Long-lived, generation-keyed warm cache for repeat query() calls.

    One instance is meant to live for the lifetime of a process serving many
    queries (for example an MCP server), shared across all query() calls.
    Never share an instance across processes.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, WarmSnapshot] = {}
        self._registry_lock = threading.Lock()

    def get_or_build(
        self,
        cache_key: str,
        db_path: str,
        current_generation_id: str | None,
        index_config: IndexConfig,
    ) -> WarmSnapshot | None:
        """Return a warm snapshot for cache_key valid at current_generation_id.

        Rebuilds when there is no cached snapshot, or the cached one's
        generation ID no longer matches (a real content/config/revision
        change occurred). Returns None — never caches — when
        current_generation_id is None (a store predating generation-ID
        support, or built without caching enabled): a None can never be
        trusted to mean "the same generation as last time".
        """
        if current_generation_id is None:
            return None
        with self._registry_lock:
            cached = self._snapshots.get(cache_key)
            if cached is not None and cached.generation_id == current_generation_id:
                return cached
            snapshot = _build_snapshot(db_path, current_generation_id, index_config)
            self._snapshots[cache_key] = snapshot
            if cached is not None:
                cached.store.close()
            return snapshot

    def close(self) -> None:
        """Close every cached snapshot's store connection."""
        with self._registry_lock:
            for snapshot in self._snapshots.values():
                snapshot.store.close()
            self._snapshots.clear()
