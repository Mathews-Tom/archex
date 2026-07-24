"""Immutable generation identity for warm-serving snapshot validation.

M2's ``QueryRuntime`` (a long-lived, in-process cache of the BM25 index,
dependency graph, and hydrated chunks for one repository) must know, cheaply
and correctly, whether an already-built in-memory snapshot still matches the
on-disk index it was built from. ``compute_generation_id`` derives a single
canonical hash from every value that would make a warm snapshot unsafe to
reuse without re-reading storage: the resolved commit/working-tree state,
the store's schema version, the store's *actual* current content (chunk and
file counts, read live rather than trusted from a metadata field that a
buggy writer could forget to update), and every retrieval-affecting field of
``IndexConfig``.

Two builds with the same generation ID are guaranteed to produce
byte-identical retrieval output for the same query and token budget: same
source revision/working-tree state, same store schema, same completed
content, and the same retrieval configuration. A generation ID is persisted
as index-store metadata by every code path that finishes publishing a
verified index (full index, and both delta-indexing branches), and is
undefined (``None``) for any store built before this module existed or left
mid-write — callers must never treat a missing generation ID as a match.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.index.store import IndexStore
    from archex.models import IndexConfig

_METADATA_KEY = "generation_id"


def _index_config_fingerprint(index_config: IndexConfig) -> str:
    """Deterministic fingerprint of every ``IndexConfig`` field affecting retrieval output."""
    fields = (
        index_config.bm25,
        index_config.vector,
        index_config.splade,
        index_config.module_prefilter,
        index_config.embedder or "",
        index_config.vector_mode.value,
        index_config.surrogate_version,
        index_config.retrieval_policy.value,
        index_config.rerank,
        index_config.rerank_model or "",
        index_config.rerank_candidate_limit,
        index_config.chunker,
        index_config.chunk_max_tokens,
        index_config.chunk_min_tokens,
        index_config.token_encoding,
        index_config.quantize_vectors,
        index_config.quantize_bits,
        index_config.identifier_fragment_tokenization,
        ",".join(index_config.semantic_evidence_providers),
        ",".join(index_config.runtime_evidence_providers),
    )
    return "|".join(str(field) for field in fields)


def compute_generation_id(
    *,
    schema_version: str,
    commit_hash: str | None,
    working_tree_signature: str | None,
    file_count: int,
    chunk_count: int,
    index_config: IndexConfig,
) -> str:
    """Canonical identity for one verified, complete index generation."""
    payload = "\n".join(
        [
            f"schema_version={schema_version}",
            f"commit_hash={commit_hash or ''}",
            f"working_tree_signature={working_tree_signature or ''}",
            f"file_count={file_count}",
            f"chunk_count={chunk_count}",
            f"index_config={_index_config_fingerprint(index_config)}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finalize_generation_id(store: IndexStore, index_config: IndexConfig) -> str:
    """Compute this store's current generation ID from live state and persist it.

    Must be called after every metadata field the identity depends on
    (``commit_hash``, ``working_tree_signature``, ``schema_version``) has
    already been written for this build or delta, and before the store is
    published, so a fresh reader observes a complete, self-consistent
    identity or none at all. Chunk and file counts are read live from the
    ``chunks`` table rather than trusted from a separately-maintained
    metadata field, so a caller that forgets to update a count field
    elsewhere still gets a correct identity here.
    """
    generation_id = compute_generation_id(
        schema_version=store.get_metadata("schema_version") or "",
        commit_hash=store.get_metadata("commit_hash"),
        working_tree_signature=store.get_metadata("working_tree_signature"),
        file_count=store.get_file_count(),
        chunk_count=store.get_chunk_count(),
        index_config=index_config,
    )
    store.set_metadata(_METADATA_KEY, generation_id)
    return generation_id


def read_generation_id(store: IndexStore) -> str | None:
    """Read the persisted generation ID for the store's current on-disk state.

    ``None`` means the store predates this module or is mid-write; callers
    must treat that as "no valid identity" and never match it against
    another ``None`` to decide a cached snapshot is still safe to reuse.
    """
    return store.get_metadata(_METADATA_KEY)
