"""BM25 keyword index: build and query a sparse text retrieval index over CodeChunks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.index.store import IndexStore
    from archex.models import CodeChunk

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    symbol_name
);
"""

_DROP_FTS_ROWS = "DELETE FROM chunks_fts;"


def escape_fts_query(query: str) -> str:
    """Sanitize query tokens and join with OR for FTS5 partial matching.

    Each token is stripped of all non-alphanumeric/underscore/dot characters
    to eliminate FTS5 special operators (*, :, (), NOT, AND, OR, NEAR).
    """
    tokens = query.split()
    safe: list[str] = []
    for token in tokens:
        cleaned = re.sub(r"[^a-zA-Z0-9_.]", "", token)
        if cleaned:
            safe.append(f'"{cleaned}"')
    return " OR ".join(safe)


class BM25Index:
    """BM25 keyword index using SQLite FTS5."""

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        store.conn.execute(_CREATE_FTS)
        store.conn.commit()

    def build(self, chunks: list[CodeChunk]) -> None:
        conn = self._store.conn
        conn.execute(_DROP_FTS_ROWS)
        conn.executemany(
            "INSERT INTO chunks_fts (chunk_id, content, symbol_name) VALUES (?, ?, ?)",
            [(c.id, c.content, c.symbol_name or "") for c in chunks],
        )
        conn.commit()

    def search(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        if not query.strip():
            return []

        escaped = escape_fts_query(query)
        if not escaped:
            return []

        conn = self._store.conn
        try:
            cur = conn.execute(
                "SELECT chunk_id, bm25(chunks_fts) AS score "
                "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (escaped, top_k),
            )
        except Exception:
            return []

        results: list[tuple[CodeChunk, float]] = []
        for chunk_id, raw_score in cur.fetchall():
            chunk = self._store.get_chunk(str(chunk_id))
            if chunk is None:
                continue
            # FTS5 bm25() returns negative values; negate for positive relevance score
            score = -float(raw_score)
            results.append((chunk, score))

        return results
