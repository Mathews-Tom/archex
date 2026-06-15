"""BM25 keyword index: build and query a sparse text retrieval index over CodeChunks."""

from __future__ import annotations

import itertools
import logging
import re
import sqlite3
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from archex.index.store import IndexStore
    from archex.models import CodeChunk

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    symbol_name,
    file_path,
    docstring,
    breadcrumbs,
    summary,
    tokenize='porter unicode61'
);
"""

_DROP_FTS_ROWS = "DELETE FROM chunks_fts;"


def _ensure_fts_schema(conn: sqlite3.Connection) -> None:
    """Ensure the FTS table has the current schema (including breadcrumbs and summary columns).

    FTS5 does not support ALTER TABLE, so we detect stale schemas
    by probing for the latest column and recreate if needed.
    """
    try:
        # Probe for the newest columns (breadcrumbs and summary)
        conn.execute("SELECT chunk_id FROM chunks_fts WHERE summary MATCH 'probe' LIMIT 0")
        conn.execute("SELECT chunk_id FROM chunks_fts WHERE breadcrumbs MATCH 'probe' LIMIT 0")
    except sqlite3.OperationalError:
        # breadcrumbs or summary column missing — drop and recreate with new schema
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.execute(_CREATE_FTS)
        conn.commit()


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "archex",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "explicitly",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "implement",
        "implementation",
        "in",
        "is",
        "it",
        "its",
        "manage",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "work",
        "you",
    }
)

_GRADUATED_THRESHOLD = 10

# BM25F column weights: (content, symbol_name, file_path, docstring, breadcrumbs, summary)
_WEIGHTS_DEFAULT: tuple[float, float, float, float, float, float] = (1.0, 10.0, 1.5, 6.0, 5.0, 8.0)
_WEIGHTS_LOW_IDF: tuple[float, float, float, float, float, float] = (1.0, 3.0, 4.0, 2.0, 3.0, 5.0)
_LOW_IDF_THRESHOLD = 2.5
_PATH_TERM_BONUS = 0.2
_RERANK_MULTIPLIER = 2


def _sanitize_tokens(query: str) -> list[str]:
    """Extract and sanitize query tokens, stripping FTS5 operators and stopwords."""
    safe: list[str] = []
    for token in query.split():
        cleaned = re.sub(r"[^a-zA-Z0-9_.]", "", token)
        if cleaned and cleaned.lower() not in _STOPWORDS:
            safe.append(f'"{cleaned}"')
    return safe


def escape_fts_query(query: str) -> str:
    """Sanitize query tokens and AND-join for FTS5 precise matching."""
    safe = _sanitize_tokens(query)
    return " AND ".join(safe) if safe else ""


class BM25Index:
    """BM25 keyword index using SQLite FTS5."""

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        store.conn.execute(_CREATE_FTS)
        _ensure_fts_schema(store.conn)
        store.conn.commit()

    @property
    def has_data(self) -> bool:
        """Check if FTS table already has indexed rows (avoids rebuild on cache hit)."""
        row = self._store.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
        return bool(row and row[0] > 0)

    def build(self, chunks: list[CodeChunk]) -> None:
        from archex.pipeline.chunker import expand_identifiers

        conn = self._store.conn
        conn.execute(_DROP_FTS_ROWS)
        conn.executemany(
            "INSERT INTO chunks_fts "
            "(chunk_id, content, symbol_name, file_path, docstring, breadcrumbs, summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.id,
                    expand_identifiers(c.content),
                    c.symbol_name or "",
                    c.file_path,
                    c.docstring or "",
                    c.breadcrumbs,
                    c.summary or "",
                )
                for c in chunks
            ],
        )
        conn.commit()

    def _execute_fts(
        self,
        escaped: str,
        top_k: int,
        weights: tuple[float, float, float, float, float, float] = _WEIGHTS_DEFAULT,
    ) -> list[tuple[str, float]]:
        """Run a single FTS5 MATCH query, returning (chunk_id, score) pairs."""
        conn = self._store.conn
        w_content, w_symbol, w_path, w_docstring, w_bc, w_summary = weights
        try:
            cur = conn.execute(
                "SELECT chunk_id, "
                f"bm25(chunks_fts, {w_content}, {w_symbol}, {w_path}, "
                f"{w_docstring}, {w_bc}, {w_summary}) AS score "
                "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (escaped, top_k),
            )
        except sqlite3.OperationalError:
            logger.warning("FTS5 query failed for: %s", escaped, exc_info=True)
            return []
        return [(str(row[0]), float(row[1])) for row in cur.fetchall()]

    def _graduated_search(
        self,
        tokens: list[str],
        top_k: int,
        weights: tuple[float, float, float, float, float, float] = _WEIGHTS_DEFAULT,
    ) -> list[tuple[str, float]]:
        """Graduated fallback: AND-all → AND-subsets → OR-all.

        1. AND all terms — if >= threshold results, return
        2. Drop one term at a time (N-1 subsets), merge results
        3. If still < threshold, try pairs of terms, merge
        4. Final fallback: OR all terms
        """
        # Stage 1: AND all terms
        if len(tokens) >= 2:
            and_query = " AND ".join(tokens)
            rows = self._execute_fts(and_query, top_k, weights)
            if len(rows) >= _GRADUATED_THRESHOLD:
                return rows

        # Stage 2: drop one term at a time (N-1 subsets)
        merged: dict[str, float] = {}
        if len(tokens) >= 3:
            for subset in itertools.combinations(tokens, len(tokens) - 1):
                sub_query = " AND ".join(subset)
                sub_rows = self._execute_fts(sub_query, top_k, weights)
                for cid, score in sub_rows:
                    # Keep the best score per chunk across subsets
                    if cid not in merged or score < merged[cid]:  # FTS5 scores are negative
                        merged[cid] = score
            if len(merged) >= _GRADUATED_THRESHOLD:
                return sorted(merged.items(), key=lambda x: x[1])[:top_k]

        # Stage 3: pairs of terms
        if len(tokens) >= 4:
            for pair in itertools.combinations(tokens, 2):
                pair_query = " AND ".join(pair)
                pair_rows = self._execute_fts(pair_query, top_k, weights)
                for cid, score in pair_rows:
                    if cid not in merged or score < merged[cid]:
                        merged[cid] = score
            if len(merged) >= _GRADUATED_THRESHOLD:
                return sorted(merged.items(), key=lambda x: x[1])[:top_k]

        # Stage 4: OR all terms
        or_query = " OR ".join(tokens)
        or_rows = self._execute_fts(or_query, top_k, weights)
        # Merge OR results with anything we found earlier
        for cid, score in or_rows:
            if cid not in merged or score < merged[cid]:
                merged[cid] = score

        return sorted(merged.items(), key=lambda x: x[1])[:top_k]

    def avg_idf(self, query: str) -> float:
        """Compute average Inverse Document Frequency of query terms against the corpus.

        Low AvgIDF (< ~2.0) means query terms are common across the corpus —
        BM25 scores will be flat and unreliable for ranking.
        High AvgIDF means terms are specific — BM25 can discriminate effectively.

        Used as a pre-retrieval Query Performance Prediction signal to decide
        whether to force fusion with vector search.
        """
        import math

        tokens = _sanitize_tokens(query)
        if not tokens:
            return 0.0

        conn = self._store.conn
        total_docs_row = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
        total_docs = total_docs_row[0] if total_docs_row else 0
        if total_docs == 0:
            return 0.0

        idfs: list[float] = []
        for token in tokens:
            try:
                row = conn.execute(
                    "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                    (token,),
                ).fetchone()
                df = row[0] if row else 0
            except sqlite3.OperationalError:
                continue
            if df > 0:
                idfs.append(math.log(total_docs / df))

        return sum(idfs) / len(idfs) if idfs else 0.0

    def _adaptive_weights(self, query: str) -> tuple[float, float, float, float, float, float]:
        """Compute BM25F column weights adapted to query term specificity.

        When query terms are common in the corpus (low avg IDF), symbol_name
        and docstring boosts amplify noise.  Shift weight toward content and
        file_path columns instead.
        """
        idf = self.avg_idf(query)
        if idf >= _LOW_IDF_THRESHOLD:
            return _WEIGHTS_DEFAULT
        # Linear interpolation: low-IDF weights → default weights
        t = max(0.0, idf / _LOW_IDF_THRESHOLD)
        return tuple(
            lo + t * (hi - lo) for lo, hi in zip(_WEIGHTS_LOW_IDF, _WEIGHTS_DEFAULT, strict=False)
        )  # type: ignore[return-value]

    @staticmethod
    def _apply_path_bonus(
        results: list[tuple[CodeChunk, float]],
        tokens: list[str],
    ) -> list[tuple[CodeChunk, float]]:
        """Boost chunks whose file path contains query terms.

        Each matching term adds a multiplicative bonus, promoting files
        whose names/directories align with the query concept
        (e.g. celery/app/task.py for query containing "task").
        """
        clean = [t.strip('"').lower() for t in tokens]
        boosted: list[tuple[CodeChunk, float]] = []
        for chunk, score in results:
            parts = set(re.split(r"[/\\._]", chunk.file_path.lower()))
            matches = sum(1 for t in clean if t in parts)
            multiplier = 1.0 + _PATH_TERM_BONUS * matches
            boosted.append((chunk, score * multiplier))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def search(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        if not query.strip():
            return []

        tokens = _sanitize_tokens(query)
        if not tokens:
            return []

        weights = self._adaptive_weights(query)
        fetch_k = top_k * _RERANK_MULTIPLIER
        rows = self._graduated_search(tokens, fetch_k, weights)

        if not rows:
            return []

        # Batch-fetch all chunks in one query instead of N individual lookups
        chunk_ids = [cid for cid, _ in rows]
        score_map = {cid: -score for cid, score in rows}
        fetched = self._store.get_chunks_by_ids(chunk_ids)

        results: list[tuple[CodeChunk, float]] = []
        for chunk in fetched:
            score = score_map.get(chunk.id, 0.0)
            results.append((chunk, score))

        # Preserve BM25 ranking order, then apply path bonus and trim
        results.sort(key=lambda r: r[1], reverse=True)
        results = self._apply_path_bonus(results, tokens)
        return results[:top_k]
