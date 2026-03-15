"""Unified index store: persist and load BM25, vector, and graph indexes to/from disk."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from archex.models import ChunkSurrogate, CodeChunk, Edge, EdgeKind, SymbolKind

if TYPE_CHECKING:
    from types import TracebackType


_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    symbol_name TEXT,
    symbol_kind TEXT,
    language TEXT NOT NULL,
    imports_context TEXT DEFAULT '',
    token_count INTEGER DEFAULT 0,
    symbol_id TEXT,
    qualified_name TEXT,
    visibility TEXT DEFAULT 'public',
    signature TEXT,
    docstring TEXT
);
"""

_CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    kind TEXT NOT NULL,
    location TEXT
);
"""

_CREATE_METADATA = """
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_CHUNK_SURROGATES = """
CREATE TABLE IF NOT EXISTS chunk_surrogates (
    chunk_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    surrogate_text TEXT NOT NULL,
    surrogate_version TEXT NOT NULL
);
"""

_CREATE_IDX_CHUNKS_FILE = "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);"
_CREATE_IDX_CHUNKS_SYMBOL_ID = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_symbol_id ON chunks(symbol_id);"
)
_CREATE_IDX_CHUNKS_SYMBOL_NAME = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_symbol_name ON chunks(symbol_name);"
)
_CREATE_IDX_CHUNKS_SYMBOL_KIND = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_symbol_kind ON chunks(symbol_kind);"
)
_CREATE_IDX_CHUNKS_LANGUAGE = "CREATE INDEX IF NOT EXISTS idx_chunks_language ON chunks(language);"
_CREATE_IDX_CHUNKS_VISIBILITY = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_visibility ON chunks(visibility);"
)
_CREATE_IDX_EDGES_SOURCE = "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);"
_CREATE_IDX_EDGES_TARGET = "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);"
_CREATE_IDX_CHUNK_SURROGATES_FILE = (
    "CREATE INDEX IF NOT EXISTS idx_chunk_surrogates_file ON chunk_surrogates(file_path);"
)

_CREATE_SYMBOLS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
    symbol_id,
    symbol_name,
    qualified_name,
    file_path,
    tokenize='unicode61'
);
"""

_SYMBOL_COLUMNS = ["symbol_id", "qualified_name", "visibility", "signature", "docstring"]


_CHUNK_SELECT = (
    "SELECT id, content, file_path, start_line, end_line, symbol_name, symbol_kind, "
    "language, imports_context, token_count, symbol_id, qualified_name, visibility, "
    "signature, docstring FROM chunks"
)


def _row_to_chunk(row: tuple[object, ...]) -> CodeChunk:
    symbol_kind_str = row[6]
    symbol_kind: SymbolKind | None = SymbolKind(symbol_kind_str) if symbol_kind_str else None
    return CodeChunk(
        id=str(row[0]),
        content=str(row[1]),
        file_path=str(row[2]),
        start_line=int(row[3]),  # type: ignore[arg-type]
        end_line=int(row[4]),  # type: ignore[arg-type]
        symbol_name=str(row[5]) if row[5] is not None else None,
        symbol_kind=symbol_kind,
        language=str(row[7]),
        imports_context=str(row[8]) if row[8] else "",
        token_count=int(row[9]) if row[9] is not None else 0,  # type: ignore[arg-type]
        symbol_id=str(row[10]) if len(row) > 10 and row[10] is not None else None,
        qualified_name=str(row[11]) if len(row) > 11 and row[11] is not None else None,
        visibility=str(row[12]) if len(row) > 12 and row[12] is not None else None,
        signature=str(row[13]) if len(row) > 13 and row[13] is not None else None,
        docstring=str(row[14]) if len(row) > 14 and row[14] is not None else None,
    )


class IndexStore:
    """SQLite-backed persistence for chunks, edges, and metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self.create_schema()
            self._migrate_schema()
        except Exception:
            self._conn.close()
            raise

    def create_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            _CREATE_CHUNKS
            + _CREATE_EDGES
            + _CREATE_METADATA
            + _CREATE_CHUNK_SURROGATES
            + _CREATE_IDX_CHUNKS_FILE
            + _CREATE_IDX_EDGES_SOURCE
            + _CREATE_IDX_EDGES_TARGET
            + _CREATE_IDX_CHUNK_SURROGATES_FILE
        )
        # Create BM25 FTS table so delete operations are safe without prior BM25Index init
        cur.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                content,
                symbol_name,
                file_path,
                tokenize='porter unicode61'
            );
        """)
        self._conn.commit()

    def _insert_chunks_no_commit(self, chunks: list[CodeChunk]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(id, content, file_path, start_line, end_line, symbol_name, symbol_kind, "
            "language, imports_context, token_count, symbol_id, qualified_name, "
            "visibility, signature, docstring) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.id,
                    c.content,
                    c.file_path,
                    c.start_line,
                    c.end_line,
                    c.symbol_name,
                    str(c.symbol_kind) if c.symbol_kind is not None else None,
                    c.language,
                    c.imports_context,
                    c.token_count,
                    c.symbol_id,
                    c.qualified_name,
                    c.visibility,
                    c.signature,
                    c.docstring,
                )
                for c in chunks
            ],
        )
        fts_rows = [
            (c.symbol_id, c.symbol_name, c.qualified_name, c.file_path)
            for c in chunks
            if c.symbol_id is not None
        ]
        if fts_rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO symbols_fts "
                "(symbol_id, symbol_name, qualified_name, file_path) "
                "VALUES (?, ?, ?, ?)",
                fts_rows,
            )

    def insert_chunks(self, chunks: list[CodeChunk]) -> None:
        self._insert_chunks_no_commit(chunks)
        self._conn.commit()

    def _insert_chunk_surrogates_no_commit(self, surrogates: list[ChunkSurrogate]) -> None:
        if not surrogates:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunk_surrogates "
            "(chunk_id, file_path, surrogate_text, surrogate_version) "
            "VALUES (?, ?, ?, ?)",
            [
                (
                    surrogate.chunk_id,
                    surrogate.file_path,
                    surrogate.surrogate_text,
                    surrogate.surrogate_version,
                )
                for surrogate in surrogates
            ],
        )

    def insert_chunk_surrogates(self, surrogates: list[ChunkSurrogate]) -> None:
        self._insert_chunk_surrogates_no_commit(surrogates)
        self._conn.commit()

    def _insert_edges_no_commit(self, edges: list[Edge]) -> None:
        self._conn.executemany(
            "INSERT INTO edges (source, target, kind, location) VALUES (?, ?, ?, ?)",
            [(e.source, e.target, str(e.kind), e.location) for e in edges],
        )

    def insert_edges(self, edges: list[Edge]) -> None:
        self._insert_edges_no_commit(edges)
        self._conn.commit()

    def delete_chunks_for_files(self, file_paths: list[str]) -> int:
        """Delete all chunks and FTS entries for the given file paths."""
        if not file_paths:
            return 0
        placeholders = ",".join("?" for _ in file_paths)
        self._conn.execute(
            f"DELETE FROM chunk_surrogates WHERE file_path IN ({placeholders})",
            file_paths,
        )
        self._conn.execute(
            f"DELETE FROM symbols_fts WHERE file_path IN ({placeholders})",
            file_paths,
        )
        self._conn.execute(
            f"DELETE FROM chunks_fts WHERE chunk_id IN "
            f"(SELECT id FROM chunks WHERE file_path IN ({placeholders}))",
            file_paths,
        )
        cur = self._conn.execute(
            f"DELETE FROM chunks WHERE file_path IN ({placeholders})",
            file_paths,
        )
        self._conn.commit()
        return cur.rowcount

    def delete_edges_for_files(self, file_paths: list[str]) -> int:
        """Delete all edges where source or target is in file_paths."""
        if not file_paths:
            return 0
        placeholders = ",".join("?" for _ in file_paths)
        cur = self._conn.execute(
            f"DELETE FROM edges WHERE source IN ({placeholders}) OR target IN ({placeholders})",
            file_paths + file_paths,
        )
        self._conn.commit()
        return cur.rowcount

    def update_file_paths(self, old_path: str, new_path: str) -> None:
        """Update file path references for a rename across chunks, edges, and FTS."""
        self._conn.execute(
            "UPDATE chunks SET file_path = ? WHERE file_path = ?",
            (new_path, old_path),
        )
        self._conn.execute(
            "UPDATE chunk_surrogates SET file_path = ? WHERE file_path = ?",
            (new_path, old_path),
        )
        self._conn.execute(
            "UPDATE edges SET source = ? WHERE source = ?",
            (new_path, old_path),
        )
        self._conn.execute(
            "UPDATE edges SET target = ? WHERE target = ?",
            (new_path, old_path),
        )
        cur = self._conn.execute(
            "SELECT symbol_id, symbol_name, qualified_name FROM symbols_fts WHERE file_path = ?",
            (old_path,),
        )
        rows = cur.fetchall()
        if rows:
            self._conn.execute(
                "DELETE FROM symbols_fts WHERE file_path = ?",
                (old_path,),
            )
            self._conn.executemany(
                "INSERT INTO symbols_fts (symbol_id, symbol_name, qualified_name, file_path) "
                "VALUES (?, ?, ?, ?)",
                [(r[0], r[1], r[2], new_path) for r in rows],
            )
        self._conn.commit()

    def delete_and_insert_for_files(
        self,
        file_paths: list[str],
        new_chunks: list[CodeChunk],
        new_edges: list[Edge],
        new_surrogates: list[ChunkSurrogate] | None = None,
    ) -> None:
        """Atomically replace chunks and edges for the given files."""
        try:
            if file_paths:
                placeholders = ",".join("?" for _ in file_paths)
                self._conn.execute(
                    f"DELETE FROM chunk_surrogates WHERE file_path IN ({placeholders})",
                    file_paths,
                )
                self._conn.execute(
                    f"DELETE FROM symbols_fts WHERE file_path IN ({placeholders})",
                    file_paths,
                )
                self._conn.execute(
                    f"DELETE FROM chunks_fts WHERE chunk_id IN "
                    f"(SELECT id FROM chunks WHERE file_path IN ({placeholders}))",
                    file_paths,
                )
                self._conn.execute(
                    f"DELETE FROM chunks WHERE file_path IN ({placeholders})",
                    file_paths,
                )
                self._conn.execute(
                    f"DELETE FROM edges WHERE source IN ({placeholders}) "
                    f"OR target IN ({placeholders})",
                    file_paths + file_paths,
                )
            self._insert_chunks_no_commit(new_chunks)
            self._insert_edges_no_commit(new_edges)
            self._insert_chunk_surrogates_no_commit(new_surrogates or [])
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get_chunks(self) -> list[CodeChunk]:
        cur = self._conn.execute(_CHUNK_SELECT)
        return [_row_to_chunk(row) for row in cur.fetchall()]

    def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        cur = self._conn.execute(f"{_CHUNK_SELECT} WHERE id = ?", (chunk_id,))
        row = cur.fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, ids: list[str]) -> list[CodeChunk]:
        """Fetch multiple chunks by ID in a single query."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cur = self._conn.execute(f"{_CHUNK_SELECT} WHERE id IN ({placeholders})", ids)
        return [_row_to_chunk(row) for row in cur.fetchall()]

    def get_chunks_for_file(self, file_path: str) -> list[CodeChunk]:
        cur = self._conn.execute(f"{_CHUNK_SELECT} WHERE file_path = ?", (file_path,))
        return [_row_to_chunk(row) for row in cur.fetchall()]

    def get_chunk_surrogate(self, chunk_id: str) -> ChunkSurrogate | None:
        cur = self._conn.execute(
            "SELECT chunk_id, file_path, surrogate_text, surrogate_version "
            "FROM chunk_surrogates WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return ChunkSurrogate(
            chunk_id=str(row[0]),
            file_path=str(row[1]),
            surrogate_text=str(row[2]),
            surrogate_version=str(row[3]),
        )

    def get_chunk_surrogates(self, chunk_ids: list[str] | None = None) -> list[ChunkSurrogate]:
        if chunk_ids is None:
            cur = self._conn.execute(
                "SELECT chunk_id, file_path, surrogate_text, surrogate_version "
                "FROM chunk_surrogates ORDER BY chunk_id"
            )
        elif not chunk_ids:
            return []
        else:
            placeholders = ",".join("?" for _ in chunk_ids)
            cur = self._conn.execute(
                "SELECT chunk_id, file_path, surrogate_text, surrogate_version "
                f"FROM chunk_surrogates WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
        return [
            ChunkSurrogate(
                chunk_id=str(row[0]),
                file_path=str(row[1]),
                surrogate_text=str(row[2]),
                surrogate_version=str(row[3]),
            )
            for row in cur.fetchall()
        ]

    def get_chunk_surrogates_for_file(self, file_path: str) -> list[ChunkSurrogate]:
        cur = self._conn.execute(
            "SELECT chunk_id, file_path, surrogate_text, surrogate_version "
            "FROM chunk_surrogates WHERE file_path = ? ORDER BY chunk_id",
            (file_path,),
        )
        return [
            ChunkSurrogate(
                chunk_id=str(row[0]),
                file_path=str(row[1]),
                surrogate_text=str(row[2]),
                surrogate_version=str(row[3]),
            )
            for row in cur.fetchall()
        ]

    def get_chunk_by_symbol_id(self, symbol_id: str) -> CodeChunk | None:
        cur = self._conn.execute(f"{_CHUNK_SELECT} WHERE symbol_id = ?", (symbol_id,))
        row = cur.fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_symbol_ids(self, symbol_ids: list[str]) -> list[CodeChunk]:
        if not symbol_ids:
            return []
        placeholders = ",".join("?" for _ in symbol_ids)
        cur = self._conn.execute(f"{_CHUNK_SELECT} WHERE symbol_id IN ({placeholders})", symbol_ids)
        return [_row_to_chunk(row) for row in cur.fetchall()]

    def get_file_metadata(self) -> list[dict[str, str | int]]:
        """Return per-file aggregates: file_path, language, line count, symbol count."""
        rows = self._conn.execute(
            "SELECT file_path, language, MAX(end_line) as lines, COUNT(*) as symbol_count "
            "FROM chunks GROUP BY file_path ORDER BY file_path"
        ).fetchall()
        return [
            {
                "file_path": str(r[0]),
                "language": str(r[1]),
                "lines": int(r[2]),
                "symbol_count": int(r[3]),
            }
            for r in rows
        ]

    def search_symbols(
        self,
        query: str,
        kind: SymbolKind | None = None,
        limit: int = 50,
    ) -> list[CodeChunk]:
        escaped = query.replace('"', '""')
        fts_query = f'"{escaped}"'
        cur = self._conn.execute(
            "SELECT symbol_id FROM symbols_fts WHERE symbols_fts MATCH ? LIMIT ?",
            (fts_query, limit),
        )
        symbol_ids = [str(row[0]) for row in cur.fetchall()]
        if not symbol_ids:
            return []
        chunks = self.get_chunks_by_symbol_ids(symbol_ids)
        if kind is not None:
            chunks = [c for c in chunks if c.symbol_kind == kind]
        return chunks

    def get_total_tokens(self) -> int:
        """Sum of token_count across all chunks in the store."""
        row = self._conn.execute("SELECT COALESCE(SUM(token_count), 0) FROM chunks").fetchone()
        return int(row[0])

    def get_file_tokens(self, file_path: str) -> int:
        """Sum of token_count for chunks in a single file."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM chunks WHERE file_path = ?",
            (file_path,),
        ).fetchone()
        return int(row[0])

    def get_files_tokens(self, file_paths: list[str]) -> int:
        """Sum of token_count across unique files (deduplicates paths)."""
        if not file_paths:
            return 0
        unique = list(set(file_paths))
        placeholders = ",".join("?" for _ in unique)
        row = self._conn.execute(
            f"SELECT COALESCE(SUM(token_count), 0) FROM chunks WHERE file_path IN ({placeholders})",
            unique,
        ).fetchone()
        return int(row[0])

    def search_chunks_by_path_keyword(
        self,
        keyword: str,
        limit: int = 50,
    ) -> list[CodeChunk]:
        """Find chunks whose file_path contains keyword as exact substring (case-insensitive)."""
        cur = self._conn.execute(
            f"{_CHUNK_SELECT} WHERE LOWER(file_path) LIKE ? LIMIT ?",
            (f"%{keyword.lower()}%", limit),
        )
        return [_row_to_chunk(row) for row in cur.fetchall()]

    def get_chunks_for_files(self, file_paths: list[str]) -> list[CodeChunk]:
        """Fetch all chunks for multiple files in a single query."""
        if not file_paths:
            return []
        placeholders = ",".join("?" for _ in file_paths)
        cur = self._conn.execute(f"{_CHUNK_SELECT} WHERE file_path IN ({placeholders})", file_paths)
        return [_row_to_chunk(row) for row in cur.fetchall()]

    def get_chunk_count(self) -> int:
        """Return total number of chunks in the store."""
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0])

    def get_file_count(self) -> int:
        """Return number of distinct files in the store."""
        row = self._conn.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()
        return int(row[0])

    def get_edges(self) -> list[Edge]:
        cur = self._conn.execute("SELECT source, target, kind, location FROM edges")
        return [
            Edge(source=str(r[0]), target=str(r[1]), kind=EdgeKind(r[2]), location=r[3])
            for r in cur.fetchall()
        ]

    def set_metadata(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def get_metadata(self, key: str) -> str | None:
        cur = self._conn.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cur.fetchone()
        return str(row[0]) if row else None

    def needs_reindex(self) -> bool:
        """Return True if the store contains chunks without symbol_ids (pre-stable-ID data)."""
        return self.get_metadata("needs_reindex") == "true"

    def clear_reindex_flag(self) -> None:
        """Clear the needs_reindex flag after a successful full re-index."""
        self._conn.execute("DELETE FROM metadata WHERE key = 'needs_reindex'")
        self._conn.commit()

    def _migrate_schema(self) -> None:
        cur = self._conn.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cur.fetchall()}
        for col in _SYMBOL_COLUMNS:
            if col not in columns:
                self._conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} TEXT")
        # Create indexes and FTS table after columns exist
        self._conn.executescript(
            _CREATE_IDX_CHUNKS_SYMBOL_ID
            + _CREATE_IDX_CHUNKS_SYMBOL_NAME
            + _CREATE_IDX_CHUNKS_SYMBOL_KIND
            + _CREATE_IDX_CHUNKS_LANGUAGE
            + _CREATE_IDX_CHUNKS_VISIBILITY
            + _CREATE_CHUNK_SURROGATES
            + _CREATE_IDX_CHUNK_SURROGATES_FILE
            + _CREATE_SYMBOLS_FTS
        )
        # Set schema version and detect stale data needing re-index
        self.set_metadata("schema_version", "3")
        cur = self._conn.execute("SELECT COUNT(*) FROM chunks WHERE symbol_id IS NULL")
        null_count = cur.fetchone()[0]
        if null_count > 0:
            self.set_metadata("needs_reindex", "true")
        self._conn.commit()

    def vector_index_path_for(
        self,
        *,
        vector_mode: str = "raw",
        surrogate_version: str = "v1",
    ) -> Path:
        """Path to a representation-specific vector index co-located with the database."""
        if vector_mode == "raw":
            return Path(self._db_path).with_suffix(".vectors.npz")
        return Path(self._db_path).with_suffix(f".{vector_mode}.{surrogate_version}.vectors.npz")

    @property
    def vector_index_path(self) -> Path:
        """Backward-compatible raw vector index path."""
        return self.vector_index_path_for()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> IndexStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
