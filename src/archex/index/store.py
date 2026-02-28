"""Unified index store: persist and load BM25, vector, and graph indexes to/from disk."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from archex.models import CodeChunk, Edge, EdgeKind, SymbolKind

if TYPE_CHECKING:
    from pathlib import Path
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
    token_count INTEGER DEFAULT 0
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

_CREATE_IDX_CHUNKS_FILE = "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);"
_CREATE_IDX_EDGES_SOURCE = "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);"
_CREATE_IDX_EDGES_TARGET = "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);"


def _row_to_chunk(row: tuple[object, ...]) -> CodeChunk:
    (
        id_,
        content,
        file_path,
        start_line,
        end_line,
        symbol_name,
        symbol_kind_str,
        language,
        imports_context,
        token_count,
    ) = row
    symbol_kind: SymbolKind | None = SymbolKind(symbol_kind_str) if symbol_kind_str else None
    return CodeChunk(
        id=str(id_),
        content=str(content),
        file_path=str(file_path),
        start_line=int(start_line),  # type: ignore[arg-type]
        end_line=int(end_line),  # type: ignore[arg-type]
        symbol_name=str(symbol_name) if symbol_name is not None else None,
        symbol_kind=symbol_kind,
        language=str(language),
        imports_context=str(imports_context) if imports_context else "",
        token_count=int(token_count) if token_count is not None else 0,  # type: ignore[arg-type]
    )


class IndexStore:
    """SQLite-backed persistence for chunks, edges, and metadata."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self.create_schema()
        except Exception:
            self._conn.close()
            raise

    def create_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            _CREATE_CHUNKS
            + _CREATE_EDGES
            + _CREATE_METADATA
            + _CREATE_IDX_CHUNKS_FILE
            + _CREATE_IDX_EDGES_SOURCE
            + _CREATE_IDX_EDGES_TARGET
        )
        self._conn.commit()

    def insert_chunks(self, chunks: list[CodeChunk]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(id, content, file_path, start_line, end_line, symbol_name, symbol_kind, "
            "language, imports_context, token_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                )
                for c in chunks
            ],
        )
        self._conn.commit()

    def insert_edges(self, edges: list[Edge]) -> None:
        self._conn.executemany(
            "INSERT INTO edges (source, target, kind, location) VALUES (?, ?, ?, ?)",
            [(e.source, e.target, str(e.kind), e.location) for e in edges],
        )
        self._conn.commit()

    def get_chunks(self) -> list[CodeChunk]:
        cur = self._conn.execute(
            "SELECT id, content, file_path, start_line, end_line, symbol_name, symbol_kind, "
            "language, imports_context, token_count FROM chunks"
        )
        return [_row_to_chunk(row) for row in cur.fetchall()]

    def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        cur = self._conn.execute(
            "SELECT id, content, file_path, start_line, end_line, symbol_name, symbol_kind, "
            "language, imports_context, token_count FROM chunks WHERE id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_ids(self, ids: list[str]) -> list[CodeChunk]:
        """Fetch multiple chunks by ID in a single query."""
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        sql = (
            "SELECT id, content, file_path, start_line, end_line, symbol_name, "
            "symbol_kind, language, imports_context, token_count "
            f"FROM chunks WHERE id IN ({placeholders})"
        )
        cur = self._conn.execute(sql, ids)
        return [_row_to_chunk(row) for row in cur.fetchall()]

    def get_chunks_for_file(self, file_path: str) -> list[CodeChunk]:
        cur = self._conn.execute(
            "SELECT id, content, file_path, start_line, end_line, symbol_name, symbol_kind, "
            "language, imports_context, token_count FROM chunks WHERE file_path = ?",
            (file_path,),
        )
        return [_row_to_chunk(row) for row in cur.fetchall()]

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
