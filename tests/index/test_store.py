"""Tests for IndexStore: SQLite persistence for chunks, edges, and metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from archex.index.store import IndexStore
from archex.models import CodeChunk, Edge, EdgeKind, SymbolKind

SAMPLE_CHUNKS = [
    CodeChunk(
        id="utils.py:calculate_sum:5",
        content="def calculate_sum(a: int, b: int) -> int:\n    return a + b",
        file_path="utils.py",
        start_line=5,
        end_line=6,
        symbol_name="calculate_sum",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=20,
    ),
    CodeChunk(
        id="auth.py:authenticate:10",
        content=(
            "def authenticate(username: str, password: str) -> bool:\n"
            "    return check_credentials(username, password)"
        ),
        file_path="auth.py",
        start_line=10,
        end_line=11,
        symbol_name="authenticate",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=25,
    ),
    CodeChunk(
        id="models.py:User:1",
        content=(
            "class User:\n"
            "    def __init__(self, name: str, email: str) -> None:\n"
            "        self.name = name\n"
            "        self.email = email"
        ),
        file_path="models.py",
        start_line=1,
        end_line=4,
        symbol_name="User",
        symbol_kind=SymbolKind.CLASS,
        language="python",
        token_count=35,
    ),
]

SAMPLE_EDGES = [
    Edge(source="auth.py", target="models.py", kind=EdgeKind.IMPORTS),
    Edge(source="utils.py", target="auth.py", kind=EdgeKind.CALLS, location="utils.py:12"),
]


@pytest.fixture
def store(tmp_path: Path) -> Generator[IndexStore, None, None]:
    db = tmp_path / "test.db"
    s = IndexStore(db)
    yield s
    s.close()


def test_insert_and_get_all_chunks(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    result = store.get_chunks()
    assert len(result) == 3
    ids = {c.id for c in result}
    assert ids == {c.id for c in SAMPLE_CHUNKS}


def test_data_integrity_round_trip(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    chunk = store.get_chunk("utils.py:calculate_sum:5")
    assert chunk is not None
    assert chunk.content == SAMPLE_CHUNKS[0].content
    assert chunk.file_path == "utils.py"
    assert chunk.start_line == 5
    assert chunk.end_line == 6
    assert chunk.symbol_name == "calculate_sum"
    assert chunk.symbol_kind == SymbolKind.FUNCTION
    assert chunk.language == "python"
    assert chunk.token_count == 20


def test_get_chunk_by_id(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    chunk = store.get_chunk("models.py:User:1")
    assert chunk is not None
    assert chunk.symbol_kind == SymbolKind.CLASS
    assert chunk.symbol_name == "User"


def test_get_chunk_missing_returns_none(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    assert store.get_chunk("nonexistent:id") is None


def test_get_chunks_for_file(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    chunks = store.get_chunks_for_file("auth.py")
    assert len(chunks) == 1
    assert chunks[0].id == "auth.py:authenticate:10"


def test_get_chunks_for_file_no_match(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    assert store.get_chunks_for_file("nonexistent.py") == []


def test_metadata_set_get(store: IndexStore) -> None:
    store.set_metadata("version", "1.0")
    store.set_metadata("built_at", "2026-01-01")
    assert store.get_metadata("version") == "1.0"
    assert store.get_metadata("built_at") == "2026-01-01"


def test_metadata_missing_key_returns_none(store: IndexStore) -> None:
    assert store.get_metadata("missing_key") is None


def test_metadata_overwrite(store: IndexStore) -> None:
    store.set_metadata("key", "first")
    store.set_metadata("key", "second")
    assert store.get_metadata("key") == "second"


def test_data_persists_after_reopen(tmp_path: Path) -> None:
    db = tmp_path / "persist.db"
    with IndexStore(db) as s:
        s.insert_chunks(SAMPLE_CHUNKS)
        s.set_metadata("tag", "test")

    with IndexStore(db) as s2:
        chunks = s2.get_chunks()
        assert len(chunks) == 3
        assert s2.get_metadata("tag") == "test"


def test_context_manager(tmp_path: Path) -> None:
    db = tmp_path / "ctx.db"
    with IndexStore(db) as s:
        s.insert_chunks([SAMPLE_CHUNKS[0]])
        result = s.get_chunks()
    assert len(result) == 1


def test_insert_edges(store: IndexStore) -> None:
    store.insert_edges(SAMPLE_EDGES)
    edges = store.get_edges()
    assert len(edges) == 2
    sources = {e.source for e in edges}
    assert "auth.py" in sources
    assert "utils.py" in sources


def test_edge_kind_preserved(store: IndexStore) -> None:
    store.insert_edges(SAMPLE_EDGES)
    edges = store.get_edges()
    kind_map = {e.source: e.kind for e in edges}
    assert kind_map["auth.py"] == EdgeKind.IMPORTS
    assert kind_map["utils.py"] == EdgeKind.CALLS


def test_empty_store_returns_empty_lists(store: IndexStore) -> None:
    assert store.get_chunks() == []
    assert store.get_edges() == []


def test_chunk_without_symbol_kind(store: IndexStore) -> None:
    chunk = CodeChunk(
        id="bare.py:module:1",
        content="# bare module",
        file_path="bare.py",
        start_line=1,
        end_line=1,
        symbol_kind=None,
        language="python",
    )
    store.insert_chunks([chunk])
    result = store.get_chunk("bare.py:module:1")
    assert result is not None
    assert result.symbol_kind is None


def test_insert_or_replace_deduplicates(store: IndexStore) -> None:
    store.insert_chunks([SAMPLE_CHUNKS[0]])
    modified = SAMPLE_CHUNKS[0].model_copy(update={"content": "# updated"})
    store.insert_chunks([modified])
    chunks = store.get_chunks()
    assert len(chunks) == 1
    assert chunks[0].content == "# updated"


def test_store_closes_connection_on_schema_failure(tmp_path: Path) -> None:
    """IndexStore closes its connection when create_schema raises, preventing resource leak."""
    from unittest.mock import patch

    db = tmp_path / "fail.db"
    with (
        patch.object(IndexStore, "create_schema", side_effect=RuntimeError("schema failure")),
        pytest.raises(RuntimeError, match="schema failure"),
    ):
        IndexStore(db)
    # Connection should be closed; the db file may or may not exist, but no hanging connection


def test_store_connection_accessible(tmp_path: Path) -> None:
    """IndexStore.conn returns the underlying sqlite3.Connection."""
    import sqlite3

    db = tmp_path / "conn.db"
    with IndexStore(db) as s:
        assert isinstance(s.conn, sqlite3.Connection)
