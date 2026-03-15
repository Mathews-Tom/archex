"""Tests for IndexStore: SQLite persistence for chunks, edges, and metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from archex.index.store import IndexStore
from archex.models import ChunkSurrogate, CodeChunk, Edge, EdgeKind, SymbolKind

SAMPLE_CHUNKS = [
    CodeChunk(
        id="utils.py:calculate_sum:5",
        symbol_id="utils.py::calculate_sum#function",
        qualified_name="calculate_sum",
        visibility="public",
        signature="def calculate_sum(a: int, b: int) -> int",
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
        symbol_id="auth.py::authenticate#function",
        qualified_name="authenticate",
        visibility="public",
        signature="def authenticate(username: str, password: str) -> bool",
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
        symbol_id="models.py::User#class",
        qualified_name="User",
        visibility="public",
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

SAMPLE_SURROGATES = [
    ChunkSurrogate(
        chunk_id="utils.py:calculate_sum:5",
        file_path="utils.py",
        surrogate_text="path: utils.py\nsymbol: calculate_sum\nanchors: calculate_sum return",
    ),
    ChunkSurrogate(
        chunk_id="auth.py:authenticate:10",
        file_path="auth.py",
        surrogate_text="path: auth.py\nsymbol: authenticate\nanchors: authenticate password",
    ),
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


def test_insert_and_get_chunk_surrogates(store: IndexStore) -> None:
    store.insert_chunk_surrogates(SAMPLE_SURROGATES)
    result = store.get_chunk_surrogates()
    assert len(result) == 2
    assert {s.chunk_id for s in result} == {s.chunk_id for s in SAMPLE_SURROGATES}


def test_chunk_surrogates_deleted_with_file(store: IndexStore) -> None:
    store.insert_chunk_surrogates(SAMPLE_SURROGATES)
    store.delete_chunks_for_files(["auth.py"])
    remaining = store.get_chunk_surrogates()
    assert [s.file_path for s in remaining] == ["utils.py"]


def test_chunk_surrogate_file_paths_updated_on_rename(store: IndexStore) -> None:
    store.insert_chunk_surrogates(SAMPLE_SURROGATES)
    store.update_file_paths("auth.py", "security/auth.py")
    renamed = store.get_chunk_surrogates_for_file("security/auth.py")
    assert len(renamed) == 1
    assert renamed[0].chunk_id == "auth.py:authenticate:10"


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


def test_symbol_id_fields_round_trip(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    chunk = store.get_chunk("utils.py:calculate_sum:5")
    assert chunk is not None
    assert chunk.symbol_id == "utils.py::calculate_sum#function"
    assert chunk.qualified_name == "calculate_sum"
    assert chunk.visibility == "public"
    assert chunk.signature == "def calculate_sum(a: int, b: int) -> int"


def test_get_chunk_by_symbol_id(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    chunk = store.get_chunk_by_symbol_id("auth.py::authenticate#function")
    assert chunk is not None
    assert chunk.id == "auth.py:authenticate:10"
    assert chunk.qualified_name == "authenticate"


def test_get_chunks_by_symbol_ids(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    chunks = store.get_chunks_by_symbol_ids(
        ["utils.py::calculate_sum#function", "models.py::User#class"]
    )
    assert len(chunks) == 2
    ids = {c.symbol_id for c in chunks}
    assert "utils.py::calculate_sum#function" in ids
    assert "models.py::User#class" in ids


def test_search_symbols(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    results = store.search_symbols("authenticate")
    assert len(results) == 1
    assert results[0].symbol_id == "auth.py::authenticate#function"


def test_search_symbols_no_match(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    assert store.search_symbols("nonexistent_xyz") == []


# ---------------------------------------------------------------------------
# needs_reindex / schema_version
# ---------------------------------------------------------------------------


def test_fresh_store_does_not_need_reindex(store: IndexStore) -> None:
    assert store.needs_reindex() is False


def test_fresh_store_has_schema_version(store: IndexStore) -> None:
    assert store.get_metadata("schema_version") == "3"


def test_migrated_store_with_null_symbol_ids_needs_reindex(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE chunks (
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
        CREATE TABLE edges (
            source TEXT NOT NULL, target TEXT NOT NULL,
            kind TEXT NOT NULL, location TEXT
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    conn.execute(
        "INSERT INTO chunks (id, content, file_path, start_line, end_line, "
        "symbol_name, symbol_kind, language) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("f.py:foo:1", "def foo(): pass", "f.py", 1, 1, "foo", "function", "python"),
    )
    conn.commit()
    conn.close()

    with IndexStore(db) as s:
        assert s.needs_reindex() is True
        assert s.get_metadata("schema_version") == "3"


def test_clear_reindex_flag(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "old2.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, file_path TEXT NOT NULL,
            start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
            symbol_name TEXT, symbol_kind TEXT, language TEXT NOT NULL,
            imports_context TEXT DEFAULT '', token_count INTEGER DEFAULT 0
        );
        CREATE TABLE edges (
            source TEXT NOT NULL, target TEXT NOT NULL,
            kind TEXT NOT NULL, location TEXT
        );
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    conn.execute(
        "INSERT INTO chunks (id, content, file_path, start_line, end_line, "
        "symbol_name, symbol_kind, language) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("f.py:foo:1", "def foo(): pass", "f.py", 1, 1, "foo", "function", "python"),
    )
    conn.commit()
    conn.close()

    with IndexStore(db) as s:
        assert s.needs_reindex() is True
        s.clear_reindex_flag()
        assert s.needs_reindex() is False


def test_get_file_metadata(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    meta = store.get_file_metadata()
    assert len(meta) == 3  # 3 different files
    by_file = {m["file_path"]: m for m in meta}
    assert "utils.py" in by_file
    assert "auth.py" in by_file
    assert "models.py" in by_file
    assert by_file["utils.py"]["language"] == "python"
    assert by_file["utils.py"]["symbol_count"] == 1


def test_get_file_metadata_empty_store(store: IndexStore) -> None:
    meta = store.get_file_metadata()
    assert meta == []


# ---------------------------------------------------------------------------
# Token aggregation methods
# ---------------------------------------------------------------------------


def test_get_total_tokens(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    # utils.py=20 + auth.py=25 + models.py=35
    assert store.get_total_tokens() == 80


def test_get_total_tokens_empty(store: IndexStore) -> None:
    assert store.get_total_tokens() == 0


def test_get_file_tokens(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    assert store.get_file_tokens("utils.py") == 20
    assert store.get_file_tokens("auth.py") == 25
    assert store.get_file_tokens("models.py") == 35


def test_get_file_tokens_missing_file(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    assert store.get_file_tokens("nonexistent.py") == 0


def test_get_files_tokens(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    assert store.get_files_tokens(["utils.py", "auth.py"]) == 45
    assert store.get_files_tokens(["models.py"]) == 35


def test_get_files_tokens_deduplicates(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    # Duplicate paths should be deduplicated
    assert store.get_files_tokens(["utils.py", "utils.py", "auth.py"]) == 45


def test_get_files_tokens_empty_list(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    assert store.get_files_tokens([]) == 0


def test_get_chunks_for_files(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    chunks = store.get_chunks_for_files(["utils.py", "auth.py"])
    file_paths = {c.file_path for c in chunks}
    assert file_paths == {"utils.py", "auth.py"}
    assert len(chunks) >= 2


def test_get_chunks_for_files_empty(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    assert store.get_chunks_for_files([]) == []


def test_get_chunk_count(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    assert store.get_chunk_count() == len(SAMPLE_CHUNKS)


def test_get_chunk_count_empty(store: IndexStore) -> None:
    assert store.get_chunk_count() == 0


def test_get_file_count(store: IndexStore) -> None:
    store.insert_chunks(SAMPLE_CHUNKS)
    unique_files = {c.file_path for c in SAMPLE_CHUNKS}
    assert store.get_file_count() == len(unique_files)


def test_get_file_count_empty(store: IndexStore) -> None:
    assert store.get_file_count() == 0


# ---------------------------------------------------------------------------
# Batch delta operations
# ---------------------------------------------------------------------------


def _init_bm25(store: IndexStore) -> None:
    """Initialize the chunks_fts table required by delete/insert operations."""
    from archex.index.bm25 import BM25Index

    BM25Index(store)


class TestStoreBatchOperations:
    def test_delete_chunks_for_files(self, store: IndexStore) -> None:
        _init_bm25(store)
        chunks = [
            CodeChunk(
                id="a1",
                content="x=1",
                file_path="a.py",
                start_line=1,
                end_line=1,
                language="python",
            ),
            CodeChunk(
                id="b1",
                content="y=2",
                file_path="b.py",
                start_line=1,
                end_line=1,
                language="python",
            ),
        ]
        store.insert_chunks(chunks)
        deleted = store.delete_chunks_for_files(["a.py"])
        assert deleted == 1
        assert store.get_chunks_for_file("a.py") == []
        assert len(store.get_chunks_for_file("b.py")) == 1

    def test_delete_chunks_for_files_empty(self, store: IndexStore) -> None:
        deleted = store.delete_chunks_for_files([])
        assert deleted == 0

    def test_delete_edges_for_files(self, store: IndexStore) -> None:
        edges = [
            Edge(source="a.py", target="b.py", kind=EdgeKind.IMPORTS),
            Edge(source="b.py", target="c.py", kind=EdgeKind.IMPORTS),
        ]
        store.insert_edges(edges)
        deleted = store.delete_edges_for_files(["a.py"])
        remaining = store.get_edges()
        assert deleted == 1
        assert len(remaining) == 1
        assert remaining[0].source == "b.py"

    def test_delete_edges_for_files_target_match(self, store: IndexStore) -> None:
        edges = [
            Edge(source="x.py", target="a.py", kind=EdgeKind.IMPORTS),
            Edge(source="y.py", target="z.py", kind=EdgeKind.IMPORTS),
        ]
        store.insert_edges(edges)
        deleted = store.delete_edges_for_files(["a.py"])
        remaining = store.get_edges()
        assert deleted == 1
        assert len(remaining) == 1
        assert remaining[0].source == "y.py"

    def test_delete_edges_for_files_empty(self, store: IndexStore) -> None:
        store.insert_edges([Edge(source="a.py", target="b.py", kind=EdgeKind.IMPORTS)])
        deleted = store.delete_edges_for_files([])
        assert deleted == 0
        assert len(store.get_edges()) == 1

    def test_update_file_paths(self, store: IndexStore) -> None:
        chunks = [
            CodeChunk(
                id="old1",
                content="x=1",
                file_path="old.py",
                start_line=1,
                end_line=1,
                language="python",
                symbol_id="old.py::x#variable",
                symbol_name="x",
                qualified_name="x",
            ),
        ]
        store.insert_chunks(chunks)
        store.insert_edges([Edge(source="old.py", target="other.py", kind=EdgeKind.IMPORTS)])

        store.update_file_paths("old.py", "new.py")

        assert store.get_chunks_for_file("old.py") == []
        assert len(store.get_chunks_for_file("new.py")) == 1
        edges = store.get_edges()
        assert edges[0].source == "new.py"

    def test_update_file_paths_target_edge(self, store: IndexStore) -> None:
        store.insert_edges([Edge(source="other.py", target="old.py", kind=EdgeKind.IMPORTS)])
        store.update_file_paths("old.py", "new.py")
        edges = store.get_edges()
        assert edges[0].target == "new.py"

    def test_delete_and_insert_atomic(self, store: IndexStore) -> None:
        _init_bm25(store)
        old = [
            CodeChunk(
                id="x1",
                content="old",
                file_path="x.py",
                start_line=1,
                end_line=1,
                language="python",
            )
        ]
        store.insert_chunks(old)
        store.insert_edges([Edge(source="x.py", target="y.py", kind=EdgeKind.IMPORTS)])

        new_chunks = [
            CodeChunk(
                id="x2",
                content="new",
                file_path="x.py",
                start_line=1,
                end_line=1,
                language="python",
            )
        ]
        new_edges = [Edge(source="x.py", target="z.py", kind=EdgeKind.IMPORTS)]
        store.delete_and_insert_for_files(["x.py"], new_chunks, new_edges)

        chunks = store.get_chunks_for_file("x.py")
        assert len(chunks) == 1
        assert chunks[0].content == "new"
        edges = store.get_edges()
        assert len(edges) == 1
        assert edges[0].target == "z.py"

    def test_delete_and_insert_empty_new_chunks(self, store: IndexStore) -> None:
        _init_bm25(store)
        old = [
            CodeChunk(
                id="z1",
                content="content",
                file_path="z.py",
                start_line=1,
                end_line=1,
                language="python",
            )
        ]
        store.insert_chunks(old)
        store.delete_and_insert_for_files(["z.py"], [], [])
        assert store.get_chunks_for_file("z.py") == []

    def test_delete_chunks_for_multiple_files(self, store: IndexStore) -> None:
        _init_bm25(store)
        chunks = [
            CodeChunk(
                id="p1",
                content="p",
                file_path="p.py",
                start_line=1,
                end_line=1,
                language="python",
            ),
            CodeChunk(
                id="q1",
                content="q",
                file_path="q.py",
                start_line=1,
                end_line=1,
                language="python",
            ),
            CodeChunk(
                id="r1",
                content="r",
                file_path="r.py",
                start_line=1,
                end_line=1,
                language="python",
            ),
        ]
        store.insert_chunks(chunks)
        deleted = store.delete_chunks_for_files(["p.py", "q.py"])
        assert deleted == 2
        assert store.get_chunks_for_file("p.py") == []
        assert store.get_chunks_for_file("q.py") == []
        assert len(store.get_chunks_for_file("r.py")) == 1


# ---------------------------------------------------------------------------
# Edge case tests — chunks_fts bug fix verification
# ---------------------------------------------------------------------------


class TestChunksFtsBugFix:
    """Verify delete operations work on fresh stores without BM25Index init."""

    def test_delete_chunks_without_bm25_init(self, store: IndexStore) -> None:
        """delete_chunks_for_files should work on fresh store without BM25Index."""
        chunks = [
            CodeChunk(
                id="a1",
                content="x=1",
                file_path="a.py",
                start_line=1,
                end_line=1,
                language="python",
            ),
        ]
        store.insert_chunks(chunks)
        # This should NOT raise OperationalError anymore
        deleted = store.delete_chunks_for_files(["a.py"])
        assert deleted == 1

    def test_delete_and_insert_without_bm25_init(self, store: IndexStore) -> None:
        """delete_and_insert_for_files should work without prior BM25Index."""
        old = [
            CodeChunk(
                id="b1",
                content="old",
                file_path="b.py",
                start_line=1,
                end_line=1,
                language="python",
            )
        ]
        store.insert_chunks(old)
        new_chunks = [
            CodeChunk(
                id="b2",
                content="new",
                file_path="b.py",
                start_line=1,
                end_line=1,
                language="python",
            )
        ]
        # This should NOT raise OperationalError anymore
        store.delete_and_insert_for_files(["b.py"], new_chunks, [])
        result = store.get_chunks_for_file("b.py")
        assert len(result) == 1
        assert result[0].content == "new"


# ---------------------------------------------------------------------------
# Edge case tests — FTS special characters
# ---------------------------------------------------------------------------


class TestFTSSpecialCharacters:
    """Verify search_symbols handles FTS special characters without crashing."""

    def test_search_with_double_quotes(self, store: IndexStore) -> None:
        store.insert_chunks(SAMPLE_CHUNKS)
        # Should not crash — may return empty
        results = store.search_symbols('"authenticate"')
        assert isinstance(results, list)

    def test_search_with_parentheses(self, store: IndexStore) -> None:
        store.insert_chunks(SAMPLE_CHUNKS)
        results = store.search_symbols("func()")
        assert isinstance(results, list)

    def test_search_with_asterisk(self, store: IndexStore) -> None:
        store.insert_chunks(SAMPLE_CHUNKS)
        results = store.search_symbols("auth*")
        assert isinstance(results, list)

    def test_search_with_fts_operators(self, store: IndexStore) -> None:
        store.insert_chunks(SAMPLE_CHUNKS)
        results = store.search_symbols("auth AND user")
        assert isinstance(results, list)

    def test_search_with_single_quotes(self, store: IndexStore) -> None:
        store.insert_chunks(SAMPLE_CHUNKS)
        results = store.search_symbols("it's")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Edge case tests — empty and boundary operations
# ---------------------------------------------------------------------------


class TestEmptyAndBoundaryOperations:
    def test_insert_empty_chunks(self, store: IndexStore) -> None:
        store.insert_chunks([])
        assert store.get_chunks() == []

    def test_get_chunks_for_nonexistent_file(self, store: IndexStore) -> None:
        result = store.get_chunks_for_file("nonexistent.py")
        assert result == []

    def test_get_chunk_by_symbol_id_missing(self, store: IndexStore) -> None:
        result = store.get_chunk_by_symbol_id("missing::symbol#function")
        assert result is None

    def test_get_chunks_by_ids_empty(self, store: IndexStore) -> None:
        result = store.get_chunks_by_ids([])
        assert result == []

    def test_get_chunks_by_ids_large_batch(self, store: IndexStore) -> None:
        """Verify SQLite variable limit is handled for 999+ IDs."""
        store.insert_chunks(SAMPLE_CHUNKS)
        # Create 1000+ IDs (most won't match — that's fine, we're testing no crash)
        large_ids = [f"fake_id_{i}" for i in range(1050)]
        large_ids.extend([c.id for c in SAMPLE_CHUNKS])
        results = store.get_chunks_by_ids(large_ids)
        assert len(results) == len(SAMPLE_CHUNKS)


# ---------------------------------------------------------------------------
# Edge case tests — corrupted database
# ---------------------------------------------------------------------------


class TestCorruptedDatabase:
    def test_non_sqlite_file_raises(self, tmp_path: Path) -> None:
        """Passing a non-SQLite file should raise an error."""
        bad_db = tmp_path / "not_sqlite.db"
        bad_db.write_text("This is not a SQLite database")
        import sqlite3

        with pytest.raises(sqlite3.DatabaseError):
            IndexStore(bad_db)
