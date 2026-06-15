"""Tests for stable symbol identifiers: generation, stability, disambiguation, and round-trip."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from archex.index.store import IndexStore
from archex.models import (
    CodeChunk,
    IndexConfig,
    ParsedFile,
    Symbol,
    SymbolKind,
    make_symbol_id,
)
from archex.parse.adapters.rust import RustAdapter
from archex.parse.engine import TreeSitterEngine
from archex.pipeline.chunker import ASTChunker

if TYPE_CHECKING:
    from collections.abc import Generator


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunker() -> ASTChunker:
    return ASTChunker()


def _chunk_python_source(
    source: bytes, symbols: list[Symbol], path: str = "test.py"
) -> list[CodeChunk]:
    parsed = ParsedFile(
        path=path,
        language="python",
        symbols=symbols,
        imports=[],
        lines=source.count(b"\n") + 1,
    )
    return _chunker().chunk_file(parsed, source)


@pytest.fixture()
def store(tmp_path: Path) -> Generator[IndexStore, None, None]:
    db = tmp_path / "test_symbols.db"
    s = IndexStore(db)
    yield s
    s.close()


@pytest.fixture()
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


# ---------------------------------------------------------------------------
# 1. Stability: same input → same symbol_id
# ---------------------------------------------------------------------------


def test_symbol_id_deterministic() -> None:
    source = b"def hello():\n    pass\n"
    sym = Symbol(
        name="hello",
        qualified_name="hello",
        kind=SymbolKind.FUNCTION,
        file_path="test.py",
        start_line=1,
        end_line=2,
    )
    chunks_a = _chunk_python_source(source, [sym])
    chunks_b = _chunk_python_source(source, [sym])

    ids_a = [c.symbol_id for c in chunks_a]
    ids_b = [c.symbol_id for c in chunks_b]
    assert ids_a == ids_b


def test_symbol_id_stable_across_rechunk() -> None:
    source = b"class Foo:\n    def bar(self):\n        pass\n"
    symbols = [
        Symbol(
            name="Foo",
            qualified_name="Foo",
            kind=SymbolKind.CLASS,
            file_path="test.py",
            start_line=1,
            end_line=1,
        ),
        Symbol(
            name="bar",
            qualified_name="Foo.bar",
            kind=SymbolKind.METHOD,
            file_path="test.py",
            start_line=2,
            end_line=3,
            parent="Foo",
        ),
    ]
    first_run = _chunk_python_source(source, symbols)
    second_run = _chunk_python_source(source, symbols)
    assert [c.symbol_id for c in first_run] == [c.symbol_id for c in second_run]


# ---------------------------------------------------------------------------
# 2. Line-shift invariance
# ---------------------------------------------------------------------------


def test_symbol_id_invariant_to_line_shift() -> None:
    source_original = b"def greet():\n    return 'hi'\n"
    source_shifted = b"\n\n\ndef greet():\n    return 'hi'\n"

    sym_orig = Symbol(
        name="greet",
        qualified_name="greet",
        kind=SymbolKind.FUNCTION,
        file_path="test.py",
        start_line=1,
        end_line=2,
    )
    sym_shifted = Symbol(
        name="greet",
        qualified_name="greet",
        kind=SymbolKind.FUNCTION,
        file_path="test.py",
        start_line=4,
        end_line=5,
    )

    chunks_orig = _chunk_python_source(source_original, [sym_orig])
    chunks_shifted = _chunk_python_source(source_shifted, [sym_shifted])

    orig_id = next(c.symbol_id for c in chunks_orig if c.symbol_name == "greet")
    shifted_id = next(c.symbol_id for c in chunks_shifted if c.symbol_name == "greet")
    assert orig_id == shifted_id

    # Old line-based IDs should differ
    orig_chunk_id = next(c.id for c in chunks_orig if c.symbol_name == "greet")
    shifted_chunk_id = next(c.id for c in chunks_shifted if c.symbol_name == "greet")
    assert orig_chunk_id != shifted_chunk_id


# ---------------------------------------------------------------------------
# 3. Disambiguation
# ---------------------------------------------------------------------------


def test_disambiguation_appends_ordinal() -> None:
    source = b"def foo():\n    x = 1\n" * 50  # large enough to split
    sym = Symbol(
        name="foo",
        qualified_name="foo",
        kind=SymbolKind.FUNCTION,
        file_path="test.py",
        start_line=1,
        end_line=100,
    )
    config = IndexConfig(chunk_max_tokens=20, chunk_min_tokens=5)
    parsed = ParsedFile(
        path="test.py",
        language="python",
        symbols=[sym],
        imports=[],
        lines=100,
    )
    chunker = ASTChunker(config)
    chunks = chunker.chunk_file(parsed, source)

    sym_chunks = [c for c in chunks if c.symbol_name == "foo"]
    assert len(sym_chunks) > 1

    # First chunk keeps original symbol_id, subsequent get @N suffix
    base_id = "test.py::foo#function"
    assert sym_chunks[0].symbol_id == base_id
    for i, chunk in enumerate(sym_chunks[1:], start=2):
        assert chunk.symbol_id == f"{base_id}@{i}"

    # All symbol_ids are unique
    all_ids = [c.symbol_id for c in chunks if c.symbol_id]
    assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# 4. Format for all adapters
# ---------------------------------------------------------------------------


def test_make_symbol_id_format_function() -> None:
    assert make_symbol_id("src/utils.py", "authenticate", SymbolKind.FUNCTION) == (
        "src/utils.py::authenticate#function"
    )


def test_make_symbol_id_format_class() -> None:
    assert make_symbol_id("src/pool.py", "ConnectionPool", SymbolKind.CLASS) == (
        "src/pool.py::ConnectionPool#class"
    )


def test_make_symbol_id_format_method() -> None:
    assert make_symbol_id("src/pool.py", "ConnectionPool.handle", SymbolKind.METHOD) == (
        "src/pool.py::ConnectionPool.handle#method"
    )


def test_make_symbol_id_format_module() -> None:
    assert make_symbol_id("src/utils.py", None, None) == "src/utils.py::_module#module"


def test_python_adapter_qualified_names(engine: TreeSitterEngine) -> None:
    fixture = FIXTURES_DIR / "python_simple" / "services" / "auth.py"
    source = fixture.read_bytes()
    from archex.parse.adapters.python import PythonAdapter

    adapter = PythonAdapter()
    tree = engine.parse_bytes(source, "python")
    symbols = adapter.extract_symbols(tree, source, str(fixture))
    methods = [s for s in symbols if s.kind == SymbolKind.METHOD]
    # Python methods use Class.method format
    for m in methods:
        assert "." in m.qualified_name


def test_rust_adapter_impl_qualified_names(engine: TreeSitterEngine) -> None:
    fixture = FIXTURES_DIR / "rust_simple" / "src" / "models.rs"
    source = fixture.read_bytes()
    adapter = RustAdapter()
    tree = engine.parse_bytes(source, "rust")
    symbols = adapter.extract_symbols(tree, source, str(fixture))
    impl_methods = [s for s in symbols if s.kind == SymbolKind.METHOD and s.parent is not None]
    # Rust impl methods from _extract_function use impl_Type::method
    # Trait method declarations use Trait.method
    for m in impl_methods:
        assert "::" in m.qualified_name or "." in m.qualified_name


def test_go_adapter_qualified_names(engine: TreeSitterEngine) -> None:
    fixture = FIXTURES_DIR / "go_simple" / "handlers.go"
    source = fixture.read_bytes()
    from archex.parse.adapters.go import GoAdapter

    adapter = GoAdapter()
    tree = engine.parse_bytes(source, "go")
    symbols = adapter.extract_symbols(tree, source, str(fixture))
    methods = [s for s in symbols if s.kind == SymbolKind.METHOD]
    for m in methods:
        assert "." in m.qualified_name


def test_typescript_adapter_qualified_names(engine: TreeSitterEngine) -> None:
    fixture = FIXTURES_DIR / "typescript_simple" / "src" / "handlers" / "auth.ts"
    source = fixture.read_bytes()
    from archex.parse.adapters.typescript import TypeScriptAdapter

    adapter = TypeScriptAdapter()
    tree = engine.parse_bytes(source, "typescript")
    symbols = adapter.extract_symbols(tree, source, str(fixture))
    methods = [s for s in symbols if s.kind == SymbolKind.METHOD]
    for m in methods:
        assert "." in m.qualified_name


# ---------------------------------------------------------------------------
# 5. Round-trip through IndexStore
# ---------------------------------------------------------------------------


def test_store_round_trip_symbol_id(store: IndexStore) -> None:
    chunk = CodeChunk(
        id="auth.py:authenticate:10",
        symbol_id="auth.py::authenticate#function",
        qualified_name="authenticate",
        visibility="public",
        signature="def authenticate(user: str) -> bool",
        docstring="Authenticate a user.",
        content="def authenticate(user: str) -> bool:\n    return True",
        file_path="auth.py",
        start_line=10,
        end_line=11,
        symbol_name="authenticate",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=20,
    )
    store.insert_chunks([chunk])
    result = store.get_chunk_by_symbol_id("auth.py::authenticate#function")
    assert result is not None
    assert result.symbol_id == "auth.py::authenticate#function"
    assert result.qualified_name == "authenticate"
    assert result.visibility == "public"
    assert result.signature == "def authenticate(user: str) -> bool"
    assert result.docstring == "Authenticate a user."


def test_store_get_chunk_by_symbol_id_missing(store: IndexStore) -> None:
    assert store.get_chunk_by_symbol_id("nonexistent") is None


# ---------------------------------------------------------------------------
# 6. FTS search
# ---------------------------------------------------------------------------


def test_fts_search_symbols(store: IndexStore) -> None:
    chunks = [
        CodeChunk(
            id="auth.py:authenticate:10",
            symbol_id="auth.py::authenticate#function",
            qualified_name="authenticate",
            content="def authenticate(): pass",
            file_path="auth.py",
            start_line=10,
            end_line=10,
            symbol_name="authenticate",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
        ),
        CodeChunk(
            id="utils.py:format:5",
            symbol_id="utils.py::format_name#function",
            qualified_name="format_name",
            content="def format_name(): pass",
            file_path="utils.py",
            start_line=5,
            end_line=5,
            symbol_name="format_name",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
        ),
    ]
    store.insert_chunks(chunks)
    results = store.search_symbols("authenticate")
    assert len(results) == 1
    assert results[0].symbol_id == "auth.py::authenticate#function"


def test_fts_search_no_results(store: IndexStore) -> None:
    assert store.search_symbols("nonexistent_symbol") == []


def test_fts_search_with_kind_filter(store: IndexStore) -> None:
    chunks = [
        CodeChunk(
            id="a.py:Foo:1",
            symbol_id="a.py::Foo#class",
            qualified_name="Foo",
            content="class Foo: pass",
            file_path="a.py",
            start_line=1,
            end_line=1,
            symbol_name="Foo",
            symbol_kind=SymbolKind.CLASS,
            language="python",
        ),
        CodeChunk(
            id="a.py:Foo.bar:3",
            symbol_id="a.py::Foo.bar#method",
            qualified_name="Foo.bar",
            content="def bar(self): pass",
            file_path="a.py",
            start_line=3,
            end_line=3,
            symbol_name="Foo.bar",
            symbol_kind=SymbolKind.METHOD,
            language="python",
        ),
    ]
    store.insert_chunks(chunks)
    # Search "Foo" with kind=CLASS should return only the class
    results = store.search_symbols("Foo", kind=SymbolKind.CLASS)
    assert len(results) == 1
    assert results[0].symbol_kind == SymbolKind.CLASS


# ---------------------------------------------------------------------------
# 7. Migration
# ---------------------------------------------------------------------------


def test_migration_adds_columns(tmp_path: Path) -> None:
    db = tmp_path / "old_schema.db"

    # Create a store with old schema (no symbol columns)
    import sqlite3

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
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            kind TEXT NOT NULL,
            location TEXT
        );
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    # Insert a row with old schema
    conn.execute(
        "INSERT INTO chunks (id, content, file_path, start_line, end_line, "
        "symbol_name, symbol_kind, language, imports_context, token_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("old.py:foo:1", "def foo(): pass", "old.py", 1, 1, "foo", "function", "python", "", 5),
    )
    conn.commit()
    conn.close()

    # Open with new IndexStore — migration should add columns
    with IndexStore(db) as store:
        chunks = store.get_chunks()
        assert len(chunks) == 1
        assert chunks[0].id == "old.py:foo:1"
        assert chunks[0].content == "def foo(): pass"
        # New columns should be None for old data
        assert chunks[0].symbol_id is None
        assert chunks[0].qualified_name is None
        assert chunks[0].visibility is None

        # Can insert new chunks with symbol_id
        new_chunk = CodeChunk(
            id="new.py:bar:1",
            symbol_id="new.py::bar#function",
            qualified_name="bar",
            content="def bar(): pass",
            file_path="new.py",
            start_line=1,
            end_line=1,
            symbol_name="bar",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
        )
        store.insert_chunks([new_chunk])
        result = store.get_chunk_by_symbol_id("new.py::bar#function")
        assert result is not None
        assert result.qualified_name == "bar"


# ---------------------------------------------------------------------------
# 8. Batch fetch
# ---------------------------------------------------------------------------


def test_batch_fetch_by_symbol_ids(store: IndexStore) -> None:
    chunks = [
        CodeChunk(
            id=f"f{i}.py:fn{i}:{i}",
            symbol_id=f"f{i}.py::fn{i}#function",
            content=f"def fn{i}(): pass",
            file_path=f"f{i}.py",
            start_line=i,
            end_line=i,
            symbol_name=f"fn{i}",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
        )
        for i in range(5)
    ]
    store.insert_chunks(chunks)
    result = store.get_chunks_by_symbol_ids(
        ["f0.py::fn0#function", "f2.py::fn2#function", "f4.py::fn4#function"]
    )
    assert len(result) == 3
    ids = {c.symbol_id for c in result}
    assert ids == {"f0.py::fn0#function", "f2.py::fn2#function", "f4.py::fn4#function"}


def test_batch_fetch_empty_list(store: IndexStore) -> None:
    assert store.get_chunks_by_symbol_ids([]) == []


# ---------------------------------------------------------------------------
# Chunker: symbol_id populated on output
# ---------------------------------------------------------------------------


def test_chunker_populates_symbol_id() -> None:
    source = b"def hello():\n    pass\n"
    sym = Symbol(
        name="hello",
        qualified_name="hello",
        kind=SymbolKind.FUNCTION,
        file_path="test.py",
        start_line=1,
        end_line=2,
    )
    chunks = _chunk_python_source(source, [sym])
    hello_chunk = next(c for c in chunks if c.symbol_name == "hello")
    assert hello_chunk.symbol_id == "test.py::hello#function"
    assert hello_chunk.qualified_name == "hello"
    assert hello_chunk.visibility == "public"


def test_chunker_file_level_chunk_symbol_id() -> None:
    source = b"X = 42\n"
    chunks = _chunk_python_source(source, [], path="mod.py")
    assert len(chunks) >= 1
    for c in chunks:
        assert c.symbol_id is not None
        assert "::_module#module" in c.symbol_id
        assert c.visibility == "public"
