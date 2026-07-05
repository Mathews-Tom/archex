"""Unit tests for Precision Symbol Tools API functions (Tier 1).

These tests mock _ensure_index to return a pre-populated IndexStore,
isolating the API function logic from the full acquire→parse→chunk pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from archex.index.store import IndexStore
from archex.models import CodeChunk, SymbolKind


def _make_store(chunks: list[CodeChunk]) -> IndexStore:
    """Create a temporary IndexStore pre-populated with chunks."""
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = IndexStore(db_path)
    store.insert_chunks(chunks)
    return store


def _chunk(
    *,
    file_path: str = "src/main.py",
    name: str = "foo",
    kind: SymbolKind = SymbolKind.FUNCTION,
    start_line: int = 1,
    end_line: int = 5,
    content: str = "def foo(): pass",
    language: str = "python",
    qualified_name: str | None = None,
    visibility: str = "public",
    signature: str | None = None,
    docstring: str | None = None,
    imports_context: str = "",
    token_count: int = 10,
) -> CodeChunk:
    qname = qualified_name or name
    sid = f"{file_path}::{qname}#{kind.value}"
    return CodeChunk(
        id=sid,
        content=content,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol_name=name,
        symbol_kind=kind,
        language=language,
        symbol_id=sid,
        qualified_name=qname,
        visibility=visibility,
        signature=signature,
        docstring=docstring,
        imports_context=imports_context,
        token_count=token_count,
    )


# Shared test chunks
CHUNKS = [
    _chunk(
        file_path="src/main.py",
        name="main",
        kind=SymbolKind.FUNCTION,
        start_line=1,
        end_line=10,
        content="def main():\n    print('hello')",
        signature="def main()",
        token_count=15,
    ),
    _chunk(
        file_path="src/main.py",
        name="MyClass",
        kind=SymbolKind.CLASS,
        start_line=12,
        end_line=30,
        content="class MyClass:\n    pass",
        qualified_name="MyClass",
        signature="class MyClass",
        token_count=20,
    ),
    _chunk(
        file_path="src/main.py",
        name="method_a",
        kind=SymbolKind.METHOD,
        start_line=15,
        end_line=20,
        content="def method_a(self): pass",
        qualified_name="MyClass.method_a",
        signature="def method_a(self)",
        token_count=12,
    ),
    _chunk(
        file_path="src/utils.py",
        name="helper",
        kind=SymbolKind.FUNCTION,
        start_line=1,
        end_line=8,
        content="def helper(): return 42",
        language="python",
        signature="def helper()",
        token_count=10,
    ),
    _chunk(
        file_path="lib/server.ts",
        name="serve",
        kind=SymbolKind.FUNCTION,
        start_line=1,
        end_line=15,
        content="function serve() {}",
        language="typescript",
        signature="function serve()",
        token_count=18,
    ),
]


@pytest.fixture
def populated_store() -> IndexStore:
    return _make_store(CHUNKS)


def _patch_ensure(store: IndexStore):
    """Return a context manager that patches _ensure_index to return the given store."""
    return patch("archex.api._ensure_index", return_value=store)


# ---------------------------------------------------------------------------
# file_tree
# ---------------------------------------------------------------------------


class TestFileTree:
    def test_returns_file_tree_with_entries(self, populated_store: IndexStore) -> None:
        from archex.api import file_tree
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_tree(source)

        assert result.total_files > 0
        assert result.root == "/fake"
        assert "python" in result.languages

    def test_language_filter(self, populated_store: IndexStore) -> None:
        from archex.api import file_tree
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_tree(source, language="typescript")

        assert result.total_files == 1
        assert "typescript" in result.languages
        assert "python" not in result.languages

    def test_language_filter_no_match(self, populated_store: IndexStore) -> None:
        from archex.api import file_tree
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_tree(source, language="rust")

        assert result.total_files == 0
        assert result.entries == []

    def test_depth_limit(self, populated_store: IndexStore) -> None:
        from archex.api import file_tree
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_tree(source, max_depth=1)

        # With max_depth=1, only top-level directories should appear (no nested files)
        assert result.total_files > 0

    def test_directory_entries_are_directories(self, populated_store: IndexStore) -> None:
        from archex.api import file_tree
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_tree(source)

        dirs = [e for e in result.entries if e.is_directory]
        files = [e for e in result.entries if not e.is_directory]
        # src/ and lib/ should be directories
        assert len(dirs) >= 2
        assert len(files) == 0  # all files are nested, not at root

    def test_tree_has_correct_languages(self, populated_store: IndexStore) -> None:
        from archex.api import file_tree
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_tree(source)

        # 2 python files (main.py, utils.py) + 1 typescript file (server.ts)
        assert result.languages.get("python") == 2
        assert result.languages.get("typescript") == 1


# ---------------------------------------------------------------------------
# file_outline
# ---------------------------------------------------------------------------


class TestFileOutline:
    def test_returns_outline_for_known_file(self, populated_store: IndexStore) -> None:
        from archex.api import file_outline
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_outline(source, file_path="src/main.py")

        assert result.file_path == "src/main.py"
        assert result.language == "python"
        assert result.lines > 0
        assert result.token_count_raw > 0

    def test_full_tier_outline_does_not_extract_references(
        self, populated_store: IndexStore
    ) -> None:
        from archex.api import file_outline
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with (
            _patch_ensure(populated_store),
            patch(
                "archex.precision._extract_file_references",
                side_effect=AssertionError("non-STRUCTURED references must not run"),
            ),
        ):
            result = file_outline(source, file_path="src/main.py")

        assert result.references == []

    def test_missing_file_returns_empty_outline(self, populated_store: IndexStore) -> None:
        from archex.api import file_outline
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_outline(source, file_path="nonexistent.py")

        assert result.symbols == []
        assert result.language == "unknown"
        assert result.lines == 0

    def test_parent_child_hierarchy(self, populated_store: IndexStore) -> None:
        from archex.api import file_outline
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_outline(source, file_path="src/main.py")

        # MyClass should be top-level with method_a as a child
        top_names = {s.name for s in result.symbols}
        assert "MyClass" in top_names
        assert "main" in top_names
        # method_a should NOT be top-level (it's a child of MyClass)
        assert "method_a" not in top_names

        # Find MyClass and verify its children
        my_class = next(s for s in result.symbols if s.name == "MyClass")
        child_names = {c.name for c in my_class.children}
        assert "method_a" in child_names

    def test_token_count_raw_sums_chunks(self, populated_store: IndexStore) -> None:
        from archex.api import file_outline
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = file_outline(source, file_path="src/main.py")

        # token_count_raw should be sum of chunk token_counts for that file
        expected = sum(c.token_count for c in CHUNKS if c.file_path == "src/main.py")
        assert result.token_count_raw == expected


# ---------------------------------------------------------------------------
# search_symbols
# ---------------------------------------------------------------------------


class TestSearchSymbols:
    def test_finds_symbols_by_name(self, populated_store: IndexStore) -> None:
        from archex.api import search_symbols
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            results = search_symbols(source, query="main")

        assert len(results) >= 1
        names = {m.name for m in results}
        assert "main" in names

    def test_respects_limit(self, populated_store: IndexStore) -> None:
        from archex.api import search_symbols
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            results = search_symbols(source, query="def", limit=1)

        assert len(results) <= 1

    def test_kind_filter(self, populated_store: IndexStore) -> None:
        from archex.api import search_symbols
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            results = search_symbols(source, query="MyClass", kind="class")

        for m in results:
            assert m.kind == SymbolKind.CLASS

    def test_language_filter(self, populated_store: IndexStore) -> None:
        from archex.api import search_symbols
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            results = search_symbols(source, query="serve", language="typescript")

        for m in results:
            assert m.file_path.endswith(".ts")

    def test_returns_symbol_match_fields(self, populated_store: IndexStore) -> None:
        from archex.api import search_symbols
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            results = search_symbols(source, query="helper")

        if results:
            m = results[0]
            assert m.symbol_id
            assert m.name == "helper"
            assert m.kind == SymbolKind.FUNCTION
            assert m.file_path == "src/utils.py"
            assert m.start_line == 1


# ---------------------------------------------------------------------------
# get_symbol
# ---------------------------------------------------------------------------


class TestGetSymbol:
    def test_returns_source_for_known_id(self, populated_store: IndexStore) -> None:
        from archex.api import get_symbol
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        sid = "src/main.py::main#function"
        with _patch_ensure(populated_store):
            result = get_symbol(source, symbol_id=sid)

        assert result is not None
        assert result.symbol_id == sid
        assert result.name == "main"
        assert result.source == "def main():\n    print('hello')"
        assert result.token_count == 15

    def test_returns_none_for_unknown_id(self, populated_store: IndexStore) -> None:
        from archex.api import get_symbol
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            result = get_symbol(source, symbol_id="nonexistent::fake#function")

        assert result is None

    def test_includes_signature_and_docstring(self) -> None:
        from archex.api import get_symbol
        from archex.models import RepoSource

        chunk = _chunk(
            name="documented",
            signature="def documented(x: int) -> str",
            docstring="A documented function.",
            content="def documented(x: int) -> str:\n    return str(x)",
        )
        store = _make_store([chunk])
        source = RepoSource(local_path="/fake")
        sid = chunk.symbol_id or chunk.id
        with _patch_ensure(store):
            result = get_symbol(source, symbol_id=sid)

        assert result is not None
        assert result.signature == "def documented(x: int) -> str"
        assert result.docstring == "A documented function."


# ---------------------------------------------------------------------------
# get_symbols_batch
# ---------------------------------------------------------------------------


class TestGetSymbolsBatch:
    def test_returns_ordered_results(self, populated_store: IndexStore) -> None:
        from archex.api import get_symbols_batch
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        ids = [
            "src/utils.py::helper#function",
            "src/main.py::main#function",
        ]
        with _patch_ensure(populated_store):
            results = get_symbols_batch(source, symbol_ids=ids)

        assert len(results) == 2
        assert results[0] is not None
        assert results[0].name == "helper"
        assert results[1] is not None
        assert results[1].name == "main"

    def test_none_for_missing_ids(self, populated_store: IndexStore) -> None:
        from archex.api import get_symbols_batch
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        ids = [
            "src/main.py::main#function",
            "nonexistent::missing#function",
        ]
        with _patch_ensure(populated_store):
            results = get_symbols_batch(source, symbol_ids=ids)

        assert len(results) == 2
        assert results[0] is not None
        assert results[0].name == "main"
        assert results[1] is None

    def test_empty_input(self, populated_store: IndexStore) -> None:
        from archex.api import get_symbols_batch
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with _patch_ensure(populated_store):
            results = get_symbols_batch(source, symbol_ids=[])

        assert results == []

    def test_rejects_over_50(self) -> None:
        from archex.api import get_symbols_batch
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        with pytest.raises(ValueError, match="Maximum 50"):
            get_symbols_batch(source, symbol_ids=["id"] * 51)

    def test_all_none_for_all_missing(self, populated_store: IndexStore) -> None:
        from archex.api import get_symbols_batch
        from archex.models import RepoSource

        source = RepoSource(local_path="/fake")
        ids = ["fake::a#function", "fake::b#function"]
        with _patch_ensure(populated_store):
            results = get_symbols_batch(source, symbol_ids=ids)

        assert len(results) == 2
        assert all(r is None for r in results)
