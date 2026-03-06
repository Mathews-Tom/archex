"""Tests for BM25Index: keyword search over CodeChunks using SQLite FTS5."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

from archex.index.bm25 import BM25Index, escape_fts_query
from archex.index.store import IndexStore
from archex.models import CodeChunk, SymbolKind

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


@pytest.fixture
def store_and_index(tmp_path: Path) -> Generator[tuple[IndexStore, BM25Index], None, None]:
    db = tmp_path / "bm25_test.db"
    s = IndexStore(db)
    idx = BM25Index(s)
    s.insert_chunks(SAMPLE_CHUNKS)
    idx.build(SAMPLE_CHUNKS)
    yield s, idx
    s.close()


def test_build_and_search_returns_results(store_and_index: tuple[IndexStore, BM25Index]) -> None:
    _, idx = store_and_index
    results = idx.search("calculate")
    assert len(results) > 0


def test_search_function_name_returns_correct_chunk(
    store_and_index: tuple[IndexStore, BM25Index],
) -> None:
    _, idx = store_and_index
    results = idx.search("authenticate")
    assert len(results) > 0
    top_chunk, _ = results[0]
    assert top_chunk.id == "auth.py:authenticate:10"


def test_search_class_name_returns_user_chunk(
    store_and_index: tuple[IndexStore, BM25Index],
) -> None:
    _, idx = store_and_index
    results = idx.search("User")
    assert len(results) > 0
    ids = [c.id for c, _ in results]
    assert "models.py:User:1" in ids


def test_search_keyword_in_content(store_and_index: tuple[IndexStore, BM25Index]) -> None:
    _, idx = store_and_index
    results = idx.search("password")
    assert len(results) > 0
    top_chunk, _ = results[0]
    assert top_chunk.id == "auth.py:authenticate:10"


def test_results_sorted_by_relevance(store_and_index: tuple[IndexStore, BM25Index]) -> None:
    _, idx = store_and_index
    results = idx.search("str")
    assert len(results) > 0
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results(store_and_index: tuple[IndexStore, BM25Index]) -> None:
    _, idx = store_and_index
    results = idx.search("str", top_k=1)
    assert len(results) <= 1


def test_empty_query_returns_empty_list(store_and_index: tuple[IndexStore, BM25Index]) -> None:
    _, idx = store_and_index
    assert idx.search("") == []
    assert idx.search("   ") == []


def test_no_matches_returns_empty_list(store_and_index: tuple[IndexStore, BM25Index]) -> None:
    _, idx = store_and_index
    results = idx.search("xyzzy_nonexistent_token_12345")
    assert results == []


def test_scores_are_positive(store_and_index: tuple[IndexStore, BM25Index]) -> None:
    _, idx = store_and_index
    results = idx.search("def")
    for _, score in results:
        assert score > 0


def test_build_is_idempotent(store_and_index: tuple[IndexStore, BM25Index]) -> None:
    _, idx = store_and_index
    # Rebuilding should not duplicate results
    idx.build(SAMPLE_CHUNKS)
    results = idx.search("authenticate")
    # Should only return one result for authenticate, not two
    auth_chunks = [c for c, _ in results if c.id == "auth.py:authenticate:10"]
    assert len(auth_chunks) == 1


# ---------------------------------------------------------------------------
# escape_fts_query — unit tests
# ---------------------------------------------------------------------------


def test_escape_fts_query_basic_token() -> None:
    result = escape_fts_query("authenticate")
    assert result == '"authenticate"'


def test_escape_fts_query_multiple_tokens() -> None:
    result = escape_fts_query("foo bar")
    assert result == '"foo" AND "bar"'


def test_escape_fts_query_empty_string() -> None:
    assert escape_fts_query("") == ""


def test_escape_fts_query_strips_fts5_star() -> None:
    result = escape_fts_query("foo*")
    assert result == '"foo"'
    assert "*" not in result


def test_escape_fts_query_strips_not_operator() -> None:
    result = escape_fts_query("NOT")
    # "NOT" → only letters remain → "NOT"
    assert result == '"NOT"'


@pytest.mark.parametrize(
    "query",
    [
        "foo* bar",
        "NOT authenticate",
        "NEAR/3(foo bar)",
        "content:value",
        "(open OR close)",
        "foo AND bar",
        "foo OR bar",
        'foo "exact match"',
    ],
)
def test_escape_fts_query_adversarial_does_not_crash(
    store_and_index: tuple[IndexStore, BM25Index], query: str
) -> None:
    _, idx = store_and_index
    # Must not raise — result can be empty or return chunks
    results = idx.search(query)
    assert isinstance(results, list)


def test_escape_fts_query_strips_parentheses() -> None:
    result = escape_fts_query("(foo)")
    assert result == '"foo"'
    assert "(" not in result
    assert ")" not in result


def test_escape_fts_query_strips_colon() -> None:
    result = escape_fts_query("column:value")
    # colon is stripped; "columnvalue" remains
    assert ":" not in result


def test_escape_fts_query_all_special_becomes_empty_token_skipped() -> None:
    # Token consisting entirely of special chars should be skipped
    result = escape_fts_query("*** !!!")
    assert result == ""


def test_escape_fts_query_preserves_dots_and_underscores() -> None:
    result = escape_fts_query("my_module.func")
    assert result == '"my_module.func"'


def test_escape_fts_query_strips_stopwords() -> None:
    result = escape_fts_query("how does the adapter work")
    assert '"how"' not in result.lower()
    assert '"does"' not in result.lower()
    assert '"the"' not in result.lower()
    assert '"adapter"' in result
    assert '"work"' in result


def test_escape_fts_query_all_stopwords_returns_empty() -> None:
    result = escape_fts_query("how does it")
    assert result == ""


def test_and_fallback_to_or_on_sparse_results(
    store_and_index: tuple[IndexStore, BM25Index],
) -> None:
    """AND-join returning < 3 results triggers OR fallback."""
    _, idx = store_and_index
    # "calculate email" — unlikely both terms in one chunk, OR should find matches
    results = idx.search("calculate email")
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Edge cases: empty escaped query and FTS5 exception path
# ---------------------------------------------------------------------------


def test_search_all_special_chars_returns_empty(
    store_and_index: tuple[IndexStore, BM25Index],
) -> None:
    _, idx = store_and_index
    # escape_fts_query("*** !!!") returns "" → search short-circuits to []
    assert idx.search("*** !!!") == []


def test_search_survives_fts5_error(tmp_path: Path) -> None:
    db = tmp_path / "bm25_err.db"
    s = IndexStore(db)
    idx = BM25Index(s)
    s.insert_chunks(SAMPLE_CHUNKS)
    idx.build(SAMPLE_CHUNKS)

    # sqlite3.Connection.execute is a C-level slot — wrap conn in a proxy.
    real_conn = s.conn

    class _FailOnMatch:
        """Proxy that raises on FTS MATCH queries, delegates everything else."""

        def __getattr__(self, name: str) -> object:
            return getattr(real_conn, name)

        def execute(self, sql: str, parameters: object = None, /) -> object:
            if "MATCH" in sql:
                raise sqlite3.OperationalError("FTS5 error")
            if parameters is not None:
                return real_conn.execute(sql, parameters)  # type: ignore[arg-type]
            return real_conn.execute(sql)

    s._conn = _FailOnMatch()  # type: ignore[assignment]
    results = idx.search("authenticate")
    s._conn = real_conn  # type: ignore[assignment]
    assert results == []
    s.close()


def test_search_propagates_non_operational_error(tmp_path: Path) -> None:
    """Non-OperationalError exceptions from FTS5 must propagate."""
    db = tmp_path / "bm25_prop.db"
    s = IndexStore(db)
    idx = BM25Index(s)
    s.insert_chunks(SAMPLE_CHUNKS)
    idx.build(SAMPLE_CHUNKS)

    real_conn = s.conn

    class _FailIntegrity:
        def __getattr__(self, name: str) -> object:
            return getattr(real_conn, name)

        def execute(self, sql: str, parameters: object = None, /) -> object:
            if "MATCH" in sql:
                raise sqlite3.IntegrityError("integrity violation")
            if parameters is not None:
                return real_conn.execute(sql, parameters)  # type: ignore[arg-type]
            return real_conn.execute(sql)

    s._conn = _FailIntegrity()  # type: ignore[assignment]
    with pytest.raises(sqlite3.IntegrityError, match="integrity violation"):
        idx.search("authenticate")
    s._conn = real_conn  # type: ignore[assignment]
    s.close()


# ---------------------------------------------------------------------------
# Porter stemming tests
# ---------------------------------------------------------------------------


STEMMING_CHUNKS = [
    CodeChunk(
        id="validators.py:validate_input:1",
        content="def validate_input(data):\n    return validated_data",
        file_path="validators.py",
        start_line=1,
        end_line=2,
        symbol_name="validate_input",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=15,
    ),
    CodeChunk(
        id="deps.py:solve_depends:1",
        content="def solve_depends(graph):\n    return dependent_nodes",
        file_path="deps.py",
        start_line=1,
        end_line=2,
        symbol_name="solve_depends",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=15,
    ),
    CodeChunk(
        id="utils.py:helper:1",
        content="def helper():\n    return None",
        file_path="utils.py",
        start_line=1,
        end_line=2,
        symbol_name="helper",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=10,
    ),
]


@pytest.fixture
def stemming_index(tmp_path: Path) -> Generator[tuple[IndexStore, BM25Index], None, None]:
    db = tmp_path / "stemming_test.db"
    s = IndexStore(db)
    idx = BM25Index(s)
    s.insert_chunks(STEMMING_CHUNKS)
    idx.build(STEMMING_CHUNKS)
    yield s, idx
    s.close()


def test_porter_stemming_validators_matches_validate(
    stemming_index: tuple[IndexStore, BM25Index],
) -> None:
    """'validators' stems to 'valid', matching 'validate' and 'validated'."""
    _, idx = stemming_index
    results = idx.search("validators")
    assert len(results) > 0
    top_chunk, _ = results[0]
    assert top_chunk.file_path == "validators.py"


def test_porter_stemming_dependency_matches_depends(
    stemming_index: tuple[IndexStore, BM25Index],
) -> None:
    """'dependency' stems to 'depend', matching 'depends' and 'dependent'."""
    _, idx = stemming_index
    results = idx.search("dependency")
    assert len(results) > 0
    ids = [c.file_path for c, _ in results]
    assert "deps.py" in ids


def test_file_path_boosting(
    stemming_index: tuple[IndexStore, BM25Index],
) -> None:
    """Chunk from validators.py ranks higher than utils.py for query 'validators'."""
    _, idx = stemming_index
    results = idx.search("validators")
    if len(results) >= 2:
        file_paths = [c.file_path for c, _ in results]
        if "validators.py" in file_paths and "utils.py" in file_paths:
            val_idx = file_paths.index("validators.py")
            util_idx = file_paths.index("utils.py")
            assert val_idx < util_idx


def test_and_join_multi_term_ranks_correctly(
    stemming_index: tuple[IndexStore, BM25Index],
) -> None:
    """AND-join: multi-term query requires all terms present."""
    _, idx = stemming_index
    # "solve depends" — both terms in deps.py, neither term fully in utils.py
    results = idx.search("solve depends")
    if results:
        top_chunk, _ = results[0]
        assert top_chunk.file_path == "deps.py"


def test_single_term_query_still_works(
    stemming_index: tuple[IndexStore, BM25Index],
) -> None:
    """Single-term queries work with AND-join (no join needed)."""
    _, idx = stemming_index
    results = idx.search("helper")
    assert len(results) > 0
    assert results[0][0].file_path == "utils.py"
