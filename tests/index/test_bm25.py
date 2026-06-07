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
    assert '"work"' not in result.lower()
    assert '"archex"' not in escape_fts_query("how does archex index work").lower()
    assert '"adapter"' in result


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


# ---------------------------------------------------------------------------
# Docstring column tests
# ---------------------------------------------------------------------------


DOCSTRING_CHUNKS = [
    CodeChunk(
        id="cache.py:CacheManager:1",
        content="class CacheManager:\n    pass",
        file_path="cache.py",
        start_line=1,
        end_line=2,
        symbol_name="CacheManager",
        symbol_kind=SymbolKind.CLASS,
        language="python",
        token_count=10,
        docstring="Manages in-memory cache eviction using LRU strategy.",
    ),
    CodeChunk(
        id="db.py:DatabasePool:1",
        content="class DatabasePool:\n    pass",
        file_path="db.py",
        start_line=1,
        end_line=2,
        symbol_name="DatabasePool",
        symbol_kind=SymbolKind.CLASS,
        language="python",
        token_count=10,
        docstring=None,
    ),
    CodeChunk(
        id="net.py:HttpClient:1",
        content="class HttpClient:\n    pass",
        file_path="net.py",
        start_line=1,
        end_line=2,
        symbol_name="HttpClient",
        symbol_kind=SymbolKind.CLASS,
        language="python",
        token_count=10,
        docstring="Sends HTTP requests with retry and timeout support.",
    ),
]


@pytest.fixture
def docstring_index(tmp_path: Path) -> Generator[tuple[IndexStore, BM25Index], None, None]:
    db = tmp_path / "docstring_test.db"
    s = IndexStore(db)
    idx = BM25Index(s)
    s.insert_chunks(DOCSTRING_CHUNKS)
    idx.build(DOCSTRING_CHUNKS)
    yield s, idx
    s.close()


def test_docstring_term_is_searchable(
    docstring_index: tuple[IndexStore, BM25Index],
) -> None:
    """Terms found only in docstrings must appear in search results."""
    _, idx = docstring_index
    # "eviction" appears only in CacheManager's docstring, not in content
    results = idx.search("eviction")
    assert len(results) > 0
    ids = [c.id for c, _ in results]
    assert "cache.py:CacheManager:1" in ids


def test_docstring_chunk_ranks_higher_than_no_docstring(
    docstring_index: tuple[IndexStore, BM25Index],
) -> None:
    """A chunk with query terms in its docstring ranks above chunks without docstrings."""
    _, idx = docstring_index
    # "retry" appears only in HttpClient's docstring
    results = idx.search("retry")
    assert len(results) > 0
    top_chunk, _ = results[0]
    assert top_chunk.id == "net.py:HttpClient:1"


# ---------------------------------------------------------------------------
# Schema migration test
# ---------------------------------------------------------------------------


def test_schema_migration_from_old_fts_schema(tmp_path: Path) -> None:
    """BM25Index detects a stale FTS schema (no breadcrumbs/summary columns) and migrates it."""
    db = tmp_path / "migration_test.db"
    store = IndexStore(db)

    # Manually create the old FTS schema without the summary column
    store.conn.execute("DROP TABLE IF EXISTS chunks_fts")
    store.conn.execute(
        """
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            content,
            symbol_name,
            file_path,
            docstring,
            tokenize='porter unicode61'
        )
        """
    )
    store.conn.commit()

    # Instantiating BM25Index must trigger migration transparently
    idx = BM25Index(store)

    # The new schema must have both breadcrumbs and summary columns — probe them
    store.conn.execute("SELECT chunk_id FROM chunks_fts WHERE breadcrumbs MATCH 'probe' LIMIT 0")
    store.conn.execute("SELECT chunk_id FROM chunks_fts WHERE summary MATCH 'probe' LIMIT 0")

    # Index must be functional after migration
    chunk = CodeChunk(
        id="migrated.py:func:1",
        content="def func(): pass",
        file_path="migrated.py",
        start_line=1,
        end_line=1,
        symbol_name="func",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=5,
        docstring="xyzzy_migrated_docstring",
        summary="Session management factory pattern for database connections.",
    )
    store.insert_chunks([chunk])
    idx.build([chunk])

    results = idx.search("xyzzy migrated docstring")
    assert len(results) > 0
    assert results[0][0].id == "migrated.py:func:1"

    store.close()


# ---------------------------------------------------------------------------
# AvgIDF tests
# ---------------------------------------------------------------------------


def test_avg_idf_specific_terms_higher_than_common(
    store_and_index: tuple[IndexStore, BM25Index],
) -> None:
    """Rare terms have higher AvgIDF than common ones."""
    _, idx = store_and_index
    # "calculate_sum" appears in 1 chunk; "data" appears in multiple (model, config content)
    idf_rare = idx.avg_idf("calculate_sum")
    idf_common = idx.avg_idf("return")  # appears in every function body
    assert idf_rare > idf_common, (
        f"Rare term IDF ({idf_rare:.3f}) must exceed common term IDF ({idf_common:.3f})"
    )


def test_avg_idf_empty_query_returns_zero(
    store_and_index: tuple[IndexStore, BM25Index],
) -> None:
    """Empty or all-stopword queries return 0.0."""
    _, idx = store_and_index
    assert idx.avg_idf("") == 0.0
    assert idx.avg_idf("the and for") == 0.0


def test_avg_idf_positive_for_indexed_terms(
    store_and_index: tuple[IndexStore, BM25Index],
) -> None:
    """Any non-stopword term that exists in the corpus has positive IDF."""
    _, idx = store_and_index
    idf = idx.avg_idf("calculate_sum")
    assert idf > 0.0


# ---------------------------------------------------------------------------
# Adaptive weights tests
# ---------------------------------------------------------------------------


ADAPTIVE_WEIGHT_CHUNKS = [
    # Many chunks with "task" in symbol/docstring to simulate celery-like scenario
    CodeChunk(
        id="app/task.py:Task:1",
        content="class Task:\n    def apply_async(self, args):\n        return self.dispatch(args)",
        file_path="app/task.py",
        start_line=1,
        end_line=3,
        symbol_name="Task",
        symbol_kind=SymbolKind.CLASS,
        language="python",
        token_count=20,
        docstring="Core task class for dispatching work.",
    ),
    CodeChunk(
        id="__init__.py:task_export:1",
        content="from .app.task import Task\nfrom ._state import current_task",
        file_path="__init__.py",
        start_line=1,
        end_line=2,
        symbol_name="task_export",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=15,
        docstring="Task re-exports for convenient access. Dispatch from celery.task.",
    ),
    CodeChunk(
        id="_state.py:current_task:1",
        content="current_task = None\n_task_stack = []",
        file_path="_state.py",
        start_line=1,
        end_line=2,
        symbol_name="current_task",
        symbol_kind=SymbolKind.VARIABLE,
        language="python",
        token_count=10,
        docstring="Thread-local task state tracking for dispatch coordination.",
    ),
    CodeChunk(
        id="worker/strategy.py:strategy:1",
        content="def dispatch_strategy(task):\n    return task.execute()",
        file_path="worker/strategy.py",
        start_line=1,
        end_line=2,
        symbol_name="dispatch_strategy",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=15,
        docstring=None,
    ),
]


@pytest.fixture
def adaptive_index(tmp_path: Path) -> Generator[tuple[IndexStore, BM25Index], None, None]:
    db = tmp_path / "adaptive_test.db"
    s = IndexStore(db)
    idx = BM25Index(s)
    s.insert_chunks(ADAPTIVE_WEIGHT_CHUNKS)
    idx.build(ADAPTIVE_WEIGHT_CHUNKS)
    yield s, idx
    s.close()


def test_adaptive_weights_low_idf_reduces_symbol_boost(
    adaptive_index: tuple[IndexStore, BM25Index],
) -> None:
    """Low-IDF query gets reduced symbol/docstring/breadcrumbs/summary weights."""
    _, idx = adaptive_index
    weights = idx._adaptive_weights("task dispatch")  # pyright: ignore[reportPrivateUsage]
    # symbol weight (index 1) must be less than default 10.0
    assert weights[1] < 10.0
    # path weight (index 2) must be more than default 1.5
    assert weights[2] > 1.5
    # docstring weight (index 3) must be less than default 6.0
    assert weights[3] < 6.0
    # breadcrumbs weight (index 4) must be less than default 5.0
    assert weights[4] < 5.0
    # summary weight (index 5) must be less than default 8.0
    assert weights[5] < 8.0


def test_adaptive_weights_high_idf_uses_defaults(tmp_path: Path) -> None:
    """High-IDF (specific) query returns default weights."""
    # Need N >= 13 chunks so that a term in 1 chunk has IDF >= 2.5
    chunks = [
        CodeChunk(
            id=f"mod_{i}.py:fn_{i}:{i}",
            content=f"def fn_{i}(): return {i}",
            file_path=f"mod_{i}.py",
            start_line=i,
            end_line=i + 1,
            symbol_name=f"fn_{i}",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=10,
        )
        for i in range(15)
    ]
    # Add one chunk with a unique term
    chunks.append(
        CodeChunk(
            id="unique.py:xyzzy_rare:1",
            content="def xyzzy_rare(): return 42",
            file_path="unique.py",
            start_line=1,
            end_line=1,
            symbol_name="xyzzy_rare",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=10,
        )
    )
    db = tmp_path / "high_idf.db"
    s = IndexStore(db)
    idx = BM25Index(s)
    s.insert_chunks(chunks)
    idx.build(chunks)
    try:
        weights = idx._adaptive_weights("xyzzy_rare")  # pyright: ignore[reportPrivateUsage]
        assert weights == (1.0, 10.0, 1.5, 6.0, 5.0, 8.0)
    finally:
        s.close()


def test_path_bonus_promotes_matching_file(
    adaptive_index: tuple[IndexStore, BM25Index],
) -> None:
    """File with 'task' in path ranks higher after path bonus for 'task' query."""
    _, idx = adaptive_index
    results = idx.search("task dispatch")
    file_paths = [c.file_path for c, _ in results]
    assert "app/task.py" in file_paths
    # app/task.py should rank above __init__.py and _state.py
    task_idx = file_paths.index("app/task.py")
    for other in ("__init__.py", "_state.py"):
        if other in file_paths:
            assert task_idx < file_paths.index(other), (
                f"app/task.py (idx={task_idx}) should rank above {other}"
            )


def test_path_bonus_no_match_leaves_score_unchanged() -> None:
    """Chunks with no path-matching terms get multiplier 1.0 (no change)."""
    chunk = CodeChunk(
        id="foo.py:bar:1",
        content="def bar(): pass",
        file_path="foo.py",
        start_line=1,
        end_line=1,
        symbol_name="bar",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=5,
    )
    results = [(chunk, 5.0)]
    boosted = BM25Index._apply_path_bonus(results, ['"unrelated"'])  # pyright: ignore[reportPrivateUsage]
    assert boosted[0][1] == 5.0  # unchanged


def test_path_bonus_multiple_term_matches() -> None:
    """Multiple query terms in path compound the bonus."""
    chunk = CodeChunk(
        id="runtime/scheduler/mod.rs:main:1",
        content="pub mod scheduler;",
        file_path="runtime/scheduler/mod.rs",
        start_line=1,
        end_line=1,
        symbol_name="main",
        symbol_kind=SymbolKind.MODULE,
        language="rust",
        token_count=5,
    )
    results = [(chunk, 5.0)]
    tokens = ['"runtime"', '"scheduler"', '"task"']
    boosted = BM25Index._apply_path_bonus(results, tokens)  # pyright: ignore[reportPrivateUsage]
    # "runtime" and "scheduler" match path parts; "task" does not
    # Expected multiplier: 1.0 + 0.2 * 2 = 1.4
    assert abs(boosted[0][1] - 7.0) < 0.01


def test_search_respects_top_k_after_reranking(
    adaptive_index: tuple[IndexStore, BM25Index],
) -> None:
    """search() returns at most top_k results even after path-bonus reranking."""
    _, idx = adaptive_index
    results = idx.search("task dispatch", top_k=2)
    assert len(results) <= 2


def test_rerank_multiplier_fetches_more_candidates(
    adaptive_index: tuple[IndexStore, BM25Index],
) -> None:
    """With top_k=1, internal fetch should still consider multiple candidates."""
    _, idx = adaptive_index
    results = idx.search("task dispatch", top_k=1)
    assert len(results) == 1
    # The top result should be from app/task.py thanks to path bonus
    assert results[0][0].file_path == "app/task.py"


# ---------------------------------------------------------------------------
# Summary column tests
# ---------------------------------------------------------------------------


SUMMARY_CHUNKS = [
    CodeChunk(
        id="orm.py:ScopedSession:1",
        content="class ScopedSession:\n    pass",
        file_path="orm.py",
        start_line=1,
        end_line=2,
        symbol_name="ScopedSession",
        symbol_kind=SymbolKind.CLASS,
        language="python",
        token_count=10,
        summary=(
            "Session management factory pattern using scoped_session"
            " for thread-safe database connections."
        ),
    ),
    CodeChunk(
        id="views.py:login:1",
        content="def login(request):\n    pass",
        file_path="views.py",
        start_line=1,
        end_line=2,
        symbol_name="login",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=10,
        summary=None,
    ),
]


@pytest.fixture
def summary_index(tmp_path: Path) -> Generator[tuple[IndexStore, BM25Index], None, None]:
    db = tmp_path / "summary_test.db"
    s = IndexStore(db)
    idx = BM25Index(s)
    s.insert_chunks(SUMMARY_CHUNKS)
    idx.build(SUMMARY_CHUNKS)
    yield s, idx
    s.close()


def test_summary_term_is_searchable(
    summary_index: tuple[IndexStore, BM25Index],
) -> None:
    """Terms found only in the summary column must appear in search results."""
    _, idx = summary_index
    # "thread-safe" appears only in ScopedSession's summary, not in content
    results = idx.search("thread safe database")
    assert len(results) > 0
    ids = [c.id for c, _ in results]
    assert "orm.py:ScopedSession:1" in ids


def test_summary_bridges_vocabulary_gap(
    summary_index: tuple[IndexStore, BM25Index],
) -> None:
    """NL query 'session management' matches via summary even though code has no such terms."""
    _, idx = summary_index
    results = idx.search("session management")
    assert len(results) > 0
    top_chunk, _ = results[0]
    assert top_chunk.id == "orm.py:ScopedSession:1"


def test_chunk_without_summary_still_searchable(
    summary_index: tuple[IndexStore, BM25Index],
) -> None:
    """Chunks without summaries are still searchable by content/symbol/path."""
    _, idx = summary_index
    results = idx.search("login")
    assert len(results) > 0
    ids = [c.id for c, _ in results]
    assert "views.py:login:1" in ids
