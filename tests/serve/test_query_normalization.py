"""Tests for query normalization (_query_terms) and symbol promotion (_symbol_search_seeds)."""

# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from archex.index.store import IndexStore
from archex.models import CodeChunk, SymbolKind
from archex.serve.context import _ARCH_SYNONYMS, _query_terms, _split_compound_token

# ---------------------------------------------------------------------------
# _split_compound_token tests
# ---------------------------------------------------------------------------


def test_split_camel_case_basic() -> None:
    # _split_compound_token receives the original mixed-case token from re.findall
    parts = _split_compound_token("queryPipeline")
    lowered = [p.lower() for p in parts]
    assert "query" in lowered
    assert "pipeline" in lowered


def test_split_camel_case_preserves_original() -> None:
    parts = _split_compound_token("queryPipeline")
    assert "queryPipeline" in parts


def test_split_pascal_case() -> None:
    parts = _split_compound_token("BM25Index")
    lowered = [p.lower() for p in parts]
    # Should detect "index" at minimum
    assert any("index" in p for p in lowered)


def test_split_snake_case() -> None:
    parts = _split_compound_token("next_function")
    assert "next" in parts
    assert "function" in parts


def test_split_snake_case_preserves_original() -> None:
    parts = _split_compound_token("next_function")
    assert "next_function" in parts


def test_single_word_returns_itself() -> None:
    parts = _split_compound_token("pipeline")
    assert parts == ["pipeline"]


def test_all_lowercase_short_token() -> None:
    parts = _split_compound_token("bm25")
    assert "bm25" in parts


# ---------------------------------------------------------------------------
# _query_terms tests
# ---------------------------------------------------------------------------


def test_camel_case_query_splits_to_components() -> None:
    terms = _query_terms("How does queryPipeline work?")
    assert "query" in terms
    assert "pipeline" in terms


def test_snake_case_query_splits_to_components() -> None:
    terms = _query_terms("describe the next_function handler")
    assert "next" in terms
    assert "function" not in terms  # "function" is a stop word


def test_stop_words_excluded() -> None:
    terms = _query_terms("How does the function work with this module?")
    for stop_word in ("how", "does", "the", "function", "work", "with", "this", "module"):
        assert stop_word not in terms


def test_bigram_compound_form_generated() -> None:
    terms = _query_terms("dependency injection")
    assert "dependency_injection" in terms


def test_arch_synonym_pipeline_expanded() -> None:
    terms = _query_terms("How does archex implement the query pipeline?")
    # "pipeline" should be present and its synonyms should be added
    assert "pipeline" in terms
    for synonym in _ARCH_SYNONYMS["pipeline"]:
        assert synonym in terms


def test_arch_synonym_middleware_expanded() -> None:
    terms = _query_terms("How does express implement the middleware chain?")
    assert "middleware" in terms
    for synonym in _ARCH_SYNONYMS["middleware"]:
        assert synonym in terms


def test_arch_synonym_injection_expanded() -> None:
    terms = _query_terms("How does FastAPI implement dependency injection?")
    assert "injection" in terms
    for synonym in _ARCH_SYNONYMS["injection"]:
        assert synonym in terms


def test_arch_synonym_routing_expanded() -> None:
    terms = _query_terms("How does gin implement routing?")
    assert "routing" in terms
    for synonym in _ARCH_SYNONYMS["routing"]:
        assert synonym in terms


def test_arch_synonym_indexing_expanded() -> None:
    terms = _query_terms("How does archex handle delta indexing?")
    assert "indexing" in terms
    for synonym in _ARCH_SYNONYMS["indexing"]:
        assert synonym in terms


def test_dependency_term_expanded_to_synonyms() -> None:
    terms = _query_terms("dependency injection for route handlers")
    assert "dependency" in terms
    for synonym in _ARCH_SYNONYMS["dependency"]:
        assert synonym in terms


def test_short_words_under_three_chars_excluded() -> None:
    terms = _query_terms("the is it go do to")
    # All are either stop words or too short — result should be empty or near-empty
    assert "is" not in terms
    assert "it" not in terms
    assert "go" not in terms


def test_regular_terms_not_affected() -> None:
    terms = _query_terms("How does authentication work?")
    assert "authentication" in terms


def test_returns_set() -> None:
    result = _query_terms("query pipeline query")
    assert isinstance(result, set)


def test_no_duplicates() -> None:
    # Sets are unique by definition, but no term should appear twice on expansion
    terms = _query_terms("pipeline pipeline pipeline")
    assert len(terms) == len(set(terms))


# ---------------------------------------------------------------------------
# _symbol_search_seeds promotion tests
# ---------------------------------------------------------------------------


def _make_store(chunks: list[CodeChunk]) -> IndexStore:
    db_path = Path(tempfile.mkdtemp()) / "test.db"
    store = IndexStore(db_path)
    store.insert_chunks(chunks)
    return store


def _chunk(
    *,
    file_path: str = "src/main.py",
    name: str = "foo",
    kind: SymbolKind = SymbolKind.FUNCTION,
    content: str = "def foo(): pass",
    qualified_name: str | None = None,
    token_count: int = 10,
) -> CodeChunk:
    qname = qualified_name or name
    sid = f"{file_path}::{qname}#{kind.value}"
    return CodeChunk(
        id=sid,
        content=content,
        file_path=file_path,
        start_line=1,
        end_line=5,
        symbol_name=name,
        # symbol_id must be set for FTS5 indexing (symbols_fts requires non-None symbol_id)
        symbol_id=sid,
        qualified_name=qname,
        symbol_kind=kind,
        language="python",
        token_count=token_count,
    )


def test_retrieval_question_expands_query_pipeline_terms() -> None:
    from archex.api import _expand_retrieval_question

    expanded = _expand_retrieval_question("How does archex implement the query pipeline?")

    assert "api" in expanded
    assert "bm25" in expanded
    assert "BM25Index" in expanded
    assert "assemble_context" in expanded
    assert "context" in expanded


def test_retrieval_question_expands_index_terms() -> None:
    from archex.api import _expand_retrieval_question

    expanded = _expand_retrieval_question(
        "How does archex explicitly build or refresh a repo-local index?"
    )

    assert "cache" in expanded
    assert "config" in expanded
    assert "project" in expanded
    assert "CacheManager" in expanded


def test_non_retrieval_question_is_not_expanded() -> None:
    from archex.api import _expand_retrieval_question

    question = "Where is User defined?"

    assert _expand_retrieval_question(question) == question


def test_path_terms_expand_query_pipeline_keywords() -> None:
    from archex.api import _extract_path_terms

    terms = _extract_path_terms("How does archex implement the query pipeline?")
    assert "api" in terms
    assert "bm25" in terms
    assert "context" in terms


def test_task_dispatch_path_terms_include_architecture_files() -> None:
    from archex.api import _extract_path_terms

    terms = _extract_path_terms("How does Celery dispatch and execute distributed tasks?")

    assert "task" in terms
    assert "worker" in terms
    assert "strategy" in terms


def test_path_match_multiplier_prefers_basename_and_segments() -> None:
    from archex.api import _path_match_multiplier

    assert _path_match_multiplier("celery/app/task.py", "task") == pytest.approx(0.85)
    assert _path_match_multiplier("celery/worker/strategy.py", "worker") == pytest.approx(0.70)
    assert _path_match_multiplier("celery/events/dispatcher.py", "patch") == pytest.approx(0.45)


def test_file_path_boost_caps_chunks_per_file() -> None:
    from archex.api import _file_path_boost

    chunks = [
        _chunk(
            file_path="src/archex/pipeline/chunker.py",
            name=f"chunk_{i}",
            qualified_name=f"chunk_{i}",
        )
        for i in range(5)
    ]
    chunks.append(
        _chunk(
            file_path="src/archex/serve/context.py",
            name="assemble_context",
            qualified_name="assemble_context",
        )
    )
    store = _make_store(chunks)
    try:
        boosted = _file_path_boost(
            store,
            "query pipeline context",
            existing_ids=set(),
            max_bm25_score=10.0,
            max_chunks_per_file=2,
        )
        boosted_paths = [chunk.file_path for chunk, _ in boosted]

        assert boosted_paths.count("src/archex/pipeline/chunker.py") == 2
        assert "src/archex/serve/context.py" in boosted_paths
    finally:
        store.close()


def test_exact_symbol_match_gets_high_boost() -> None:
    """Exact symbol_name equality → 0.60× max_bm25_score boost.

    _symbol_search_seeds filters to bigrams and single words >= 6 chars.
    For the query "solve_dependencies inject route", "inject" (6 chars) is a
    search term.  A symbol named exactly "inject" matches exactly.
    """
    from archex.api import _symbol_search_seeds

    chunk = _chunk(name="inject", file_path="fastapi/dependencies/utils.py")
    store = _make_store([chunk])
    try:
        seeds = _symbol_search_seeds(store, "inject resolve depends", max_bm25_score=1.0)
        matched = [s for c, s in seeds if c.symbol_name == "inject"]
        assert len(matched) >= 1
        assert matched[0] == pytest.approx(0.60, abs=0.01)
    finally:
        store.close()


def test_partial_symbol_match_gets_low_boost() -> None:
    """Partial/qualified-name match → 0.15× max_bm25_score boost."""
    from archex.api import _symbol_search_seeds

    # Symbol name contains the term as a substring but is not equal to it
    chunk = _chunk(
        name="get_injection_scope",
        file_path="fastapi/dependencies/utils.py",
        qualified_name="get_injection_scope",
    )
    store = _make_store([chunk])
    try:
        # "injection" is a 9-char term so it's included; symbol name contains it
        seeds = _symbol_search_seeds(store, "injection resolver handler", max_bm25_score=1.0)
        matched = [s for c, s in seeds if c.symbol_name == "get_injection_scope"]
        assert len(matched) >= 1
        # "get_injection_scope" != "injection" → partial match → 0.15×
        assert matched[0] == pytest.approx(0.15, abs=0.01)
    finally:
        store.close()


def test_exact_match_scores_higher_than_partial() -> None:
    """Exact symbol match should produce a higher score than a partial match."""
    from archex.api import _SYMBOL_EXACT_BOOST, _SYMBOL_PARTIAL_BOOST

    assert _SYMBOL_EXACT_BOOST > _SYMBOL_PARTIAL_BOOST


def test_no_symbol_match_returns_empty() -> None:
    """Query terms that don't match any symbols produce no seeds."""
    from archex.api import _symbol_search_seeds

    chunk = _chunk(name="unrelated_symbol", file_path="src/unrelated.py")
    store = _make_store([chunk])
    try:
        seeds = _symbol_search_seeds(store, "dependency injection", max_bm25_score=1.0)
        matched_names = {c.symbol_name for c, _ in seeds}
        assert "unrelated_symbol" not in matched_names
    finally:
        store.close()


def test_file_path_only_match_excluded() -> None:
    """FTS5 hits where symbol_name/qualified_name don't contain the term are excluded."""
    from archex.api import _symbol_search_seeds

    # Symbol name is completely unrelated to the query; only the file path might match
    chunk = _chunk(
        name="completely_irrelevant",
        file_path="src/dependency.py",
        qualified_name="completely_irrelevant",
    )
    store = _make_store([chunk])
    try:
        seeds = _symbol_search_seeds(store, "dependency injection", max_bm25_score=1.0)
        matched_names = {c.symbol_name for c, _ in seeds}
        assert "completely_irrelevant" not in matched_names
    finally:
        store.close()


def test_exact_match_scales_with_max_bm25() -> None:
    """Exact match boost scales proportionally with max_bm25_score."""
    from archex.api import _symbol_search_seeds

    chunk = _chunk(name="inject", file_path="src/di.py")
    store = _make_store([chunk])
    try:
        seeds_at_2 = _symbol_search_seeds(store, "inject resolve", max_bm25_score=2.0)
        matched = [s for c, s in seeds_at_2 if c.symbol_name == "inject"]
        assert len(matched) >= 1
        assert matched[0] == pytest.approx(0.60 * 2.0, abs=0.01)
    finally:
        store.close()
