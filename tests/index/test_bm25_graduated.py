"""Tests for BM25Index graduated fallback search stages.

The graduated search has 4 stages:
1. AND all terms (return if >= 10 results)
2. N-1 subsets (drop one term at a time)
3. Pairs of terms (requires >= 4 tokens)
4. OR all terms (final fallback)

These tests create enough chunks to exercise each stage boundary.
"""

# pyright: reportPrivateUsage=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING

from archex.index.bm25 import BM25Index, _sanitize_tokens
from archex.index.store import IndexStore
from archex.models import CodeChunk, SymbolKind

if TYPE_CHECKING:
    from pathlib import Path


def _make_chunk(idx: int, content: str, file_path: str = "module.py") -> CodeChunk:
    return CodeChunk(
        id=f"{file_path}:func_{idx}:{idx}",
        content=content,
        file_path=file_path,
        start_line=idx,
        end_line=idx + 5,
        symbol_name=f"func_{idx}",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=20,
    )


def _build_index(tmp_path: Path, chunks: list[CodeChunk]) -> tuple[IndexStore, BM25Index]:
    db = tmp_path / "graduated.db"
    store = IndexStore(db)
    idx = BM25Index(store)
    store.insert_chunks(chunks)
    idx.build(chunks)
    return store, idx


class TestGraduatedSearchStages:
    """Exercise graduated fallback: AND-all → N-1 subsets → pairs → OR-all."""

    def test_and_all_returns_early_when_enough_results(self, tmp_path: Path) -> None:
        """Stage 1: AND-all produces >= 10 results, no fallback needed."""
        # Create 15 chunks all containing "alpha" and "beta"
        chunks = [
            _make_chunk(
                i,
                f"def func_{i}(alpha, beta): return alpha + beta + {i}",
                f"mod_{i}.py",
            )
            for i in range(15)
        ]
        store, idx = _build_index(tmp_path, chunks)
        try:
            results = idx.search("alpha beta")
            # All 15 chunks match both terms
            assert len(results) >= 10
        finally:
            store.close()

    def test_n_minus_1_subsets_with_three_terms(self, tmp_path: Path) -> None:
        """Stage 2: AND-all returns < 10, N-1 subsets gather enough results.

        With 3 terms (A, B, C): subsets are (A,B), (A,C), (B,C).
        We create chunks that match various pairs but not all three.
        """
        chunks = []
        # 5 chunks with "alpha" and "beta" (no "gamma")
        for i in range(5):
            chunks.append(_make_chunk(i, f"alpha beta handler_{i}", f"ab_{i}.py"))
        # 5 chunks with "alpha" and "gamma" (no "beta")
        for i in range(5, 10):
            chunks.append(_make_chunk(i, f"alpha gamma processor_{i}", f"ag_{i}.py"))
        # 3 chunks with "beta" and "gamma" (no "alpha")
        for i in range(10, 13):
            chunks.append(_make_chunk(i, f"beta gamma worker_{i}", f"bg_{i}.py"))

        store, idx = _build_index(tmp_path, chunks)
        try:
            # "alpha beta gamma" — AND-all matches 0 chunks (no chunk has all 3)
            # N-1 subsets: (alpha,beta)=5, (alpha,gamma)=5, (beta,gamma)=3 → merged ≥ 10
            results = idx.search("alpha beta gamma")
            assert len(results) >= 10
        finally:
            store.close()

    def test_pairs_stage_with_four_terms(self, tmp_path: Path) -> None:
        """Stage 3: AND-all < 10, N-1 < 10, pairs gather enough.

        With 4 terms, N-1 subsets are groups of 3 (hard to match).
        Pairs are groups of 2 (easier to match).
        """
        chunks = []
        # Chunks matching various pairs but not triplets
        for i in range(4):
            chunks.append(_make_chunk(i, f"alpha beta item_{i}", f"ab_{i}.py"))
        for i in range(4, 8):
            chunks.append(_make_chunk(i, f"gamma delta item_{i}", f"cd_{i}.py"))
        for i in range(8, 12):
            chunks.append(_make_chunk(i, f"alpha gamma item_{i}", f"ac_{i}.py"))

        store, idx = _build_index(tmp_path, chunks)
        try:
            results = idx.search("alpha beta gamma delta")
            # Pairs should find enough combined results
            assert len(results) >= 8
        finally:
            store.close()

    def test_or_fallback_final_stage(self, tmp_path: Path) -> None:
        """Stage 4: All previous stages fail to reach threshold, OR-all gathers results."""
        # Each chunk has exactly one unique term
        chunks = [
            _make_chunk(0, "alpha unique_content_zero", "a.py"),
            _make_chunk(1, "beta unique_content_one", "b.py"),
        ]

        store, idx = _build_index(tmp_path, chunks)
        try:
            # "alpha beta" — AND returns 0, OR returns 2
            results = idx.search("alpha beta")
            assert len(results) == 2
        finally:
            store.close()

    def test_single_token_bypasses_and_stage(self, tmp_path: Path) -> None:
        """Single token skips AND-all (needs >= 2 tokens) and goes to OR."""
        chunks = [_make_chunk(0, "alpha handler", "a.py")]

        store, idx = _build_index(tmp_path, chunks)
        try:
            results = idx.search("alpha")
            assert len(results) == 1
        finally:
            store.close()


class TestSanitizeTokensEdgeCases:
    def test_preserves_dotted_identifiers(self) -> None:
        tokens = _sanitize_tokens("os.path.join")
        assert '"os.path.join"' in tokens

    def test_strips_all_special_chars(self) -> None:
        tokens = _sanitize_tokens("@#$%")
        assert tokens == []

    def test_mixed_valid_and_invalid(self) -> None:
        tokens = _sanitize_tokens("valid @@@ also_valid")
        assert '"valid"' in tokens
        assert '"also_valid"' in tokens
        assert len(tokens) == 2

    def test_stopwords_removed(self) -> None:
        tokens = _sanitize_tokens("the quick fox")
        assert '"the"' not in tokens
        assert '"quick"' in tokens
        assert '"fox"' in tokens

    def test_preserves_underscored_identifiers(self) -> None:
        tokens = _sanitize_tokens("__init__ _private")
        assert '"__init__"' in tokens
        assert '"_private"' in tokens

    def test_numeric_tokens(self) -> None:
        tokens = _sanitize_tokens("func123 42")
        assert '"func123"' in tokens
        assert '"42"' in tokens


class TestHasData:
    def test_empty_index_has_no_data(self, tmp_path: Path) -> None:
        db = tmp_path / "empty.db"
        store = IndexStore(db)
        idx = BM25Index(store)
        assert idx.has_data is False
        store.close()

    def test_populated_index_has_data(self, tmp_path: Path) -> None:
        chunks = [_make_chunk(0, "hello world", "a.py")]
        store, idx = _build_index(tmp_path, chunks)
        try:
            assert idx.has_data is True
        finally:
            store.close()
