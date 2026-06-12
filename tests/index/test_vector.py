"""Tests for VectorIndex and reciprocal_rank_fusion."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from archex.exceptions import ArchexIndexError
from archex.index.fusion import (
    confidence_weighted_rrf,
    reciprocal_rank_fusion,
    relative_score_fusion,
    should_fuse,
)
from archex.index.store import IndexStore
from archex.index.vector import VectorIndex
from archex.models import CodeChunk, SymbolKind

if TYPE_CHECKING:
    from pathlib import Path


class FakeEmbedder:
    """Deterministic test embedder using content hashes for reproducible vectors."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            raw: list[float] = []
            for i in range(self._dim):
                byte_val = h[i % len(h)]
                raw.append((byte_val / 255.0) * 2 - 1)
            results.append(raw)
        return results

    @property
    def dimension(self) -> int:
        return self._dim


class CountingEmbedder(FakeEmbedder):
    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim)
        self.encoded_batches: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.encoded_batches.append(texts)
        return super().encode(texts)


class QueryPrefixEmbedder:
    dimension = 2

    def __init__(self) -> None:
        self.encoded_documents: list[str] = []
        self.encoded_queries: list[str] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.encoded_documents.extend(texts)
        return [self._vector(text) for text in texts]

    def encode_queries(self, queries: list[str]) -> list[list[float]]:
        self.encoded_queries.extend(queries)
        return [self._vector(f"query:{query}") for query in queries]

    def _vector(self, text: str) -> list[float]:
        if text in {"target", "query:target"}:
            return [1.0, 0.0]
        return [0.0, 1.0]


QUERY_PREFIX_CHUNKS = [
    CodeChunk(
        id="target.py:target:1",
        content="target",
        file_path="target.py",
        start_line=1,
        end_line=1,
        symbol_name="target",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=1,
    ),
    CodeChunk(
        id="other.py:other:1",
        content="other",
        file_path="other.py",
        start_line=1,
        end_line=1,
        symbol_name="other",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=1,
    ),
]


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
    CodeChunk(
        id="math.py:add:1",
        content="def add(x: int, y: int) -> int:\n    return x + y",
        file_path="math.py",
        start_line=1,
        end_line=2,
        symbol_name="add",
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=18,
    ),
]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=64)


@pytest.fixture
def built_index(embedder: FakeEmbedder) -> VectorIndex:
    idx = VectorIndex()
    idx.build(SAMPLE_CHUNKS, embedder)
    return idx


class TestVectorIndexBuild:
    def test_build_sets_size(self, built_index: VectorIndex) -> None:
        assert built_index.size == 4

    def test_build_empty_chunks(self, embedder: FakeEmbedder) -> None:
        idx = VectorIndex()
        idx.build([], embedder)
        assert idx.size == 0

    def test_build_single_chunk(self, embedder: FakeEmbedder) -> None:
        idx = VectorIndex()
        idx.build([SAMPLE_CHUNKS[0]], embedder)
        assert idx.size == 1

    def test_build_reuses_cached_vectors_by_content_hash(self) -> None:
        embedder = CountingEmbedder(dim=64)
        first = VectorIndex()
        hits, misses = first.build(SAMPLE_CHUNKS[:2], embedder)
        assert (hits, misses) == (0, 2)
        assert embedder.encoded_batches == [[SAMPLE_CHUNKS[0].content, SAMPLE_CHUNKS[1].content]]

        second = VectorIndex()
        changed = SAMPLE_CHUNKS[1].model_copy(
            update={
                "id": "auth.py:authenticate:12",
                "content": (
                    "def authenticate(username: str, password: str) -> bool:\n    return True"
                ),
            }
        )
        hits, misses = second.build(
            [SAMPLE_CHUNKS[0], changed],
            embedder,
            cached_vectors_by_content_hash=first.vectors_by_content_hash(),
        )

        assert (hits, misses) == (1, 1)
        assert embedder.encoded_batches[-1] == [changed.content]


class TestVectorIndexSearch:
    def test_search_returns_results(self, built_index: VectorIndex, embedder: FakeEmbedder) -> None:
        results = built_index.search("calculate sum", embedder)
        assert len(results) > 0

    def test_search_returns_tuples_of_chunk_and_score(
        self, built_index: VectorIndex, embedder: FakeEmbedder
    ) -> None:
        results = built_index.search("function", embedder)
        for chunk, score in results:
            assert isinstance(chunk, CodeChunk)
            assert isinstance(score, float)

    def test_search_scores_are_positive(
        self, built_index: VectorIndex, embedder: FakeEmbedder
    ) -> None:
        results = built_index.search("user model", embedder)
        for _, score in results:
            assert score > 0

    def test_search_top_k_limits_results(
        self, built_index: VectorIndex, embedder: FakeEmbedder
    ) -> None:
        results = built_index.search("def", embedder, top_k=2)
        assert len(results) <= 2

    def test_search_empty_index(self, embedder: FakeEmbedder) -> None:
        idx = VectorIndex()
        results = idx.search("test", embedder)
        assert results == []

    def test_search_results_sorted_by_score(
        self, built_index: VectorIndex, embedder: FakeEmbedder
    ) -> None:
        results = built_index.search("function def", embedder)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_similar_content_ranks_higher(
        self, built_index: VectorIndex, embedder: FakeEmbedder
    ) -> None:
        # Query with exact content should return that chunk first
        results = built_index.search(
            "def calculate_sum(a: int, b: int) -> int:\n    return a + b",
            embedder,
        )
        assert len(results) > 0
        assert results[0][0].id == "utils.py:calculate_sum:5"

    def test_search_uses_embedder_query_encoder(self) -> None:
        embedder = QueryPrefixEmbedder()
        index = VectorIndex()
        index.build(QUERY_PREFIX_CHUNKS, embedder)

        results = index.search("target", embedder, top_k=1)

        assert results[0][0].id == "target.py:target:1"
        assert embedder.encoded_queries == ["target"]
        assert embedder.encoded_documents == ["target", "other"]


class TestVectorIndexRerank:
    def test_rerank_uses_embedder_query_encoder(self) -> None:
        embedder = QueryPrefixEmbedder()
        index = VectorIndex()

        results = index.rerank("target", list(reversed(QUERY_PREFIX_CHUNKS)), embedder)

        assert results[0][0].id == "target.py:target:1"
        assert embedder.encoded_queries == ["target"]
        assert embedder.encoded_documents == ["other", "target"]


class TestVectorIndexPersistence:
    def test_save_and_load_roundtrip(
        self, built_index: VectorIndex, embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        save_path = tmp_path / "vectors.npz"
        built_index.save(save_path)

        loaded = VectorIndex()
        loaded.load(save_path, SAMPLE_CHUNKS)
        assert loaded.size == built_index.size

        # Search results should match
        original = built_index.search("user", embedder)
        restored = loaded.search("user", embedder)
        assert len(original) == len(restored)
        for (oc, os_), (rc, rs) in zip(original, restored, strict=True):
            assert oc.id == rc.id
            assert abs(os_ - rs) < 1e-5

    def test_save_empty_index_raises(self, tmp_path: Path) -> None:
        idx = VectorIndex()
        with pytest.raises(ArchexIndexError, match="empty"):
            idx.save(tmp_path / "empty.npz")

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        idx = VectorIndex()
        with pytest.raises(ArchexIndexError, match="not found"):
            idx.load(tmp_path / "nonexistent.npz", [])

    def test_save_creates_parent_dirs(self, built_index: VectorIndex, tmp_path: Path) -> None:
        save_path = tmp_path / "sub" / "dir" / "vectors.npz"
        built_index.save(save_path)
        assert save_path.exists()

    def test_save_load_with_long_chunk_ids(self, embedder: FakeEmbedder, tmp_path: Path) -> None:
        long_id = "a/b/c/d/e/f/g/h/i/j/k/l/m/n.py:VeryLongClassName_WithSuffix:12345"
        chunk = CodeChunk(
            id=long_id,
            content="def foo(): pass",
            file_path="a/b/c/d/e/f/g/h/i/j/k/l/m/n.py",
            start_line=12345,
            end_line=12346,
            symbol_name="VeryLongClassName_WithSuffix",
            symbol_kind=SymbolKind.CLASS,
            language="python",
            token_count=5,
        )
        idx = VectorIndex()
        idx.build([chunk], embedder)
        save_path = tmp_path / "long_ids.npz"
        idx.save(save_path)

        loaded = VectorIndex()
        loaded.load(save_path, [chunk])
        assert loaded.size == 1
        results = loaded.search(chunk.content, embedder)
        assert len(results) == 1
        assert results[0][0].id == long_id

    def test_load_rejects_pickle_payload(self, tmp_path: Path) -> None:
        # numpy's savez with dtype=object uses pickle; allow_pickle=False must reject it
        bad_path = tmp_path / "bad.npz"
        vectors = np.zeros((1, 4), dtype=np.float32)
        # object array that would hold a pickle payload
        malicious = np.empty(1, dtype=object)
        malicious[0] = "injected"
        np.savez_compressed(str(bad_path), vectors=vectors, chunk_ids=malicious)

        idx = VectorIndex()
        with pytest.raises(ValueError):
            # allow_pickle=False causes ValueError on object dtype arrays
            idx.load(bad_path, [])

    def test_load_validates_chunk_id_length_matches_vectors(
        self, embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        idx = VectorIndex()
        idx.build(SAMPLE_CHUNKS, embedder)
        save_path = tmp_path / "vectors.npz"
        idx.save(save_path)

        # Manually write a corrupt npz with mismatched lengths
        data = np.load(str(save_path), allow_pickle=False)
        corrupt_path = tmp_path / "corrupt.npz"
        np.savez_compressed(
            str(corrupt_path),
            vectors=data["vectors"],
            chunk_ids=np.array(["only_one_id"], dtype="U512"),
        )

        loaded = VectorIndex()
        with pytest.raises(ArchexIndexError, match="corrupt"):
            loaded.load(corrupt_path, SAMPLE_CHUNKS)


class TestVectorIndexEmbedderManifest:
    def test_save_load_with_matching_embedder(self, embedder: FakeEmbedder, tmp_path: Path) -> None:
        """save/load round-trip succeeds when embedder name and dim match."""
        idx = VectorIndex()
        idx.build(SAMPLE_CHUNKS[:2], embedder)
        p = tmp_path / "vec.npz"
        idx.save(p, embedder_name="fake", vector_dim=64)

        idx2 = VectorIndex()
        idx2.load(p, SAMPLE_CHUNKS[:2], embedder_name="fake", vector_dim=64)
        assert idx2.size == 2

    def test_load_rejects_mismatched_embedder_name(
        self, embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """load raises ArchexIndexError when embedder name doesn't match."""
        idx = VectorIndex()
        idx.build([SAMPLE_CHUNKS[0]], embedder)
        p = tmp_path / "vec.npz"
        idx.save(p, embedder_name="model-a", vector_dim=64)

        idx2 = VectorIndex()
        with pytest.raises(ArchexIndexError, match="Embedder mismatch"):
            idx2.load(p, [SAMPLE_CHUNKS[0]], embedder_name="model-b", vector_dim=64)

    def test_load_rejects_mismatched_vector_dim(
        self, embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """load raises ArchexIndexError when vector dimension doesn't match."""
        idx = VectorIndex()
        idx.build([SAMPLE_CHUNKS[0]], embedder)
        p = tmp_path / "vec.npz"
        idx.save(p, embedder_name="fake", vector_dim=64)

        idx2 = VectorIndex()
        with pytest.raises(ArchexIndexError, match="Vector dim mismatch"):
            idx2.load(p, [SAMPLE_CHUNKS[0]], embedder_name="fake", vector_dim=768)

    def test_load_tolerates_legacy_npz_without_meta(self, tmp_path: Path) -> None:
        """load succeeds for legacy .npz files that lack embedder_meta."""
        p = tmp_path / "legacy.npz"
        np.savez_compressed(
            str(p),
            vectors=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            chunk_ids=np.array(["c1"], dtype="U512"),
        )
        idx = VectorIndex()
        idx.load(p, [], embedder_name="anything", vector_dim=3)
        assert idx.size == 1


class TestReciprocalRankFusion:
    def test_rrf_merges_results(self) -> None:
        bm25: list[tuple[CodeChunk, float]] = [
            (SAMPLE_CHUNKS[0], 5.0),
            (SAMPLE_CHUNKS[1], 3.0),
        ]
        vector: list[tuple[CodeChunk, float]] = [
            (SAMPLE_CHUNKS[1], 0.9),
            (SAMPLE_CHUNKS[2], 0.8),
        ]
        merged = reciprocal_rank_fusion(bm25, vector)
        ids = [c.id for c, _ in merged]
        # chunk[1] appears in both lists, should rank highest
        assert ids[0] == SAMPLE_CHUNKS[1].id
        assert len(merged) == 3

    def test_rrf_deduplicates(self) -> None:
        bm25: list[tuple[CodeChunk, float]] = [
            (SAMPLE_CHUNKS[0], 5.0),
        ]
        vector: list[tuple[CodeChunk, float]] = [
            (SAMPLE_CHUNKS[0], 0.9),
        ]
        merged = reciprocal_rank_fusion(bm25, vector)
        assert len(merged) == 1

    def test_rrf_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([], []) == []

    def test_rrf_one_empty_list(self) -> None:
        bm25: list[tuple[CodeChunk, float]] = [
            (SAMPLE_CHUNKS[0], 5.0),
            (SAMPLE_CHUNKS[1], 3.0),
        ]
        merged = reciprocal_rank_fusion(bm25, [])
        assert len(merged) == 2

    def test_rrf_scores_are_positive(self) -> None:
        bm25: list[tuple[CodeChunk, float]] = [(SAMPLE_CHUNKS[0], 1.0)]
        vector: list[tuple[CodeChunk, float]] = [(SAMPLE_CHUNKS[1], 0.5)]
        merged = reciprocal_rank_fusion(bm25, vector)
        for _, score in merged:
            assert score > 0

    def test_rrf_custom_k(self) -> None:
        bm25: list[tuple[CodeChunk, float]] = [(SAMPLE_CHUNKS[0], 1.0)]
        vector: list[tuple[CodeChunk, float]] = [(SAMPLE_CHUNKS[0], 0.5)]
        result_k10 = reciprocal_rank_fusion(bm25, vector, k=10)
        result_k100 = reciprocal_rank_fusion(bm25, vector, k=100)
        # Higher k → lower individual scores
        assert result_k10[0][1] > result_k100[0][1]

    def test_rrf_preserves_all_chunks(self) -> None:
        bm25: list[tuple[CodeChunk, float]] = [
            (SAMPLE_CHUNKS[0], 5.0),
            (SAMPLE_CHUNKS[1], 3.0),
        ]
        vector: list[tuple[CodeChunk, float]] = [
            (SAMPLE_CHUNKS[2], 0.9),
            (SAMPLE_CHUNKS[3], 0.8),
        ]
        merged = reciprocal_rank_fusion(bm25, vector)
        assert len(merged) == 4


class TestVectorPersistenceRoundTrip:
    """Tests for build → save → load → search round-trip via IndexStore.vector_index_path."""

    def test_build_save_load_search_matches_original(
        self, embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """build → save → load → search produces the same results as build → search."""
        idx = VectorIndex()
        idx.build(SAMPLE_CHUNKS, embedder)
        expected = idx.search("user authentication", embedder)

        save_path = tmp_path / "vec.npz"
        idx.save(save_path, embedder_name="fake", vector_dim=64)

        loaded = VectorIndex()
        loaded.load(save_path, SAMPLE_CHUNKS, embedder_name="fake", vector_dim=64)
        actual = loaded.search("user authentication", embedder)

        assert len(actual) == len(expected)
        for (ec, es), (ac, as_) in zip(expected, actual, strict=True):
            assert ec.id == ac.id
            assert abs(es - as_) < 1e-5

    def test_save_creates_npz_at_expected_path(
        self, embedder: FakeEmbedder, tmp_path: Path
    ) -> None:
        """save() writes the .npz file to the specified path."""
        save_path = tmp_path / "index.vectors.npz"
        idx = VectorIndex()
        idx.build(SAMPLE_CHUNKS[:2], embedder)
        idx.save(save_path, embedder_name="fake", vector_dim=64)
        assert save_path.exists()

    def test_load_wrong_embedder_name_raises(self, embedder: FakeEmbedder, tmp_path: Path) -> None:
        """load raises ArchexIndexError when embedder_name doesn't match saved metadata."""
        save_path = tmp_path / "vec.npz"
        idx = VectorIndex()
        idx.build([SAMPLE_CHUNKS[0]], embedder)
        idx.save(save_path, embedder_name="model-x", vector_dim=64)

        loader = VectorIndex()
        with pytest.raises(ArchexIndexError, match="Embedder mismatch"):
            loader.load(save_path, [SAMPLE_CHUNKS[0]], embedder_name="model-y", vector_dim=64)

    def test_index_store_vector_index_path_returns_correct_path(self, tmp_path: Path) -> None:
        """IndexStore.vector_index_path is db_path with .vectors.npz suffix."""
        db_path = tmp_path / "index.db"
        store = IndexStore(db_path)
        try:
            expected = tmp_path / "index.vectors.npz"
            assert store.vector_index_path == expected
        finally:
            store.close()


class TestConfidenceWeightedRRF:
    """Tests for confidence_weighted_rrf weight adaptation."""

    def _make_results(
        self,
        chunks: list[CodeChunk],
        scores: list[float],
    ) -> list[tuple[CodeChunk, float]]:
        return list(zip(chunks, scores, strict=True))

    def test_high_agreement_favors_bm25(self) -> None:
        """signal_agreement > 0.7 → bm25_weight=0.85, vector_weight=0.15."""
        bm25 = self._make_results(SAMPLE_CHUNKS[:2], [5.0, 3.0])
        vec = self._make_results(SAMPLE_CHUNKS[:2], [0.9, 0.8])
        _, bw, vw = confidence_weighted_rrf(bm25, vec, signal_agreement=0.8, bm25_score_cv=0.5)
        assert bw == 0.85
        assert vw == 0.15

    def test_low_agreement_favors_vector(self) -> None:
        """signal_agreement < 0.3 → bm25_weight=0.35, vector_weight=0.65."""
        bm25 = self._make_results(SAMPLE_CHUNKS[:2], [5.0, 3.0])
        vec = self._make_results(SAMPLE_CHUNKS[2:], [0.9, 0.8])
        _, bw, vw = confidence_weighted_rrf(bm25, vec, signal_agreement=0.1, bm25_score_cv=0.5)
        assert bw == 0.35
        assert vw == 0.65

    def test_medium_agreement_low_cv_favors_vector(self) -> None:
        """Medium agreement with low BM25 CV (<=0.3) → bm25=0.40, vector=0.60."""
        bm25 = self._make_results(SAMPLE_CHUNKS[:2], [5.0, 3.0])
        vec = self._make_results(SAMPLE_CHUNKS[1:3], [0.9, 0.8])
        _, bw, vw = confidence_weighted_rrf(bm25, vec, signal_agreement=0.5, bm25_score_cv=0.2)
        assert bw == 0.40
        assert vw == 0.60

    def test_medium_agreement_high_cv_equal_weights(self) -> None:
        """Medium agreement with high BM25 CV (>0.3) → bm25=0.50, vector=0.50."""
        bm25 = self._make_results(SAMPLE_CHUNKS[:2], [5.0, 3.0])
        vec = self._make_results(SAMPLE_CHUNKS[1:3], [0.9, 0.8])
        _, bw, vw = confidence_weighted_rrf(bm25, vec, signal_agreement=0.5, bm25_score_cv=0.6)
        assert bw == 0.50
        assert vw == 0.50

    def test_weights_sum_to_one(self) -> None:
        """BM25 and vector weights always sum to 1.0 across all branches."""
        cases = [
            (0.8, 0.5),  # high agreement
            (0.5, 0.6),  # medium + high CV
            (0.5, 0.2),  # medium + low CV
            (0.1, 0.5),  # low agreement
            (0.3, 0.3),  # boundary: agreement exactly 0.3
            (0.7, 0.3),  # boundary: agreement exactly 0.7
        ]
        bm25 = self._make_results(SAMPLE_CHUNKS[:2], [5.0, 3.0])
        vec = self._make_results(SAMPLE_CHUNKS[:2], [0.9, 0.8])
        for agreement, cv in cases:
            _, bw, vw = confidence_weighted_rrf(
                bm25,
                vec,
                signal_agreement=agreement,
                bm25_score_cv=cv,
            )
            assert abs(bw + vw - 1.0) < 1e-9, f"weights don't sum to 1 for {agreement=}, {cv=}"

    def test_fused_results_deduplicate_chunks(self) -> None:
        """Chunks appearing in both lists are deduplicated."""
        bm25 = self._make_results([SAMPLE_CHUNKS[0]], [5.0])
        vec = self._make_results([SAMPLE_CHUNKS[0]], [0.9])
        fused, _, _ = confidence_weighted_rrf(bm25, vec, signal_agreement=0.8, bm25_score_cv=0.5)
        assert len(fused) == 1

    def test_fused_results_sorted_descending(self) -> None:
        """Fused results are sorted by score descending."""
        bm25 = self._make_results(SAMPLE_CHUNKS[:3], [5.0, 3.0, 1.0])
        vec = self._make_results(SAMPLE_CHUNKS[:3], [0.9, 0.8, 0.7])
        fused, _, _ = confidence_weighted_rrf(bm25, vec, signal_agreement=0.5, bm25_score_cv=0.4)
        fused_scores = [s for _, s in fused]
        assert fused_scores == sorted(fused_scores, reverse=True)

    def test_empty_bm25_returns_vector_only(self) -> None:
        """Empty BM25 results: fused output comes entirely from vector results."""
        vec = self._make_results(SAMPLE_CHUNKS[:2], [0.9, 0.8])
        fused, _, _ = confidence_weighted_rrf([], vec, signal_agreement=0.0, bm25_score_cv=0.0)
        assert len(fused) == 2

    def test_empty_vector_returns_bm25_only(self) -> None:
        """Empty vector results: fused output comes entirely from BM25 results."""
        bm25 = self._make_results(SAMPLE_CHUNKS[:2], [5.0, 3.0])
        fused, _, _ = confidence_weighted_rrf(bm25, [], signal_agreement=0.0, bm25_score_cv=0.5)
        assert len(fused) == 2


def _make_chunks_with_paths(
    paths: list[str],
    scores: list[float] | None = None,
) -> list[tuple[CodeChunk, float]]:
    """Build (chunk, score) pairs with distinct file paths for should_fuse tests."""
    results: list[tuple[CodeChunk, float]] = []
    for i, path in enumerate(paths):
        chunk = CodeChunk(
            id=f"{path}:fn:{i}",
            content=f"def fn_{i}(): pass",
            file_path=path,
            start_line=1,
            end_line=1,
            symbol_name=f"fn_{i}",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=10,
        )
        score = scores[i] if scores is not None else float(10 - i)
        results.append((chunk, score))
    return results


class TestShouldFuse:
    """Tests for the should_fuse() gating function."""

    def test_high_cv_high_agreement_skips_fusion(self) -> None:
        """BM25 CV > threshold AND agreement > threshold → skip fusion (BM25 confident)."""
        # Same 5 paths in both lists → high Jaccard; wide score spread → high CV
        paths = ["a.py", "b.py", "c.py", "d.py", "e.py"]
        bm25 = _make_chunks_with_paths(paths)
        # Manually override scores for high CV (large spread)
        bm25 = [(c, float(score * 10)) for c, score in bm25]  # scores: 90, 80, 70, 60, 50
        vec = _make_chunks_with_paths(paths)

        apply, reason = should_fuse(bm25, vec, cv_threshold=0.1, agreement_threshold=0.5)
        assert not apply
        assert reason.startswith("bm25_confident:")

    def test_low_cv_low_agreement_applies_fusion(self) -> None:
        """BM25 CV below threshold AND agreement below threshold → apply fusion."""
        bm25_paths = ["a.py", "b.py", "c.py", "d.py", "e.py"]
        vec_paths = ["x.py", "y.py", "z.py", "w.py", "v.py"]
        # Flat scores → low CV
        bm25 = [(c, 1.0) for c, _ in _make_chunks_with_paths(bm25_paths)]
        vec = _make_chunks_with_paths(vec_paths)

        apply, reason = should_fuse(bm25, vec)
        assert apply
        assert reason.startswith("fusion_needed:")

    def test_high_cv_low_agreement_applies_fusion(self) -> None:
        """High CV but low agreement → apply fusion (signals disagree, vector has novel hits)."""
        bm25_paths = ["a.py", "b.py", "c.py", "d.py", "e.py"]
        vec_paths = ["x.py", "y.py", "z.py", "w.py", "v.py"]
        # Wide score spread → high CV
        bm25 = [
            (c, float((5 - i) * 100))
            for i, (c, _) in enumerate(_make_chunks_with_paths(bm25_paths))
        ]
        vec = _make_chunks_with_paths(vec_paths)

        apply, reason = should_fuse(bm25, vec, cv_threshold=0.1, agreement_threshold=0.99)
        assert apply
        assert reason.startswith("fusion_needed:")

    def test_low_cv_high_agreement_applies_fusion(self) -> None:
        """Low CV AND high agreement → apply fusion (BM25 ambiguous, even if signals agree)."""
        paths = ["a.py", "b.py", "c.py", "d.py", "e.py"]
        # Flat scores → low CV
        bm25 = [(c, 1.0) for c, _ in _make_chunks_with_paths(paths)]
        vec = _make_chunks_with_paths(paths)

        apply, reason = should_fuse(bm25, vec, cv_threshold=0.5, agreement_threshold=0.5)
        assert apply
        assert reason.startswith("fusion_needed:")

    def test_too_few_bm25_results_skips_fusion(self) -> None:
        """Fewer than min_results BM25 results → skip fusion."""
        bm25 = _make_chunks_with_paths(["a.py", "b.py"])  # only 2
        vec = _make_chunks_with_paths(["a.py", "b.py", "c.py", "d.py"])

        apply, reason = should_fuse(bm25, vec, min_results=3)
        assert not apply
        assert reason.startswith("too_few_bm25_results:")
        assert "2" in reason

    def test_too_few_vector_results_skips_fusion(self) -> None:
        """Fewer than min_results vector results → skip fusion."""
        bm25 = _make_chunks_with_paths(["a.py", "b.py", "c.py", "d.py"])
        vec = _make_chunks_with_paths(["x.py"])  # only 1

        apply, reason = should_fuse(bm25, vec, min_results=3)
        assert not apply
        assert reason.startswith("too_few_vector_results:")
        assert "1" in reason

    def test_empty_vector_results_skips_fusion(self) -> None:
        """Empty vector list → skip fusion."""
        bm25 = _make_chunks_with_paths(["a.py", "b.py", "c.py", "d.py"])

        apply, reason = should_fuse(bm25, [], min_results=3)
        assert not apply
        assert reason.startswith("too_few_vector_results:0")

    def test_returns_tuple_of_bool_and_str(self) -> None:
        """Return type is always (bool, str)."""
        bm25 = _make_chunks_with_paths(["a.py", "b.py", "c.py"])
        vec = _make_chunks_with_paths(["a.py", "b.py", "c.py"])
        result = should_fuse(bm25, vec)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_low_avg_idf_forces_fusion(self) -> None:
        """Low AvgIDF forces fusion regardless of CV and agreement."""
        # High CV + high agreement would normally skip fusion
        bm25 = _make_chunks_with_paths(["a.py", "b.py", "c.py"], scores=[10.0, 1.0, 0.5])
        vec = _make_chunks_with_paths(["a.py", "b.py", "c.py"])
        apply, reason = should_fuse(bm25, vec, avg_idf=1.0, idf_threshold=2.0)
        assert apply is True
        assert "low_idf_force_fusion" in reason

    def test_high_avg_idf_does_not_force_fusion(self) -> None:
        """High AvgIDF defers to the existing CV/agreement gate."""
        bm25 = _make_chunks_with_paths(["a.py", "b.py", "c.py"], scores=[10.0, 1.0, 0.5])
        vec = _make_chunks_with_paths(["a.py", "b.py", "c.py"])
        # With high IDF, the function falls through to the CV/agreement check
        _, reason = should_fuse(
            bm25, vec, avg_idf=5.0, idf_threshold=2.0, cv_threshold=0.1, agreement_threshold=0.5
        )
        assert "low_idf_force_fusion" not in reason

    def test_avg_idf_none_uses_existing_gate(self) -> None:
        """When avg_idf is None, the existing CV/agreement gate is used unchanged."""
        bm25 = _make_chunks_with_paths(["a.py", "b.py", "c.py"])
        vec = _make_chunks_with_paths(["a.py", "b.py", "c.py"])
        result_no_idf = should_fuse(bm25, vec)
        result_none_idf = should_fuse(bm25, vec, avg_idf=None)
        assert result_no_idf == result_none_idf


class TestRelativeScoreFusion:
    """Tests for relative_score_fusion."""

    def test_rsf_preserves_score_magnitude(self) -> None:
        # Arrange: chunk_a is BM25-only top scorer; chunk_b appears in both
        chunk_a = CodeChunk(
            id="a.py:fn:0",
            content="def fn_a(): pass",
            file_path="a.py",
            start_line=1,
            end_line=1,
            symbol_name="fn_a",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=5,
        )
        chunk_b = CodeChunk(
            id="b.py:fn:1",
            content="def fn_b(): pass",
            file_path="b.py",
            start_line=1,
            end_line=1,
            symbol_name="fn_b",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=5,
        )
        chunk_c = CodeChunk(
            id="c.py:fn:2",
            content="def fn_c(): pass",
            file_path="c.py",
            start_line=1,
            end_line=1,
            symbol_name="fn_c",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=5,
        )
        chunk_d = CodeChunk(
            id="d.py:fn:3",
            content="def fn_d(): pass",
            file_path="d.py",
            start_line=1,
            end_line=1,
            symbol_name="fn_d",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=5,
        )
        bm25_results: list[tuple[CodeChunk, float]] = [
            (chunk_a, 10.0),
            (chunk_b, 5.0),
            (chunk_c, 1.0),
        ]
        vector_results: list[tuple[CodeChunk, float]] = [
            (chunk_b, 0.9),
            (chunk_c, 0.8),
            (chunk_d, 0.7),
        ]

        # Act
        fused = relative_score_fusion(bm25_results, vector_results)
        ids = [c.id for c, _ in fused]

        # Assert: chunk_a (BM25-only dominant) and chunk_b (both signals) in top 2
        assert chunk_a.id in ids[:2]
        assert chunk_b.id in ids[:2]

    def test_rsf_equal_weights(self) -> None:
        # Arrange: identical scores in both signals
        chunk_a = CodeChunk(
            id="a.py:fn:0",
            content="def fn_a(): pass",
            file_path="a.py",
            start_line=1,
            end_line=1,
            symbol_name="fn_a",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=5,
        )
        chunk_b = CodeChunk(
            id="b.py:fn:1",
            content="def fn_b(): pass",
            file_path="b.py",
            start_line=1,
            end_line=1,
            symbol_name="fn_b",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=5,
        )
        results: list[tuple[CodeChunk, float]] = [(chunk_a, 1.0), (chunk_b, 0.5)]

        # Act: fuse with equal weights using same list for both signals
        fused = relative_score_fusion(results, results, bm25_weight=0.5, vector_weight=0.5)
        scores = {c.id: s for c, s in fused}

        # Assert: symmetric — both get the same normalized + weighted score
        # chunk_a is top in both signals (normalized to 1.0), chunk_b to 0.0
        assert scores[chunk_a.id] > scores[chunk_b.id]

    def test_rsf_empty_bm25_returns_vector_signal(self) -> None:
        chunk_a = CodeChunk(
            id="a.py:fn:0",
            content="def fn_a(): pass",
            file_path="a.py",
            start_line=1,
            end_line=1,
            symbol_name="fn_a",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=5,
        )
        vector_results: list[tuple[CodeChunk, float]] = [(chunk_a, 0.9)]

        fused = relative_score_fusion([], vector_results)
        assert len(fused) == 1
        assert fused[0][0].id == chunk_a.id

    def test_rsf_empty_vector_returns_bm25_signal(self) -> None:
        chunk_a = CodeChunk(
            id="a.py:fn:0",
            content="def fn_a(): pass",
            file_path="a.py",
            start_line=1,
            end_line=1,
            symbol_name="fn_a",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=5,
        )
        bm25_results: list[tuple[CodeChunk, float]] = [(chunk_a, 5.0)]

        fused = relative_score_fusion(bm25_results, [])
        assert len(fused) == 1
        assert fused[0][0].id == chunk_a.id

    def test_rsf_flat_scores_normalize_to_full_signal(self) -> None:
        # Arrange: all equal positive scores → normalized to 1.0 per signal
        chunks = [
            CodeChunk(
                id=f"f{i}.py:fn:{i}",
                content=f"def fn_{i}(): pass",
                file_path=f"f{i}.py",
                start_line=1,
                end_line=1,
                symbol_name=f"fn_{i}",
                symbol_kind=SymbolKind.FUNCTION,
                language="python",
                token_count=5,
            )
            for i in range(3)
        ]
        bm25_results: list[tuple[CodeChunk, float]] = [(c, 1.0) for c in chunks]
        vector_results: list[tuple[CodeChunk, float]] = [(c, 1.0) for c in chunks]

        fused = relative_score_fusion(bm25_results, vector_results)
        # All chunks get 0.5*1.0 + 0.5*1.0 = 1.0 each
        for _, score in fused:
            assert abs(score - 1.0) < 1e-9
