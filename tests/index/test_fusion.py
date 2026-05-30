"""Tests for archex.index.fusion — fusion strategies and gating."""

from __future__ import annotations

import pytest

from archex.index.fusion import (
    adaptive_rsf,
    adaptive_rsf_weights,
    bm25_score_cv,
    confidence_weighted_rrf,
    normalize_scores,
    reciprocal_rank_fusion,
    relative_score_fusion,
    should_fuse,
)
from archex.models import CodeChunk, SymbolKind


def _chunk(id: str, file_path: str = "a.py") -> CodeChunk:
    return CodeChunk(
        id=id,
        content=f"# {id}",
        file_path=file_path,
        start_line=1,
        end_line=2,
        symbol_name=id,
        symbol_kind=SymbolKind.FUNCTION,
        language="python",
        token_count=10,
    )


def _results(
    specs: list[tuple[str, str, float]],
) -> list[tuple[CodeChunk, float]]:
    """Build results from (id, file_path, score) triples."""
    return [(_chunk(id, fp), score) for id, fp, score in specs]


# ---------------------------------------------------------------------------
# bm25_score_cv
# ---------------------------------------------------------------------------


class TestBm25ScoreCV:
    def test_high_cv_with_spread_scores(self) -> None:
        results = _results([("a", "a.py", 10.0), ("b", "b.py", 1.0), ("c", "c.py", 0.1)])
        cv = bm25_score_cv(results)
        assert cv > 0.8

    def test_low_cv_with_flat_scores(self) -> None:
        results = _results([("a", "a.py", 5.0), ("b", "b.py", 4.9), ("c", "c.py", 4.8)])
        cv = bm25_score_cv(results)
        assert cv < 0.05

    def test_fewer_than_two_returns_zero(self) -> None:
        results = _results([("a", "a.py", 5.0)])
        assert bm25_score_cv(results) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert bm25_score_cv([]) == 0.0

    def test_zero_mean_returns_zero(self) -> None:
        results = _results([("a", "a.py", 0.0), ("b", "b.py", 0.0)])
        assert bm25_score_cv(results) == 0.0


# ---------------------------------------------------------------------------
# should_fuse (relaxed thresholds: cv>0.8 AND agreement>0.8 to skip)
# ---------------------------------------------------------------------------


class TestShouldFuse:
    def _bm25_spread(self) -> list[tuple[CodeChunk, float]]:
        """BM25 results with very high CV (>0.8)."""
        return _results([("a", "a.py", 10.0), ("b", "b.py", 1.0), ("c", "c.py", 0.1)])

    def _bm25_flat(self) -> list[tuple[CodeChunk, float]]:
        """BM25 results with low CV (<0.1)."""
        return _results([("a", "a.py", 5.0), ("b", "b.py", 4.9), ("c", "c.py", 4.8)])

    def _vec_same_files(self) -> list[tuple[CodeChunk, float]]:
        """Vector results with same files as BM25 spread — high agreement."""
        return _results([("a", "a.py", 0.9), ("b", "b.py", 0.8), ("c", "c.py", 0.7)])

    def _vec_different_files(self) -> list[tuple[CodeChunk, float]]:
        """Vector results with completely different files — zero agreement."""
        return _results([("x", "x.py", 0.9), ("y", "y.py", 0.8), ("z", "z.py", 0.7)])

    def test_high_cv_high_agreement_skips_fusion(self) -> None:
        fuse, reason = should_fuse(self._bm25_spread(), self._vec_same_files())
        assert not fuse
        assert "bm25_confident" in reason

    def test_high_cv_low_agreement_applies_fusion(self) -> None:
        fuse, reason = should_fuse(self._bm25_spread(), self._vec_different_files())
        assert fuse
        assert "fusion_needed" in reason

    def test_low_cv_high_agreement_applies_fusion(self) -> None:
        """With relaxed thresholds, low CV (<0.8) always triggers fusion."""
        fuse, reason = should_fuse(self._bm25_flat(), self._vec_same_files())
        assert fuse
        assert "fusion_needed" in reason

    def test_low_cv_low_agreement_applies_fusion(self) -> None:
        fuse, reason = should_fuse(self._bm25_flat(), self._vec_different_files())
        assert fuse
        assert "fusion_needed" in reason

    def test_low_idf_forces_fusion(self) -> None:
        fuse, reason = should_fuse(self._bm25_spread(), self._vec_same_files(), avg_idf=1.5)
        assert fuse
        assert "low_idf_force_fusion" in reason

    def test_high_idf_does_not_force_fusion(self) -> None:
        """High IDF doesn't force — falls through to CV/agreement check."""
        fuse, _ = should_fuse(self._bm25_spread(), self._vec_same_files(), avg_idf=3.0)
        # High CV + high agreement → skipped
        assert not fuse

    def test_too_few_bm25_skips(self) -> None:
        bm25 = _results([("a", "a.py", 5.0), ("b", "b.py", 4.0)])
        vec = self._vec_same_files()
        fuse, reason = should_fuse(bm25, vec)
        assert not fuse
        assert "too_few_bm25" in reason

    def test_too_few_vector_skips(self) -> None:
        bm25 = self._bm25_flat()
        vec = _results([("a", "a.py", 0.9)])
        fuse, reason = should_fuse(bm25, vec)
        assert not fuse
        assert "too_few_vector" in reason

    def test_empty_vector_skips(self) -> None:
        fuse, reason = should_fuse(self._bm25_flat(), [])
        assert not fuse
        assert "too_few_vector" in reason

    def test_moderate_cv_applies_fusion(self) -> None:
        """CV between 0.5-0.8 should now trigger fusion (old threshold was 0.5)."""
        bm25 = _results([("a", "a.py", 8.0), ("b", "b.py", 3.0), ("c", "c.py", 2.5)])
        vec = self._vec_same_files()
        fuse, reason = should_fuse(bm25, vec)
        assert fuse
        assert "fusion_needed" in reason


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------


class TestNormalizeScores:
    def test_normalize_spread_scores(self) -> None:
        results = _results([("a", "a.py", 10.0), ("b", "b.py", 5.0), ("c", "c.py", 0.0)])
        norm = normalize_scores(results)
        assert norm["a"] == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
        assert norm["b"] == pytest.approx(0.5)  # pyright: ignore[reportUnknownMemberType]
        assert norm["c"] == pytest.approx(0.0)  # pyright: ignore[reportUnknownMemberType]

    def test_lowest_positive_score_remains_positive(self) -> None:
        results = _results([("a", "a.py", 10.0), ("b", "b.py", 5.0), ("c", "c.py", 1.0)])
        norm = normalize_scores(results)
        assert norm["a"] == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
        assert norm["b"] == pytest.approx(0.5)  # pyright: ignore[reportUnknownMemberType]
        assert norm["c"] == pytest.approx(0.1)  # pyright: ignore[reportUnknownMemberType]

    def test_normalize_flat_scores(self) -> None:
        results = _results([("a", "a.py", 5.0), ("b", "b.py", 5.0)])
        norm = normalize_scores(results)
        assert norm["a"] == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
        assert norm["b"] == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]

    def test_normalize_empty(self) -> None:
        assert normalize_scores([]) == {}


# ---------------------------------------------------------------------------
# adaptive_rsf_weights
# ---------------------------------------------------------------------------


class TestAdaptiveRSFWeights:
    def test_high_agreement(self) -> None:
        bw, vw = adaptive_rsf_weights(signal_agreement=0.8, bm25_cv=0.5)
        assert bw == 0.60
        assert vw == 0.40

    def test_medium_agreement_high_cv(self) -> None:
        bw, vw = adaptive_rsf_weights(signal_agreement=0.5, bm25_cv=0.5)
        assert bw == 0.50
        assert vw == 0.50

    def test_medium_agreement_low_cv(self) -> None:
        bw, vw = adaptive_rsf_weights(signal_agreement=0.5, bm25_cv=0.2)
        assert bw == 0.40
        assert vw == 0.60

    def test_low_agreement(self) -> None:
        bw, vw = adaptive_rsf_weights(signal_agreement=0.2, bm25_cv=0.5)
        assert bw == 0.35
        assert vw == 0.65

    def test_weights_sum_to_one(self) -> None:
        for agreement in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for cv in [0.1, 0.3, 0.5, 0.8]:
                bw, vw = adaptive_rsf_weights(agreement, cv)
                assert bw + vw == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# relative_score_fusion
# ---------------------------------------------------------------------------


class TestRelativeScoreFusion:
    def test_rsf_preserves_score_magnitude(self) -> None:
        """Chunk in both signals should rank above chunks in only one."""
        bm25 = _results([("a", "a.py", 10.0), ("b", "b.py", 2.0)])
        vec = _results([("a", "a.py", 0.9), ("c", "c.py", 0.5)])
        fused = relative_score_fusion(bm25, vec, bm25_weight=0.5, vector_weight=0.5)
        ids = [c.id for c, _ in fused]
        # a appears in both with high scores → boosted to top
        assert ids[0] == "a"

    def test_rsf_vector_unique_hits_surface(self) -> None:
        """Chunks only in vector results should still appear in fused output."""
        bm25 = _results([("a", "a.py", 10.0), ("b", "b.py", 5.0)])
        vec = _results([("c", "c.py", 0.95), ("d", "d.py", 0.90)])
        fused = relative_score_fusion(bm25, vec, bm25_weight=0.5, vector_weight=0.5)
        ids = {c.id for c, _ in fused}
        assert "c" in ids
        assert "d" in ids

    def test_rsf_empty_bm25(self) -> None:
        vec = _results([("a", "a.py", 0.9), ("b", "b.py", 0.5)])
        fused = relative_score_fusion([], vec, bm25_weight=0.5, vector_weight=0.5)
        assert len(fused) == 2
        assert fused[0][0].id == "a"

    def test_rsf_empty_vector(self) -> None:
        bm25 = _results([("a", "a.py", 10.0), ("b", "b.py", 5.0)])
        fused = relative_score_fusion(bm25, [], bm25_weight=0.5, vector_weight=0.5)
        assert len(fused) == 2
        assert fused[0][0].id == "a"

    def test_rsf_sorted_descending(self) -> None:
        bm25 = _results([("a", "a.py", 10.0), ("b", "b.py", 5.0), ("c", "c.py", 1.0)])
        vec = _results([("c", "c.py", 0.9), ("b", "b.py", 0.5), ("a", "a.py", 0.1)])
        fused = relative_score_fusion(bm25, vec)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# adaptive_rsf
# ---------------------------------------------------------------------------


class TestAdaptiveRSF:
    def test_returns_weights_and_results(self) -> None:
        bm25 = _results([("a", "a.py", 10.0), ("b", "b.py", 5.0), ("c", "c.py", 1.0)])
        vec = _results([("a", "a.py", 0.9), ("d", "d.py", 0.8), ("c", "c.py", 0.3)])
        fused, bw, vw = adaptive_rsf(bm25, vec, signal_agreement=0.5, bm25_cv=0.4)
        assert bw + vw == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]
        assert len(fused) == 4  # a, b, c, d

    def test_high_agreement_weights(self) -> None:
        bm25 = _results([("a", "a.py", 10.0), ("b", "b.py", 5.0)])
        vec = _results([("a", "a.py", 0.9), ("b", "b.py", 0.8)])
        _, bw, vw = adaptive_rsf(bm25, vec, signal_agreement=0.8, bm25_cv=0.5)
        assert bw == 0.60
        assert vw == 0.40

    def test_low_agreement_weights(self) -> None:
        bm25 = _results([("a", "a.py", 10.0)])
        vec = _results([("b", "b.py", 0.9)])
        _, bw, vw = adaptive_rsf(bm25, vec, signal_agreement=0.1, bm25_cv=0.5)
        assert bw == 0.35
        assert vw == 0.65

    def test_vector_unique_hits_rank_with_low_agreement(self) -> None:
        """When signals disagree, vector-only chunks should rank competitively."""
        bm25 = _results([("a", "a.py", 10.0), ("b", "b.py", 5.0), ("c", "c.py", 1.0)])
        vec = _results([("x", "x.py", 0.95), ("y", "y.py", 0.90), ("z", "z.py", 0.85)])
        fused, _, _ = adaptive_rsf(bm25, vec, signal_agreement=0.1, bm25_cv=0.3)
        ids = [c.id for c, _ in fused]
        # With 0.35/0.65 weights and no overlap, vector top hit should rank high
        # x gets 0.65*1.0 = 0.65, a gets 0.35*1.0 = 0.35
        assert ids[0] == "x"


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion (preserved for backward compat)
# ---------------------------------------------------------------------------


class TestRRF:
    def test_rrf_merges_and_deduplicates(self) -> None:
        bm25 = _results([("a", "a.py", 10.0), ("b", "b.py", 5.0)])
        vec = _results([("b", "b.py", 0.9), ("c", "c.py", 0.8)])
        fused = reciprocal_rank_fusion(bm25, vec)
        ids = [c.id for c, _ in fused]
        assert len(ids) == 3
        # b appears in both → highest RRF score
        assert ids[0] == "b"

    def test_rrf_scores_positive(self) -> None:
        bm25 = _results([("a", "a.py", 10.0)])
        vec = _results([("b", "b.py", 0.9)])
        fused = reciprocal_rank_fusion(bm25, vec)
        assert all(s > 0 for _, s in fused)

    def test_rrf_empty_inputs(self) -> None:
        assert reciprocal_rank_fusion([], []) == []


# ---------------------------------------------------------------------------
# confidence_weighted_rrf (preserved for backward compat)
# ---------------------------------------------------------------------------


class TestConfidenceWeightedRRF:
    def test_high_agreement_weights(self) -> None:
        bm25 = _results([("a", "a.py", 10.0)])
        vec = _results([("a", "a.py", 0.9)])
        _, bw, vw = confidence_weighted_rrf(bm25, vec, signal_agreement=0.8, bm25_score_cv=0.5)
        assert bw == 0.85
        assert vw == 0.15

    def test_low_agreement_weights(self) -> None:
        bm25 = _results([("a", "a.py", 10.0)])
        vec = _results([("b", "b.py", 0.9)])
        _, bw, vw = confidence_weighted_rrf(bm25, vec, signal_agreement=0.2, bm25_score_cv=0.5)
        assert bw == 0.35
        assert vw == 0.65
