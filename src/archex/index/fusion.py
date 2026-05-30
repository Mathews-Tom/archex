"""Fusion strategies for merging BM25 and vector retrieval signals.

Provides Reciprocal Rank Fusion (RRF), Relative Score Fusion (RSF),
confidence-aware weighted variants, and a pre-retrieval gate (should_fuse)
that decides whether fusion adds value for a given query.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from archex.models import CodeChunk


def bm25_score_cv(bm25_results: list[tuple[CodeChunk, float]], top_n: int = 10) -> float:
    """Compute coefficient of variation of BM25 top-N scores.

    Low CV indicates flat BM25 scores (vocabulary ambiguity — no clear winner).
    High CV indicates a clear BM25 signal with well-separated scores.
    Returns 0.0 when fewer than 2 results are present or mean is near zero.
    """
    scores = [score for _, score in bm25_results[:top_n]]
    if len(scores) < 2:
        return 0.0
    arr = np.array(scores, dtype=np.float64)
    mean = float(arr.mean())
    if mean < 1e-9:
        return 0.0
    return float(arr.std() / mean)


def should_fuse(
    bm25_results: list[tuple[CodeChunk, float]],
    vector_results: list[tuple[CodeChunk, float]],
    *,
    avg_idf: float | None = None,
    idf_threshold: float = 2.0,
    cv_threshold: float = 0.8,
    agreement_threshold: float = 0.8,
    min_results: int = 3,
) -> tuple[bool, str]:
    """Decide whether to apply fusion based on BM25 confidence signals.

    Returns (should_apply, reason) where reason explains the decision.

    Fusion is FORCED when:
    - avg_idf is provided and below idf_threshold (query terms are too common
      in the corpus for BM25 to discriminate — a pre-retrieval QPP signal)

    Fusion is SKIPPED only when:
    - BM25 score CV > cv_threshold (very clear score separation)
      AND signal agreement > agreement_threshold (both signals strongly agree)
    - Fewer than min_results from either signal

    Thresholds are deliberately permissive: fusion should be the default,
    skipped only when BM25 is unambiguously confident AND vector agrees.
    """
    if len(bm25_results) < min_results:
        return False, f"too_few_bm25_results:{len(bm25_results)}"
    if not vector_results or len(vector_results) < min_results:
        return False, f"too_few_vector_results:{len(vector_results) if vector_results else 0}"

    # Pre-retrieval gate: low AvgIDF means query terms are common across the
    # corpus — BM25 scores will be flat regardless of result quality.
    # Force fusion unconditionally to let vector search disambiguate.
    if avg_idf is not None and avg_idf < idf_threshold:
        return True, f"low_idf_force_fusion:avg_idf={avg_idf:.3f}"

    cv = bm25_score_cv(bm25_results)

    # Compute signal agreement (Jaccard of top-20 file paths)
    k_agree = 20
    bm25_top = {chunk.file_path for chunk, _ in bm25_results[:k_agree]}
    vec_top = {chunk.file_path for chunk, _ in vector_results[:k_agree]}
    union = bm25_top | vec_top
    agreement = len(bm25_top & vec_top) / len(union) if union else 0.0

    if cv > cv_threshold and agreement > agreement_threshold:
        return False, f"bm25_confident:cv={cv:.3f},agreement={agreement:.3f}"

    return True, f"fusion_needed:cv={cv:.3f},agreement={agreement:.3f}"


def reciprocal_rank_fusion(
    bm25_results: list[tuple[CodeChunk, float]],
    vector_results: list[tuple[CodeChunk, float]],
    k: int = 60,
) -> list[tuple[CodeChunk, float]]:
    """Merge BM25 and vector search results using Reciprocal Rank Fusion.

    Each result's RRF score is 1/(k + rank), summed across both lists.
    Higher k values reduce the impact of high-ranking items.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, CodeChunk] = {}

    for rank, (chunk, _) in enumerate(bm25_results):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    for rank, (chunk, _) in enumerate(vector_results):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]


def confidence_weighted_rrf(
    bm25_results: list[tuple[CodeChunk, float]],
    vector_results: list[tuple[CodeChunk, float]],
    signal_agreement: float,
    bm25_score_cv: float,
    k: int = 60,
) -> tuple[list[tuple[CodeChunk, float]], float, float]:
    """Merge BM25 and vector results using weighted Reciprocal Rank Fusion.

    Weight schedule:
    - agreement > 0.7: bm25=0.85, vector=0.15 (both agree, trust faster signal)
    - 0.3-0.7, CV > 0.3: bm25=0.50, vector=0.50 (mixed, clear BM25 spread)
    - 0.3-0.7, CV <= 0.3: bm25=0.40, vector=0.60 (mixed, flat BM25, vector disambiguates)
    - agreement < 0.3: bm25=0.35, vector=0.65 (strong disagreement, vector has novel hits)

    Returns (fused_results, bm25_weight, vector_weight).
    """
    if signal_agreement > 0.7:
        bm25_weight, vector_weight = 0.85, 0.15
    elif signal_agreement >= 0.3:
        if bm25_score_cv > 0.3:
            bm25_weight, vector_weight = 0.50, 0.50
        else:
            bm25_weight, vector_weight = 0.40, 0.60
    else:
        bm25_weight, vector_weight = 0.35, 0.65

    scores: dict[str, float] = {}
    chunk_map: dict[str, CodeChunk] = {}

    for rank, (chunk, _) in enumerate(bm25_results):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + bm25_weight / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    for rank, (chunk, _) in enumerate(vector_results):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + vector_weight / (k + rank + 1)
        chunk_map[chunk.id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    fused = [(chunk_map[cid], scores[cid]) for cid in sorted_ids]
    return fused, bm25_weight, vector_weight


def normalize_scores(
    results: list[tuple[CodeChunk, float]],
) -> dict[str, float]:
    """Normalize scores by the maximum score, preserving retrieved positive evidence."""
    if not results:
        return {}
    scores = [s for _, s in results]
    max_s = max(scores)
    if max_s < 1e-9:
        return {c.id: 0.0 for c, _ in results}
    return {c.id: s / max_s for c, s in results}


def adaptive_rsf_weights(
    signal_agreement: float,
    bm25_cv: float,
) -> tuple[float, float]:
    """Compute RSF weights based on signal agreement and BM25 score spread.

    Weight schedule (more balanced than RRF — vector always gets meaningful weight):
    - agreement > 0.7: bm25=0.60, vector=0.40 (both agree, slight BM25 preference)
    - 0.3-0.7, CV > 0.3: bm25=0.50, vector=0.50 (mixed signals, equal weight)
    - 0.3-0.7, CV <= 0.3: bm25=0.40, vector=0.60 (flat BM25, vector disambiguates)
    - agreement < 0.3: bm25=0.35, vector=0.65 (strong disagreement, trust vector novelty)
    """
    if signal_agreement > 0.7:
        return 0.60, 0.40
    if signal_agreement >= 0.3:
        if bm25_cv > 0.3:
            return 0.50, 0.50
        return 0.40, 0.60
    return 0.35, 0.65


def relative_score_fusion(
    bm25_results: list[tuple[CodeChunk, float]],
    vector_results: list[tuple[CodeChunk, float]],
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> list[tuple[CodeChunk, float]]:
    """Fuse BM25 and vector results using Relative Score Fusion.

    Normalizes each signal's scores independently to [0, 1],
    preserving within-signal score distances. Weighted combination.

    Unlike RRF which uses only rank position, RSF preserves score
    magnitude — when BM25 has a clear winner (0.80 vs 0.40), that
    gap is reflected in the fused score.
    """
    bm25_norm = normalize_scores(bm25_results)
    vec_norm = normalize_scores(vector_results)
    chunk_map = {c.id: c for c, _ in bm25_results}
    chunk_map.update({c.id: c for c, _ in vector_results})

    all_ids = set(bm25_norm) | set(vec_norm)
    fused = {
        cid: bm25_weight * bm25_norm.get(cid, 0.0) + vector_weight * vec_norm.get(cid, 0.0)
        for cid in all_ids
    }
    sorted_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [(chunk_map[cid], fused[cid]) for cid in sorted_ids]


def adaptive_rsf(
    bm25_results: list[tuple[CodeChunk, float]],
    vector_results: list[tuple[CodeChunk, float]],
    signal_agreement: float,
    bm25_cv: float,
) -> tuple[list[tuple[CodeChunk, float]], float, float]:
    """Fuse BM25 and vector results using RSF with adaptive weights.

    Combines RSF's score-magnitude preservation with confidence-aware
    weight scheduling. Returns (fused_results, bm25_weight, vector_weight).
    """
    bm25_weight, vector_weight = adaptive_rsf_weights(signal_agreement, bm25_cv)
    fused = relative_score_fusion(bm25_results, vector_results, bm25_weight, vector_weight)
    return fused, bm25_weight, vector_weight
