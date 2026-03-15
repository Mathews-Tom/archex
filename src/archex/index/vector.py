"""Vector index: build and query a dense embedding index over CodeChunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from archex.exceptions import ArchexIndexError
from archex.models import ChunkSurrogate, VectorMode

if TYPE_CHECKING:
    from pathlib import Path

    from archex.index.embeddings.base import Embedder
    from archex.models import CodeChunk


class VectorIndex:
    """Dense vector index for semantic search over CodeChunks.

    Vectors are L2-normalized at build time so search uses dot product
    (equivalent to cosine similarity on normalized vectors).
    """

    def __init__(self) -> None:
        self._vectors: np.ndarray[tuple[int, int], np.dtype[np.float32]] | None = None
        self._chunk_ids: list[str] = []
        self._chunks_by_id: dict[str, CodeChunk] = {}

    def build(
        self,
        chunks: list[CodeChunk],
        embedder: Embedder,
        *,
        surrogates_by_chunk_id: dict[str, ChunkSurrogate] | None = None,
        vector_mode: VectorMode = VectorMode.RAW,
    ) -> None:
        """Encode all chunks and store normalized vectors."""
        if not chunks:
            self._vectors = None
            self._chunk_ids = []
            self._chunks_by_id = {}
            return

        texts = [
            surrogates_by_chunk_id[c.id].surrogate_text
            if (
                vector_mode == VectorMode.SURROGATE
                and surrogates_by_chunk_id
                and c.id in surrogates_by_chunk_id
            )
            else c.content
            for c in chunks
        ]
        raw_embeddings = embedder.encode(texts)
        vectors = np.array(raw_embeddings, dtype=np.float32)

        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        vectors = vectors / norms

        self._vectors = vectors
        self._chunk_ids = [c.id for c in chunks]
        self._chunks_by_id = {c.id: c for c in chunks}

    def search(
        self, query: str, embedder: Embedder, top_k: int = 30
    ) -> list[tuple[CodeChunk, float]]:
        """Search for chunks most similar to the query string."""
        if self._vectors is None or len(self._chunk_ids) == 0:
            return []

        query_vec = np.array(embedder.encode([query])[0], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm < 1e-9:
            return []
        query_vec = query_vec / norm

        # Dot product on normalized vectors = cosine similarity
        similarities = self._vectors @ query_vec
        k = min(top_k, len(self._chunk_ids))
        # O(N) argpartition for top-k selection, then sort only the k selected
        if k < len(similarities):
            part_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = part_indices[np.argsort(similarities[part_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        results: list[tuple[CodeChunk, float]] = []
        for idx in top_indices:
            chunk_id = self._chunk_ids[int(idx)]
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            score = float(similarities[int(idx)])
            if score > 0:
                results.append((chunk, score))

        return results

    @property
    def dim(self) -> int:
        """Return the vector dimension, or 0 if not built."""
        if self._vectors is not None:
            return int(self._vectors.shape[1])
        return 0

    def save(
        self,
        path: Path,
        *,
        embedder_name: str = "",
        vector_dim: int = 0,
        vector_mode: VectorMode = VectorMode.RAW,
        surrogate_version: str = "v1",
    ) -> None:
        """Save vectors and chunk IDs to a compressed numpy file."""
        if self._vectors is None:
            raise ArchexIndexError("Cannot save empty vector index")

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            vectors=self._vectors,
            chunk_ids=np.array(self._chunk_ids, dtype="U512"),
            embedder_meta=np.array([embedder_name, str(vector_dim)], dtype="U256"),
            vector_meta=np.array([str(vector_mode), surrogate_version], dtype="U256"),
        )

    def load(
        self,
        path: Path,
        chunks: list[CodeChunk],
        *,
        embedder_name: str = "",
        vector_dim: int = 0,
        vector_mode: VectorMode = VectorMode.RAW,
        surrogate_version: str = "v1",
    ) -> None:
        """Load vectors from disk and rebuild the chunk lookup map."""
        if not path.exists():
            suffix = ".npz"
            p = path if path.suffix == suffix else path.with_suffix(suffix)
            if not p.exists():
                raise ArchexIndexError(f"Vector index file not found: {path}")
            path = p

        data = np.load(str(path), allow_pickle=False)
        vectors = data["vectors"].astype(np.float32, copy=False)
        chunk_ids = list(data["chunk_ids"])
        if vectors.shape[0] != len(chunk_ids):
            raise ArchexIndexError(
                f"Vector index corrupt: {vectors.shape[0]} vectors but {len(chunk_ids)} chunk IDs"
            )

        if "embedder_meta" in data:
            stored = list(data["embedder_meta"])
            if len(stored) >= 2:
                stored_name, stored_dim = str(stored[0]), int(stored[1])
                if embedder_name and stored_name and stored_name != embedder_name:
                    raise ArchexIndexError(
                        f"Embedder mismatch: cached={stored_name}, current={embedder_name}"
                    )
                if vector_dim > 0 and stored_dim > 0 and stored_dim != vector_dim:
                    raise ArchexIndexError(
                        f"Vector dim mismatch: cached={stored_dim}, current={vector_dim}"
                    )
        if "vector_meta" in data:
            stored_mode, stored_version = [str(v) for v in data["vector_meta"][:2]]
            if stored_mode and stored_mode != str(vector_mode):
                raise ArchexIndexError(
                    f"Vector mode mismatch: cached={stored_mode}, current={vector_mode}"
                )
            if stored_mode == str(VectorMode.SURROGATE) and stored_version != surrogate_version:
                raise ArchexIndexError(
                    "Surrogate version mismatch: "
                    f"cached={stored_version}, current={surrogate_version}"
                )

        self._vectors = vectors
        self._chunk_ids = chunk_ids
        self._chunks_by_id = {c.id: c for c in chunks}

    def rerank(
        self,
        query: str,
        candidates: list[CodeChunk],
        embedder: Embedder,
        *,
        surrogates_by_chunk_id: dict[str, ChunkSurrogate] | None = None,
        vector_mode: VectorMode = VectorMode.RAW,
    ) -> list[tuple[CodeChunk, float]]:
        """Embed only the candidate chunks and query, return sorted by cosine similarity.

        Two-stage retrieval: BM25 generates candidates, this method reranks them
        using dense vector similarity. Embeds len(candidates)+1 texts instead of
        the full corpus — orders of magnitude faster than build()+search().
        """
        if not candidates:
            return []

        texts = [query] + [
            surrogates_by_chunk_id[c.id].surrogate_text
            if (
                vector_mode == VectorMode.SURROGATE
                and surrogates_by_chunk_id
                and c.id in surrogates_by_chunk_id
            )
            else c.content
            for c in candidates
        ]
        encode_np = getattr(embedder, "encode_ndarray", None)
        if encode_np is not None:
            vecs = encode_np(texts)
        else:
            vecs = np.array(embedder.encode(texts), dtype=np.float32)

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        vecs = vecs / norms

        query_vec = vecs[0]
        candidate_vecs = vecs[1:]

        similarities = candidate_vecs @ query_vec
        order = np.argsort(similarities)[::-1]

        results: list[tuple[CodeChunk, float]] = []
        for idx in order:
            score = float(similarities[int(idx)])
            if score > 0:
                results.append((candidates[int(idx)], score))
        return results

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunk_ids)


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
    cv_threshold: float = 0.5,
    agreement_threshold: float = 0.6,
    min_results: int = 3,
) -> tuple[bool, str]:
    """Decide whether to apply fusion based on BM25 confidence signals.

    Returns (should_apply, reason) where reason explains the decision.

    Fusion is SKIPPED when:
    - BM25 score CV > cv_threshold (clear score separation — BM25 is confident)
      AND signal agreement > agreement_threshold (both signals agree, so vector adds no info)
    - Fewer than min_results from either signal

    Fusion is APPLIED when:
    - BM25 CV is low (flat scores — vocabulary ambiguity)
    - OR signal agreement is low (signals disagree — vector found different files)
    """
    if len(bm25_results) < min_results:
        return False, f"too_few_bm25_results:{len(bm25_results)}"
    if not vector_results or len(vector_results) < min_results:
        return False, f"too_few_vector_results:{len(vector_results) if vector_results else 0}"

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
