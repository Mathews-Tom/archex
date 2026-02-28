"""Vector index: build and query a dense embedding index over CodeChunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from archex.exceptions import ArchexIndexError

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

    def build(self, chunks: list[CodeChunk], embedder: Embedder) -> None:
        """Encode all chunks and store normalized vectors."""
        if not chunks:
            self._vectors = None
            self._chunk_ids = []
            self._chunks_by_id = {}
            return

        texts = [c.content for c in chunks]
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
        top_indices = np.argsort(similarities)[::-1][:k]

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

    def save(self, path: Path) -> None:
        """Save vectors and chunk IDs to a compressed numpy file."""
        if self._vectors is None:
            raise ArchexIndexError("Cannot save empty vector index")

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            vectors=self._vectors,
            chunk_ids=np.array(self._chunk_ids, dtype="U512"),
        )

    def load(self, path: Path, chunks: list[CodeChunk]) -> None:
        """Load vectors from disk and rebuild the chunk lookup map."""
        if not path.exists():
            suffix = ".npz"
            p = path if path.suffix == suffix else path.with_suffix(suffix)
            if not p.exists():
                raise ArchexIndexError(f"Vector index file not found: {path}")
            path = p

        data = np.load(str(path), allow_pickle=False)
        vectors = data["vectors"].astype(np.float32)
        chunk_ids = list(data["chunk_ids"])
        if vectors.shape[0] != len(chunk_ids):
            raise ArchexIndexError(
                f"Vector index corrupt: {vectors.shape[0]} vectors but {len(chunk_ids)} chunk IDs"
            )
        self._vectors = vectors
        self._chunk_ids = chunk_ids
        self._chunks_by_id = {c.id: c for c in chunks}

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
