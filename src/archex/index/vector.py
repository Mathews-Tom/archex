"""Vector index: build and query a dense embedding index over CodeChunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from archex.exceptions import ArchexIndexError
from archex.index.quantize import (
    DEFAULT_BITS,
    pack_codes,
    quantize_vectors,
    quantized_dot_product,
    unpack_codes,
)
from archex.models import ChunkSurrogate, CodeChunk, VectorMode

if TYPE_CHECKING:
    from pathlib import Path

    from archex.index.embeddings.base import Embedder


def _chunk_embedding_text(chunk: CodeChunk) -> str:
    """Build the text to embed for a chunk, prepending breadcrumbs if present."""
    if chunk.breadcrumbs:
        return chunk.breadcrumbs + "\n" + chunk.content
    return chunk.content


def _encode_query(embedder: Embedder, query: str) -> list[float]:
    encode_queries = getattr(embedder, "encode_queries", None)
    if encode_queries is not None:
        return encode_queries([query])[0]
    return embedder.encode([query])[0]


class VectorIndex:
    """Dense vector index for semantic search over CodeChunks.

    Vectors are L2-normalized at build time so search uses dot product
    (equivalent to cosine similarity on normalized vectors).

    Supports optional TurboQuant quantization for 8x+ storage reduction
    with minimal recall degradation. When quantized, vectors are stored
    as packed bit codes with per-vector scale parameters.
    """

    def __init__(self, *, quantize: bool = False, quantize_bits: int = DEFAULT_BITS) -> None:
        self._vectors: np.ndarray[tuple[int, int], np.dtype[np.float32]] | None = None
        self._chunk_ids: list[str] = []
        self._chunks_by_id: dict[str, CodeChunk] = {}
        # Quantization state
        self._quantize = quantize
        self._quantize_bits = quantize_bits
        self._quantized_codes: np.ndarray | None = None
        self._quantized_norms: np.ndarray | None = None
        self._quantized_scale: np.ndarray | None = None
        self._quantized_dim: int = 0

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
            self._quantized_codes = None
            self._quantized_norms = None
            self._quantized_scale = None
            self._quantized_dim = 0
            return

        texts = [
            surrogates_by_chunk_id[c.id].surrogate_text
            if (
                vector_mode == VectorMode.SURROGATE
                and surrogates_by_chunk_id
                and c.id in surrogates_by_chunk_id
            )
            else _chunk_embedding_text(c)
            for c in chunks
        ]
        raw_embeddings = embedder.encode(texts)
        vectors = np.array(raw_embeddings, dtype=np.float32)

        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        vectors = vectors / norms

        self._chunk_ids = [c.id for c in chunks]
        self._chunks_by_id = {c.id: c for c in chunks}

        if self._quantize:
            codes, q_norms, q_scale = quantize_vectors(vectors, bits=self._quantize_bits)
            self._quantized_codes = codes
            self._quantized_norms = q_norms
            self._quantized_scale = q_scale
            self._quantized_dim = vectors.shape[1]
            self._vectors = None
        else:
            self._vectors = vectors
            self._quantized_codes = None
            self._quantized_norms = None
            self._quantized_scale = None
            self._quantized_dim = 0

    @property
    def is_quantized(self) -> bool:
        """Whether the index is using quantized storage."""
        return self._quantized_codes is not None

    def search(
        self, query: str, embedder: Embedder, top_k: int = 30
    ) -> list[tuple[CodeChunk, float]]:
        """Search for chunks most similar to the query string."""
        if len(self._chunk_ids) == 0:
            return []
        if self._vectors is None and self._quantized_codes is None:
            return []

        query_vec = np.array(_encode_query(embedder, query), dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm < 1e-9:
            return []
        query_vec = query_vec / norm

        # Compute similarities via quantized or unquantized path
        if self._quantized_codes is not None:
            similarities = quantized_dot_product(
                query_vec,
                self._quantized_codes,
                self._quantized_norms,  # type: ignore[arg-type]
                self._quantized_scale,  # type: ignore[arg-type]
                bits=self._quantize_bits,
            )
        else:
            assert self._vectors is not None
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
        if self._quantized_dim > 0:
            return self._quantized_dim
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
        if self._vectors is None and self._quantized_codes is None:
            raise ArchexIndexError("Cannot save empty vector index")

        path.parent.mkdir(parents=True, exist_ok=True)

        common = {
            "chunk_ids": np.array(self._chunk_ids, dtype="U512"),
            "embedder_meta": np.array([embedder_name, str(vector_dim)], dtype="U256"),
            "vector_meta": np.array([str(vector_mode), surrogate_version], dtype="U256"),
        }

        if self._quantized_codes is not None:
            packed = pack_codes(self._quantized_codes, self._quantize_bits)
            np.savez_compressed(
                str(path),
                **common,  # pyright: ignore[reportArgumentType]
                quantized_packed=packed,
                quantized_norms=self._quantized_norms,  # pyright: ignore[reportArgumentType]
                quantized_scale=self._quantized_scale,  # pyright: ignore[reportArgumentType]
                quantize_meta=np.array(
                    [str(self._quantize_bits), str(self._quantized_dim)], dtype="U64"
                ),
            )
        else:
            np.savez_compressed(
                str(path),
                **common,  # pyright: ignore[reportArgumentType]
                vectors=self._vectors,  # pyright: ignore[reportArgumentType]
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
        """Load vectors from disk and rebuild the chunk lookup map.

        Supports both quantized and unquantized .npz formats. Old files
        without quantization metadata are loaded as unquantized float32.
        """
        if not path.exists():
            suffix = ".npz"
            p = path if path.suffix == suffix else path.with_suffix(suffix)
            if not p.exists():
                raise ArchexIndexError(f"Vector index file not found: {path}")
            path = p

        data = np.load(str(path), allow_pickle=False)
        chunk_ids = list(data["chunk_ids"])

        # Validate embedder/vector metadata (shared by both formats)
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

        # Load quantized or unquantized format
        if "quantize_meta" in data:
            meta = list(data["quantize_meta"])
            bits = int(meta[0])
            q_dim = int(meta[1])
            packed = data["quantized_packed"]
            codes = unpack_codes(packed, q_dim, bits)
            if codes.shape[0] != len(chunk_ids):
                raise ArchexIndexError(
                    f"Vector index corrupt: {codes.shape[0]} vectors but {len(chunk_ids)} chunk IDs"
                )
            self._quantized_codes = codes
            self._quantized_norms = data["quantized_norms"].astype(np.float32, copy=False)
            self._quantized_scale = data["quantized_scale"].astype(np.float32, copy=False)
            self._quantize_bits = bits
            self._quantized_dim = q_dim
            self._quantize = True
            self._vectors = None
        else:
            vectors = data["vectors"].astype(np.float32, copy=False)
            if vectors.shape[0] != len(chunk_ids):
                raise ArchexIndexError(
                    f"Vector index corrupt: {vectors.shape[0]} vectors "
                    f"but {len(chunk_ids)} chunk IDs"
                )
            self._vectors = vectors
            self._quantized_codes = None
            self._quantized_norms = None
            self._quantized_scale = None
            self._quantized_dim = 0

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

        candidate_texts = [
            surrogates_by_chunk_id[c.id].surrogate_text
            if (
                vector_mode == VectorMode.SURROGATE
                and surrogates_by_chunk_id
                and c.id in surrogates_by_chunk_id
            )
            else _chunk_embedding_text(c)
            for c in candidates
        ]
        query_vec_raw = np.array(_encode_query(embedder, query), dtype=np.float32)
        encode_np = getattr(embedder, "encode_ndarray", None)
        if encode_np is not None:
            candidate_vecs_raw = encode_np(candidate_texts)
        else:
            candidate_vecs_raw = np.array(embedder.encode(candidate_texts), dtype=np.float32)
        vecs = np.vstack([query_vec_raw, candidate_vecs_raw])

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
