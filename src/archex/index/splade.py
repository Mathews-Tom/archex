"""SPLADE learned sparse retrieval: build and query a sparse expansion index over CodeChunks.

SPLADE uses a masked language model head to expand each chunk into a sparse vector
over the full vocabulary. At query time the same expansion is applied to the query,
and the score is the dot product of the two sparse vectors.

This bridges the vocabulary gap that limits BM25 on natural-language → code queries:
"session management" activates tokens like ``Session``, ``scoped_session``, ``commit``
because the MLM head learned those co-occurrence patterns during pre-training.
"""
# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from archex.exceptions import ArchexIndexError
from archex.index.huggingface import resolve_hf_model_path

if TYPE_CHECKING:
    from pathlib import Path

    from archex.index.store import IndexStore
    from archex.models import CodeChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
_BATCH_SIZE = 32
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_SPLADE_VECTORS = """
CREATE TABLE IF NOT EXISTS splade_vectors (
    chunk_id TEXT PRIMARY KEY,
    sparse_json TEXT NOT NULL
);
"""

_CREATE_SPLADE_INVERTED = """
CREATE TABLE IF NOT EXISTS splade_inverted (
    term_id INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    weight REAL NOT NULL
);
"""

_CREATE_SPLADE_INVERTED_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_splade_inv_term ON splade_inverted(term_id);"
)

_CREATE_SPLADE_META = """
CREATE TABLE IF NOT EXISTS splade_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Encoder protocol + implementation
# ---------------------------------------------------------------------------


@runtime_checkable
class SparseEncoder(Protocol):
    """Protocol for sparse vector encoders — enables testing without the real model."""

    def encode(self, texts: list[str]) -> list[dict[int, float]]: ...


class SPLADEEncoder:
    """Encode text into SPLADE sparse vectors using a pre-trained MLM head.

    Lazy-loads the model on first call to ``encode`` so import is cheap.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        cached = _MODEL_CACHE.get(self._model_name)
        if cached is not None:
            self._tokenizer, self._model = cached
            return

        model_path = resolve_hf_model_path(self._model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        model.eval()
        if torch.backends.mps.is_available():
            model.to("mps")
        self._tokenizer = tokenizer
        self._model = model
        _MODEL_CACHE[self._model_name] = (tokenizer, model)

    def warm(self) -> None:
        """Load tokenizer and model without encoding text."""
        if self._model is None or self._tokenizer is None:
            self._load()

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            self._load()
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise ArchexIndexError("SPLADE tokenizer unavailable after loading")
        return int(tokenizer.vocab_size)

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        """Encode texts into sparse vectors (dict mapping token_id → weight).

        Uses the SPLADE max-pooling formulation:
            w_j = max_i(log(1 + ReLU(logits_ij))) for each vocabulary token j
        where i iterates over input token positions.
        """
        import torch

        if self._model is None or self._tokenizer is None:
            self._load()

        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            raise ArchexIndexError("SPLADE model unavailable after loading")

        device = next(model.parameters()).device
        results: list[dict[int, float]] = []

        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch_texts = texts[batch_start : batch_start + _BATCH_SIZE]
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}

            with torch.no_grad():
                output = model(**tokens)

            # SPLADE aggregation: log(1 + ReLU(logits)), max-pool over positions
            logits = output.logits  # (batch, seq_len, vocab_size)
            activated = torch.log1p(torch.relu(logits))

            # Mask padding positions before max-pooling
            attention_mask = tokens["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
            activated = activated * attention_mask

            sparse_vecs = activated.max(dim=1).values  # (batch, vocab_size)

            for vec in sparse_vecs:
                vec_cpu = vec.cpu().numpy()
                nonzero_idx = np.nonzero(vec_cpu)[0]
                sparse_dict = {int(idx): float(vec_cpu[idx]) for idx in nonzero_idx}
                results.append(sparse_dict)

        return results

    def decode_token_ids(self, token_ids: list[int]) -> list[str]:
        """Convert token IDs back to string tokens (for debugging/inspection)."""
        if self._tokenizer is None:
            self._load()
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise ArchexIndexError("SPLADE tokenizer unavailable after loading")
        return [tokenizer.decode([tid]).strip() for tid in token_ids]


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class SPLADEIndex:
    """SPLADE learned sparse retrieval index.

    Interface mirrors BM25Index: ``build(chunks)`` then ``search(query, top_k)``.

    Storage uses two SQLite tables:
    - ``splade_vectors``: chunk_id → JSON sparse vector (for persistence/inspection)
    - ``splade_inverted``: inverted index (term_id, chunk_id, weight) for fast query

    The inverted index layout enables efficient query-time scoring:
    for each non-zero query term, fetch all matching chunks and accumulate
    the dot product incrementally.
    """

    def __init__(
        self,
        store: IndexStore,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        encoder: SparseEncoder | None = None,
    ) -> None:
        self._store = store
        self._model_name = model_name
        self._encoder: SparseEncoder = encoder or SPLADEEncoder(model_name)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = self._store.conn
        conn.executescript(
            _CREATE_SPLADE_VECTORS
            + _CREATE_SPLADE_INVERTED
            + _CREATE_SPLADE_INVERTED_IDX
            + _CREATE_SPLADE_META
        )
        conn.commit()

    @property
    def has_data(self) -> bool:
        """Check if SPLADE index already has data."""
        row = self._store.conn.execute("SELECT COUNT(*) FROM splade_vectors").fetchone()
        return bool(row and row[0] > 0)

    @property
    def model_name(self) -> str:
        return self._model_name

    def build(self, chunks: list[CodeChunk]) -> None:
        """Encode all chunks into sparse vectors and build the inverted index."""
        from archex.index.chunker import expand_identifiers

        conn = self._store.conn

        # Clear existing data
        conn.execute("DELETE FROM splade_vectors")
        conn.execute("DELETE FROM splade_inverted")

        if not chunks:
            conn.commit()
            return

        # Prepare texts with identifier expansion (same as BM25)
        texts = [expand_identifiers(c.content) for c in chunks]

        # Encode in batches
        sparse_vecs = self._encoder.encode(texts)

        # Insert sparse vectors and inverted index entries
        vector_rows: list[tuple[str, str]] = []
        inverted_rows: list[tuple[int, str, float]] = []

        for chunk, sparse_dict in zip(chunks, sparse_vecs, strict=False):
            vector_rows.append((chunk.id, json.dumps(sparse_dict)))
            for term_id, weight in sparse_dict.items():
                inverted_rows.append((term_id, chunk.id, weight))

        conn.executemany(
            "INSERT INTO splade_vectors (chunk_id, sparse_json) VALUES (?, ?)",
            vector_rows,
        )
        conn.executemany(
            "INSERT INTO splade_inverted (term_id, chunk_id, weight) VALUES (?, ?, ?)",
            inverted_rows,
        )

        # Store metadata
        conn.execute(
            "INSERT OR REPLACE INTO splade_meta (key, value) VALUES (?, ?)",
            ("model_name", self._model_name),
        )
        conn.execute(
            "INSERT OR REPLACE INTO splade_meta (key, value) VALUES (?, ?)",
            ("chunk_count", str(len(chunks))),
        )
        conn.commit()

        logger.info(
            "SPLADE index built: %d chunks, %d inverted entries",
            len(chunks),
            len(inverted_rows),
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[CodeChunk, float]]:
        """Search the SPLADE index by encoding the query and computing dot products.

        Scoring: for each non-zero query term, look up chunks with non-zero
        document weight for that term. The chunk score is the sum of
        query_weight * doc_weight across all matching terms (dot product).
        """
        if not query.strip():
            return []

        conn = self._store.conn

        # Encode query into sparse vector
        query_sparse = self._encoder.encode([query])[0]
        if not query_sparse:
            return []

        # Accumulate dot products via inverted index
        scores: dict[str, float] = {}
        for term_id, q_weight in query_sparse.items():
            cur = conn.execute(
                "SELECT chunk_id, weight FROM splade_inverted WHERE term_id = ?",
                (term_id,),
            )
            for chunk_id, d_weight in cur.fetchall():
                scores[chunk_id] = scores.get(chunk_id, 0.0) + q_weight * float(d_weight)

        if not scores:
            return []

        # Sort by score descending and fetch top_k chunks
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        chunk_ids = [cid for cid, _ in ranked]
        score_map = dict(ranked)

        fetched = self._store.get_chunks_by_ids(chunk_ids)
        results: list[tuple[CodeChunk, float]] = []
        for chunk in fetched:
            score = score_map.get(chunk.id, 0.0)
            results.append((chunk, score))

        results.sort(key=lambda r: r[1], reverse=True)
        return results

    def get_sparse_vector(self, chunk_id: str) -> dict[int, float]:
        """Retrieve the stored sparse vector for a chunk (for debugging/inspection)."""
        row = self._store.conn.execute(
            "SELECT sparse_json FROM splade_vectors WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise ArchexIndexError(f"No SPLADE vector for chunk: {chunk_id}")
        raw: dict[str, float] = json.loads(row[0])
        return {int(k): v for k, v in raw.items()}

    def save(self, path: Path) -> None:
        """Export SPLADE index to a compressed numpy file for portability.

        The SQLite tables are the primary store; this enables offline transfer.
        """
        conn = self._store.conn
        rows = conn.execute("SELECT chunk_id, sparse_json FROM splade_vectors").fetchall()
        if not rows:
            raise ArchexIndexError("Cannot save empty SPLADE index")

        chunk_ids = [r[0] for r in rows]
        sparse_jsons = [r[1] for r in rows]

        np.savez_compressed(
            str(path),
            chunk_ids=np.array(chunk_ids, dtype="U512"),
            sparse_jsons=np.array(sparse_jsons, dtype="U"),
            model_name=np.array([self._model_name], dtype="U256"),
        )

    def load(self, path: Path, chunks: list[CodeChunk]) -> None:
        """Load SPLADE index from a numpy file and rebuild SQLite tables."""
        if not path.exists():
            raise ArchexIndexError(f"SPLADE index file not found: {path}")

        data = np.load(str(path), allow_pickle=False)
        chunk_ids = list(data["chunk_ids"])
        sparse_jsons = list(data["sparse_jsons"])
        stored_model = str(data["model_name"][0])

        if stored_model != self._model_name:
            raise ArchexIndexError(
                f"SPLADE model mismatch: cached={stored_model}, current={self._model_name}"
            )

        conn = self._store.conn
        conn.execute("DELETE FROM splade_vectors")
        conn.execute("DELETE FROM splade_inverted")

        vector_rows: list[tuple[str, str]] = list(zip(chunk_ids, sparse_jsons, strict=False))
        inverted_rows: list[tuple[int, str, float]] = []

        for chunk_id, sparse_json in zip(chunk_ids, sparse_jsons, strict=False):
            sparse_dict: dict[str, float] = json.loads(sparse_json)
            for term_id_str, weight in sparse_dict.items():
                inverted_rows.append((int(term_id_str), chunk_id, weight))

        conn.executemany(
            "INSERT INTO splade_vectors (chunk_id, sparse_json) VALUES (?, ?)",
            vector_rows,
        )
        conn.executemany(
            "INSERT INTO splade_inverted (term_id, chunk_id, weight) VALUES (?, ?, ?)",
            inverted_rows,
        )
        conn.commit()

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        row = self._store.conn.execute("SELECT COUNT(*) FROM splade_vectors").fetchone()
        return int(row[0]) if row else 0
