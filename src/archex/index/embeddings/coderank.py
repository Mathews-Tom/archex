"""CodeRankEmbed embedding provider: code-optimized embeddings via sentence-transformers."""

from __future__ import annotations

import logging
import sys
from typing import Any

from archex.exceptions import ArchexIndexError

logger = logging.getLogger(__name__)

HF_MODEL_ID = "nomic-ai/CodeRankEmbed"
HF_MODEL_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
QUERY_PREFIX = "Represent this query for searching relevant code: "


def _best_device() -> str:
    """Pick the best available torch device: mps > cuda > cpu."""
    try:
        import torch  # type: ignore[import-untyped]

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class CodeRankEmbedder:
    """Embedding provider using nomic-ai/CodeRankEmbed for code search.

    CodeRankEmbed is trained on CoRNStack (21M curated code pairs) and
    achieves MRR 77.9 on CodeSearchNet — a +10.7 point improvement over
    Jina v2 base code. Same dimension (768), similar parameter count (137M).

    Important: queries require the prefix "Represent this query for
    searching relevant code: " but documents do not.
    """

    def __init__(
        self,
        model_name: str = HF_MODEL_ID,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self._dimension: int | None = None

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model on first encode call."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ArchexIndexError(
                "CodeRankEmbedder requires sentence-transformers. "
                "Install with: uv add 'archex[vector-torch]'"
            ) from e

        device = _best_device()
        print(
            f"Loading embedding model '{self._model_name}' on {device} "
            "(downloading if not cached)...",
            file=sys.stderr,
            flush=True,
        )
        self._model = SentenceTransformer(
            self._model_name,
            revision=HF_MODEL_REVISION if self._model_name == HF_MODEL_ID else None,
            trust_remote_code=True,
            device=device,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Loaded %s via sentence-transformers on %s (dim=%d)",
            self._model_name,
            device,
            self._dimension,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into embedding vectors.

        For document encoding (indexing), texts are passed as-is.
        For query encoding, the caller should prepend QUERY_PREFIX.
        """
        self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()  # type: ignore[no-any-return]

    def encode_queries(self, queries: list[str]) -> list[list[float]]:
        """Encode queries with the required prefix for asymmetric search."""
        prefixed = [f"{QUERY_PREFIX}{q}" for q in queries]
        return self.encode(prefixed)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        if self._dimension is None:
            raise ArchexIndexError("Model dimension unavailable after loading")
        return self._dimension
