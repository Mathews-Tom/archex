"""Nomic embedding provider: nomic-embed-code via sentence-transformers."""

from __future__ import annotations

import logging
import sys
from typing import Any

from archex.exceptions import ArchexIndexError

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "nomic-ai/nomic-embed-code"


def _best_device() -> str:
    """Pick the best available torch device: mps > cuda > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class NomicCodeEmbedder:
    """Embedding using nomic-embed-code for code-specific semantic search.

    Uses sentence-transformers (requires ``archex[vector-torch]``) which
    handles model download automatically on first use.
    Automatically selects MPS/CUDA/CPU device.
    """

    def __init__(
        self,
        model_name: str = _HF_MODEL_ID,
        batch_size: int = 32,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self._dimension: int | None = None
        self._backend: str | None = None

    def _load_model(self) -> None:
        """Lazy-load the model on first encode call."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            device = _best_device()
            print(
                f"Loading embedding model '{self._model_name}' on {device} "
                "(downloading if not cached)...",
                file=sys.stderr,
                flush=True,
            )
            self._model = SentenceTransformer(
                self._model_name, trust_remote_code=True, device=device
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._backend = "sentence-transformers"
            logger.info(
                "Loaded %s via sentence-transformers on %s (dim=%d)",
                self._model_name,
                device,
                self._dimension,
            )
            return
        except ImportError:
            pass

        raise ArchexIndexError(
            f"NomicCodeEmbedder requires sentence-transformers to load "
            f"'{self._model_name}'. Install with: uv add 'archex[vector-torch]'"
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into embedding vectors."""
        self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()  # type: ignore[no-any-return]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        if self._dimension is None:
            raise ArchexIndexError("Model dimension unavailable after loading")
        return self._dimension
