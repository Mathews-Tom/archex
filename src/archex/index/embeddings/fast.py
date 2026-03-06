"""FastEmbed embedding provider: lightweight ONNX-based embeddings without PyTorch."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from archex.exceptions import ArchexIndexError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-code"


class FastEmbedder:
    """Embedding provider using fastembed (ONNX Runtime backend).

    No PyTorch required. Uses quantized models for fast CPU inference
    with ~10x less memory than sentence-transformers.

    Requires ``archex[vector-fast]``: ``uv add 'archex[vector-fast]'``
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self._dimension: int | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from fastembed import TextEmbedding  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise ArchexIndexError(
                "FastEmbedder requires fastembed. Install with: uv add 'archex[vector-fast]'"
            ) from e

        self._model = TextEmbedding(model_name=self._model_name)
        # Probe dimension from a dummy encode
        probe: list[Any] = list(self._model.embed(["dim_probe"]))
        self._dimension = int(len(probe[0]))
        logger.info(
            "Loaded %s via fastembed/ONNX (dim=%d)",
            self._model_name,
            self._dimension,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        embeddings = list(self._model.embed(texts, batch_size=self._batch_size))
        return [e.tolist() for e in embeddings]

    def encode_ndarray(self, texts: list[str]) -> np.ndarray:
        """Encode texts and return a numpy array directly, skipping list conversion."""
        self._load_model()
        embeddings = list(self._model.embed(texts, batch_size=self._batch_size))
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        if self._dimension is None:
            raise ArchexIndexError("Model dimension unavailable after loading")
        return self._dimension
