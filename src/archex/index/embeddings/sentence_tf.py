"""SentenceTransformers embedding provider: encode via sentence-transformers library."""

from __future__ import annotations

from typing import Any

from archex.exceptions import ArchexIndexError


class SentenceTransformerEmbedder:
    """Embedding provider using the sentence-transformers library.

    Requires the `vector-torch` extra: ``uv add archex[vector-torch]``
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> None:
        try:
            import sentence_transformers as _st  # pyright: ignore[reportUnusedImport]  # noqa: F401
        except ImportError as e:
            raise ArchexIndexError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "Install with: uv add 'archex[vector-torch]'"
            ) from e

        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None
        self._dimension: int | None = None

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model on first encode call."""
        if self._model is not None:
            return

        from sentence_transformers import (
            SentenceTransformer,  # pyright: ignore[reportMissingImports]
        )

        self._model = SentenceTransformer(self._model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into embedding vectors."""
        self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()  # type: ignore[no-any-return]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        if self._dimension is None:
            raise ArchexIndexError("Model dimension unavailable after loading")
        return self._dimension
