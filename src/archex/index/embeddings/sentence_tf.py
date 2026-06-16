"""SentenceTransformers embedding provider: encode via sentence-transformers library."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from archex.exceptions import ArchexIndexError
from archex.index.model_policy import custom_remote_code_profile, remote_code_trust_value


class SentenceTransformerEmbedder:
    """Embedding provider using the sentence-transformers library.

    Requires the `vector-torch` extra: ``uv add archex[vector-torch]``
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        trust_remote_code: bool = False,
        allow_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        model_kwargs: Mapping[str, object] = MappingProxyType({}),
        config_kwargs: Mapping[str, object] = MappingProxyType({}),
        max_seq_length: int | None = None,
    ) -> None:
        try:
            import sentence_transformers as _st  # pyright: ignore[reportUnusedImport]  # noqa: F401
        except ImportError as e:
            raise ArchexIndexError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "Install with: uv add 'archex[vector-torch]'"
            ) from e

        if trust_remote_code:
            code_revision = config_kwargs.get("code_revision")
            trust_remote_code = remote_code_trust_value(
                custom_remote_code_profile(
                    component="embedder",
                    provider="sentence-transformers",
                    model_name=model_name,
                    model_revision=revision,
                    code_revision=str(code_revision) if code_revision is not None else None,
                ),
                allow_remote_code=allow_remote_code,
            )
        self._model_name = model_name
        self._batch_size = batch_size
        self._trust_remote_code = trust_remote_code
        self._revision = revision
        self._local_files_only = local_files_only
        self._model_kwargs = dict(model_kwargs)
        self._config_kwargs = dict(config_kwargs)
        self._max_seq_length = max_seq_length
        self._model: Any = None
        self._dimension: int | None = None

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model on first encode call."""
        if self._model is not None:
            return

        from sentence_transformers import (
            SentenceTransformer,  # pyright: ignore[reportMissingImports]
        )

        from archex.index.embeddings.nomic import (
            _best_device,  # pyright: ignore[reportPrivateUsage]
        )

        self._model = SentenceTransformer(
            self._model_name,
            device=_best_device(),
            trust_remote_code=self._trust_remote_code,
            revision=self._revision,
            local_files_only=self._local_files_only,
            model_kwargs=self._model_kwargs,
            config_kwargs=self._config_kwargs,
        )
        if self._max_seq_length is not None:
            self._model.max_seq_length = self._max_seq_length
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
