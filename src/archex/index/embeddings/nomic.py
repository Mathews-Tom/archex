"""Nomic embedding provider: local nomic-embed-code model via ONNX runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from archex.exceptions import ArchexIndexError

_DEFAULT_MODEL_DIR = Path.home() / ".archex" / "models"
_MODEL_NAME = "nomic-embed-code-v1"


class NomicCodeEmbedder:
    """Local embedding using nomic-embed-code ONNX model.

    Requires the `vector` extra: ``uv add 'archex[vector]'``
    """

    def __init__(
        self,
        model_dir: Path = _DEFAULT_MODEL_DIR,
        batch_size: int = 32,
        cache_dir: str | None = None,
    ) -> None:
        try:
            import onnxruntime as _ort  # pyright: ignore[reportMissingTypeStubs,reportUnusedImport]  # noqa: F401
            import tokenizers as _tok  # pyright: ignore[reportUnusedImport]  # noqa: F401
        except ImportError as e:
            raise ArchexIndexError(
                "NomicCodeEmbedder requires onnxruntime and tokenizers. "
                "Install with: uv add 'archex[vector]'"
            ) from e

        effective_dir = Path(cache_dir).expanduser() if cache_dir is not None else model_dir
        self._model_dir = effective_dir / _MODEL_NAME
        self._batch_size = batch_size
        self._session: Any = None
        self._tokenizer: Any = None
        self._dimension = 768

    def _load_model(self) -> None:
        """Lazy-load ONNX session and tokenizer on first encode call."""
        if self._session is not None:
            return

        import onnxruntime as ort  # pyright: ignore[reportMissingTypeStubs]

        model_path = self._model_dir / "model.onnx"
        tokenizer_path = self._model_dir / "tokenizer.json"

        if not model_path.exists():
            raise ArchexIndexError(
                f"ONNX model not found at {model_path}. "
                f"Download nomic-embed-code to {self._model_dir}/"
            )
        if not tokenizer_path.exists():
            raise ArchexIndexError(
                f"Tokenizer not found at {tokenizer_path}. "
                f"Download nomic-embed-code tokenizer to {self._model_dir}/"
            )

        self._session = ort.InferenceSession(str(model_path))  # pyright: ignore[reportUnknownMemberType]
        tokenizer_mod: Any = __import__("tokenizers")
        self._tokenizer = tokenizer_mod.Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(length=512)
        self._tokenizer.enable_truncation(max_length=512)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into embedding vectors using the ONNX model."""
        import numpy as np

        self._load_model()

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            encoded: Any = self._tokenizer.encode_batch(batch)
            input_ids: Any = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask: Any = np.array([e.attention_mask for e in encoded], dtype=np.int64)

            outputs: Any = self._session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            # Mean pooling over token dimension
            token_embeddings: Any = outputs[0]
            mask_expanded: Any = attention_mask[:, :, np.newaxis].astype(np.float32)
            summed: Any = (token_embeddings * mask_expanded).sum(axis=1)
            counts: Any = mask_expanded.sum(axis=1).clip(min=1e-9)
            embeddings: Any = summed / counts

            # L2 normalize
            norms: Any = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
            normalized: Any = embeddings / norms
            result: list[list[float]] = normalized.tolist()
            all_embeddings.extend(result)

        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimension
