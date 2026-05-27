"""Cross-encoder reranking stage for candidate refinement."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from archex.exceptions import ArchexIndexError

if TYPE_CHECKING:
    from archex.models import CodeChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Maximum content length passed to the cross-encoder per chunk.
# MiniLM has a 512-token context window (~4 chars/token avg).
MAX_CONTENT_CHARS = 4096

# Default number of top candidates to keep after reranking.
# Sized to cover ~8-10 files x 3-4 chunks each, giving downstream
# scoring enough diversity without losing cross-encoder precision.
DEFAULT_TOP_K = 30

_MODEL_CACHE: dict[str, Any] = {}


def is_available() -> bool:
    """Return True if cross-encoder dependencies are installed."""
    try:
        import sentence_transformers as _st  # noqa: F401  # pyright: ignore[reportUnusedImport]

        return True
    except ImportError:
        return False


class CrossEncoderReranker:
    """Rerank candidates using a cross-encoder model.

    Cross-encoders compute full query-chunk attention, capturing
    token-level interactions that bi-encoder similarity misses.
    Applied as a post-fusion refinement stage over the top-N
    candidates to improve precision without affecting recall.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        cached_model = _MODEL_CACHE.get(self._model_name)
        if cached_model is not None:
            self._model = cached_model
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ArchexIndexError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: uv add 'archex[vector-torch]'"
            ) from e

        print(
            f"Loading reranker model '{self._model_name}' (downloading if not cached)...",
            file=sys.stderr,
            flush=True,
        )
        self._model = CrossEncoder(self._model_name, trust_remote_code=True)
        _MODEL_CACHE[self._model_name] = self._model
        logger.info("Loaded cross-encoder reranker: %s", self._model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[CodeChunk, float]],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[tuple[CodeChunk, float]]:
        """Rerank candidates by cross-encoder relevance score.

        Args:
            query: The search query.
            candidates: (chunk, score) pairs from prior retrieval stages.
            top_k: Maximum number of results to return.

        Returns:
            Re-scored (chunk, cross_encoder_score) pairs sorted by relevance.
        """
        if not candidates:
            return []

        self._load_model()

        pairs = [(query, chunk.content[:MAX_CONTENT_CHARS]) for chunk, _ in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()

        scored = sorted(
            zip(candidates, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, float(ce_score)) for (chunk, _), ce_score in scored[:top_k]]
