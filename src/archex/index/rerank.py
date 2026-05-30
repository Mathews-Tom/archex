"""Cross-encoder reranking stage for candidate refinement."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from archex.exceptions import ArchexIndexError

if TYPE_CHECKING:
    from archex.models import CodeChunk

logger = logging.getLogger(__name__)

JINA_RERANKER_MODEL = "jinaai/jina-reranker-v3"
JINA_RERANKER_REVISION = "10fb694fc21f7a710a563ff1eb977a460f3868e4"
DEFAULT_MODEL = JINA_RERANKER_MODEL
MODEL_REVISIONS = {
    JINA_RERANKER_MODEL: JINA_RERANKER_REVISION,
}

# Maximum content length passed to the cross-encoder per chunk.
# jinaai/jina-reranker-v3 supports an 8192-token window. Use a conservative
# ~4096-token chunk slice so larger code chunks score while query overhead fits.
MAX_CONTENT_CHARS = 16384

# Default number of top candidates to keep after reranking.
# Sized to cover ~8-10 files x 3-4 chunks each, giving downstream
# scoring enough diversity without losing cross-encoder precision.
DEFAULT_TOP_K = 30

_MODEL_CACHE: dict[str, Any] = {}


def _ensure_padding_token(cross_encoder: Any) -> None:
    tokenizer = getattr(cross_encoder, "tokenizer", None)
    if tokenizer is None:
        raise ArchexIndexError(f"Cross-encoder model '{cross_encoder}' has no tokenizer.")

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is None or eos_token_id is None:
            raise ArchexIndexError(
                f"Cross-encoder model '{cross_encoder}' has no padding token or EOS token."
            )

        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id
        pad_token_id = eos_token_id

    if pad_token_id is None:
        raise ArchexIndexError(f"Cross-encoder model '{cross_encoder}' has no padding token id.")

    model = getattr(cross_encoder, "model", None)
    config = getattr(model, "config", None)
    if config is None:
        raise ArchexIndexError(f"Cross-encoder model '{cross_encoder}' has no config.")
    config.pad_token_id = pad_token_id


def _best_device() -> str:
    """Pick the best available torch device for cross-encoder reranking."""
    try:
        import torch  # type: ignore[import-untyped]

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


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
            _ensure_padding_token(cached_model)
            self._model = cached_model
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ArchexIndexError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: uv add 'archex[vector-torch]'"
            ) from e

        device = _best_device()
        print(
            f"Loading reranker model '{self._model_name}' on {device} "
            "(downloading if not cached)...",
            file=sys.stderr,
            flush=True,
        )
        self._model = CrossEncoder(
            self._model_name,
            revision=MODEL_REVISIONS.get(self._model_name),
            trust_remote_code=True,
            device=device,
        )
        _ensure_padding_token(self._model)
        _MODEL_CACHE[self._model_name] = self._model
        logger.info("Loaded cross-encoder reranker: %s on %s", self._model_name, device)

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
