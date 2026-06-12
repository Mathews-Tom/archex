"""Cross-encoder reranking stage for candidate refinement."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

from archex.exceptions import ArchexIndexError
from archex.index.huggingface import resolve_hf_model_path

if TYPE_CHECKING:
    from archex.models import CodeChunk

logger = logging.getLogger(__name__)

JINA_RERANKER_MODEL = "jinaai/jina-reranker-v3"
JINA_RERANKER_REVISION = "10fb694fc21f7a710a563ff1eb977a460f3868e4"
DEFAULT_MODEL = JINA_RERANKER_MODEL
MODEL_REVISIONS = {
    JINA_RERANKER_MODEL: JINA_RERANKER_REVISION,
}

# Maximum content length passed to the reranker per chunk.
# The Jina listwise reranker on Apple Silicon scales sharply with total prompt
# text; keeping each candidate near 1k chars preserves the representative code
# region while holding warm rerank latency under the benchmark gate.
MAX_CONTENT_CHARS = 1024

# Default number of top candidates to keep after reranking.
# Sized to cover ~8-10 files x 3-4 chunks each, giving downstream scoring
# enough diversity without losing cross-encoder precision.
DEFAULT_TOP_K = 30

# Maximum candidates sent through the expensive model. The caller preserves the
# full candidate pool and treats rerank output as a score boost, so a small
# window is enough to bias the highest-confidence hits without paying the
# latency cost of listwise scoring across the entire frontier.
RERANK_CANDIDATE_LIMIT = 4

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
    """Return True if default cross-encoder dependencies are installed."""
    try:
        import transformers as _transformers  # noqa: F401  # pyright: ignore[reportUnusedImport]

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

    def _uses_transformers_rerank_api(self) -> bool:
        return self._model_name == JINA_RERANKER_MODEL

    def _load_model(self) -> None:
        if self._model is not None:
            return

        cached_model = _MODEL_CACHE.get(self._model_name)
        if cached_model is not None:
            if not self._uses_transformers_rerank_api():
                _ensure_padding_token(cached_model)
            self._model = cached_model
            return

        device = _best_device()
        revision = MODEL_REVISIONS.get(self._model_name)
        model_path = (
            self._model_name
            if self._uses_transformers_rerank_api()
            else resolve_hf_model_path(self._model_name, revision=revision)
        )
        print(
            f"Loading reranker model '{self._model_name}' on {device} "
            "(downloading if not cached)...",
            file=sys.stderr,
            flush=True,
        )
        if self._uses_transformers_rerank_api():
            try:
                from transformers import AutoModel  # type: ignore[import-untyped]
            except ImportError as e:
                raise ArchexIndexError(
                    "CrossEncoderReranker default model requires transformers. "
                    "Install with: uv add 'archex[splade]'"
                ) from e

            auto_model: Any = AutoModel
            model: Any = auto_model.from_pretrained(
                model_path,
                revision=revision,
                dtype="auto",
                trust_remote_code=True,
            )
            if not hasattr(model, "rerank"):
                raise ArchexIndexError(
                    f"Reranker model '{self._model_name}' does not expose rerank()."
                )
            model.eval()
            if device != "cpu":
                model.to(device)
            self._model = model
        else:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ArchexIndexError(
                    "CrossEncoderReranker custom models require sentence-transformers. "
                    "Install with: uv add 'archex[vector-torch]'"
                ) from e

            cross_encoder: Any = CrossEncoder
            self._model = cross_encoder(
                model_path,
                trust_remote_code=True,
                device=device,
            )
            _ensure_padding_token(self._model)
        _MODEL_CACHE[self._model_name] = self._model
        logger.info("Loaded cross-encoder reranker: %s on %s", self._model_name, device)

    def warm(self) -> None:
        """Load the reranker model without scoring candidates."""
        self._load_model()

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

        documents = [chunk.content[:MAX_CONTENT_CHARS] for chunk, _ in candidates]
        if self._uses_transformers_rerank_api():
            results = self._model.rerank(query, documents, top_n=min(top_k, len(documents)))
            ranked: list[tuple[CodeChunk, float]] = []
            for result in results:
                index = int(result["index"])
                ranked.append(
                    (
                        candidates[index][0],
                        float(result["relevance_score"]),
                    )
                )
            return ranked

        pairs = list(zip([query] * len(candidates), documents, strict=False))
        scores: list[float] = self._model.predict(pairs).tolist()

        scored = sorted(
            zip(candidates, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, float(ce_score)) for (chunk, _), ce_score in scored[:top_k]]
