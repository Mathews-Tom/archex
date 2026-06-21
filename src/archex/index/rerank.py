"""Cross-encoder reranking stage for candidate refinement."""

from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from archex.exceptions import ArchexIndexError, ConfigError
from archex.index.fusion import bm25_score_cv
from archex.index.huggingface import resolve_hf_model_path
from archex.index.model_policy import (
    JINA_RERANKER_MODEL,
    JINA_RERANKER_REVISION,
    remote_code_trust_value,
    reranker_security_profile,
)

if TYPE_CHECKING:
    from archex.models import CodeChunk

logger = logging.getLogger(__name__)

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


def loaded_reranker_model_names() -> list[str]:
    """Return names of cross-encoder models already loaded in this process.

    Lets callers reuse an in-process reranker without triggering a model load or
    download: an empty list means no local reranker is available right now.
    """
    return list(_MODEL_CACHE)


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

    def __init__(self, model_name: str = DEFAULT_MODEL, *, allow_remote_code: bool = False) -> None:
        self._model_name = model_name
        self._allow_remote_code = allow_remote_code
        self._model: Any = None

    def _uses_transformers_rerank_api(self) -> bool:
        return self._model_name == JINA_RERANKER_MODEL

    def _load_model(self) -> None:
        if self._model is not None:
            return

        profile = reranker_security_profile(self._model_name)
        trust_remote_code = remote_code_trust_value(
            profile,
            allow_remote_code=self._allow_remote_code,
        )

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
                trust_remote_code=trust_remote_code,
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
                trust_remote_code=False,
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


# --------------------------------------------------------------------------- #
# Low-latency conditional reranker (opt-in, benchmark/experimental lane)
# --------------------------------------------------------------------------- #
#
# Full cross-encoder rerank on every query missed the p95 budget (Jina v3 16.5 s,
# MiniLM 3.9 s; see docs/RETRIEVAL_DEFAULT_DECISIONS.md). The conditional reranker
# spends the cross-encoder only where it can pay off: when the BM25 ranking is
# *ambiguous*. It reuses the same query-performance-prediction signals the fusion
# gate uses (BM25 score CV and AvgIDF, ``index/fusion.py``), so the heavy path
# runs rarely and the common confident-BM25 query stays model-free and fast. A
# hard per-stage latency cap aborts a slow model pass and falls back to the
# original order, so the rerank stage is bounded regardless of model or
# candidate-set surprises. It is opt-in only and never on the default path; the
# remote-code policy is enforced by the wrapped ``CrossEncoderReranker``.
#
# It is designed for a small distilled cross-encoder (Ettin-17M/68M class) loaded
# through the local sentence-transformers path (``trust_remote_code=False``,
# optionally an ONNX/INT8 backend), but it is model-agnostic: the operator
# supplies the model via ``rerank_model``.

# Hard wall-clock ceiling (ms) for one conditional rerank stage. The model pass
# runs in a worker thread and the caller is released at this cap, so the rerank
# stage the pipeline observes is bounded even when the model misbehaves; the
# orphaned pass finishes in the background and its result is discarded.
CONDITIONAL_RERANK_LATENCY_CAP_MS = 1500.0

# BM25 is treated as ambiguous — and the cross-encoder fires — when score
# separation is weak (CV at or below this) or query terms are too common for BM25
# to discriminate (AvgIDF below this). The thresholds mirror the fusion QPP gate
# (``index/fusion.py``: ``should_fuse`` uses cv_threshold=0.8, idf_threshold=2.0)
# so the conditional rerank fires on the same low-confidence-BM25 queries fusion
# already treats as needing help.
AMBIGUOUS_BM25_CV_THRESHOLD = 0.8
AMBIGUOUS_BM25_IDF_THRESHOLD = 2.0

# Default head size sent through the model when the lane fires. The remaining
# candidates keep their retrieval order, bounding model work structurally.
CONDITIONAL_RERANK_CANDIDATE_LIMIT = 30


@dataclass(frozen=True)
class ConditionalRerankDecision:
    """Whether BM25 is ambiguous enough to spend the cross-encoder, and why."""

    should_rerank: bool
    reason: str
    bm25_cv: float
    avg_idf: float | None


def bm25_is_ambiguous(
    bm25_results: list[tuple[CodeChunk, float]],
    *,
    avg_idf: float | None = None,
    cv_threshold: float = AMBIGUOUS_BM25_CV_THRESHOLD,
    idf_threshold: float = AMBIGUOUS_BM25_IDF_THRESHOLD,
    min_results: int = 2,
) -> ConditionalRerankDecision:
    """Decide whether BM25 is ambiguous enough to invoke the cross-encoder.

    Reuses the fusion QPP signals (``index/fusion.py``): AvgIDF is a pre-retrieval
    gate (common query terms flatten BM25 scores) and the BM25 score CV measures
    post-retrieval separation. BM25 is ambiguous (rerank) when AvgIDF is low or
    the score CV is weak; it is confident (skip the model) only when terms are
    discriminative and the top scores separate clearly. With too few results the
    rerank cannot help, so it is skipped.
    """
    cv = bm25_score_cv(bm25_results)
    if len(bm25_results) < min_results:
        return ConditionalRerankDecision(
            should_rerank=False,
            reason=f"too_few_results:{len(bm25_results)}",
            bm25_cv=cv,
            avg_idf=avg_idf,
        )
    if avg_idf is not None and avg_idf < idf_threshold:
        return ConditionalRerankDecision(
            should_rerank=True,
            reason=f"low_idf:avg_idf={avg_idf:.3f}",
            bm25_cv=cv,
            avg_idf=avg_idf,
        )
    if cv <= cv_threshold:
        return ConditionalRerankDecision(
            should_rerank=True,
            reason=f"flat_bm25:cv={cv:.3f}",
            bm25_cv=cv,
            avg_idf=avg_idf,
        )
    return ConditionalRerankDecision(
        should_rerank=False,
        reason=f"confident_bm25:cv={cv:.3f}",
        bm25_cv=cv,
        avg_idf=avg_idf,
    )


@dataclass(frozen=True)
class ConditionalRerankResult:
    """Outcome of a conditional rerank pass over a candidate set.

    ``results`` is the (possibly reordered) full candidate list; ``invoked`` is
    whether the cross-encoder ran; ``status`` records the disposition
    (``skipped:confident_bm25`` / ``skipped:no_candidates`` / ``applied`` /
    ``aborted:latency``); ``rerank_ms`` is the measured model-stage wall time
    (``0.0`` when the model did not run).
    """

    results: list[tuple[CodeChunk, float]]
    decision: ConditionalRerankDecision
    invoked: bool
    status: str
    rerank_ms: float


class ConditionalReranker:
    """Invoke a cross-encoder only when BM25 is ambiguous, under a latency cap.

    Wraps a :class:`CrossEncoderReranker`. The confident-BM25 path returns the
    candidates unchanged without touching the model (no load, no scoring), so the
    common query stays fast; the model is spent only on the ambiguous tail. The
    cross-encoder reorders the top ``candidate_limit`` candidates; the rest keep
    their retrieval order. The model pass runs in a worker thread and the caller
    is released at ``latency_cap_ms``, so the rerank stage is wall-clock bounded;
    on a latency abort the original retrieval order is kept.
    """

    def __init__(
        self,
        reranker: CrossEncoderReranker,
        *,
        latency_cap_ms: float = CONDITIONAL_RERANK_LATENCY_CAP_MS,
        candidate_limit: int = CONDITIONAL_RERANK_CANDIDATE_LIMIT,
    ) -> None:
        self._reranker = reranker
        self._latency_cap_ms = latency_cap_ms
        self._candidate_limit = candidate_limit

    @property
    def latency_cap_ms(self) -> float:
        return self._latency_cap_ms

    def rerank_if_ambiguous(
        self,
        query: str,
        candidates: list[tuple[CodeChunk, float]],
        bm25_results: list[tuple[CodeChunk, float]],
        *,
        avg_idf: float | None = None,
    ) -> ConditionalRerankResult:
        """Rerank ``candidates`` only when ``bm25_results`` look ambiguous.

        ``bm25_results`` are the BM25-ranked candidates used for the ambiguity
        decision (typically the same retrieval pool as ``candidates``). The model
        never runs on a confident-BM25 query, so the default fast path is free.
        """
        decision = bm25_is_ambiguous(bm25_results, avg_idf=avg_idf)
        if not decision.should_rerank:
            return ConditionalRerankResult(
                results=candidates,
                decision=decision,
                invoked=False,
                status="skipped:confident_bm25",
                rerank_ms=0.0,
            )
        head = candidates[: self._candidate_limit]
        tail = candidates[self._candidate_limit :]
        if not head:
            return ConditionalRerankResult(
                results=candidates,
                decision=decision,
                invoked=False,
                status="skipped:no_candidates",
                rerank_ms=0.0,
            )
        start = time.perf_counter()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="conditional-rerank")
        future = executor.submit(self._reranker.rerank, query, head, len(head))
        try:
            reranked = future.result(timeout=self._latency_cap_ms / 1000.0)
        except FuturesTimeoutError:
            # The caller is released at the cap; the orphaned model pass keeps
            # running in the background and its result is discarded, so the rerank
            # stage the pipeline observes never exceeds the cap.
            rerank_ms = (time.perf_counter() - start) * 1000.0
            return ConditionalRerankResult(
                results=candidates,
                decision=decision,
                invoked=True,
                status="aborted:latency",
                rerank_ms=rerank_ms,
            )
        finally:
            executor.shutdown(wait=False)
        rerank_ms = (time.perf_counter() - start) * 1000.0
        return ConditionalRerankResult(
            results=reranked + tail,
            decision=decision,
            invoked=True,
            status="applied",
            rerank_ms=rerank_ms,
        )


def maybe_conditional_reranker(
    *,
    enabled: bool,
    model_name: str | None = None,
    allow_remote_code: bool = False,
    latency_cap_ms: float = CONDITIONAL_RERANK_LATENCY_CAP_MS,
    candidate_limit: int = CONDITIONAL_RERANK_CANDIDATE_LIMIT,
) -> ConditionalReranker | None:
    """Build a :class:`ConditionalReranker` only when the lane is opted in.

    Returns ``None`` when ``enabled`` is false, so callers that leave the lane off
    (every default path) never construct a reranker. When enabled, an explicit
    ``model_name`` is required: the lane is designed for an operator-supplied small
    distilled local cross-encoder, so it never silently falls back to the
    remote-code default reranker. The wrapped ``CrossEncoderReranker`` still
    enforces the remote-code opt-in policy at load time.
    """
    if not enabled:
        return None
    if not model_name:
        raise ConfigError(
            "Conditional reranker is enabled but no rerank model was supplied. "
            "Set an explicit model (a small distilled local cross-encoder is "
            "recommended), e.g. via --rerank-model."
        )
    reranker = CrossEncoderReranker(
        model_name=model_name,
        allow_remote_code=allow_remote_code,
    )
    return ConditionalReranker(
        reranker,
        latency_cap_ms=latency_cap_ms,
        candidate_limit=candidate_limit,
    )
