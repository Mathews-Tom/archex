"""Benchmark preflight for model-backed retrieval components."""

from __future__ import annotations

from collections.abc import Sequence

from archex.benchmark.models import BenchmarkRetrievalOptions, Strategy

_VECTOR_MODEL_STRATEGIES: frozenset[Strategy] = frozenset(
    {
        Strategy.ARCHEX_QUERY_VECTOR,
        Strategy.SURROGATE_VECTOR,
        Strategy.ARCHEX_QUERY_FUSION,
        Strategy.ARCHEX_QUERY_FUSION_RERANK,
        Strategy.CROSS_LAYER_FUSION,
    }
)


def warm_benchmark_models(
    strategies: Sequence[Strategy],
    retrieval_options: BenchmarkRetrievalOptions,
) -> list[str]:
    """Load opt-in benchmark model dependencies before the task loop starts."""
    warmed: list[str] = []

    if any(strategy in _VECTOR_MODEL_STRATEGIES for strategy in strategies):
        from archex.index.embeddings import default_embedder_registry
        from archex.models import IndexConfig

        default_embedder_registry.load_entry_points()
        embedder = default_embedder_registry.create(
            IndexConfig(vector=True, embedder=retrieval_options.embedder)
        )
        if embedder is None:
            msg = "Benchmark vector strategy requires a configured embedder"
            raise ValueError(msg)
        warm = getattr(embedder, "warm", None)
        if callable(warm):
            warm()
        else:
            _ = embedder.dimension
        warmed.append(retrieval_options.embedder)

    if retrieval_options.splade:
        from archex.index.splade import DEFAULT_MODEL_NAME, SPLADEEncoder

        SPLADEEncoder().warm()
        warmed.append(DEFAULT_MODEL_NAME)

    if Strategy.ARCHEX_QUERY_FUSION_RERANK in strategies:
        from archex.index.rerank import DEFAULT_MODEL, CrossEncoderReranker

        rerank_model = retrieval_options.rerank_model or DEFAULT_MODEL
        CrossEncoderReranker(model_name=rerank_model).warm()
        warmed.append(rerank_model)

    return warmed
