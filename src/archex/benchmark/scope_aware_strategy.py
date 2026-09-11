"""Benchmark-only execution path for R26 Candidate A."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from archex.api import (
    _compute_dynamic_budget,  # pyright: ignore[reportPrivateUsage]
    _ensure_index,  # pyright: ignore[reportPrivateUsage]
    _finalize_context_bundle,  # pyright: ignore[reportPrivateUsage]
    _freshness_for_query,  # pyright: ignore[reportPrivateUsage]
    query,
)
from archex.benchmark.scope_aware_candidate import rank_scopes, resolve_scope_map
from archex.benchmark.scope_aware_receipt import (
    ScopeAwareReceipt,
    build_multi_scope_receipt,
    build_single_scope_receipt,
)
from archex.cache import CacheManager
from archex.exceptions import ConfigError
from archex.index.graph import DependencyGraph
from archex.models import (
    Config,
    ContextBundle,
    IndexConfig,
    PipelineTiming,
    RepoSource,
    RetrievalPolicy,
    VectorMode,
)
from archex.receipt import index_revision_from_store

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkTask
    from archex.index.store import IndexStore
    from archex.models import CodeChunk
from archex.serve.context import assemble_context


@dataclass(frozen=True, slots=True)
class ScopeAwareOutcome:
    """Candidate payload, strict sidecar, and product timing."""

    bundle: ContextBundle
    receipt: ScopeAwareReceipt
    timing: PipelineTiming


def frozen_scope_aware_config(
    *,
    languages: list[str] | None,
    cache_dir: str = "~/.archex/cache",
) -> Config:
    """Return the R27 indexing runtime configuration, independent of CLI options."""
    return Config(
        languages=languages,
        cache=True,
        cache_dir=cache_dir,
        max_file_size=10_000_000,
        parallel=False,
        worktree_seed=False,
    )


def frozen_scope_aware_index_config() -> IndexConfig:
    """Return the exact R27 candidate index configuration."""
    return IndexConfig(
        allow_remote_code=False,
        bm25=True,
        chunk_max_tokens=500,
        chunk_min_tokens=50,
        chunker="default",
        documentation_evidence_providers=[],
        embedder=None,
        history_evidence_providers=[],
        identifier_fragment_tokenization=False,
        module_prefilter=False,
        quantize_bits=4,
        quantize_vectors=True,
        rerank=False,
        rerank_candidate_limit=4,
        rerank_model=None,
        retrieval_policy=RetrievalPolicy.AUTO,
        runtime_evidence_providers=[],
        semantic_evidence_providers=[],
        splade=False,
        surrogate_version="v1",
        token_encoding="cl100k_base",
        vector=False,
        vector_mode=VectorMode.RAW,
    )


def scope_aware_repo_source(
    task: BenchmarkTask,
    repo_path: Path,
    *,
    config: Config,
    index_config: IndexConfig,
) -> RepoSource:
    """Build a distinct cache identity from the frozen candidate configuration."""
    commit = (
        CacheManager.git_head(str(repo_path))
        if task.commit == "HEAD"
        else task.commit or CacheManager.git_head(str(repo_path))
    )
    if not commit:
        raise ConfigError(
            f"Benchmark task {task.task_id!r} has no commit and {repo_path} has no git HEAD"
        )
    identity_payload = {
        "strategy": "scope_aware_candidate:v1",
        "config": config.model_dump(mode="json"),
        "index_config": index_config.model_dump(mode="json"),
        "include_paths": sorted(task.include_paths),
    }
    digest = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return RepoSource(
        local_path=str(repo_path),
        stable_identity=f"{task.repo}@{commit}#scope-aware-candidate:{digest}",
    )


def _repository_totals(store: IndexStore, chunks: list[CodeChunk]) -> tuple[int, int]:
    stored_tokens = store.get_metadata("repo_total_tokens")
    total_repo_tokens = (
        int(stored_tokens) if stored_tokens is not None else store.get_chunk_token_total()
    )
    stored_count = store.get_metadata("chunk_count")
    chunk_count = int(stored_count) if stored_count is not None else len(chunks)
    return total_repo_tokens, chunk_count


def execute_scope_aware_query(
    source: RepoSource,
    question: str,
    *,
    repo_root: Path,
    task_id: str,
    repository_id: str,
    token_budget: int,
    config: Config,
    index_config: IndexConfig,
) -> ScopeAwareOutcome:
    """Execute Candidate A without changing product retrieval or payload models."""
    started_at = time.perf_counter()
    discovery_timing = PipelineTiming()
    store = _ensure_index(
        source,
        config,
        timing=discovery_timing,
        index_config=index_config,
    )
    try:
        all_chunks = store.get_chunks()
        scope_map = resolve_scope_map(repo_root, all_chunks)
        if len(scope_map.scopes) == 1:
            timing = PipelineTiming()
            bundle = query(
                source,
                question,
                token_budget=token_budget,
                config=config,
                index_config=index_config,
                timing=timing,
                explicit_token_budget=True,
                refresh=False,
            )
            single_receipt = build_single_scope_receipt(
                task_id=task_id,
                repository_id=repository_id,
                scope_map=scope_map,
                bundle=bundle,
            )
            return ScopeAwareOutcome(bundle=bundle, receipt=single_receipt, timing=timing)

        total_repo_tokens, chunk_count = _repository_totals(store, all_chunks)
        index_revision = index_revision_from_store(store)
        timing = discovery_timing
        timing.index_ms = (time.perf_counter() - started_at) * 1000.0
        search_started = time.perf_counter()
        resolved_map, ranking = rank_scopes(
            store=store,
            repo_root=repo_root,
            chunks=all_chunks,
            query=question,
        )
        if ranking is None or len(resolved_map.scopes) < 2:
            raise RuntimeError("multi-scope candidate unexpectedly entered the bypass path")
        timing.search_ms = (time.perf_counter() - search_started) * 1000.0
        post_search_at = time.perf_counter()

        graph = DependencyGraph.from_edges(store.get_edges())
        if graph.file_edge_count == 0 and graph.file_count > 1:
            graph.add_co_directory_edges()
        effective_budget = _compute_dynamic_budget(total_repo_tokens, token_budget, None)
        bundle = assemble_context(
            search_results=[
                (candidate.chunk, candidate.normalized_score) for candidate in ranking.candidates
            ],
            graph=graph,
            all_chunks=all_chunks,
            question=question,
            token_budget=effective_budget,
            apply_intent_budget=False,
        )
        bundle = _finalize_context_bundle(
            bundle,
            label="scope-aware-candidate",
            started_at=started_at,
            timing=timing,
            trace=None,
            index_config=index_config,
            chunk_count=chunk_count,
            total_repo_tokens=total_repo_tokens,
            metadata_timing=discovery_timing,
            index_revision=index_revision,
            freshness=_freshness_for_query(False),
            post_search_at=post_search_at,
            semantic_providers=store.get_semantic_provider_receipts(),
            runtime_providers=store.get_runtime_provider_receipts(),
            history_providers=store.get_history_provider_receipts(),
            history_change_cards=store.get_history_change_cards(),
            history_coupling_observations=store.get_history_coupling_observations(),
            documentation_providers=store.get_documentation_provider_receipts(),
        )
        multi_receipt = build_multi_scope_receipt(
            task_id=task_id,
            repository_id=repository_id,
            ranking=ranking,
            shared_idf_identity=(f"bm25-fts5:repository-wide:index-revision:{index_revision}"),
        )
        return ScopeAwareOutcome(bundle=bundle, receipt=multi_receipt, timing=timing)
    finally:
        store.close()
