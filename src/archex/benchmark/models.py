"""Benchmark data models: tasks, results, reports, and strategy enum."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, model_validator

from archex.models import (  # noqa: TCH001 — Pydantic needs at runtime
    DeltaMeta,
    PipelineTiming,
    VectorMode,
)


class Strategy(StrEnum):
    RAW_FILES = "raw_files"
    RAW_GREPPED = "raw_grepped"
    ARCHEX_QUERY = "archex_query"
    ARCHEX_QUERY_VECTOR = "archex_query_vector"
    SURROGATE_VECTOR = "surrogate_vector"
    ARCHEX_QUERY_FUSION = "archex_query_fusion"
    CROSS_LAYER_FUSION = "cross_layer_fusion"
    ARCHEX_QUERY_FUSION_RERANK = "archex_query_fusion_rerank"
    ARCHEX_SYMBOL_LOOKUP = "archex_symbol_lookup"


class TaskCategory(StrEnum):
    SELF = "self"
    EXTERNAL_FRAMEWORK = "external-framework"
    EXTERNAL_LARGE = "external-large"
    ARCHITECTURE_BROAD = "architecture-broad"
    FRAMEWORK_SEMANTIC = "framework-semantic"


class BenchmarkTask(BaseModel):
    task_id: str
    repo: str
    commit: str
    question: str
    expected_files: list[str]
    expected_symbols: list[str] = []
    token_budget: int = 8192
    keywords: list[str] = []
    languages: list[str] | None = None
    include_paths: list[str] = []
    category: TaskCategory | None = None

    @model_validator(mode="after")
    def _validate_include_paths(self) -> BenchmarkTask:
        for path in self.include_paths:
            if not path or path.startswith("/") or ".." in path.split("/"):
                msg = f"include_paths entries must be relative paths: {path!r}"
                raise ValueError(msg)
        return self


class BenchmarkRetrievalOptions(BaseModel):
    splade: bool = False
    module_prefilter: bool = False
    embedder: str = "jina-v2"


class BenchmarkResult(BaseModel):
    task_id: str
    strategy: Strategy
    tokens_total: int
    tool_calls: int
    files_accessed: int
    recall: float
    precision: float
    f1_score: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    map_score: float = 0.0
    symbol_recall: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    token_efficiency: float = 0.0
    tokens_raw_baseline: int = 0
    savings_vs_raw: float
    wall_time_ms: float
    cached: bool
    timing: PipelineTiming | None = None
    timestamp: str
    # Seed vs expansion diagnostics
    unique_ranked_files: int = 0
    seed_files: list[str] = []
    expanded_files: list[str] = []
    expansion_ratio: float = 0.0
    seed_recall: float = 0.0
    seed_precision: float = 0.0
    expansion_eligible_seeds: int = 0
    expansion_candidates_found: int = 0
    expansion_import_neighbor_edges: int = 0
    expansion_same_module_candidates: int = 0
    expansion_hub_candidates: int = 0
    expansion_test_candidates_skipped: int = 0
    expansion_zero_candidate_reason: str = ""
    category: TaskCategory | None = None
    vector_mode: VectorMode = VectorMode.RAW
    surrogate_version: str | None = None
    cache_state: str = "cold"


class BenchmarkReport(BaseModel):
    task_id: str
    repo: str
    question: str
    results: list[BenchmarkResult]
    baseline_tokens: int
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Delta benchmarking models
# ---------------------------------------------------------------------------


class DeltaStrategy(StrEnum):
    DELTA_INDEX = "delta_index"
    FULL_REINDEX = "full_reindex"


class DeltaBenchmarkTask(BaseModel):
    task_id: str
    repo: str
    base_commit: str
    delta_commit: str
    expected_delta: list[str] = []
    language: str = "python"


class DeltaBenchmarkResult(BaseModel):
    task_id: str
    strategy: DeltaStrategy
    delta_files: int
    total_files: int
    delta_pct: float
    delta_time_ms: float
    full_reindex_time_ms: float
    speedup_factor: float
    correctness: bool
    chunks_updated: int
    chunks_unchanged: int
    edges_updated: int
    timestamp: str
    delta_meta: DeltaMeta | None = None
