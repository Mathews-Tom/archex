"""Benchmark data models: tasks, results, reports, and strategy enum."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from archex.models import (  # noqa: TCH001 — Pydantic needs at runtime
    ChunkerName,
    DeltaMeta,
    PipelineTiming,
    SymbolKind,
    VectorMode,
)


class Strategy(StrEnum):
    RAW_FILES = "raw_files"
    RAW_GREPPED = "raw_grepped"
    RAW_RIPGREP = "raw_ripgrep"
    ARCHEX_QUERY = "archex_query"
    ARCHEX_SCOUT_FETCH = "archex_scout_fetch"
    ARCHEX_QUERY_VECTOR = "archex_query_vector"
    SURROGATE_VECTOR = "surrogate_vector"
    ARCHEX_QUERY_FUSION = "archex_query_fusion"
    ARCHEX_QUERY_HYBRID = "archex_query_hybrid"
    ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT = "archex_query_hybrid_quantized_4bit"
    CROSS_LAYER_FUSION = "cross_layer_fusion"
    ARCHEX_QUERY_FUSION_RERANK = "archex_query_fusion_rerank"
    EXTERNAL_MCP = "external_mcp"


class TaskCategory(StrEnum):
    SELF = "self"
    EXTERNAL_FRAMEWORK = "external-framework"
    EXTERNAL_LARGE = "external-large"
    ARCHITECTURE_BROAD = "architecture-broad"
    FRAMEWORK_SEMANTIC = "framework-semantic"


class BenchmarkSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BundleOnlyAllowedContext(StrEnum):
    BUNDLE_ONLY = "bundle-only"
    BUNDLE_PLUS_FRONTIER = "bundle-plus-frontier"


class BundleOnlyEvaluatorCommand(BenchmarkSpecModel):
    command: str
    args: list[str] = []
    timeout_seconds: float = Field(default=600.0, gt=0)

    @model_validator(mode="after")
    def _validate_command(self) -> BundleOnlyEvaluatorCommand:
        if not self.command.strip():
            msg = "bundle-only evaluator command must not be empty"
            raise ValueError(msg)
        return self


class BundleOnlyEvaluation(BenchmarkSpecModel):
    expected_answer: str | None = None
    deterministic_rubric: str | None = None
    allowed_context_policy: BundleOnlyAllowedContext = BundleOnlyAllowedContext.BUNDLE_ONLY
    evaluator_command: BundleOnlyEvaluatorCommand | None = None

    @model_validator(mode="after")
    def _validate_grader(self) -> BundleOnlyEvaluation:
        has_expected_answer = self.expected_answer is not None and bool(
            self.expected_answer.strip()
        )
        has_rubric = self.deterministic_rubric is not None and bool(
            self.deterministic_rubric.strip()
        )
        if has_expected_answer == has_rubric:
            msg = (
                "bundle-only evaluation must define exactly one of "
                "expected_answer or deterministic_rubric"
            )
            raise ValueError(msg)
        return self


class RegionGranularity(StrEnum):
    FILE = "file"
    SYMBOL = "symbol"
    BLOCK = "block"
    LINE_RANGE = "line_range"


class ExpectedRegion(BenchmarkSpecModel):
    """Optional ground-truth code region for region/line-level retrieval scoring.

    A region is either a path plus an inclusive line range (``line_range``
    granularity) or a path plus a symbol/block handle (``symbol``/``block``
    granularity). ``file`` granularity scores at whole-file level only.
    """

    path: str
    granularity: RegionGranularity = RegionGranularity.LINE_RANGE
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    symbol: str | None = None
    notes: str | None = None
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _validate_region(self) -> ExpectedRegion:
        if not self.path or self.path.startswith("/") or ".." in self.path.split("/"):
            msg = f"expected_regions path must be a relative repo path: {self.path!r}"
            raise ValueError(msg)
        if (self.start_line is None) != (self.end_line is None):
            msg = "expected_regions line range requires both start_line and end_line"
            raise ValueError(msg)
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            msg = (
                f"expected_regions end_line ({self.end_line}) must be "
                f">= start_line ({self.start_line})"
            )
            raise ValueError(msg)
        has_range = self.start_line is not None
        has_symbol = self.symbol is not None and bool(self.symbol.strip())
        granularity = self.granularity
        needs_symbol = granularity in (RegionGranularity.SYMBOL, RegionGranularity.BLOCK)
        if granularity is RegionGranularity.LINE_RANGE and not has_range:
            msg = "line_range regions require start_line and end_line"
            raise ValueError(msg)
        if needs_symbol and not has_symbol:
            msg = f"{granularity.value} regions require a symbol handle"
            raise ValueError(msg)
        if granularity is RegionGranularity.FILE and (has_range or has_symbol):
            msg = "file regions must not declare a line range or symbol handle"
            raise ValueError(msg)
        return self


class BenchmarkTask(BenchmarkSpecModel):
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
    bundle_only_eval: BundleOnlyEvaluation | None = None
    expected_regions: list[ExpectedRegion] = []

    @model_validator(mode="after")
    def _validate_include_paths(self) -> BenchmarkTask:
        for path in self.include_paths:
            if not path or path.startswith("/") or ".." in path.split("/"):
                msg = f"include_paths entries must be relative paths: {path!r}"
                raise ValueError(msg)
        return self


class ArchitectureExpectedModule(BenchmarkSpecModel):
    name: str
    root_path: str
    files: list[str]
    responsibility_terms: list[str] = []


class ArchitectureExpectedPattern(BenchmarkSpecModel):
    name: str
    evidence_symbols: list[str] = []


class ArchitectureExpectedInterface(BenchmarkSpecModel):
    name: str
    file_path: str
    kind: SymbolKind | None = None


class ArchitectureExpectedDecision(BenchmarkSpecModel):
    decision_terms: list[str]


class ArchitectureOracle(BenchmarkSpecModel):
    modules: list[ArchitectureExpectedModule] = []
    patterns: list[ArchitectureExpectedPattern] = []
    interfaces: list[ArchitectureExpectedInterface] = []
    decisions: list[ArchitectureExpectedDecision] = []


class ArchitectureBenchmarkTask(BenchmarkSpecModel):
    task_id: str
    repo: str
    commit: str
    question: str
    include_paths: list[str]
    languages: list[str] | None = None
    arch_oracle: ArchitectureOracle

    @model_validator(mode="after")
    def _validate_include_paths(self) -> ArchitectureBenchmarkTask:
        if not self.include_paths:
            msg = "architecture benchmark tasks must declare include_paths"
            raise ValueError(msg)
        for path in self.include_paths:
            if not path or path.startswith("/") or ".." in path.split("/"):
                msg = f"include_paths entries must be relative paths: {path!r}"
                raise ValueError(msg)
        return self


class ArchitectureDimensionScores(BaseModel):
    boundary_precision: float = 1.0
    boundary_recall: float = 1.0
    boundary_f1: float = 1.0
    responsibility_recall: float = 1.0
    pattern_precision: float = 1.0
    pattern_recall: float = 1.0
    interface_completeness: float = 1.0
    decision_recall: float = 1.0
    overall: float = 1.0


class ArchitectureBenchmarkResult(BaseModel):
    task_id: str
    repo: str
    commit: str
    scores: ArchitectureDimensionScores
    detected_modules: list[str] = []
    detected_patterns: list[str] = []
    detected_interfaces: list[str] = []
    detected_decisions: list[str] = []
    advisory: bool = True


class BenchmarkRetrievalOptions(BaseModel):
    splade: bool = False
    module_prefilter: bool = False
    embedder: str = "jina-v2"
    rerank_model: str | None = None
    allow_remote_code: bool = False
    freshness: bool = False
    chunker: ChunkerName = "default"
    bm25_chunker: ChunkerName | None = None
    vector_chunker: ChunkerName | None = None
    rerank_candidate_limit: int | None = None


class ExternalToolCommandConfig(BenchmarkSpecModel):
    command: str
    args: list[str] = []
    timeout_seconds: float = 600.0


class ExternalToolBenchmarkConfig(BenchmarkSpecModel):
    name: str
    version: str
    command: str
    args: list[str] = []
    embedder: str
    cwd: str | None = None
    env: dict[str, str] = {}
    search_tool: str = "search"
    query_argument: str = "query"
    path_argument: str | None = "paths"
    language_argument: str | None = "languages"
    limit_argument: str | None = "limit"
    limit: int = 10
    extra_arguments: dict[str, object] = {}
    timeout_seconds: float = 120.0
    bootstrap_commands: list[ExternalToolCommandConfig] = []


class HeadToHeadArchexConfig(BenchmarkSpecModel):
    strategy: Strategy = Strategy.ARCHEX_QUERY
    embedder: str = "jina-v2"
    local_models_only: bool = True


class HeadToHeadManifest(BenchmarkSpecModel):
    manifest_version: int = 1
    name: str
    hardware_notes: str
    task_subset: list[str]
    archex: HeadToHeadArchexConfig = Field(default_factory=HeadToHeadArchexConfig)
    raw_read_strategy: Strategy = Strategy.RAW_RIPGREP
    external_tools: list[ExternalToolBenchmarkConfig]


class TaskCompletionResult(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class BenchmarkResult(BaseModel):
    task_id: str
    strategy: Strategy
    strategy_label: str | None = None
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
    expansion_reason_counts: dict[str, int] = {}
    expanded_file_reasons: dict[str, list[str]] = {}
    result_files: list[str] = []
    required_file_recall: float = 0.0
    missed_required_file_rate: float = 0.0
    missed_required_task_rate: float = 0.0
    all_required_files_present: bool = False
    required_files_present: list[str] = []
    required_files_missing: list[str] = []
    bundle_only_success: TaskCompletionResult | None = None
    needed_files_outside_returned: list[str] | None = None
    needed_files_in_frontier_cut: list[str] | None = None
    needed_files_in_top_candidates: list[str] | None = None
    safe_to_act_false_positive: bool | None = None
    post_bundle_search_turns: int | None = None
    post_bundle_read_turns: int | None = None
    task_completion_result: TaskCompletionResult = TaskCompletionResult.UNKNOWN
    completion_preserved: bool | None = None
    receipt_accuracy: bool | None = None
    bundle_completion_tokens: int = 0
    bundle_completion_files: list[str] = []
    token_efficiency_with_completion: float = 0.0
    cold_start_ms: float = 0.0
    warm_latency_ms: float = 0.0
    provenance: dict[str, str] = {}
    freshness_latency_ms: float = 0.0
    freshness_measured: bool = False
    freshness_correct: bool = False
    chunker: ChunkerName = "default"
    index_chunk_count: int = 0
    mean_chunk_tokens: float = 0.0


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


class DeltaBenchmarkTask(BenchmarkSpecModel):
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
