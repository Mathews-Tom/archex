"""Benchmark data models: tasks, results, reports, and strategy enum."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

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
    ARCHEX_QUERY_TASK_AWARE = "archex_query_task_aware"
    ARCHEX_QUERY_COMPRESSED = "archex_query_compressed"
    ARCHEX_QUERY_EFFICIENCY_PACKED = "archex_query_efficiency_packed"
    ARCHEX_QUERY_DUAL_TRANSFORM = "archex_query_dual_transform"
    ARCHEX_QUERY_BOUNDED_RERANK = "archex_query_bounded_rerank"
    ARCHEX_QUERY_SUMMARY_SIDECAR = "archex_query_summary_sidecar"
    ARCHEX_QUERY_GRAPH_MULTIHOP = "archex_query_graph_multihop"
    ARCHEX_QUERY_COVERAGE_CANDIDATE = "archex_query_coverage_candidate"
    ARCHEX_QUERY_RANK_CANDIDATE = "archex_query_rank_candidate"
    ARCHEX_QUERY_CONTEXT_CANDIDATE = "archex_query_context_candidate"
    ARCHEX_QUERY_CONDITIONAL_RERANK = "archex_query_conditional_rerank"
    ARCHEX_QUERY_SYMBOLIC_RERANK = "archex_query_symbolic_rerank"
    ARCHEX_QUERY_DIVERSITY_PACKED = "archex_query_diversity_packed"
    ARCHEX_QUERY_PROFILE_FAST = "archex_query_profile_fast"
    ARCHEX_QUERY_PROFILE_BALANCED = "archex_query_profile_balanced"
    ARCHEX_QUERY_SEMANTIC = "archex_query_semantic"
    ARCHEX_QUERY_RUNTIME_EVIDENCE = "archex_query_runtime_evidence"
    ARCHEX_QUERY_HISTORY_EVIDENCE = "archex_query_history_evidence"
    ARCHEX_QUERY_DOCUMENTATION_EVIDENCE = "archex_query_documentation_evidence"
    EXTERNAL_MCP = "external_mcp"


class TaskCategory(StrEnum):
    SELF = "self"
    EXTERNAL_FRAMEWORK = "external-framework"
    EXTERNAL_LARGE = "external-large"
    ARCHITECTURE_BROAD = "architecture-broad"
    FRAMEWORK_SEMANTIC = "framework-semantic"


class TaskFamily(StrEnum):
    """Task shape, orthogonal to ``TaskCategory`` (which is repo type).

    ``comprehension`` tasks ask "how does X work?" and are graded on whole-bundle
    retrieval quality. ``localization`` tasks are issue-to-edit fault-localization
    tasks graded on file- and symbol-level localization of the regions to edit.
    The reporting layer can group the two families separately so a localization
    result is never folded into a comprehension-mixed aggregate.
    """

    COMPREHENSION = "comprehension"
    LOCALIZATION = "localization"


class RepoSizeClass(StrEnum):
    """Coarse repository-size bucket for M3 scorecard slicing.

    Classification counts non-blank lines across recognized source
    extensions in the checked-out task repository at benchmark run time —
    a deterministic, offline proxy for scale, not a precise LOC audit.
    """

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


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
    family: TaskFamily = TaskFamily.COMPREHENSION
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


class SymbolicRerankMode(StrEnum):
    """Combination form for the ``archex_query_symbolic_rerank`` lane.

    Both forms apply the same never-demote guard (a strong-symbolic-evidence
    candidate can never fall below its pre-rerank rank); the mode only selects
    the base reordering signal the guard is applied on top of:

    - ``BLEND`` (default): order the head by ``alpha * norm(cross_encoder) +
      (1 - alpha) * norm(symbolic_evidence)``, then floor. Validated winner: it
      lets the symbolic evidence both lift and floor candidates.
    - ``GUARD``: order the head by the cross-encoder score alone, then floor.
      The conservative never-demote-only alternative.
    """

    GUARD = "guard"
    BLEND = "blend"


class BenchmarkRetrievalOptions(BaseModel):
    splade: bool = False
    module_prefilter: bool = False
    embedder: str = "jina-v2"
    rerank_model: str | None = None
    allow_remote_code: bool = False
    freshness: bool = False
    # Discard one cache-populating execution per indexed strategy before timing.
    # This is an explicit local-operator measurement mode, not product behavior.
    warm_cache: bool = False
    chunker: ChunkerName = "default"
    bm25_chunker: ChunkerName | None = None
    vector_chunker: ChunkerName | None = None
    rerank_candidate_limit: int | None = None
    # Explicit offline opt-in for the summary-sidecar lane: filesystem path of a
    # prebuilt sidecar. None disables summary-first selection (no auto-generation).
    summary_sidecar_path: str | None = None
    # archex_query_symbolic_rerank lane: combination form (guard vs blend) and
    # the blend weight on the cross-encoder vs the symbolic evidence signal.
    symbolic_rerank_mode: SymbolicRerankMode = SymbolicRerankMode.BLEND
    symbolic_rerank_alpha: float = 0.5


class BenchmarkEvidenceManifest(BenchmarkSpecModel):
    """Immutable identity and integrity record for one benchmark evidence directory."""

    evidence_version: Literal[1] = 1
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    archex_version: str = Field(min_length=1)
    task_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_ids: list[str] = Field(min_length=1)
    strategies: list[Strategy] = Field(min_length=1)
    retrieval_options: BenchmarkRetrievalOptions
    generated_at: str = Field(min_length=1)
    hardware_advisory: str = Field(min_length=1)
    report_hashes: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_coverage(self) -> BenchmarkEvidenceManifest:
        if len(self.task_ids) != len(set(self.task_ids)):
            msg = "task_ids must not contain duplicates"
            raise ValueError(msg)
        if any(not task_id.strip() for task_id in self.task_ids):
            msg = "task_ids must not contain empty values"
            raise ValueError(msg)
        if len(self.strategies) != len(set(self.strategies)):
            msg = "strategies must not contain duplicates"
            raise ValueError(msg)
        if set(self.report_hashes) != set(self.task_ids):
            msg = "report_hashes keys must match task_ids exactly"
            raise ValueError(msg)
        for task_id, digest in self.report_hashes.items():
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                msg = f"report_hashes[{task_id!r}] must be a lowercase SHA-256 digest"
                raise ValueError(msg)
        return self


class ComparisonLayerType(StrEnum):
    """Where a comparison lane operates.

    ``retrieval`` lanes are retrieval/selection engines (archex, cocoindex-code).
    ``graph-memory`` lanes are graph/memory-layer workflows (Graphify); they may
    include retrieval-like answers, but the comparison must keep their build/query
    semantics explicit instead of presenting them as direct retrieval equivalents.
    ``compression`` lanes are post-selection context-management layers (Headroom);
    they are not retrieval engines. ``baseline`` is the raw filesystem read/grep lane.
    """

    RETRIEVAL = "retrieval"
    GRAPH_MEMORY = "graph-memory"
    COMPRESSION = "compression"
    BASELINE = "baseline"


class CompressionLayerMode(StrEnum):
    """How a compression layer is composed with retrieval in the comparison.

    ``headroom_only_on_raw_context`` compresses raw/broad context with no archex
    selection (does compression alone rescue broad context?). ``archex_plus_headroom``
    compresses an archex-selected bundle after selection (is the pair composable?).
    """

    HEADROOM_ONLY_ON_RAW_CONTEXT = "headroom_only_on_raw_context"
    ARCHEX_PLUS_HEADROOM = "archex_plus_headroom"


class GraphifyLaneName(StrEnum):
    GRAPHIFY_BUILD_PLUS_QUERY = "graphify_build_plus_query"
    GRAPHIFY_QUERY_WARM = "graphify_query_warm"


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
    # External MCP tools modeled here are retrieval/selection engines. The layer
    # type is explicit metadata so reports never frame a retrieval engine and a
    # compression layer as the same kind of system.
    layer_type: ComparisonLayerType = ComparisonLayerType.RETRIEVAL
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


class CompressionLayerConfig(BenchmarkSpecModel):
    """A post-selection compression layer (Headroom-style) in the comparison.

    This is a compression/context-management layer, not a retrieval engine, so
    ``layer_type`` defaults to ``compression`` and the manifest validator rejects
    any other value. ``artifact_dir`` selects artifact-adapter mode: operator-run
    compression outputs are imported instead of running the binary in-session.
    """

    name: str
    version: str
    command: str
    args: list[str] = []
    layer_type: ComparisonLayerType = ComparisonLayerType.COMPRESSION
    modes: list[CompressionLayerMode] = Field(
        default_factory=lambda: [
            CompressionLayerMode.HEADROOM_ONLY_ON_RAW_CONTEXT,
            CompressionLayerMode.ARCHEX_PLUS_HEADROOM,
        ]
    )
    compression_settings: dict[str, str] = {}
    env: dict[str, str] = {}
    timeout_seconds: float = 120.0
    artifact_dir: str | None = None


class GraphifyLaneConfig(BenchmarkSpecModel):
    """One Graphify public lane in the competitive comparison manifest.

    Graphify is modeled separately from retrieval engines because the public
    comparison must keep graph-build cost distinct from warm graph-query cost.
    ``artifact_dir`` selects operator-artifact import mode. Without it, the
    adapter executes ``command``/``args`` locally and expects a lane-result JSON
    document on stdout.
    """

    name: GraphifyLaneName
    package_name: str = "graphifyy"
    version: str
    command: str
    args: list[str] = []
    layer_type: ComparisonLayerType = ComparisonLayerType.GRAPH_MEMORY
    includes_build_cost: bool
    env: dict[str, str] = {}
    timeout_seconds: float = Field(default=600.0, gt=0)
    artifact_dir: str | None = None
    operational_notes: str | None = None

    @model_validator(mode="after")
    def _validate_lane_semantics(self) -> GraphifyLaneConfig:
        if not self.package_name.strip():
            msg = "graphify package_name must not be empty"
            raise ValueError(msg)
        if not self.command.strip():
            msg = "graphify command must not be empty"
            raise ValueError(msg)
        if self.artifact_dir is not None and not self.artifact_dir.strip():
            msg = "graphify artifact_dir must not be empty"
            raise ValueError(msg)
        expected = self.name is GraphifyLaneName.GRAPHIFY_BUILD_PLUS_QUERY
        if self.includes_build_cost is not expected:
            if expected:
                msg = "graphify_build_plus_query must include build cost"
            else:
                msg = "graphify_query_warm must not include build cost"
            raise ValueError(msg)
        return self


class HeadToHeadArchexConfig(BenchmarkSpecModel):
    strategy: Strategy = Strategy.ARCHEX_QUERY
    # Improved/benchmark-only archex lanes compared beside the current default
    # (e.g. archex_query_compressed, archex_query_efficiency_packed). Empty keeps
    # the historical single-lane behavior.
    candidate_strategies: list[Strategy] = []
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
    compression_layers: list[CompressionLayerConfig] = []
    graphify_lanes: list[GraphifyLaneConfig] = []


class TaskCompletionResult(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class QueryTiming(BaseModel):
    """Explicit timing representation for query execution."""

    cold_ms: float | None = None
    warm_ms: float | None = None
    wall_ms: float | None = None


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
    wall_time_ms: float | None = None
    cached: bool = False
    timing: PipelineTiming | None = None
    timestamp: str
    # Seed vs expansion diagnostics
    unique_ranked_files: int = 0
    # Fraction of ranked (pre-dedup) files that were duplicates of a path
    # already ranked higher (e.g. re-surfaced by both a seed hit and a
    # graph-expansion edge). 0.0 when every returned file is unique.
    duplicate_rate: float = 0.0
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
    family: TaskFamily = TaskFamily.COMPREHENSION
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
    provenance: dict[str, str] = {}
    warm_latency_ms: float | None = None
    freshness_latency_ms: float = 0.0
    freshness_measured: bool = False
    freshness_correct: bool = False
    chunker: ChunkerName = "default"
    index_chunk_count: int = 0
    mean_chunk_tokens: float = 0.0
    # Coarse repository-size bucket, computed once from the checked-out task
    # repository. None ("unmeasured") for benchmark-only lanes that do not
    # resolve the archex_query-family assembly path.
    repo_size_class: RepoSizeClass | None = None
    # Optional region/line-level retrieval-quality metrics. Populated only when a
    # task declares expected_regions; otherwise left as None ("unknown").
    region_recall: float | None = None
    region_precision: float | None = None
    region_f1: float | None = None
    line_recall: float | None = None
    line_precision: float | None = None
    ranked_region_mrr: float | None = None
    ranked_region_ndcg: float | None = None
    context_noise_ratio: float | None = None
    useful_tokens: int | None = None
    wasted_tokens: int | None = None
    relevance_per_1k_tokens: float | None = None
    # Optional post-retrieval compression metrics. Populated only by
    # archex_query_compressed; None ("unknown") for every other strategy so
    # archex_query and other lanes are unaffected. Retrieval metrics above stay
    # attributable to the uncompressed retrieval set.
    bundle_tokens_uncompressed: int | None = None
    bundle_tokens_compressed: int | None = None
    bundle_compression_ratio: float | None = None
    required_context_compressed_tokens: int | None = None
    required_context_passthrough_tokens: int | None = None
    compression_hidden_required_region_count: int | None = None
    token_efficiency_with_compression_and_completion: float | None = None
    # Optional retrieval-aware packing metrics. Populated only by
    # archex_query_efficiency_packed; None ("unknown") for every other strategy.
    # The packed lane reuses the bundle_tokens_uncompressed/compressed/ratio and
    # token_efficiency_with_compression_and_completion fields above to report its
    # token reduction; these add the packing-decision provenance counts and the
    # delivered relevance-per-token of the packed bundle.
    packed_relevance_per_1k_tokens: float | None = None
    packing_included_regions: int | None = None
    packing_compressed_regions: int | None = None
    packing_elided_regions: int | None = None
    packing_skipped_regions: int | None = None
    # Optional low-latency conditional-rerank metric. Populated only by
    # archex_query_conditional_rerank; the on-disk size (bytes) of the rerank
    # model artifact when it resolves to a local path, else None ("unmeasured").
    rerank_model_storage_bytes: int | None = None
    # Optional query-adaptive diversity-packing metric. Populated only by
    # archex_query_diversity_packed; the count of redundant regions de-selected by
    # MMR diversity (distinct from budget skips). None for every other strategy.
    diversity_deselected_regions: int | None = None


class BenchmarkProvenance(BaseModel):
    """Reproducible run provenance for benchmark reports."""

    archex_version: str
    commit: str | None = None
    generation_time: str
    config: dict[str, str | bool | int | float | list[str]] = {}
    hardware: str | None = None
    sample_count: int = 1


class BenchmarkReport(BaseModel):
    task_id: str
    repo: str
    question: str
    results: list[BenchmarkResult]
    baseline_tokens: int
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    provenance: BenchmarkProvenance | None = None


class CompressionLayerResult(BaseModel):
    """Result of applying a compression layer (Headroom-style) to a lane's context.

    A compression layer is not a retrieval engine: ``layer_type`` is fixed to
    ``compression`` and ``source_lane`` records which retrieval/baseline lane
    produced the context that was compressed. ``source_passthrough`` records
    whether the layer protected (passed through) the source/RAG content rather
    than compressing it. Reuses the bundle compression-token fields so reports
    can show compression ratio beside the retrieval lanes' token efficiency.
    """

    task_id: str
    lane_label: str
    layer_type: ComparisonLayerType = ComparisonLayerType.COMPRESSION
    mode: CompressionLayerMode
    source_lane: str
    source_passthrough: bool
    bundle_tokens_uncompressed: int
    bundle_tokens_compressed: int
    bundle_compression_ratio: float
    provenance: dict[str, str] = {}
    timestamp: str


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
