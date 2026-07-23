"""All Pydantic data models for archex: enums, input, intermediate, index, and output types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from archex.index.quantize import SUPPORTED_BITS
from archex.integrations.lsap_models import LSAPEnrichment  # noqa: TCH001

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SymbolKind(StrEnum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    TYPE = "type"
    VARIABLE = "variable"
    CONSTANT = "constant"
    INTERFACE = "interface"
    ENUM = "enum"
    MODULE = "module"


class Visibility(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"


class EdgeKind(StrEnum):
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    USES_TYPE = "uses_type"
    EXPORTS = "exports"
    CO_DIRECTORY = "co_directory"


class EdgeConfidence(StrEnum):
    EXTRACTED = "extracted"
    HEURISTIC = "heuristic"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"


class PatternCategory(StrEnum):
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    CREATIONAL = "creational"


class ChangeStatus(StrEnum):
    MODIFIED = "M"
    ADDED = "A"
    DELETED = "D"
    RENAMED = "R"


class VectorMode(StrEnum):
    RAW = "raw"
    SURROGATE = "surrogate"


class RetrievalPolicy(StrEnum):
    AUTO = "auto"
    BM25_ONLY = "bm25_only"
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    CROSS_LAYER = "cross_layer"


class RetrievalProfile(StrEnum):
    """Named retrieval-cost/quality tradeoff, mapped to IndexConfig feature flags.

    FAST: bm25 only — zero vector/model thread work, the cheapest and always
        available tier. Equivalent to IndexConfig()'s own defaults.
    BALANCED: adds module-responsibility prefiltering, a pure structural
        signal computed from the dependency graph — still no model calls.
    DEEP: adds vector search and cross-encoder reranking for maximum quality
        at the highest cost. Falls back to the bm25-only path per the
        existing embedder/reranker-unavailable behavior if neither is
        configured.
    """

    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


class LanguageTier(StrEnum):
    FULL = "full"
    STRUCTURED = "structured"
    CHUNK_ONLY = "chunk-only"
    UNKNOWN = "unknown"


class ContextFreshness(StrEnum):
    CLEAN = "clean"
    DIRTY = "dirty"
    WATCH_ACTIVE = "watch_active"
    WATCH_UNAVAILABLE = "watch_unavailable"
    UNKNOWN = "unknown"


class ContextCompletenessStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


class ContextCompletenessReason(StrEnum):
    COMPLETE = "complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEPENDENCY_FRONTIER_CUT = "dependency_frontier_cut"
    DUPLICATE_SUPPRESSED = "duplicate_suppressed"
    NO_CANDIDATES = "no_candidates"
    STALE_INDEX = "stale_index"
    UNSUPPORTED_GRAMMAR = "unsupported_grammar"
    UNKNOWN = "unknown"


class ContextRecommendedAction(StrEnum):
    USE_BUNDLE = "use_bundle"
    NARROW_QUERY = "narrow_query"
    RAISE_BUDGET = "raise_budget"
    REFRESH_INDEX = "refresh_index"
    FETCH_SKIPPED_CANDIDATE = "fetch_skipped_candidate"
    MANUAL_REVIEW = "manual_review"


class ContextSkippedReason(StrEnum):
    BELOW_THRESHOLD = "below_threshold"
    DEPENDENCY_FRONTIER_CUT = "dependency_frontier_cut"
    DUPLICATE = "duplicate"
    OVER_BUDGET = "over_budget"
    STALE_INDEX = "stale_index"
    TEST_DEPRIORITIZED = "test_deprioritized"
    UNSUPPORTED_GRAMMAR = "unsupported_grammar"


class ContextOmittedEdgeReason(StrEnum):
    OVER_BUDGET = "over_budget"
    UNSUPPORTED_GRAMMAR = "unsupported_grammar"
    STALE_INDEX = "stale_index"
    BELOW_THRESHOLD = "below_threshold"


class CompressionMode(StrEnum):
    """Deterministic post-retrieval compression mode applied to a returned region."""

    PASSTHROUGH_REQUIRED = "passthrough_required"
    STRUCTURAL_CODE_ELISION = "structural_code_elision"
    COMMENT_AND_WHITESPACE_SLIMMING = "comment_and_whitespace_slimming"
    LARGE_LITERAL_SUMMARIZATION = "large_literal_summarization"
    JSON_LOG_SMART_CRUSHING = "json_log_smart_crushing"


class CompressionLossRisk(StrEnum):
    """How risky a compression mode is for an editing/debugging agent."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SymbolId = str


def make_symbol_id(
    file_path: str,
    qualified_name: str | None,
    kind: SymbolKind | None,
) -> SymbolId:
    """Build a stable, line-independent symbol identifier.

    Format: ``file_path::qualified_name#kind``
    File-level (no symbol): ``file_path::_module#module``
    """
    name = qualified_name if qualified_name is not None else "_module"
    kind_str = str(kind) if kind is not None else "module"
    return f"{file_path}::{name}#{kind_str}"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class RepoSource(BaseModel):
    url: str | None = None
    local_path: str | None = None
    target: str | None = None
    commit: str | None = None
    stable_identity: str | None = None

    @model_validator(mode="after")
    def _require_source(self) -> RepoSource:
        if self.url is not None and not self.url.strip():
            raise ValueError("url must not be empty")
        if self.local_path is not None and not self.local_path.strip():
            raise ValueError("local_path must not be empty")
        if self.stable_identity is not None and not self.stable_identity.strip():
            raise ValueError("stable_identity must not be empty")
        if not self.url and not self.local_path:
            raise ValueError("RepoSource requires either 'url' or 'local_path'")
        return self


class Config(BaseModel):
    languages: list[str] | None = None
    depth: Literal["shallow", "full"] = "full"
    cache: bool = True
    cache_dir: str = "~/.archex/cache"
    max_file_size: int = 10_000_000
    parallel: bool = True
    strict: bool = False
    delta_threshold: float = 0.5

    @model_validator(mode="after")
    def _validate_config(self) -> Config:
        if self.max_file_size <= 0:
            raise ValueError("max_file_size must be > 0")
        if self.delta_threshold < 0.0 or self.delta_threshold > 1.0:
            raise ValueError("delta_threshold must be between 0.0 and 1.0")
        return self


ChunkerName = Literal["default", "cast"]


class IndexConfig(BaseModel):
    bm25: bool = True
    vector: bool = False
    splade: bool = False
    module_prefilter: bool = False
    embedder: str | None = None
    vector_mode: VectorMode = VectorMode.RAW
    surrogate_version: str = "v1"
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.AUTO
    rerank: bool = False
    rerank_model: str | None = None
    rerank_candidate_limit: int = 4
    chunker: ChunkerName = "default"
    chunk_max_tokens: int = 500
    chunk_min_tokens: int = 50
    token_encoding: str = "cl100k_base"
    allow_remote_code: bool = False
    quantize_vectors: bool = True
    quantize_bits: int = 4
    #: Additive camelCase/PascalCase identifier-fragment splitting for the BM25
    #: symbol_name/breadcrumbs FTS columns (M17). Defaults off: measured on the
    #: self-repo identifier-fragment benchmark corpus, it regressed previously
    #: passing tasks (fragment collisions among related PascalCase symbols
    #: outweighed the intended recall gain) — see
    #: benchmarks/results/m17_identifier_bm25/DECISION.md.
    identifier_fragment_tokenization: bool = False

    @model_validator(mode="after")
    def _validate_index_config(self) -> IndexConfig:
        if not self.bm25 and not self.vector and not self.splade:
            raise ValueError("At least one of bm25, vector, or splade must be enabled")
        if self.module_prefilter and not self.bm25:
            raise ValueError("module_prefilter requires bm25 retrieval")
        if self.chunk_max_tokens <= 0:
            raise ValueError("chunk_max_tokens must be > 0")
        if self.chunk_min_tokens < 0:
            raise ValueError("chunk_min_tokens must be >= 0")
        if self.chunk_min_tokens > self.chunk_max_tokens:
            raise ValueError("chunk_min_tokens must be <= chunk_max_tokens")
        if not self.surrogate_version.strip():
            raise ValueError("surrogate_version must not be empty")
        if self.rerank_candidate_limit < 1:
            raise ValueError("rerank_candidate_limit must be at least 1")
        if self.quantize_bits not in SUPPORTED_BITS:
            raise ValueError(f"quantize_bits must be one of {SUPPORTED_BITS}")
        return self


class ScoringWeights(BaseModel):
    relevance: float = 0.80
    structural: float = 0.08
    type_coverage: float = 0.04
    cohesion: float = 0.08

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ScoringWeights:
        if self.relevance < 0 or self.structural < 0 or self.type_coverage < 0 or self.cohesion < 0:
            raise ValueError("Scoring weights must be non-negative")
        total = self.relevance + self.structural + self.type_coverage + self.cohesion
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")
        return self


# ---------------------------------------------------------------------------
# Intermediate models
# ---------------------------------------------------------------------------


class RepoMetadata(BaseModel):
    url: str | None = None
    local_path: str | None = None
    commit_hash: str | None = None
    languages: dict[str, int] = {}
    total_files: int = 0
    total_lines: int = 0


class DiscoveredFile(BaseModel):
    path: str
    absolute_path: str
    language: str
    size_bytes: int = 0


class DiscoveryResult(BaseModel):
    files: list[DiscoveredFile]
    exclusions: list[dict[str, Any]]


class Parameter(BaseModel):
    name: str
    type_annotation: str | None = None
    default: str | None = None
    required: bool = True


class SymbolRef(BaseModel):
    name: str
    qualified_name: str
    file_path: str
    kind: SymbolKind
    symbol_id: SymbolId | None = None


class Symbol(BaseModel):
    name: str
    qualified_name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    end_line: int
    visibility: Visibility = Visibility.PUBLIC
    producer_name: str | None = None
    producer_version: str | None = None
    evidence: list[str] = []
    signature: str | None = None
    docstring: str | None = None
    decorators: list[str] = []
    parent: str | None = None


class ImportStatement(BaseModel):
    module: str
    symbols: list[str] = []
    alias: str | None = None
    file_path: str
    line: int
    is_relative: bool = False
    resolved_path: str | None = None


class ChunkRange(BaseModel):
    start_line: int
    end_line: int


class ParsedFile(BaseModel):
    path: str
    language: str
    symbols: list[Symbol] = []
    imports: list[ImportStatement] = []
    chunk_ranges: list[ChunkRange] = []
    lines: int = 0
    tokens: int = 0

    producer_name: str | None = None
    producer_version: str | None = None
    evidence: list[str] = []


class FileChange(BaseModel):
    """A single file change between two commits."""

    path: str
    status: ChangeStatus
    old_path: str | None = None


class DeltaManifest(BaseModel):
    """Change manifest between two commits."""

    base_commit: str
    current_commit: str
    changes: list[FileChange] = []

    @property
    def modified_files(self) -> list[str]:
        return [c.path for c in self.changes if c.status == ChangeStatus.MODIFIED]

    @property
    def added_files(self) -> list[str]:
        return [c.path for c in self.changes if c.status == ChangeStatus.ADDED]

    @property
    def deleted_files(self) -> list[str]:
        return [c.path for c in self.changes if c.status == ChangeStatus.DELETED]

    @property
    def renamed_files(self) -> list[tuple[str, str]]:
        return [
            (c.old_path or c.path, c.path) for c in self.changes if c.status == ChangeStatus.RENAMED
        ]

    @property
    def all_affected_files(self) -> set[str]:
        paths: set[str] = set()
        for c in self.changes:
            paths.add(c.path)
            if c.old_path:
                paths.add(c.old_path)
        return paths


class DeltaMeta(BaseModel):
    """Delta indexing metrics for _meta response."""

    base_commit: str
    current_commit: str
    files_modified: int
    files_added: int
    files_deleted: int
    files_renamed: int
    files_unchanged: int
    delta_time_ms: float
    full_reindex_avoided: bool


# ---------------------------------------------------------------------------
# Index models
# ---------------------------------------------------------------------------


class IndexGenerationManifest(BaseModel):
    version: str
    created_at: str
    cache_key: str
    index_config: IndexConfig
    stores: dict[str, bool]
    chunks_count: int
    files_count: int
    edges_count: int
    excluded_files: list[dict[str, Any]]


class CodeChunk(BaseModel):
    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None = None
    symbol_kind: SymbolKind | None = None
    language: str
    imports_context: str = ""
    token_count: int = 0
    symbol_id: SymbolId | None = None
    qualified_name: str | None = None
    visibility: str | None = None
    signature: str | None = None
    docstring: str | None = None
    breadcrumbs: str = ""
    summary: str | None = None


class ChunkSurrogate(BaseModel):
    chunk_id: str
    file_path: str
    surrogate_text: str
    surrogate_version: str = "v1"


class Edge(BaseModel):
    source: str
    target: str
    kind: EdgeKind
    location: str | None = None
    confidence: EdgeConfidence = EdgeConfidence.EXTRACTED
    confidence_score: float = 1.0
    evidence: list[str] = []

    @model_validator(mode="after")
    def _validate_confidence_score(self) -> Edge:
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return self


# ---------------------------------------------------------------------------
# Output models — ArchProfile
# ---------------------------------------------------------------------------


class LanguageStats(BaseModel):
    files: int = 0
    lines: int = 0
    symbols: int = 0
    percentage: float = 0.0
    tier: LanguageTier = LanguageTier.UNKNOWN


class CodebaseStats(BaseModel):
    total_files: int = 0
    total_lines: int = 0
    languages: dict[str, LanguageStats] = {}
    module_count: int = 0
    symbol_count: int = 0
    external_dep_count: int = 0
    internal_edge_count: int = 0


class Module(BaseModel):
    name: str
    root_path: str
    files: list[str] = []
    exports: list[SymbolRef] = []
    internal_deps: list[str] = []
    external_deps: list[str] = []
    responsibility: str | None = None
    cohesion_score: float = 0.0
    file_count: int = 0
    line_count: int = 0


class PatternEvidence(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    symbol: str
    explanation: str


class DetectedPattern(BaseModel):
    name: str
    display_name: str
    confidence: float
    evidence: list[PatternEvidence] = []
    description: str
    category: PatternCategory


class Interface(BaseModel):
    symbol: SymbolRef
    signature: str
    parameters: list[Parameter] = []
    return_type: str | None = None
    docstring: str | None = None
    used_by: list[str] = []


class ArchDecision(BaseModel):
    decision: str
    alternatives: list[str] = []
    evidence: list[str] = []
    implications: list[str] = []
    source: Literal["structural"] = "structural"


class DependencyGraphSummary(BaseModel):
    nodes: int = 0
    edges: int = 0
    file_count: int = 0
    symbol_count: int = 0


class ArchProfile(BaseModel):
    repo: RepoMetadata
    module_map: list[Module] = []
    dependency_graph: DependencyGraphSummary = DependencyGraphSummary()
    pattern_catalog: list[DetectedPattern] = []
    interface_surface: list[Interface] = []
    decision_log: list[ArchDecision] = []
    stats: CodebaseStats = CodebaseStats()

    def to_dict(self) -> dict[str, Any]:
        """Return the profile as a plain dict."""
        return self.model_dump()

    def to_json(self) -> str:
        """Serialize the profile to a JSON string."""
        return self.model_dump_json(indent=2)

    def to_markdown(self) -> str:
        """Render the profile as a Markdown document."""
        lines: list[str] = []
        repo = self.repo
        name = repo.url or repo.local_path or "unknown"
        lines.append(f"# Architecture Profile: {name}")
        lines.append("")

        if repo.commit_hash:
            lines.append(f"**Commit:** `{repo.commit_hash}`")
            lines.append("")

        lines.append("## Stats")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Files | {self.stats.total_files} |")
        lines.append(f"| Lines | {self.stats.total_lines} |")
        lines.append(f"| Symbols | {self.stats.symbol_count} |")
        lines.append(f"| Modules | {self.stats.module_count} |")
        lines.append(f"| Internal edges | {self.stats.internal_edge_count} |")
        lines.append(f"| External deps | {self.stats.external_dep_count} |")
        lines.append("")

        if self.stats.languages:
            lines.append("## Languages")
            lines.append("")
            lines.append("| Language | Files | Lines | % |")
            lines.append("|----------|-------|-------|---|")
            for lang, ls in sorted(self.stats.languages.items()):
                lines.append(f"| {lang} | {ls.files} | {ls.lines} | {ls.percentage:.1f} |")
            lines.append("")

        if self.module_map:
            lines.append("## Modules")
            lines.append("")
            for mod in self.module_map:
                lines.append(f"### {mod.name}")
                lines.append(f"- **Root:** `{mod.root_path}`")
                lines.append(f"- **Files:** {mod.file_count}")
                lines.append(f"- **Lines:** {mod.line_count}")
                lines.append(f"- **Cohesion:** {mod.cohesion_score:.2f}")
                if mod.exports:
                    exports_str = ", ".join(f"`{e.name}`" for e in mod.exports[:10])
                    lines.append(f"- **Exports:** {exports_str}")
                lines.append("")

        if self.pattern_catalog:
            lines.append("## Detected Patterns")
            lines.append("")
            lines.append("| Pattern | Category | Confidence | Evidence |")
            lines.append("|---------|----------|------------|----------|")
            for pat in self.pattern_catalog:
                evidence_count = len(pat.evidence)
                lines.append(
                    f"| {pat.display_name} | {pat.category} "
                    f"| {pat.confidence:.0%} | {evidence_count} items |"
                )
            lines.append("")

        if self.interface_surface:
            lines.append("## Interface Surface")
            lines.append("")
            for iface in self.interface_surface:
                lines.append(f"- `{iface.signature}` ({iface.symbol.file_path})")
            lines.append("")

        if self.decision_log:
            lines.append("## Architecture Decisions")
            lines.append("")
            for dec in self.decision_log:
                lines.append(f"- **{dec.decision}** ({dec.source})")
                if dec.alternatives:
                    lines.append(f"  - Alternatives: {', '.join(dec.alternatives)}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output models — ContextBundle
# ---------------------------------------------------------------------------


class TypeDefinition(BaseModel):
    symbol: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    referenced_by: list[str] = []


class DependencySummary(BaseModel):
    internal: list[str] = []
    external: list[str] = []


class StructuralContext(BaseModel):
    relevant_modules: list[str] = []
    entry_points: list[str] = []
    call_chain: list[str] | None = None
    file_tree: str = ""
    file_dependency_subgraph: dict[str, list[str]] = {}


class RankedChunk(BaseModel):
    chunk: CodeChunk
    relevance_score: float = 0.0
    structural_score: float = 0.0
    type_coverage_score: float = 0.0
    cohesion_score: float = 0.0
    final_score: float = 0.0


class RetrievalMetadata(BaseModel):
    expanded_query: str | None = None
    expansion_provenance: dict[str, str] = {}

    candidates_found: int = 0
    candidates_after_expansion: int = 0
    chunks_included: int = 0
    chunks_dropped: int = 0
    strategy: str = ""
    retrieval_time_ms: float = 0.0
    assembly_time_ms: float = 0.0
    signal_agreement: float | None = None
    fusion_bm25_weight: float | None = None
    fusion_vector_weight: float | None = None
    # Expansion diagnostics
    seed_files_found: int = 0
    seed_file_paths: list[str] = []
    expanded_file_paths: list[str] = []
    expansion_eligible_seeds: int = 0
    expansion_candidates_found: int = 0
    expansion_files_added: int = 0
    expansion_zero_candidate_reason: str = ""
    expansion_import_neighbor_edges: int = 0
    expansion_same_module_candidates: int = 0
    expansion_hub_candidates: int = 0
    expansion_test_candidates_skipped: int = 0
    # Fusion gating
    fusion_skipped: bool = False
    fusion_skip_reason: str = ""
    bm25_cv: float | None = None
    splade_results: int = 0
    splade_used: bool = False
    splade_fusion_skipped: bool = False
    splade_fusion_skip_reason: str = ""
    lexical_confidence: str = ""  # "high", "medium", "low"
    vector_mode: VectorMode = VectorMode.RAW
    surrogate_version: str | None = None
    expansion_reason_counts: dict[str, int] = {}
    expanded_file_reasons: dict[str, list[str]] = {}
    index_stale: bool = False
    refresh_time_ms: float = 0.0
    refresh_files_changed: int = 0
    refresh_skipped_reason: str = ""
    chunker: ChunkerName = "default"
    index_chunk_count: int = 0
    mean_chunk_tokens: float = 0.0
    #: Named profile (fast/balanced/deep) resolved for this query, when one was
    #: used to construct the effective IndexConfig. None when the caller
    #: supplied an explicit IndexConfig or the auto-loaded repo config as-is.
    retrieval_profile: str | None = None


class ContextReceiptTokenBudget(BaseModel):
    requested: int
    consumed: int


class CompressionMetadata(BaseModel):
    """Optional provenance for a post-retrieval compression step on one region.

    Absent on uncompressed rows. When present, the original region stays exactly
    retrievable via ``fetch_original_handle`` + ``original_content_hash`` so
    compression never hides editable evidence. ``compression_ratio`` is the
    fraction of original tokens retained (1.0 means nothing was removed); a
    ``passthrough_required`` row keeps the original content and hashes unchanged.
    """

    compression_mode: CompressionMode
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    original_content_hash: str
    compressed_content_hash: str
    fetch_original_handle: str
    compression_loss_risk: CompressionLossRisk = CompressionLossRisk.NONE

    @model_validator(mode="after")
    def _validate_compression(self) -> CompressionMetadata:
        if self.original_tokens < 0 or self.compressed_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.compression_ratio < 0.0:
            raise ValueError("compression_ratio must be non-negative")
        if (
            self.compression_mode is CompressionMode.PASSTHROUGH_REQUIRED
            and self.compressed_content_hash != self.original_content_hash
        ):
            raise ValueError("passthrough_required rows must keep the original content hash")
        return self

    @property
    def is_compressed(self) -> bool:
        """True when the displayed content differs from the original region."""
        return self.compressed_content_hash != self.original_content_hash


class ContextReceiptItem(BaseModel):
    handle: str
    file_path: str
    start_line: int
    end_line: int
    content_hash: str
    symbols: list[str] = []
    score: float = 0.0
    reason_codes: list[str] = []
    compression: CompressionMetadata | None = None


class ContextReceiptEdge(BaseModel):
    source: str
    target: str
    kind: EdgeKind
    source_path: str | None = None
    target_path: str | None = None
    confidence: EdgeConfidence | None = None
    confidence_score: float | None = None
    evidence: list[str] = []
    reason: ContextOmittedEdgeReason | None = None

    @model_validator(mode="after")
    def _validate_confidence_score(self) -> ContextReceiptEdge:
        if self.confidence_score is not None and not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return self


class ContextSkippedCandidate(BaseModel):
    file_path: str
    reason: ContextSkippedReason
    handle: str | None = None
    symbol: str | None = None
    score: float = 0.0
    detail: str = ""


class ContextReceipt(BaseModel):
    query: str
    expanded_query: str | None = None
    expansion_provenance: dict[str, str] = {}
    token_budget: ContextReceiptTokenBudget
    index_revision: str
    freshness: ContextFreshness = ContextFreshness.UNKNOWN
    freshness_checked_at: str | None = None
    index_fresh_at: str | None = None
    watch_fresh_at: str | None = None
    returned_context: list[ContextReceiptItem] = []
    included_edges: list[ContextReceiptEdge] = []
    omitted_edges: list[ContextReceiptEdge] = []
    skipped_candidates: list[ContextSkippedCandidate] = []
    returned_total: int = 0
    skipped_total: int = 0
    included_edges_total: int = 0
    omitted_edges_total: int = 0
    context_complete: ContextCompletenessStatus = ContextCompletenessStatus.UNKNOWN
    context_complete_reason: ContextCompletenessReason = ContextCompletenessReason.UNKNOWN
    recommended_next_action: ContextRecommendedAction = ContextRecommendedAction.MANUAL_REVIEW


class ContextBundle(BaseModel):
    query: str
    chunks: list[RankedChunk] = []
    structural_context: StructuralContext = StructuralContext()
    type_definitions: list[TypeDefinition] = []
    dependency_summary: DependencySummary = DependencySummary()
    token_count: int = 0
    token_budget: int = 0
    truncated: bool = False
    retrieval_metadata: RetrievalMetadata = RetrievalMetadata()
    receipt: ContextReceipt | None = None

    def to_prompt(self, format: str = "xml", *, full: bool = False) -> str:
        """Render the context bundle as an LLM prompt string.

        `full` affects `format="json"` and `format="toon"`: by default
        these renderers drop unset/empty chunk fields; `full=True`
        restores every field. `format="toon"` requires the optional
        `toons` package (`uv add 'archex[toon]'`).
        """
        from archex.serve.renderers.json import render_json
        from archex.serve.renderers.markdown import render_markdown
        from archex.serve.renderers.xml import render_xml

        if format == "xml":
            return render_xml(self)
        if format == "markdown":
            return render_markdown(self)
        if format == "json":
            return render_json(self, full=full)
        if format == "toon":
            from archex.serve.renderers.toon import render_toon

            return render_toon(self, full=full)
        raise ValueError(f"Unknown format: {format}")

    def to_dict(self) -> dict[str, Any]:
        """Return the bundle as a plain dict."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Output models — Comparison
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Output models — Precision Symbol Tools (Tier 1)
# ---------------------------------------------------------------------------


class FileTreeEntry(BaseModel):
    """Single entry in an annotated repository file tree."""

    path: str
    language: str | None = None
    lines: int = 0
    symbol_count: int = 0
    is_directory: bool = False
    children: list[FileTreeEntry] = []


class FileTree(BaseModel):
    """Annotated file tree of a repository."""

    root: str
    entries: list[FileTreeEntry]
    total_files: int
    languages: dict[str, int]


class SymbolOutline(BaseModel):
    """Symbol metadata without source code — used in file outlines."""

    symbol_id: str
    name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    end_line: int
    signature: str | None = None
    visibility: Visibility = Visibility.PUBLIC
    docstring: str | None = None
    children: list[SymbolOutline] = []


class FileOutline(BaseModel):
    """Symbol hierarchy, structural ranges, and local references for a single file."""

    file_path: str
    language: str
    lines: int
    symbols: list[SymbolOutline]
    token_count_raw: int
    outline_ranges: list[ChunkRange] = []
    references: list[ImportStatement] = []


class SymbolMatch(BaseModel):
    """Search result for symbol search — metadata only."""

    symbol_id: str
    name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    signature: str | None = None
    visibility: Visibility = Visibility.PUBLIC
    relevance_score: float = 0.0


class SymbolSource(BaseModel):
    """Full symbol with source code — returned by get_symbol."""

    symbol_id: str
    name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    end_line: int
    signature: str | None = None
    visibility: Visibility = Visibility.PUBLIC
    docstring: str | None = None
    source: str
    imports_context: str = ""
    token_count: int = 0
    lsap_enrichment: LSAPEnrichment | None = None


class TokenMeta(BaseModel):
    """Token efficiency metrics included in every tool response."""

    tokens_returned: int
    tokens_raw_equivalent: int
    savings_pct: float
    strategy: str
    tool_name: str
    cached: bool = False
    index_time_ms: float = 0.0
    query_time_ms: float = 0.0
    delta: DeltaMeta | None = None


@dataclass
class PipelineTiming:
    """Per-phase timing breakdown populated by API functions."""

    acquire_ms: float = 0.0
    parse_ms: float = 0.0
    index_ms: float = 0.0
    search_ms: float = 0.0
    assemble_ms: float = 0.0
    total_ms: float = 0.0
    cached: bool = False
    delta_ms: float = 0.0
    delta_meta: DeltaMeta | None = None
    delta_attempted: bool = False
    delta_succeeded: bool = False
    parse_failure_count: int = 0
    vector_used: bool = False
    vector_build_ms: float = 0.0
    vector_index_ms: float = 0.0
    strategy: str = ""  # "full", "cached", "delta"


# ---------------------------------------------------------------------------
# Output models — Comparison
# ---------------------------------------------------------------------------


class DimensionComparison(BaseModel):
    dimension: str
    repo_a_approach: str
    repo_b_approach: str
    evidence_a: list[str] = []
    evidence_b: list[str] = []
    trade_offs: list[str] = []


class ComparisonResult(BaseModel):
    repo_a: RepoMetadata
    repo_b: RepoMetadata
    dimensions: list[DimensionComparison] = []
    summary: str = ""
