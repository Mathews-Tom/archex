"""All Pydantic data models for archex: enums, input, intermediate, index, and output types."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel

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


class PatternCategory(StrEnum):
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    CREATIONAL = "creational"


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class RepoSource(BaseModel):
    url: str | None = None
    local_path: str | None = None
    target: str | None = None
    commit: str | None = None
    sparse: bool = False


class Config(BaseModel):
    languages: list[str] | None = None
    depth: Literal["shallow", "full"] = "full"
    enrich: bool = False
    provider: str | None = None
    provider_config: dict[str, Any] = {}
    cache: bool = True
    cache_dir: str = "~/.archex/cache"


class IndexConfig(BaseModel):
    bm25: bool = True
    vector: bool = False
    embedder: str | None = None
    chunk_max_tokens: int = 500
    chunk_min_tokens: int = 50
    token_encoding: str = "cl100k_base"


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


class Symbol(BaseModel):
    name: str
    qualified_name: str
    kind: SymbolKind
    file_path: str
    start_line: int
    end_line: int
    visibility: Visibility = Visibility.PUBLIC
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


class ParsedFile(BaseModel):
    path: str
    language: str
    symbols: list[Symbol] = []
    imports: list[ImportStatement] = []
    lines: int = 0
    tokens: int = 0


# ---------------------------------------------------------------------------
# Index models
# ---------------------------------------------------------------------------


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
    module: str | None = None


class Edge(BaseModel):
    source: str
    target: str
    kind: EdgeKind
    location: str | None = None


# ---------------------------------------------------------------------------
# Output models — ArchProfile
# ---------------------------------------------------------------------------


class LanguageStats(BaseModel):
    files: int = 0
    lines: int = 0
    symbols: int = 0
    percentage: float = 0.0


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
    source: Literal["structural", "llm_inferred"] = "structural"


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
        # TODO: Implement in Phase 3
        raise NotImplementedError

    def to_markdown(self) -> str:
        """Render the profile as a Markdown document."""
        # TODO: Implement in Phase 3
        raise NotImplementedError

    def to_json(self) -> str:
        """Serialize the profile to a JSON string."""
        # TODO: Implement in Phase 3
        raise NotImplementedError


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
    final_score: float = 0.0


class RetrievalMetadata(BaseModel):
    candidates_found: int = 0
    candidates_after_expansion: int = 0
    chunks_included: int = 0
    chunks_dropped: int = 0
    strategy: str = ""
    retrieval_time_ms: float = 0.0
    assembly_time_ms: float = 0.0


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

    def to_prompt(self, format: str = "xml") -> str:
        """Render the context bundle as an LLM prompt string."""
        # TODO: Implement in Phase 2
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return the bundle as a plain dict."""
        # TODO: Implement in Phase 2
        raise NotImplementedError


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
