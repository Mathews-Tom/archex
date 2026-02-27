# archex — System Design

> Complete architecture, data models, workflows, and technical decisions.

---

## 1. Architecture

### 1.1 Pipeline Overview

archex operates as a five-stage pipeline. Each stage is independently composable and produces inspectable intermediate outputs.

```mermaid
graph TD
    subgraph S1["① Acquire"]
        Clone["Git Clone<br/>(shallow / sparse)"]
        Discover["File Discovery<br/>Language detection<br/>Ignore patterns"]
    end

    subgraph S2["② Parse"]
        AST["AST Extraction<br/>(tree-sitter)"]
        Symbols["Symbol Extraction<br/>Functions, classes,<br/>types, exports"]
        Imports["Import Resolution<br/>Module → File mapping"]
    end

    subgraph S3["③ Index"]
        Chunk["AST-Aware<br/>Chunking"]
        Graph["Dependency Graph<br/>(NetworkX)"]
        BM25["BM25 Index<br/>(SQLite FTS5)"]
        Vec["Vector Index<br/>(ONNX + Nomic)"]
    end

    subgraph S4["④ Analyze"]
        Modules["Module Boundary<br/>Detection<br/>(Louvain)"]
        Patterns["Pattern<br/>Recognition<br/>(Rule-based)"]
        Interfaces["Interface<br/>Surface<br/>Extraction"]
        Decisions["Trade-off<br/>Inference<br/>(LLM opt-in)"]
    end

    subgraph S5["⑤ Serve"]
        AP["ArchProfile"]
        CB["ContextBundle"]
        Cmp["ComparisonResult"]
    end

    Clone --> Discover --> AST --> Symbols --> Imports
    Imports --> Chunk
    Imports --> Graph
    Chunk --> BM25
    Chunk --> Vec
    Graph --> Modules --> Patterns
    Symbols --> Interfaces
    Patterns --> Decisions

    Modules --> AP
    Interfaces --> AP
    Decisions --> AP
    Graph --> AP

    BM25 --> CB
    Vec --> CB
    Graph --> CB
    Interfaces --> CB

    AP --> Cmp
    CB --> Cmp

    style S1 fill:#e8f0fe,stroke:#1a73e8
    style S2 fill:#e8f5e9,stroke:#34a853
    style S3 fill:#fff8e1,stroke:#f9ab00
    style S4 fill:#fce4ec,stroke:#e91e63
    style S5 fill:#f3e5f5,stroke:#9c27b0
```

### 1.2 Package Layout

```text
archex/
├── __init__.py             # Public re-exports: analyze, query, compare
├── api.py                  # Top-level public API functions
├── config.py               # Configuration dataclass + defaults
├── models.py               # All data models (Pydantic)
├── cache.py                # Index cache management, TTL, cleanup
├── exceptions.py           # Structured exception hierarchy
│
├── acquire/                # ① Source Acquisition
│   ├── __init__.py
│   ├── git.py              # clone_repo(), shallow_clone(), sparse_checkout()
│   ├── local.py            # open_local(), validate_repo_path()
│   └── discovery.py        # discover_files(), detect_languages(), apply_ignores()
│
├── parse/                  # ② AST Parsing & Symbol Extraction
│   ├── __init__.py
│   ├── engine.py           # TreeSitterEngine: parse orchestration
│   ├── symbols.py          # extract_symbols() → list[Symbol]
│   ├── imports.py          # parse_imports(), resolve_imports()
│   └── adapters/           # Language-specific adapters
│       ├── __init__.py     # Adapter registry
│       ├── base.py         # LanguageAdapter protocol
│       ├── python.py       # Python adapter
│       ├── typescript.py   # TypeScript/JavaScript adapter
│       ├── go.py           # Go adapter
│       └── rust.py         # Rust adapter
│
├── index/                  # ③ Indexing & Chunking
│   ├── __init__.py
│   ├── chunker.py          # ASTChunker: syntax-boundary chunking
│   ├── graph.py            # DependencyGraph: NetworkX wrapper
│   ├── bm25.py             # BM25Index: SQLite FTS5 wrapper
│   ├── vector.py           # VectorIndex: embedding + cosine similarity
│   ├── store.py            # IndexStore: SQLite persistence for all indexes
│   └── embeddings/         # Embedding model backends
│       ├── __init__.py
│       ├── base.py         # Embedder protocol
│       ├── nomic.py        # NomicCodeEmbedder (ONNX, default)
│       ├── sentence_tf.py  # SentenceTransformerEmbedder (torch)
│       └── api.py          # APIEmbedder (Voyage, OpenAI, etc.)
│
├── analyze/                # ④ Structural Analysis
│   ├── __init__.py
│   ├── modules.py          # detect_modules() via Louvain community detection
│   ├── patterns.py         # detect_patterns() via rule-based matching
│   ├── interfaces.py       # extract_interfaces() → public API surface
│   └── decisions.py        # infer_decisions() → trade-off analysis (LLM opt-in)
│
├── serve/                  # ⑤ Output Assembly
│   ├── __init__.py
│   ├── profile.py          # build_profile() → ArchProfile
│   ├── context.py          # assemble_context() → ContextBundle
│   ├── compare.py          # compare_repos() → ComparisonResult
│   └── renderers/          # Output format renderers
│       ├── __init__.py
│       ├── xml.py          # XML-tagged prompt format (default)
│       ├── markdown.py     # Markdown format
│       └── json.py         # JSON format
│
├── providers/              # LLM Provider Abstraction
│   ├── __init__.py
│   ├── base.py             # LLMProvider protocol
│   ├── anthropic.py        # Anthropic provider
│   ├── openai.py           # OpenAI provider
│   └── openrouter.py       # OpenRouter provider
│
├── integrations/           # Framework Integrations (optional)
│   ├── __init__.py
│   ├── langchain.py        # ArchexRetriever (LangChain compatible)
│   ├── llamaindex.py       # ArchexQueryEngine (LlamaIndex compatible)
│   └── mcp.py              # MCP tool definitions
│
└── cli/                    # CLI Entry Point
    ├── __init__.py
    ├── main.py             # click group definition
    ├── analyze_cmd.py      # archex analyze
    ├── query_cmd.py        # archex query
    ├── compare_cmd.py      # archex compare
    └── cache_cmd.py        # archex cache
```

### 1.3 Dependency Architecture

```mermaid
graph BT
    subgraph Core["Core (always installed)"]
        Models["models.py"]
        Config["config.py"]
        Acquire["acquire/"]
        Parse["parse/"]
        IndexMod["index/<br/>(BM25 only)"]
        Analyze["analyze/"]
        Serve["serve/"]
        CLI["cli/"]
    end

    subgraph External["External Dependencies"]
        TS["tree-sitter"]
        TK["tiktoken"]
        PD["pydantic"]
        NX["networkx"]
        CK["click"]
        SQ["sqlite3 (stdlib)"]
    end

    subgraph Optional["Optional Extras"]
        ONNX["onnxruntime<br/>(archex[vector])"]
        Torch["sentence-transformers<br/>(archex[vector-torch])"]
        Voyage["voyageai<br/>(archex[voyage])"]
        OAI["openai<br/>(archex[openai])"]
        Anth["anthropic<br/>(archex[anthropic])"]
    end

    Parse --> TS
    Serve --> TK
    Models --> PD
    IndexMod --> NX
    CLI --> CK
    IndexMod --> SQ

    style Core fill:#e8f0fe,stroke:#1a73e8
    style Optional fill:#fff8e1,stroke:#f9ab00
```

---

## 2. Data Models

All models use Pydantic v2 for validation, serialization, and schema generation.

### 2.1 Core Input Models

```python
class RepoSource:
    """Input source specification."""
    url: str | None              # Git URL (https or ssh)
    local_path: Path | None      # Local directory path
    target: str | None           # Sub-path for monorepo scoping
    commit: str | None           # Pin to specific commit (default: HEAD)
    sparse: bool = False         # Use sparse checkout for monorepos

class Config:
    """Library-level configuration."""
    languages: list[str] | None = None          # Restrict to specific languages
    depth: Literal["shallow", "full"] = "full"  # Analysis depth
    enrich: bool = False                        # Enable LLM enrichment
    provider: str | None = None                 # LLM provider name
    provider_config: dict | None = None         # Provider-specific config
    cache: bool = True                          # Enable index caching
    cache_dir: Path = Path.home() / ".archex" / "cache"

class IndexConfig:
    """Index construction configuration."""
    bm25: bool = True                           # Enable BM25 keyword index
    vector: bool = False                        # Enable vector embedding index
    embedder: Embedder | None = None            # Custom embedding model
    chunk_max_tokens: int = 500                 # Max tokens per chunk
    chunk_min_tokens: int = 50                  # Min tokens (merge threshold)
    token_encoding: str = "cl100k_base"         # tiktoken encoding
```

### 2.2 Intermediate Models (Pipeline Outputs)

```python
class RepoMetadata:
    """Metadata about a cloned/opened repository."""
    url: str | None
    local_path: Path
    commit_hash: str
    languages: dict[str, int]    # language → file count
    total_files: int
    total_lines: int

class ParsedFile:
    """A single file after AST parsing."""
    path: str                    # Relative path within repo
    language: str                # Detected language
    symbols: list[Symbol]        # Extracted symbols
    imports: list[ImportStatement]
    lines: int
    tokens: int                  # Estimated token count

class Symbol:
    """A named code entity extracted from the AST."""
    name: str                    # Symbol name (e.g., "ConnectionPool")
    qualified_name: str          # Fully qualified (e.g., "httpx._pool.ConnectionPool")
    kind: SymbolKind             # function | class | type | variable | constant | interface
    file_path: str
    start_line: int
    end_line: int
    visibility: Visibility       # public | internal | private
    signature: str | None        # Function signature or class declaration line
    docstring: str | None
    decorators: list[str]        # @staticmethod, @dataclass, etc.
    parent: str | None           # Enclosing class/module qualified name

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

class ImportStatement:
    """A parsed import statement."""
    module: str                  # The imported module/package
    symbols: list[str]           # Specific imported symbols (empty = whole module)
    alias: str | None            # Import alias (as)
    file_path: str               # File containing this import
    line: int
    is_relative: bool            # Relative import (Python: from . import x)
    resolved_path: str | None    # Resolved file path within repo (None if external)
```

### 2.3 Index Models

```python
class CodeChunk:
    """A syntax-aligned unit of source code."""
    id: str                      # Unique chunk identifier
    content: str                 # The actual source code
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str | None      # Enclosing symbol name
    symbol_kind: SymbolKind | None
    language: str
    imports_context: str         # Relevant import lines prepended
    token_count: int
    module: str | None           # Assigned module name

class Edge:
    """A directed dependency edge."""
    source: str                  # Qualified symbol name or file path
    target: str                  # Qualified symbol name or file path
    kind: EdgeKind               # imports | calls | inherits | implements | uses_type
    location: str | None         # source_file:line_number

class EdgeKind(StrEnum):
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    USES_TYPE = "uses_type"
    EXPORTS = "exports"
```

### 2.4 Output Models

#### ArchProfile (Human-Facing)

```python
class ArchProfile:
    """Complete architectural intelligence for a codebase."""
    repo: RepoMetadata
    module_map: list[Module]
    dependency_graph: DependencyGraphSummary  # Serializable summary
    pattern_catalog: list[DetectedPattern]
    interface_surface: list[Interface]
    decision_log: list[ArchDecision]
    stats: CodebaseStats

    def to_dict(self) -> dict: ...
    def to_markdown(self) -> str: ...
    def to_json(self) -> str: ...

class Module:
    """A detected logical module within the codebase."""
    name: str
    root_path: str
    files: list[str]
    exports: list[SymbolRef]     # References to public symbols
    internal_deps: list[str]     # Dependencies on other detected modules
    external_deps: list[str]     # Third-party package dependencies
    responsibility: str | None   # LLM-inferred description (opt-in)
    cohesion_score: float        # Intra-module coupling density (0-1)
    file_count: int
    line_count: int

class DetectedPattern:
    """An architectural pattern found in the codebase."""
    name: str                    # e.g., "middleware_chain"
    display_name: str            # e.g., "Middleware Chain"
    confidence: float            # 0.0 - 1.0
    evidence: list[PatternEvidence]
    description: str
    category: PatternCategory    # structural | behavioral | creational

class PatternEvidence:
    """Supporting evidence for a pattern detection."""
    file_path: str
    start_line: int
    end_line: int
    symbol: str
    explanation: str             # Why this code supports the pattern

class PatternCategory(StrEnum):
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    CREATIONAL = "creational"

class Interface:
    """A public-facing API contract."""
    symbol: SymbolRef
    signature: str
    parameters: list[Parameter]
    return_type: str | None
    docstring: str | None
    used_by: list[str]           # Internal consumers

class ArchDecision:
    """An inferred architectural trade-off."""
    decision: str
    alternatives: list[str]
    evidence: list[str]          # File paths + code locations
    implications: list[str]
    source: Literal["structural", "llm_inferred"]

class CodebaseStats:
    total_files: int
    total_lines: int
    languages: dict[str, LanguageStats]
    module_count: int
    symbol_count: int
    external_dep_count: int
    internal_edge_count: int

class LanguageStats:
    files: int
    lines: int
    symbols: int
    percentage: float            # Percentage of total lines
```

#### ContextBundle (Agent-Facing)

```python
class ContextBundle:
    """Token-budget-aware code context for agent consumption."""
    query: str
    chunks: list[RankedChunk]
    structural_context: StructuralContext
    type_definitions: list[TypeDefinition]
    dependency_summary: DependencySummary
    token_count: int
    token_budget: int
    truncated: bool
    retrieval_metadata: RetrievalMetadata

    def to_prompt(self, format: str = "xml") -> str: ...
    def to_dict(self) -> dict: ...

class RankedChunk:
    """A code chunk with retrieval scoring."""
    chunk: CodeChunk
    relevance_score: float       # BM25 / vector similarity score
    structural_score: float      # PageRank centrality
    type_coverage_score: float   # How many referenced types are included
    final_score: float           # Weighted composite

class StructuralContext:
    """Lightweight structural metadata for agent orientation."""
    relevant_modules: list[str]
    entry_points: list[str]
    call_chain: list[str] | None
    file_tree: str               # ASCII file tree of relevant files
    file_dependency_subgraph: dict[str, list[str]]

class TypeDefinition:
    """A type/interface definition included for reference."""
    symbol: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    referenced_by: list[str]     # Chunks that reference this type

class DependencySummary:
    """Summary of dependencies relevant to the query."""
    internal: list[str]          # Internal modules/symbols involved
    external: list[str]          # Third-party packages involved

class RetrievalMetadata:
    """Diagnostics about the retrieval process."""
    candidates_found: int
    candidates_after_expansion: int
    chunks_included: int
    chunks_dropped: int
    strategy: str                # "bm25" | "vector" | "hybrid" | "graph"
    retrieval_time_ms: float
    assembly_time_ms: float
```

#### ComparisonResult

```python
class ComparisonResult:
    """Cross-repo architectural comparison."""
    repo_a: RepoMetadata
    repo_b: RepoMetadata
    dimensions: list[DimensionComparison]
    summary: str                 # LLM-generated summary

class DimensionComparison:
    """Comparison along a single architectural dimension."""
    dimension: str               # e.g., "error_handling"
    repo_a_approach: str         # Description of repo A's approach
    repo_b_approach: str         # Description of repo B's approach
    evidence_a: list[str]        # File paths + code refs from repo A
    evidence_b: list[str]        # File paths + code refs from repo B
    trade_offs: list[str]        # Key differences and their implications
```

---

## 3. Stage-by-Stage Design

### 3.1 Stage ① — Acquire

**Responsibility:** Clone or open a repository, enumerate source files, detect languages, apply ignore rules.

```mermaid
graph TD
    Input["RepoSource<br/>(URL or local path)"]
    Input --> IsLocal{"Local path?"}
    IsLocal -->|Yes| Validate["Validate path<br/>exists + is git repo"]
    IsLocal -->|No| HasTarget{"Has target<br/>(monorepo)?"}
    HasTarget -->|No| ShallowClone["git clone --depth 1"]
    HasTarget -->|Yes + sparse| SparseClone["git clone --filter=blob:none<br/>--sparse + sparse-checkout"]
    HasTarget -->|Yes + !sparse| ShallowClone

    Validate --> Discover
    ShallowClone --> Discover
    SparseClone --> Discover

    Discover["discover_files()"]
    Discover --> DetectLang["Detect language<br/>per file (extension +<br/>tree-sitter probe)"]
    DetectLang --> ApplyIgnore["Apply ignore rules<br/>.gitignore + defaults"]
    ApplyIgnore --> Output["list[DiscoveredFile]"]
```

**Default ignore rules** (applied in addition to `.gitignore`):

```python
DEFAULT_IGNORES = [
    "node_modules/", ".git/", "__pycache__/", ".venv/", "venv/",
    "dist/", "build/", ".next/", ".nuxt/", "target/",
    "*.min.js", "*.min.css", "*.map", "*.lock",
    "package-lock.json", "bun.lock", "yarn.lock", "pnpm-lock.yaml",
    "*.pb.go", "*_generated.*", "*.g.dart",      # Generated code
    "vendor/", "third_party/",                     # Vendored deps
    "*.svg", "*.png", "*.jpg", "*.gif", "*.ico",  # Binary/media
    "*.wasm", "*.pyc", "*.so", "*.dylib",          # Compiled artifacts
]
```

**Monorepo detection:**

```python
MONOREPO_SIGNALS = {
    "turbo.json": "turborepo",
    "nx.json": "nx",
    "lerna.json": "lerna",
    "pnpm-workspace.yaml": "pnpm",
    "pants.toml": "pants",
    "BUILD": "bazel",
    "WORKSPACE": "bazel",
}

def detect_monorepo(repo_path: Path) -> MonorepoInfo | None:
    """Detect monorepo structure and sub-package locations."""
    # 1. Check for known tool configs
    # 2. Check for workspace declarations in package.json / Cargo.toml / go.work
    # 3. Count package manifests at different directory levels
    # Returns MonorepoInfo with tool, workspace_root, sub_packages
```

### 3.2 Stage ② — Parse

**Responsibility:** Parse source files into ASTs, extract symbols and imports, resolve import paths to files within the repo.

```mermaid
graph TD
    Files["list[DiscoveredFile]"]
    Files --> Group["Group by language"]
    Group --> GetAdapter["Get LanguageAdapter<br/>for each group"]
    GetAdapter --> ParseAST["Parse each file<br/>with tree-sitter"]
    ParseAST --> Extract["adapter.extract_symbols()<br/>adapter.parse_imports()"]
    Extract --> Resolve["resolve_imports()<br/>Map import → file path"]
    Resolve --> Output["list[ParsedFile]"]
```

**LanguageAdapter Protocol:**

```python
class LanguageAdapter(Protocol):
    language_id: str
    file_extensions: list[str]
    tree_sitter_name: str

    def extract_symbols(self, tree: Tree, source: bytes, file_path: str) -> list[Symbol]:
        """Extract all named symbols from the AST."""
        ...

    def parse_imports(self, tree: Tree, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract and parse all import statements."""
        ...

    def resolve_import(
        self, imp: ImportStatement, file_map: dict[str, Path]
    ) -> str | None:
        """Resolve an import to a file path within the repo. None if external."""
        ...

    def detect_entry_points(self, files: list[ParsedFile]) -> list[str]:
        """Identify application entry points."""
        ...

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Determine if a symbol is public, internal, or private."""
        ...
```

**Python adapter — symbol extraction via tree-sitter queries:**

```python
# tree-sitter query patterns for Python symbol extraction
PYTHON_QUERIES = {
    "functions": """
        (function_definition
            name: (identifier) @name
            parameters: (parameters) @params
            return_type: (type)? @return_type
            body: (block) @body
        ) @definition
    """,
    "classes": """
        (class_definition
            name: (identifier) @name
            superclasses: (argument_list)? @bases
            body: (block) @body
        ) @definition
    """,
    "imports": """
        [
            (import_statement) @import
            (import_from_statement) @import
        ]
    """,
    "type_aliases": """
        (type_alias_statement
            name: (type) @name
            value: (type) @value
        ) @definition
    """,
}
```

**Import resolution strategy (Python-specific):**

```text
from httpx._pool import ConnectionPool
  → module = "httpx._pool"
  → symbols = ["ConnectionPool"]
  → Check: is "httpx" a directory in the repo?
    → Yes → resolved_path = "httpx/_pool.py"
    → No  → resolved_path = None (external dependency)

from . import _pool
  → Relative import from current package
  → resolved_path = resolve relative to importing file's directory

import asyncio
  → stdlib → resolved_path = None (external)
```

### 3.3 Stage ③ — Index

**Responsibility:** Chunk parsed files into syntax-aligned units, build dependency graph, construct BM25 and optional vector indexes, persist to SQLite.

#### 3.3.1 AST-Aware Chunking

```mermaid
graph TD
    PF["ParsedFile"]
    PF --> Walk["Walk top-level<br/>AST nodes"]
    Walk --> ForEach["For each node"]
    ForEach --> CountTokens["Count tokens<br/>(tiktoken)"]
    CountTokens --> Check{"tokens ≤<br/>max_chunk?"}
    Check -->|Yes| EmitChunk["Emit as CodeChunk"]
    Check -->|No| HasChildren{"Has child<br/>nodes?"}
    HasChildren -->|Yes| Recurse["Recurse into<br/>children"]
    HasChildren -->|No| ForceSplit["Split at<br/>line boundary"]
    Recurse --> ForEach
    ForceSplit --> EmitChunk
    EmitChunk --> Annotate["Annotate:<br/>• symbol name/kind<br/>• import context<br/>• module membership"]
    Annotate --> MergeCheck{"Adjacent chunks<br/>< min_chunk?"}
    MergeCheck -->|Yes| Merge["Merge with<br/>neighbor chunk"]
    MergeCheck -->|No| Final["Final CodeChunk"]
    Merge --> Final
```

**Chunking rules:**

| Rule                    | Behavior                                                                                                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Boundary alignment**  | Chunks always start/end at AST node boundaries (function, class, type def, top-level statement)                                                     |
| **Import prepending**   | Each chunk includes relevant import lines as prefix (imports that resolve to symbols used in the chunk)                                             |
| **Class handling**      | Small classes (≤ max_chunk): one chunk for the whole class. Large classes: one chunk per method, with the class header + `__init__` always included |
| **Decorator inclusion** | Decorators are always included with their target symbol                                                                                             |
| **Comment association** | Leading comment blocks (docstrings, header comments) stay with their associated symbol                                                              |
| **Merge threshold**     | Adjacent chunks below `min_chunk` tokens are merged, respecting a combined max of `max_chunk * 1.5`                                                 |

#### 3.3.2 Dependency Graph Construction

```python
class DependencyGraph:
    """Multi-level dependency graph backed by NetworkX."""

    def __init__(self):
        self._file_graph = nx.DiGraph()      # File A → File B (imports)
        self._symbol_graph = nx.DiGraph()    # Symbol A → Symbol B (calls, inherits, etc.)

    @classmethod
    def from_parsed_files(cls, files: list[ParsedFile]) -> "DependencyGraph":
        """Construct graph from parsed files with resolved imports."""
        graph = cls()
        for f in files:
            graph._file_graph.add_node(f.path)
            for imp in f.imports:
                if imp.resolved_path:
                    graph._file_graph.add_edge(
                        f.path, imp.resolved_path,
                        kind="imports", symbols=imp.symbols
                    )
            for sym in f.symbols:
                graph._symbol_graph.add_node(sym.qualified_name, file=f.path)
        # Second pass: resolve symbol-level edges
        # (calls, inherits, uses_type from AST analysis)
        return graph

    def detect_modules(self, directory_prior: dict[str, str] | None = None) -> list[Module]:
        """Louvain community detection with directory bias."""
        ...

    def subgraph_for_files(self, files: list[str]) -> "DependencyGraph":
        """Extract subgraph containing only specified files + their edges."""
        ...

    def neighborhood(self, symbol: str, hops: int = 1) -> set[str]:
        """Return all symbols within N hops (both directions)."""
        predecessors = set(nx.ancestors(self._symbol_graph, symbol)) if hops > 0 else set()
        successors = set(nx.descendants(self._symbol_graph, symbol)) if hops > 0 else set()
        # Filter to within hop limit via BFS
        ...

    def structural_centrality(self) -> dict[str, float]:
        """PageRank-based importance scores for all symbols."""
        return nx.pagerank(self._symbol_graph, alpha=0.85)

    def to_sqlite(self, conn: sqlite3.Connection) -> None: ...

    @classmethod
    def from_sqlite(cls, conn: sqlite3.Connection) -> "DependencyGraph": ...
```

#### 3.3.3 BM25 Index

Uses SQLite FTS5 (full-text search) — zero external dependencies, built into Python's `sqlite3`.

```sql
-- Schema
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id,
    content,
    symbol_name,
    file_path,
    tokenize='porter unicode61'
);

-- Query (BM25 ranking is built into FTS5)
SELECT chunk_id, rank
FROM chunks_fts
WHERE chunks_fts MATCH ?
ORDER BY rank
LIMIT ?;
```

#### 3.3.4 Vector Index

Optional, activated via `IndexConfig(vector=True)`.

```python
class VectorIndex:
    """Simple embedding-based similarity search."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.vectors: np.ndarray | None = None  # (n_chunks, dim)
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[CodeChunk]) -> None:
        texts = [c.imports_context + "\n" + c.content for c in chunks]
        self.vectors = self.embedder.embed(texts)
        self.chunk_ids = [c.id for c in chunks]

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        query_vec = self.embedder.embed([query])[0]
        similarities = self.vectors @ query_vec  # Cosine sim (vectors are normalized)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.chunk_ids[i], float(similarities[i])) for i in top_indices]

    def save(self, path: Path) -> None:
        np.save(path / "vectors.npy", self.vectors)
        (path / "chunk_ids.json").write_text(json.dumps(self.chunk_ids))

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "VectorIndex": ...
```

#### 3.3.5 Index Persistence (SQLite)

Single SQLite database per indexed repo:

```sql
-- Chunks table
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    symbol_name TEXT,
    symbol_kind TEXT,
    language TEXT NOT NULL,
    imports_context TEXT,
    token_count INTEGER NOT NULL,
    module TEXT
);

-- Symbols table
CREATE TABLE symbols (
    qualified_name TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    visibility TEXT,
    signature TEXT,
    docstring TEXT,
    parent TEXT
);

-- Edges table
CREATE TABLE edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    kind TEXT NOT NULL,
    location TEXT,
    PRIMARY KEY (source, target, kind)
);

-- Metadata table
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Keys: commit_hash, indexed_at, config_hash, languages, stats

-- FTS5 index (see BM25 section above)
```

### 3.4 Stage ④ — Analyze

**Responsibility:** Detect module boundaries, recognize architectural patterns, extract public interface surface, infer trade-offs.

#### 3.4.1 Module Boundary Detection

```mermaid
graph TD
    FG["File-level<br/>dependency graph"] --> Undirect["Convert to<br/>undirected graph"]
    Undirect --> Weight["Weight edges by<br/>import density"]
    Weight --> DirPrior["Apply directory<br/>structure as prior"]
    DirPrior --> Louvain["Louvain community<br/>detection"]
    Louvain --> Communities["Raw communities<br/>(sets of files)"]
    Communities --> Name["Infer module name<br/>from common path prefix"]
    Name --> Exports["Classify exports<br/>via LanguageAdapter"]
    Exports --> Cohesion["Calculate cohesion<br/>score per module"]
    Cohesion --> Output["list[Module]"]
```

**Edge weighting:**

```python
def weight_file_edge(source: str, target: str, edge_data: dict) -> float:
    """Weight an edge for community detection."""
    base = 1.0
    # More imported symbols = stronger coupling
    base += len(edge_data.get("symbols", [])) * 0.5
    # Same directory = bias toward same module
    if Path(source).parent == Path(target).parent:
        base *= 1.5
    return base
```

**Cohesion score:** Ratio of intra-module edges to total edges touching the module. High cohesion (>0.7) means the module is well-bounded. Low cohesion (<0.3) suggests the module may need splitting.

#### 3.4.2 Pattern Recognition

Rule-based detection operating on the dependency graph structure and AST node signatures.

```python
class PatternDetector(Protocol):
    """Interface for all pattern detectors."""
    name: str
    display_name: str
    category: PatternCategory

    def detect(
        self,
        graph: DependencyGraph,
        symbols: list[Symbol],
        modules: list[Module],
    ) -> DetectedPattern | None:
        """Return a DetectedPattern if found, None otherwise."""
        ...
```

**Shipped detectors:**

| Pattern                                | Detection Strategy                                                                                                                                                                                               |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Middleware Chain**                   | Find linear call chains where each function has signature `(request, next_handler)` or `(ctx, next)`. Chain length ≥ 3 required. Confidence scales with chain length.                                            |
| **Plugin / Extension System**          | Registry pattern: a collection (dict/list) that accepts heterogeneous callables sharing a common interface. Look for `register()` / `add()` methods that accept callables or classes implementing a shared base. |
| **Event Bus / Pub-Sub**                | Emit/subscribe pairs. Functions named `emit`/`dispatch`/`publish` + `on`/`subscribe`/`listen`. String-keyed dispatch to callback collections.                                                                    |
| **Repository / DAO**                   | Classes whose methods map to CRUD operations (get/list/create/update/delete). Constructor takes a connection/client/session parameter.                                                                           |
| **Strategy Pattern**                   | Multiple concrete implementations of the same abstract base class or protocol. Runtime selection via factory or config.                                                                                          |
| **Builder Pattern**                    | Method chaining returning `self`. Fluent API with a terminal `build()` / `create()` / `compile()` method.                                                                                                        |
| **Dependency Injection**               | Constructor parameters typed as protocols/abstract base classes. External wiring in a composition root file.                                                                                                     |
| **Pipeline / Chain of Responsibility** | Sequential processing stages with uniform input/output types. Each stage transforms data and passes to next.                                                                                                     |
| **Factory Pattern**                    | Functions or methods that return instances of different concrete types based on input parameters. Centralized creation logic.                                                                                    |
| **Singleton / Module-Level State**     | Module-level instances, `_instance` patterns, `__new__` overrides, or global configuration objects.                                                                                                              |

### 3.5 Stage ⑤ — Serve

**Responsibility:** Assemble final output objects (ArchProfile, ContextBundle, ComparisonResult), render in requested format.

#### 3.5.1 ContextBundle Assembly (Token Budget Packing)

```mermaid
graph TD
    Q["Query + Budget"]
    Q --> Retrieve["① RETRIEVE<br/>BM25 top-50<br/>+ Vector top-30<br/>(if enabled)"]
    Retrieve --> Dedup["Deduplicate<br/>candidates"]
    Dedup --> Expand["② EXPAND<br/>1-hop dependency<br/>neighborhood"]
    Expand --> Score["③ RANK<br/>final_score =<br/>relevance × 0.6 +<br/>centrality × 0.2 +<br/>type_coverage × 0.2"]
    Score --> Sort["Sort by<br/>final_score desc"]
    Sort --> Pack["④ PACK<br/>Greedy bin-packing"]
    Pack --> TypeResolve["⑤ RESOLVE TYPES<br/>Add referenced<br/>type definitions"]
    TypeResolve --> Structure["⑥ ATTACH CONTEXT<br/>File tree, modules,<br/>call chain, deps"]
    Structure --> Output["ContextBundle"]
```

**Pack algorithm detail:**

```python
def pack_context(
    ranked_chunks: list[RankedChunk],
    type_defs: dict[str, TypeDefinition],
    budget: int,
) -> tuple[list[RankedChunk], list[TypeDefinition], int]:
    """Greedy bin-packing with overlap detection."""

    included_chunks: list[RankedChunk] = []
    included_types: list[TypeDefinition] = []
    included_content: set[str] = set()  # For overlap detection
    remaining = budget

    # Reserve ~15% of budget for structural context preamble
    reserved_for_context = int(budget * 0.15)
    remaining -= reserved_for_context

    for chunk in ranked_chunks:
        if remaining <= 0:
            break

        # Skip if >80% content overlap with already-included chunk
        if _overlap_ratio(chunk.chunk.content, included_content) > 0.8:
            continue

        chunk_cost = chunk.chunk.token_count

        # Include referenced type definitions not yet added
        referenced_types = _find_referenced_types(chunk.chunk, type_defs)
        type_cost = sum(
            td.token_count for td in referenced_types
            if td.symbol not in {t.symbol for t in included_types}
        )

        total_cost = chunk_cost + type_cost
        if total_cost > remaining:
            # Try without types
            if chunk_cost <= remaining:
                total_cost = chunk_cost
                referenced_types = []
            else:
                continue

        included_chunks.append(chunk)
        included_types.extend(referenced_types)
        included_content.add(chunk.chunk.content)
        remaining -= total_cost

    tokens_used = budget - remaining
    return included_chunks, included_types, tokens_used
```

#### 3.5.2 Prompt Rendering (XML Format)

```xml
<codebase_context repo="{repo_url}" commit="{commit_hash}" query="{query}">

<file_map>
{ascii_file_tree_of_relevant_files}
</file_map>

<module_context>
Relevant modules: {comma_separated_module_names}
Entry point: {entry_point_symbol}
Call chain: {A → B → C → D}
</module_context>

<chunks>
<chunk file="{path}" lines="{start}-{end}" symbol="{name}" type="{kind}" relevance="{score:.2f}">
{source_code_with_imports_prepended}
</chunk>
<!-- ... more chunks ordered by relevance ... -->
</chunks>

<types>
<type file="{path}" lines="{start}-{end}" symbol="{name}">
{full_type_definition}
</type>
<!-- ... referenced type definitions ... -->
</types>

<dependencies>
Internal: {comma_separated_internal_deps}
External: {comma_separated_external_packages}
</dependencies>

</codebase_context>
```

---

## 4. Storage Architecture

### 4.1 Cache Layout

```text
~/.archex/
├── config.toml                # Global configuration
├── cache/
│   ├── encode--httpx/         # Repo: github.com/encode/httpx
│   │   ├── index.db           # SQLite: chunks, symbols, edges, FTS5
│   │   ├── meta.json          # Commit hash, timestamp, config, stats
│   │   └── vectors.npy        # Optional: embedding matrix
│   │
│   ├── psf--requests/         # Repo: github.com/psf/requests
│   │   ├── index.db
│   │   ├── meta.json
│   │   └── vectors.npy
│   │
│   └── local--my-project/     # Local repo: /Users/tom/my-project
│       ├── index.db
│       └── meta.json
│
└── models/                    # Cached ONNX model files
    └── nomic-embed-code/
        ├── model.onnx
        └── tokenizer.json
```

### 4.2 Cache Invalidation

```mermaid
graph TD
    Request["analyze() or query()"]
    Request --> CacheExists{"Cache exists<br/>for this repo?"}
    CacheExists -->|No| FullBuild["Full pipeline:<br/>clone → parse → index"]
    CacheExists -->|Yes| CheckCommit{"HEAD commit<br/>matches cache?"}
    CheckCommit -->|Yes| UseCache["Load from cache"]
    CheckCommit -->|No| Incremental["Incremental update:<br/>git diff → re-parse<br/>changed files only"]
    Incremental --> UpdateCache["Update index.db<br/>+ meta.json"]
    FullBuild --> WriteCache["Write cache"]
```

### 4.3 Cache Key Strategy

```python
def cache_key(source: RepoSource) -> str:
    """Deterministic cache key for a repo source."""
    if source.url:
        # github.com/encode/httpx → encode--httpx
        parts = urlparse(source.url).path.strip("/").split("/")
        key = "--".join(parts[-2:])  # org--repo
    else:
        # /Users/tom/projects/my-app → local--my-app
        key = f"local--{source.local_path.name}"

    if source.target:
        # Scoped monorepo: encode--next.js--packages--next
        key += "--" + source.target.replace("/", "--")

    return key
```

---

## 5. Error Handling

### 5.1 Exception Hierarchy

```python
class ArchexError(Exception):
    """Base exception for all archex errors."""

class AcquireError(ArchexError):
    """Errors during source acquisition."""

class CloneError(AcquireError):
    """Git clone failed (network, auth, not found)."""
    url: str
    exit_code: int
    stderr: str

class PrivateRepoError(AcquireError):
    """Repository requires authentication."""
    url: str

class ParseError(ArchexError):
    """Errors during AST parsing."""

class UnsupportedLanguageError(ParseError):
    """No adapter registered for this language."""
    language: str
    file_path: str

class IndexError(ArchexError):
    """Errors during index construction."""

class ProviderError(ArchexError):
    """Errors from LLM or embedding providers."""
    provider: str
    status_code: int | None

class CacheError(ArchexError):
    """Errors reading/writing the index cache."""
```

### 5.2 Graceful Degradation

| Failure                    | Behavior                                              |
| -------------------------- | ----------------------------------------------------- |
| Single file fails to parse | Skip file, log warning, continue with remaining files |
| Language adapter not found | Skip files of that language, log warning              |
| Import resolution fails    | Mark edge as unresolved, don't add to graph           |
| Vector index build fails   | Fall back to BM25-only retrieval, log warning         |
| LLM provider unreachable   | Skip enrichment, return structural-only results       |
| Cache corrupted            | Delete cache, rebuild from scratch                    |
| Token budget too small     | Return what fits, set `truncated=True`                |

---

## 6. Performance Considerations

### 6.1 Expected Performance

| Repo Size                  | Files | Parse Time | Index Time | Query Time |
| -------------------------- | ----- | ---------- | ---------- | ---------- |
| Small (e.g., click)        | ~50   | < 1s       | < 1s       | < 200ms    |
| Medium (e.g., httpx)       | ~200  | 2-5s       | 2-3s       | < 500ms    |
| Large (e.g., FastAPI)      | ~500  | 5-10s      | 5-8s       | < 1s       |
| Very Large (e.g., Next.js) | ~5000 | 30-60s     | 20-40s     | 1-3s       |

_Parse + index is a one-time cost, amortized by caching. Query time is per-request._

### 6.2 Optimization Strategies

| Strategy                        | Where Applied                                                      |
| ------------------------------- | ------------------------------------------------------------------ |
| **Shallow clone** (`--depth 1`) | Acquire: skip git history, clone only HEAD                         |
| **Sparse checkout**             | Acquire: clone only targeted monorepo sub-package                  |
| **Parallel file parsing**       | Parse: `concurrent.futures.ProcessPoolExecutor` for AST extraction |
| **Cached NetworkX graph**       | Index: reconstruct graph once per session, cache in memory         |
| **FTS5 for BM25**               | Index: SQLite handles ranking natively, no Python-side scoring     |
| **Batch embedding**             | Index: embed all chunks in a single ONNX inference call            |
| **Incremental re-indexing**     | Cache: only re-parse files changed since last commit               |
| **Lazy vector index**           | Query: vector index loaded on first query, not on import           |

---

## 7. Security Considerations

| Concern                    | Mitigation                                                                                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Arbitrary repo cloning** | Validate URL format. Only support https:// and git:// protocols. Block file:// and ssh:// by default (configurable).                                                             |
| **Path traversal**         | All file paths are resolved relative to clone root. Reject paths containing `..`.                                                                                                |
| **LLM prompt injection**   | No source code content is ever passed to LLMs as instructions. Code appears only in structured data blocks. LLM enrichment prompts use system messages with explicit boundaries. |
| **API key management**     | Keys stored in `~/.archex/config.toml` with restricted permissions (0600). Never logged. Never included in cache files.                                                          |
| **Disk space**             | Default cache TTL of 7 days. `archex cache clean` for manual management. Warning at 5GB total cache size.                                                                        |
| **No code execution**      | archex never executes cloned code. No `eval()`, no subprocess calls on repo content. Tree-sitter parsing is static analysis only.                                                |
