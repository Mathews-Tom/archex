# Changelog

## 0.6.0 (2026-03-12)

### Retrieval Quality (8-Phase Improvement Plan)

**Measurement Repair**
- Deduplicate ranked files before computing MRR, nDCG, MAP in benchmark strategies
- Add benchmark fields: seed_files, expanded_files, unique_ranked_files, expansion_ratio, seed_precision, seed_recall
- Persist PipelineTiming breakdowns in benchmark results
- Split benchmark summaries by category bucket (self, external-framework, external-large, architecture-broad, framework-semantic)

**Cache & Performance**
- Read-only warm-cache queries: skip FTS rebuild on cache hit
- Remote HEAD resolution via `git ls-remote` for URL sources
- Structured JSON cache metadata with backward-compatible parsing

**Retrieval Precision**
- Expansion gating: seeds below 10% of max BM25 score don't trigger graph expansion
- Score-relative file cutoff (FILE_SCORE_CUTOFF=0.15): files below 15% of top file excluded
- MAX_EXPANSION_FILES reduced from 10 to 5

**Retrieval Recall**
- Query normalization: camelCase/snake_case splitting, bigram compound generation
- Architecture-intent synonym expansion (8 keyword categories)
- Symbol exact-match boost raised to 0.60x (from 0.15x for partial matches)

**Quality Gates & Benchmark Expansion**
- Quality gate with configurable thresholds: recall>=0.60, precision>=0.20, F1>=0.30, MRR>=0.55
- Latency warning system (warn_latency_ms=5000.0)
- Benchmark corpus expanded from 11 to 25 tasks across 5 difficulty categories
- Parallel CI benchmark jobs (BM25 and hybrid strategies)

### Unified Artifact Pipeline
- Chunker moved from `index/` to `pipeline/` as canonical location (backward-compat shim at old path)
- `produce_artifacts()` unified entry point: parse -> import-resolve -> chunk -> edge-build
- `ArtifactBundle` dataclass for typed pipeline output

### Observability
- `observe.py` stdlib-only module: `PipelineTrace`, `StepTiming`, `TraceCollector`
- `traced_step` / `traced_operation` context managers for timing pipeline steps
- Instrumented `api.query()` and `serve.context.assemble_context()` with step-level tracing
- Service role documentation in `api.py` (6 roles: acquisition, parsing, indexing, retrieval, analysis, observability)

### Test Coverage
- 85% coverage threshold enforced in pytest configuration
- Coverage: 90% -> 92.52% with 29 new tests
- BM25 graduated fallback stages 2-4 now covered
- Full query -> assemble_context -> render pipeline integration tests
- Pipeline service parse + chunk fully covered

### Stats
- 1541 tests, 92% coverage (85% minimum enforced)
- 25 benchmark tasks across Python, Go, Rust, JS/TS
- BM25 mean recall: 0.58, mean MRR: 0.69 (across 25 tasks)

## 0.5.0 (2026-03-04)

### Delta Indexing

- **3-path cache decision:** `_ensure_index` checks exact cache hit → delta update → full re-index
- **`compute_delta()`:** Git diff (`--name-status -M`) between commits produces `DeltaManifest` with adds, modifies, deletes, renames
- **`apply_delta()`:** Surgical store update — renames, deletions, re-parse changed files, atomic store upsert, graph update, BM25 rebuild, metadata refresh
- **`compute_mtime_delta()`:** Mtime-based fallback for non-git repos
- **`delta_threshold` config:** If >50% files changed, fall back to full re-index (default 0.5)

### Language Expansion

- **Java adapter:** Visibility defaults to INTERNAL (package-private), interface members default to PUBLIC, full symbol/import/entry-point support
- **Kotlin adapter:** Visibility defaults to PUBLIC, extension functions, companion objects, data/sealed classes
- **C# adapter:** Namespace-qualified names, 6 visibility levels mapped to 3, properties/events/delegates, top-level statement detection
- **Swift adapter:** Default INTERNAL visibility, extensions, actors, protocols, `@main`/`@UIApplicationMain`/`XCTestCase` entry points
- **Shared JVM helpers:** `_jvm_helpers.py` with `resolve_jvm_import`, `map_jvm_visibility`, `detect_jvm_convention`

### Infrastructure

- **Engine fallback:** `_try_language_pack()` in `engine.py` for grammars not available as standalone (Swift)
- **Pipeline service:** `pipeline/service.py` module
- **Grammar deps:** `tree-sitter-java`, `tree-sitter-kotlin`, `tree-sitter-c-sharp`, `tree-sitter-language-pack` (optional extra)

### Models

- **`strict` field on Config**
- **Delta models:** `ChangeStatus` (StrEnum), `FileChange`, `DeltaManifest` (with computed properties), `DeltaMeta`
- **`DeltaIndexError`** exception

### Store

- **`delete_chunks_for_files()`** — remove chunks by file path
- **`delete_edges_for_files()`** — remove edges by file path
- **`update_file_paths()`** — rename file paths in chunks/edges
- **`delete_and_insert_for_files()`** — atomic delete + re-insert for changed files

### Stats

- 1274 tests, 92% coverage

## 0.4.0 (2026-03-01)

### Refactoring

- **Shared tree-sitter helpers:** Extract duplicate `_text`/`_type`/`_children`/`_field`/`_start_line`/`_end_line` accessors from all four language adapters into a single `ts_node` module
- **Dead code removal:** Remove unused `get_adapter()`, `_extract_interfaces()`, redundant `add_node` calls in `DependencyGraph.from_edges()`, unused `index_config` param from `api.analyze()`
- **Dependency deduplication:** Use sets instead of lists in `_build_module_from_community` for O(1) membership checks
- **Chunker optimization:** Remove unnecessary `sorted()` call on already-ordered covered ranges
- **Vector load:** `copy=False` on `numpy.astype` for zero-copy when array is already float32

### Error handling & logging

- **`infer_decisions()`:** Log LLM enrichment failures with `logger.warning()` instead of silently catching
- **`BM25Index.search()`:** Log FTS5 query failures instead of silently returning empty results
- **`SentenceTransformerEmbedder.dimension`:** Replace bare `assert` with explicit `ArchexIndexError`

### Configuration & standards

- **`DEFAULT_CACHE_DIR` constant:** Centralized in `config.py`, used by all CLI cache commands
- **Config validation:** `model_fields` over `hasattr` for Pydantic v2 correctness
- **`validate_dimensions()`:** Extracted from `compare_repos()` for reuse by MCP integration
- **Install instructions:** All `pip install` references replaced with `uv add`

### Testing

- 3 new test files: `test_config.py`, `test_adapter_registry.py`, `test_renderers.py`
- 7 extended test files covering parse, index, analyze, serve, and acquire layers
- 538 → 641 tests (+103), 84% → 90% coverage

## 0.3.0 (2026-03-01)

### Phase 6a — Harden

- **Git URL validation:** `_validate_url()` restricts to `http://`, `https://`, local paths only
- **Branch name validation:** Regex guard rejects injection characters and `-` prefix
- **FTS5 query escaping:** Strip non-alphanumeric characters from BM25 query tokens
- **Cache key validation:** Enforce `^[0-9a-f]{64}$` pattern in `db_path()` and `meta_path()`
- **Vector safety:** `allow_pickle=False` and `dtype='U512'` for `.npz` persistence, length validation on load
- **File size guard:** `max_file_size` config in `discover_files()` and `parse_file()`
- **Store safety:** `IndexStore.__init__` wrapped in try/except for connection cleanup on failure
- **Parse logging:** `symbols.py` and `imports.py` log warnings on parse failures
- **MCP validation:** Dimension list validated against `SUPPORTED_DIMENSIONS` before `compare()`
- **MCP event loop:** `asyncio.get_event_loop()` → `asyncio.get_running_loop()`
- **CLI error handling:** API calls wrapped in `try/except ArchexError` → `click.ClickException`
- **Embeddings timeout:** `timeout=30` added to API `urlopen()`
- **Compare CLI:** `assert isinstance(...)` replaced with explicit type check

### Phase 6b — Performance

- **Cache-first query:** `query()` checks cache BEFORE parsing — cache hit skips entire parse pipeline
- **Graph round-trip:** `DependencyGraph.from_edges()` classmethod reconstructs graph from stored edges
- **Batch fetch:** `IndexStore.get_chunks_by_ids()` with `WHERE id IN (...)`, used in `BM25Index.search()`
- **Parallel config:** `Config.parallel` flag passed to `extract_symbols()` and `parse_imports()`
- **Parallel compare:** `ThreadPoolExecutor(max_workers=2)` runs both `analyze()` calls concurrently
- **O(N) top-k:** `np.argpartition` replaces `np.argsort` in VectorIndex search
- **Vector cache:** `CacheManager.vector_path()` persists vector indices across queries
- **Centrality cache:** Lazy `_centrality_cache` on `DependencyGraph`, invalidated on mutation
- **Chunker optimization:** Source split once in `chunk_file()`, pre-split lines passed downstream
- **Git-aware cache:** Cache key includes git HEAD commit hash for local repos

### Phase 6c — Wire & Polish

- **Hybrid retrieval wired:** VectorIndex built and searched in cache-miss query path, results passed through RRF to `assemble_context()`
- **`resolve_source()` utility:** Extracted from 4 inline copies, fixes `query_cmd` bug (`startswith("http")` → `startswith("http://")`)
- **Compare CLI routing:** Routes through `api.compare()` instead of manual `analyze()` x2
- **MCP dimension fix:** `testing_strategy` → `testing`, `dependency_management` → `state_management`, `configuration_management` → `configuration`
- **Dead field removal:** `CodeChunk.module` removed from models, store schema, and chunker
- **RepoSource validator:** `model_validator(mode="after")` requires `url` or `local_path`
- **`load_config()`:** Reads `~/.archex/config.toml` via `tomllib` + `ARCHEX_*` env vars
- **Provider model IDs:** Centralized in `DEFAULT_MODELS` dict in `config.py`
- **Pipeline logging:** `logging.getLogger(__name__)` with timing at all stage boundaries
- **Test improvements:** Cache CLI tests, `__version__` import in test_cli

### Phase 6d — Extensibility

- **`ScoringWeights` model:** Parameterized context scoring (relevance=0.6, structural=0.3, type_coverage=0.1) with sum-to-1 validator, accepted in `assemble_context()` and `query()`
- **`PatternRegistry`:** `register()` decorator, `load_entry_points()` for `archex.pattern_detectors` group, optional `registry` param in `detect_patterns()`
- **`AdapterRegistry`:** `register()`, `build_all()`, `load_entry_points()` for `archex.language_adapters` group, public `adapter_classes` property
- **`Chunker` Protocol:** `runtime_checkable`, accepted as optional `chunker` param in `query()`
- **Entry points:** `archex.language_adapters` and `archex.pattern_detectors` groups declared in `pyproject.toml`
- **Integration tests:** 12 end-to-end tests covering analyze, query (BM25, caching, custom weights, hybrid fallback), compare (default + specific dimensions), full analyze→query pipeline
- 538 tests, 84% coverage

## 0.2.0 (2026-02-28)

### Phase 5 — Ecosystem

- **MCP server:** 3 tools (analyze_repo, query_repo, compare_repos) with async stdio transport, `archex mcp` CLI command
- **LangChain integration:** `ArchexRetriever(BaseRetriever)` mapping RankedChunks to Documents
- **LlamaIndex integration:** `ArchexRetriever(BaseRetriever)` mapping RankedChunks to NodeWithScore
- **Parallel parsing:** `extract_symbols()` and `parse_imports()` accept `parallel=True` for ProcessPoolExecutor concurrency
- **ONNX model caching:** `NomicCodeEmbedder` supports `cache_dir` for persistent model storage at `~/.archex/models/`
- **New optional deps:** `archex[mcp]`, `archex[langchain]`, `archex[llamaindex]`
- 422 tests, 81% coverage

## 0.1.0 (2026-02-28)

### Phase 0 — Scaffold

- Project structure with hatchling build system, CLI stub, CI config
- Test fixtures: `python_simple`, `python_patterns`, `typescript_simple`, `monorepo_simple`
- Tooling: ruff, pyright (strict), pytest + pytest-cov, pre-commit

### Phase 1 — Foundation

- **Acquire:** `clone_repo()`, `open_local()`, `discover_files()` with git ls-files + rglob fallback
- **Parse:** `TreeSitterEngine` with cached Language/Parser, `PythonAdapter` (full AST walk)
- **Index:** `DependencyGraph` wrapping dual NetworkX DiGraphs (file-level, symbol-level), SQLite round-trip, PageRank centrality
- **Serve:** Basic `ArchProfile` assembly with stats, dependency summary, interface surface
- **API:** `analyze()` pipeline — discover, parse, resolve, graph, profile
- **CLI:** `archex analyze <source> --format json|markdown`
- **Models:** Complete Pydantic v2 model hierarchy, exception classes, StrEnum types
- 107 tests, 88% coverage

### Phase 2 — Retrieval

- **Chunker:** `ASTChunker` with symbol-based boundaries, import context, small-chunk merging, tiktoken counting
- **BM25:** SQLite FTS5 index with OR-joined query tokens
- **Store:** `IndexStore` with WAL mode, chunks/edges/metadata tables
- **Cache:** `CacheManager` with TTL, WAL checkpoint before copy, FTS rebuild on load
- **Context assembly:** BM25 search, graph neighborhood expansion, composite scoring (0.6 relevance + 0.3 structural + 0.1 type), greedy bin-packing
- **Renderers:** XML (CDATA), JSON (model_dump), Markdown
- **API:** `query()` pipeline — acquire, parse, chunk, index, search, assemble
- **CLI:** `archex query`, `archex cache list|clean|info`
- 174 tests, 85% coverage

### Phase 3 — Intelligence

- **Modules:** Louvain community detection on file-level dependency graph
- **Patterns:** Rule-based detection — middleware chain, plugin system, event bus, repository/DAO, strategy
- **Interfaces:** Public API surface extraction with usage counts
- **TypeScript adapter:** ES modules, CommonJS, type-only imports, re-exports, index resolution
- **LLM enrichment:** Optional `Provider` protocol (Anthropic, OpenAI, OpenRouter), structured output
- **Decisions:** `infer_decisions()` with structural evidence + optional LLM inference
- Full `ArchProfile` assembly with modules, patterns, interfaces, decisions

### Phase 4 — Compare + Polish

- **Go adapter:** Functions, methods (pointer/value receivers), structs, interfaces, const/var, Go visibility (uppercase = public)
- **Rust adapter:** fn, struct, enum, trait, impl blocks, const/static, macro_rules, pub/pub(crate)/pub(super) visibility
- **Vector index:** Numpy-based cosine similarity (L2-norm at build, dot at search), `.npz` persistence
- **Embedder protocol:** `encode(texts) -> list[list[float]]`, `dimension -> int` — Nomic Code ONNX, API (OpenAI-compatible), SentenceTransformers backends
- **Hybrid retrieval:** Reciprocal rank fusion merging BM25 + vector results by chunk ID
- **Comparison engine:** `compare_repos()` across 6 structural dimensions (api_surface, concurrency, configuration, error_handling, state_management, testing), no LLM required
- **CLI polish:** `--timing` flag on analyze/query, `--strategy bm25|hybrid` on query, `--dimensions` on compare
- 372 tests, 81% coverage
