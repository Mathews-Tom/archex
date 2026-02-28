# Changelog

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
