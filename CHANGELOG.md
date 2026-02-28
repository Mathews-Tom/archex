# Changelog

## 0.2.0 (2026-02-28)

### Phase 5 — Ecosystem

- **MCP server:** 3 tools (analyze_repo, query_repo, compare_repos) with async stdio transport, `archex mcp` CLI command
- **LangChain integration:** `ArchexRetriever(BaseRetriever)` mapping RankedChunks to Documents
- **LlamaIndex integration:** `ArchexRetriever(BaseRetriever)` mapping RankedChunks to NodeWithScore
- **Parallel parsing:** `extract_symbols()` and `parse_imports()` accept `parallel=True` for ProcessPoolExecutor concurrency
- **ONNX model caching:** `NomicCodeEmbedder` supports `cache_dir` for persistent model storage at `~/.archex/models/`
- **Publish workflow:** GitHub Actions publishes to PyPI on `v*` tag push
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
