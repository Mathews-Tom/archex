# Changelog

## [Unreleased]

### Changed

- **README adoption refresh:** Reframed the opening around context-window burn, kept the quickstart as the one always-visible happy path, collapsed only the secondary Docker/extras blocks, and added a metric-anchored workflow translation under Measured results without changing product behavior or benchmark numbers.
- **install-client default-global + `--dry-run`:** `archex install-client <client>` now installs at global (user) scope by default; pass a `[SOURCE]` repo path or `--scope project` for a repo-local install. Removed the `--write` flag — configs are written by default and `--dry-run` previews the exact target and config without touching the filesystem. Writes stay non-destructive (merge into existing config, never clobber unrelated sections) and are idempotent: re-running with an identical `archex` entry is a no-op, while a different existing entry is left untouched and refused. Updated the installation trust contract, compatibility matrix, and README to match.

### Fixed

- **Sticky global metrics-health warning:** `archex metrics` could surface a permanent `Metrics warning: record: Path does not exist: /repo` (or similar) on every repo, because the machine-local health flag in `~/.archex/usage.sqlite` is global and was never cleared once set. Two root causes are fixed: (1) successful recording now self-heals the health flag, so any later successful query/scout on any repo clears a stale warning; (2) expected source-unavailability when computing the optional whole-repo/raw token baselines (an `AcquireError`/missing path for a source that is not a usable local repo) is no longer latched as a metrics failure. Added `archex metrics repair` to clear a stale warning manually without deleting accumulated savings data.

## [0.13.2] - 2026-06-19

### Fixed

- **Missing scipy dependency:** `scipy` was absent from `[project.dependencies]` (declared only in the dev group), causing `ModuleNotFoundError: No module named 'scipy'` when `archex query` ran on a plain `archex[mcp]` install. networkx ≥ 3.3 routes `nx.pagerank()` — used by `DependencyGraph.structural_centrality` on the core query path — through `_pagerank_scipy`, which does a bare `import numpy` then `import scipy` at call time. v0.12.1 promoted `numpy` but missed `scipy` on the very next line of the same function, and CI installs `--all-extras` (which always pulls scipy transitively), so the gap stayed invisible. Promoted `scipy>=1.11.2` to a declared core dependency and removed the redundant dev entry. Added a minimal-install CI smoke job that installs only core dependencies and runs `archex index`/`archex query` end-to-end so missing runtime dependencies fail CI. Fixes #266.

## [0.13.1] - 2026-06-18

### Fixed

- **Tree-sitter language-pack compatibility:** Capped `tree-sitter-language-pack` below 1.0 and added a parser guard so `archex index` does not crash with `TypeError: Parser(...) expected tree_sitter.Language` when a resolver installs the incompatible Rust-backed 1.x package line.


## [0.13.0] - 2026-06-18

### Added

- **TurboQuant vector storage:** Added `IndexConfig.quantize_vectors` and `quantize_bits`, wired them through stored vector index build/search/load paths, exposed `archex index --quantize-vectors/--no-quantize-vectors` and `--quantize-bits`, and made 4-bit TurboQuant the default storage mode for vector indexes after measured recall/MRR parity.
- **Quantized retrieval benchmark lane:** Added `archex_query_hybrid_quantized_4bit`, cache isolation for quantized vector artifacts, measured `.npz` size provenance, and baseline comparison reporting for recall, MRR, F1, required-file recall, latency, and compression.
- **Raw-ripgrep trust baseline:** Added the public raw-ripgrep/read lane and required-file trust fields so external comparisons distinguish retrieval coverage from safe-to-act confidence.
- **Bundle-only evaluator lane:** Added `archex benchmark bundle-eval` for operator-supplied local evaluators that receive only the rendered bundle and receipt JSON, with missing-needed-file attribution and no hosted evaluator behavior.

### Changed

- **README v0.13 refresh:** Updated the project front page for receipts, raw-ripgrep evidence, bundle-only evaluation, TurboQuant defaults, and clearer measured-results column names.
- **Benchmark reports:** Surface required-file recall, missed-task/missed-file rates, completion-preserved metrics, bundle-only safety results, and quantized storage provenance without conflating those lanes.

### Fixed

- **Badge cache busting:** Bumped README badge cache keys for the v0.13 release so PyPI and Star History embeds stop showing stale v0.12-era cached assets.

## [0.12.1] - 2026-06-17

### Fixed

- **Missing numpy dependency:** `numpy` was absent from `[project.dependencies]`, causing `ModuleNotFoundError: No module named 'numpy'` when `archex query` ran on a plain `uv tool install archex` or `archex[mcp]` install. networkx ≥ 3.3 routes `nx.pagerank()` through its scipy implementation, which does a bare `import numpy` at call time; numpy is also a direct runtime dependency in `index/fusion.py`, `index/quantize.py`, `index/vector.py`, and `index/splade.py`. Promoted `numpy>=1.24` to a declared core dependency and removed the redundant entry from dev deps. Fixes #266.

### Removed

- **einops phantom dependency:** `einops` was declared as a core dependency since 007c00e but was never imported by archex at any point in its history. It arrives transitively via `sentence-transformers` when Jina reranker models are loaded; archex itself has no direct use of it. Removed from `[project.dependencies]` and dropped from the lockfile.
- **`vector` optional extra:** The `archex[vector]` extra (`onnxruntime>=1.17, tokenizers>=0.15`) had no backing code — the ONNX embedder it was built for was removed in a prior cleanup and never replaced. Local ONNX-backed embedding is provided by `archex[vector-fast]` (FastEmbed). Removed the extra from `[project.optional-dependencies]` and from the `all` bundle; updated README, OVERVIEW, SYSTEM_DESIGN, and WHY_ARCHEX docs to reference `vector-fast` instead.

## [0.12.0] - 2026-06-17

### Added

- **Local metrics foundation:** Added the machine-local SQLite usage ledger, repo registry, metrics policy controls, metrics health reporting, and the `archex metrics` CLI for summary, inspect, export, delete, and trace management.
- **CLI and MCP usage accounting:** Added anonymous token-savings accounting for `query`, `scout`, and the eligible structural CLI/MCP surfaces without changing their public output contracts.
- **Python API opt-in:** Added explicit `record_usage_event(...)` for Python callers that want local usage recording without making normal API calls write to disk.
- **Trust documentation:** Added `docs/LOCAL_METRICS.md` with the exact savings formulas, privacy boundary, default-off behavior, and operator controls.

### Changed

- **Telemetry contract:** Flipped local metrics from default-on to default-off so archex ships with no telemetry by default; users must explicitly enable local metrics before any ledger writes occur.
- **README and trust surfaces:** Updated the README and installation trust contract to describe optional local metrics, detailed trace opt-in, local-only storage, and the distinction between headline savings and whole-repo upper-bound context metrics.
- **Operational visibility:** Extended `status` and `doctor` to surface metrics health and recorder state alongside the new local metrics control plane.

### Fixed

- **Metrics release hardening:** Fixed branch-specific test enablement, import drift, and Ruff formatting mismatches discovered while landing the metrics stack so the release line is green across all Python CI variants and Docker checks.

## [0.11.0] - 2026-06-16

### Added

- **Context receipts:** Added first-class deterministic receipts for query and scout flows, including query inputs, token budgets, index revision, freshness state, returned context, included and omitted dependency edges, skipped candidate reason codes, completeness status, completeness reason, and recommended next action.
- **Retrieval provenance:** Threaded receipt construction through retrieval and scout assembly without extra model calls, with returned file handles, line ranges, content hashes, symbols, scores, duplicate suppression, unsupported-grammar skips, stale-index detection, budget exhaustion, and dependency-frontier cuts.
- **Output surfaces:** Exposed receipts through `archex query --format json`, compact receipt metadata in `archex query --format xml`, scout summaries and fetch handles, and the MCP query/scout tool responses.
- **Benchmark proof metrics:** Added required-file recall, missed-required-file rate, all-required-files-present, post-bundle search/read turn counts when observable, task completion result, completion preservation, receipt accuracy where ground truth exists, and a per-task missed-required-file appendix.
- **Compatibility and install paths:** Added `docs/CLIENT_COMPATIBILITY_MATRIX.md` and first-party `archex install-client` support for Claude Code, Codex headless, Pi/OMP, OpenCode, and Cursor, with preview-before-write behavior and client-shaped config snippets.
- **Enterprise installation trust:** Added `SECURITY.md`, `docs/INSTALLATION_TRUST_CONTRACT.md`, model security reporting in `archex doctor`, and remote-code model loading gates for safer local deployment.
- **Brand assets:** Added the verified-context banner and standalone logo assets under `assets/` for README and project surfaces.

### Changed

- **README messaging:** Reframed the public pitch around verified, provenance-backed code context instead of token savings alone, with links to receipt docs, benchmark evidence, compatibility, security, install trust, and client setup surfaces.
- **Benchmark reporting:** Updated benchmark reports and fixtures so public proof waits for computed required-file metrics instead of implying safe-to-act confidence from token efficiency alone.
- **Client documentation:** Marked compatibility rows by tested status, setup command/config, watch support, freshness semantics, known limitations, and last verified date; untested clients remain explicitly unverified.
- **Docker full image:** Stopped downloading the FastEmbed model during image build; the full image now keeps runtime local-FastEmbed support without making CI depend on model registry availability.
- **Retrieval defaults docs:** Folded the retrieval-default evidence-gate rationale into `docs/RETRIEVAL_DEFAULT_DECISIONS.md`.
- **Changelog format:** Adopted the grouped release-entry style used by the referenced OMP changelog for new release entries while leaving older historical entries intact.

### Fixed

- **Docker publishing path:** Fixed the release-blocking full-image build failure caused by build-time FastEmbed prewarm.
- **README assets:** Deduplicated overlapping Start and Quick links, switched README visuals to Markdown image syntax, and placed the banner at the top of the project page.
- **Security diagnostics:** Made `archex doctor` surface model-cache trust state instead of leaving remote-code posture implicit.

### Removed

- **Obsolete planning docs:** Removed the temporary brand asset design spec and the standalone retrieval-default ADR after moving durable content into maintained docs.

## 0.10.2 (2026-06-15)

### Changed

- Cleaned release documentation authority, removed stale compatibility paths and local-first contradictions, and refreshed release assets to display the `v0.10` series.
- Consolidated benchmark query strategy execution, isolated MCP tool dispatch, and centralized query bundle finalization without changing public APIs.

### Removed

- Removed the Codecov README badge because coverage is enforced by local/CI pytest gates rather than published through Codecov.
- Removed obsolete hosted API embedder prototype, stale chunker compatibility shim, old benchmark readiness doc, and unimplemented symbol-lookup benchmark strategy.

## 0.10.1 (2026-06-15)

### Fixed

- Added GitHub Container Registry login to the Docker image workflow so main-branch pushes can publish `ghcr.io/mathews-tom/archex:slim` and `ghcr.io/mathews-tom/archex:full` instead of failing with anonymous-token `403 Forbidden`.
- Expanded the release notes and changelog coverage for the intended `0.10.x` release line so the record includes all shipped changes since `0.9.0`.

## 0.10.0 (2026-06-15)

### Added

- Added selectable cAST chunking plus chunker benchmark infrastructure and supporting cache/test alignment.
- Added `archex doctor` diagnostics for index health, staleness, model cache presence, grammar availability, MCP registration, and `.archex/` disk usage.
- Added the in-repo Claude Code skill and `/archex` command workflow.
- Added slim and full Docker images, including the warm-container MCP pattern documented in the README and system design.
- Added the public archex vs. cocoindex-code comparison page with same-task head-to-head methodology and C1 evidence links.

### Changed

- Refreshed the README as the project front page with a concise hero, proof bar, audience routing, quickstart, trust and operations guidance, measured-results pointers, and current CLI/MCP/Python/Docker usage.
- Removed unnecessary local-repo positional arguments from README examples now that repo-local CLI commands default to the current working directory.
- Updated system design and roadmap documentation so shipped post-roadmap surfaces, historical execution records, and retrieval-default decision authority have one explicit documentation hierarchy.
- Let `archex symbol <symbol-id>` default to the current directory, matching the query, scout, and symbols local-default behavior.
- Supersedes the mistaken `0.9.1` patch release with the intended minor release number.

## 0.9.1 (2026-06-15)

### Changed

- Published with patch numbering by mistake; superseded by `0.10.0`.

## 0.9.0 (2026-06-13)

### Added

- Added working-tree delta indexing, query auto-refresh, and MCP watch mode so dirty local edits refresh indexes without full re-indexing.
- Added graph-query core, CLI commands, and MCP tools for neighborhood, path, and symbol dependency exploration.
- Added the scout protocol with token-capped map/fetch flows across Python API, CLI, MCP, and benchmark coverage.
- Added head-to-head benchmark harnesses and reports for external MCP adapter comparisons.
- Expanded tree-sitter language coverage through the language-pack runtime, chunk-only grammars, language-tier fixtures, and unknown-text fallback handling.

### Changed

- Improved graph expansion selectivity, benchmark diagnostics, architecture extraction hardening, and framework-semantic query normalization.
- Persisted edge-confidence provenance and embedding content-cache reuse to reduce redundant work across refreshes.

### Fixed

- Fixed stale-index query paths, MCP watch refresh rearming, delta vector persistence, and implementation-slice coverage runs.

## 0.8.0 (2026-06-09)

### Added

- Added architecture-quality benchmark tasks under `benchmarks/arch_tasks/` with hand-labeled module, pattern, interface, and decision oracles.
- Added `archex benchmark arch run`, `archex benchmark arch report`, and advisory `archex benchmark arch gate` commands for scoring analyze/explain output quality against labeled fixtures.

### Changed

- Improved Strategy pattern detection to aggregate protocol, concrete, and context evidence across files while rejecting unrelated shared-method false positives.
- Updated the benchmark pipeline entrypoint so `bash scripts/benchmark_pipeline.sh` works from any current working directory and still mirrors output to `.docs/pipeline.log`.
- Reframed the README benchmark pitch around `with archex` versus raw-file reading instead of previous-benchmark versus current-benchmark deltas.

### Fixed

- Fixed the implementation-gate pytest slice so `uv run pytest tests/analyze/ tests/benchmark/ -q` no longer fails on the repository-wide coverage floor after test bodies pass.

## 0.7.0 (2026-06-08)

### Added

- Added product-default token-efficiency gating for benchmark runs. `archex_query` now fails the gate when its measured savings falls below the calibrated floor.
- Added baseline-aware benchmark gating with `archex benchmark gate --baseline <dir>`. Baseline mode keeps token efficiency as a hard floor, fails recall regressions against the accepted baseline, and reports brittle absolute non-token rows as warnings.
- Added intent-routed context budgets so definition/symbol lookups produce tighter bundles while broad architecture questions keep larger budgets.
- Added `scripts/benchmark_pipeline.sh` as the standardized local benchmark, gate, and dogfood runner.

### Changed

- Reframed retrieval optimization around fewer tokens per retrieval/explanation query without recall regression.
- Changed benchmark `token_efficiency` to higher-is-better savings: `1 - returned_tokens / accessed_file_tokens`, clamped to `0..1`.
- Improved context assembly with file-diverse packing, nested line-range suppression, contextual query expansion, production-before-test ordering, and stronger self-query path alignment.
- Switched MCP savings accounting to use the honest raw candidate-file denominator and subtract only XML envelope scaffolding overhead.
- Refreshed the README around the product pitch, current local benchmark results, context-bundle architecture, and baseline-aware release gates.
- Removed unused Personalized PageRank graph-expansion helpers. Measured candidate ordering showed no recall gain on `archex_graph_expansion` or the external-large benchmark bucket, and introduced a `rust_tokio_runtime` candidate-recall regression risk.
- Corrected retrieval-pipeline documentation to describe the wired deterministic dependency expansion path instead of stale PPR wording.
- Kept the fast vector benchmark default after a CodeRankEmbed full-suite run crossed six hours at task 19/35; CodeRankEmbed remains pinned and configurable as `embedder="coderank"` for targeted evaluation.
- Switched the opt-in cross-encoder reranker default from `cross-encoder/ms-marco-MiniLM-L-6-v2` to pinned `jinaai/jina-reranker-v3`; the MiniLM model remains selectable via `IndexConfig(rerank_model=...)`.

### Fixed

- Fixed self-query retrieval regressions for repo index, query pipeline, MCP lifecycle, cache lifecycle, project reset, and pattern-detection questions.
- Fixed explicit `8192` budget handling in LangChain and LlamaIndex integrations so a caller-supplied default-sized budget remains an override.
- Fixed MCP `_meta.savings_pct` under-reporting by accounting for seed, expanded, and returned candidate files instead of only final bundle files.

## 0.6.2 (2026-05-25)

### Removed

- **HTTP API server removed.** Deleted the `archex serve` command, the FastAPI app (`serve/app.py`), the bundled dashboard (`serve/static/`), the API tests, and `fastapi`/`uvicorn` from dependencies. The server had no consumers — added in a single commit, never iterated, undocumented until 0.6.1 — and duplicated the MCP and CLI surfaces while taxing every `import archex` through an eager `serve/__init__` import. Use the MCP server (`archex mcp`), the Python API, or the CLI instead.

BREAKING CHANGE: the `archex serve` command and the `archex.serve.create_app` HTTP API are removed; `fastapi` and `uvicorn` are no longer dependencies. Removing the eager `serve/__init__` import also means `import archex` no longer loads FastAPI.

## 0.6.1 (2026-05-25)

### Changed

- Removed the redundant `web` optional-dependency extra. `fastapi` and `uvicorn` are core dependencies, so the HTTP API server (`archex serve`) works on a bare install; `archex[web]` is no longer a recognized extra.
- Simplified `archex serve` to import `uvicorn` directly instead of guarding an import that cannot fail (it is a core dependency).

### Fixed

- Corrected the cross-encoder reranker model name in the 0.6.0 changelog entry: the default is `cross-encoder/ms-marco-MiniLM-L-6-v2`, not Jina Reranker v2.

### Docs

- Refreshed the README to the current system: documented the `archex serve` HTTP API and its endpoints, the wired retrieval pipeline (BM25F, confidence-weighted RRF + adaptive RSF behind an AvgIDF fusion gate, intent-based weight routing, deterministic dependency expansion, opt-in `cross-encoder/ms-marco-MiniLM` reranking), the `vector-fast` and `graph` extras, and the `benchmark triage`/`readiness` subcommands. Corrected stale benchmark-task and test counts and removed dead links to gitignored artifacts.

## 0.6.0 (2026-05-21)

### Removed — LLM dependencies (breaking)

archex is now fully local, fully deterministic, no API keys, no per-query cost. All LLM-dependent surfaces have been removed from the core. This is a deliberate scope correction: the structural and retrieval-quality engineering (BM25F, vector embeddings, RRF/RSF fusion, cross-encoder reranking, dependency expansion, intent classification) was already at or above the measured contribution of the LLM lanes for most tasks, and the LLM lanes imposed adoption friction that did not pay back.

**Deleted from the codebase:**

- `src/archex/providers/` — `LLMProvider` protocol, `get_provider` factory, all provider implementations (`anthropic`, `openai`, `openrouter`).
- `src/archex/pipeline/summarize.py` — index-time LLM chunk summarisation (rolled back from the historical Phase 14.1 work).
- `src/archex/serve/query.py` — ReCo-style query augmentation (`augment_query`, rolled back from the historical Phase 13.1 work).
- `archex_query_fusion_rerank_augment` benchmark strategy and the `--augment` CLI flag.
- `openai` and `anthropic` extras in `pyproject.toml`.

**Breaking changes to public API:**

- `Config.provider`, `Config.provider_config`, `Config.enrich` — fields removed. Passing them raises `pydantic.ValidationError`.
- `analyze(..., enrich=True)` — the `enrich` kwarg no longer exists. `analyze()` produces structural `ArchDecision` records only.
- `ArchDecision.source` — `Literal["structural", "llm_inferred"]` collapsed to `Literal["structural"]`. Code constructing `ArchDecision(source="llm_inferred")` must change to `"structural"`.
- `infer_decisions(patterns, modules, interfaces)` — `provider` kwarg removed.
- `produce_artifacts(...)` — `llm_provider` kwarg removed.

**Migration:**

- Remove `provider=`, `provider_config=`, `enrich=`, and `llm_provider=` kwargs from any direct API calls.
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` are no longer consulted by archex.
- If you used `archex benchmark run --augment`, the LLM-free equivalent is `archex benchmark run --query-fusion --rerank` (BM25F + vector + RRF/RSF + local cross-encoder).

**What stayed (still local, still LLM-free):**

- Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — local sentence-transformers model, not an LLM API.
- Vector embeddings (FastEmbed, CodeRankEmbed) — local ONNX/Torch.
- All structural analysis (Louvain modules, pattern catalog, interface extraction, dependency graph expansion).
- The full 25-task benchmark corpus, now runnable end-to-end with zero API calls.

### Added — Per-dimension compare templates

`compare()` is now a per-dimension package (`src/archex/serve/compare/`). Each dimension (`error_handling`, `api_surface`, `state_management`, `concurrency`, `testing`, `configuration`) ships its own `Evidence` dataclass, structural extractor, and renderer.

- Output replaces generic "N indicator(s) detected" wording with quantitative metrics: pattern counts, return-type/docstring coverage percentages, async-interface ratios, persistence/test/config library presence, file-ratio metrics.
- `DimensionComparison` and `ComparisonResult` contracts unchanged; existing CLI, MCP, and JSON renderers continue to work.
- Trade-off classifier emits specific deltas (`"substantially more"`, `"comparable"`, `"none"`) and library-presence callouts (`"A adopts X not seen in B"`).

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
