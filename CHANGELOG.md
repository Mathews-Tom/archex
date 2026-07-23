# Changelog

## [Unreleased]

## [0.23.0] - 2026-07-24

### Added

- **M5: local explorer for the canonical diff-review artifact.** Added `archex explore ARTIFACT [--graph GRAPH] [--port N]`: a loopback-only, session-token-gated local HTTP server rendering a previously exported `AnalysisArtifactV1` (`archex report diff --format json`) and an optional `ArchGraph` (`archex graph export`). Five read-only views project the artifact without any new source parsing or graph-edge construction: Module Map (node counts aggregated by module, not a force graph), Target Neighborhood (a bounded traversal via the existing `GraphQuery.neighbors`, reached through a plain HTML search form), Diff Review, Receipt Inspector (freshness/completeness/confidence/evidence/exclusions/unknowns), and Index Health (schema/parser/config identity). Every response carries a restrictive Content-Security-Policy plus `X-Content-Type-Options`/`X-Frame-Options`/`Referrer-Policy`/`Cache-Control` headers; every request is validated against the server's own `Host` header (DNS-rebinding defense) and a per-process session token (hardened cookie: `HttpOnly`, `SameSite=Strict`); the bind address is hardcoded to loopback (`127.0.0.1`/`::1`) with no wider-bind option; and the server is GET-only with no client-side JavaScript at all. Declared 10k/100k-node projection budgets and a new-contributor orientation usability proxy (real HTTP navigation timing against an objectively correct answer, not self-reported satisfaction) are documented in `docs/EXPLORER_USABILITY_EVIDENCE.md` and reproducible via `scripts/m5_explorer_projection_benchmark.py` / `scripts/m5_explorer_usability_evidence.py`.

## [0.22.0] - 2026-07-24

This release combines three deferred release trains that were never individually tagged or published — `v0.20.0` (M2), `v0.21.0` (M3, M10), and `v0.22.0` (M4) — into a single cut, per `.docs/DEVELOPMENT_PLAN.md` §2.1's release-preparation policy. Each entry below is labeled with the version it was originally planned for.

### Added

- **M2 (originally targeted `v0.20.0`): warm QueryRuntime, generation-safe caching, and named retrieval profiles.** Repeat `query()` calls against the same repository generation within one long-lived process (the MCP server) now reuse a cached BM25 index, dependency graph, and hydrated chunk list instead of re-hydrating from SQLite on every call. A new canonical `generation_id` (`archex.serve.generation`) — derived from schema version, resolved commit/working-tree state, live chunk/file counts, and every retrieval-affecting `IndexConfig` field — lets `QueryRuntime` (`archex.serve.runtime`) validate a cached snapshot with one cheap metadata read; a real content, config, or revision change always produces a different `generation_id` and forces a rebuild, so no query can serve stale data past a real change. `run_stdio_server` now owns one `QueryRuntime` for its process lifetime; `query()`, MCP's `handle_query_repo`/`query_repo` tool, and the CLI's `archex query` command all accept it as an explicit, opt-in parameter (default `None`, byte-equivalent to the pre-M2 path). Also adds named `fast`/`balanced`/`deep` retrieval profiles (`RetrievalProfile`, `archex.serve.profiles.index_config_for_profile`, `archex query --profile`, MCP `query_repo(profile=...)`) as convenience `IndexConfig` presets — `fast` is bm25-only (equivalent to `IndexConfig()`'s own defaults, zero vector/model thread work), `balanced` adds module-responsibility prefiltering, `deep` adds vector search and cross-encoder reranking; `RetrievalMetadata.retrieval_profile` records which profile actually determined the effective config for receipt route decisions. The parser pipeline's per-file symbol/import extraction now reads each file's bytes exactly once (`TreeSitterEngine.read_and_parse_file`) instead of once for the tree-sitter parse and again for symbol/chunk-range/import extraction. **Measured** (`scripts/m2_warm_runtime_benchmark.py`, `benchmarks/evidence/m2/`): on the self-repo (186 files, ~59k lines), warm p95 latency improved 25.5% (407ms → 304ms) with recall/required-file-recall byte-identical to the pre-M2 path across all 16 self-benchmark tasks and no RSS regression (−0.6%); on a larger external fixture (django/django, 2927 files, ~524k lines), warm p95 improved 47.1% (1089ms → 576ms) with no RSS regression (+0.02%). "Phase-level progress and partial-ready lexical status" and "background incremental publication retaining the verified previous snapshot" are deferred to a follow-up round — both require a materially different async-orchestration architecture beyond generation-keyed caching.

- **M10 (originally targeted `v0.21.0`): primary agent-facing context() facade.** Added `archex context` (CLI), the `context` MCP tool, and `archex.api.context()` (Python) as the documented primary agent path: `query, intent, profile, filters, budgets, handles` in; a compact candidate map, exact fetch handles, selected code, relation paths, a route decision, a receipt, and a recommended next action out. It is a thin facade over the existing `query()` runtime — no new ranking, provider, or UI behavior — and every existing specialized tool (`query`, `scout`, `symbol`, `query_repo`, `scout_repo`, ...) remains fully supported and unchanged. New deterministic post-retrieval `filters` (include/exclude path globs, language allowlist) move excluded candidates into the existing receipt's `skipped_candidates` under a new `filter_excluded` reason instead of silently dropping them. The facade reaches a useful result on a plain, no-model install (`.smoke-venv/bin/archex context "how does archex score and rank retrieval candidates"`), exercised in CI alongside the existing `archex query` clean-install smoke.

- **M4 (originally targeted `v0.22.0`): AnalysisArtifactV1 and static diff review.** Added `archex report diff --base <ref> --format json|markdown|html` and `archex report delta --base <ref> --format json|markdown`: one versioned, read-only `AnalysisArtifactV1` (`archex.report.artifact`) projecting the existing verified index, dependency graph, and deterministic impact classifier, with full identity/provenance (archex/source/index/parser/chunker/config versions), freshness, completeness, confidence, evidence locations, and exclusion/unknown counts. JSON, Markdown (with a bounded Mermaid structure diagram), and a single self-contained offline static HTML renderer all project the same semantics without reinterpreting them; the artifact is source-redacted by construction (path/line/handle identity only, never raw source text), and every list is bounded with a companion `*_total` field. The HTML renderer's path/line references are clickable `vscode://file/` links built from the artifact's own `source_root` and hunk/symbol line numbers -- a purely local, offline URI scheme, never a network request. `archex report delta` compacts an artifact into a CI-log-sized summary; `.github/workflows/report-diff.yml` is a read-only (`permissions: contents: read`), commit-SHA-pinned example that builds and uploads both on pull requests.

### Changed

- **M3 (originally targeted `v0.21.0`, folded into M10's release per plan; no standalone release triggered by this milestone alone): established a sealed, task-family-specific external quality frontier.** Formalized the pinned-external-repo task convention into an enforceable policy (`archex.benchmark.external_corpus`: `is_pinned_commit` rejects floating refs, `find_vocabulary_leaks`/`find_ci_sealed_references` prove sealed `task_id`s never key production logic or CI execution) and added a sealed chronological holdout corpus (`benchmarks/sealed_tasks/`) gated behind a new `--allow-sealed-corpus` flag on `benchmark run`/`gate`, so sealed evidence can never fold into a default/bounded run by accident. Added `archex benchmark scorecard`, producing scorecards sliced by language, repository size, query intent, and task family from two new per-result metrics (`duplicate_rate`, `repo_size_class`), each row carrying the exact `task_ids` behind it for audit. Added a fixed-agent downstream trajectory signal (`post_bundle_search_turns`) modeling an agent that must search for each missing required file rather than an oracle that already knows where it is, wired automatically into every `archex_query`-family benchmark result with no external evaluator, LLM call, or network access. Added a candidate lane matrix comparing the default/cAST chunkers against the product's real `fast`/`balanced` retrieval profiles, and hardened the promotion gate with zero-recall, per-language-family, and fixed-agent regression checks so an improved aggregate can never mask a hidden regression. **Verdict: NO-GO for automatic default promotion of any candidate** — `fast`/`balanced` matched `archex_query`'s default bit-for-bit on recall/F1/MRR/downstream-success but could not clear the gate (an unmeasured warm-latency interaction under this environment's runner, documented rather than root-caused), and cAST measurably regressed recall/F1/downstream-success and failed its own absolute-threshold gate outright. `archex_query` (default chunker) remains the product default; see `docs/external-quality-frontier.md` for the full reproducible run.

### Fixed

- **M0.3 (deferred pending the M0.4 recovery decision; released now since M0.4 reached a terminal NO-GO rather than remaining open): deterministic ranking and result-set recovery candidate.** Adds a benchmark-only `archex_query_rank_candidate` lane, built on M0.2's coverage candidate, that reorders identifier-tier-evidenced files toward the front of the candidate-admitted tail (the base query's own ranking is never displaced) and bounds the seed/neighbor admission caps when direct evidence is concentrated. Two same-revision 64-task local runs show zero required-file-recall regressions, M0.2's five-target coverage fully preserved, and measured (if modest) precision/F1 gains with zero regressions on any task. `archex_query` remains unchanged; the candidate is retained as evidence for M0.4, not promoted.

- **M1 (deferred pending the M0.4 recovery decision; released now since M0.4 reached a terminal NO-GO rather than remaining open): benchmark and runtime measurements distinguished timing and retained provenance.** `BenchmarkResult`'s timing fields (`wall_time_ms`, `warm_latency_ms`) are now nullable; renderers, aggregate reports, and latency gates render `n/a` for unmeasured timing rather than substituting cold wall time for warm latency. `query()`'s runtime timing phases (`acquire`/`parse`/`index`/`search`/`assemble`) now account for at least 95% of measured wall time on cold, cached-search, and cached-passthrough paths (previously as low as 45% on cached search, since index-preparation work between the cache-hit check and the search phase went unmeasured). Every published `BenchmarkReport` records reproducible provenance (`BenchmarkProvenance`): Archex version, evaluated source commit, generation timestamp, hardware manifest, and full retrieval/task/strategy configuration. Production query expansion (`_expand_retrieval_question`) now returns and propagates its own provenance through `RetrievalMetadata`/`ContextReceipt`, and a regression guard confirms production code (outside the benchmark engine) never hardcodes external-benchmark vocabulary.

## [0.19.2] - 2026-07-11

### Fixed

- **M0: verified index publication and lifecycle integrity.** Index generation now publishes verified retrieval state atomically with a generation manifest and provenance for parsed symbols, edges, unresolved imports, and exclusions. Total parser-worker failure fails closed; strict parsing reaches the orchestrator; repository identity is canonical across equivalent local paths; reset removes SQLite WAL/SHM sidecars; `doctor` validates enabled retrieval stores; and CI exercises a clean-install BM25 index/query smoke path.


## [0.19.1] - 2026-07-10

### Fixed

- **Ephemeral index-store scratch directories leaked on every warm-cache query.** `_ensure_index()`'s cache-hit path, both delta-index attempt stores, `_full_reindex_in_place()`'s fallback rebuild, and `sync_imported_artifact()` each opened an `IndexStore` backed by a `tempfile.mkdtemp()` scratch directory that was never reliably closed — the cache-hit path in particular runs on nearly every warm-cache `archex query`/`archex scout` call, so the leak accumulated on ordinary use, not just error paths. All five sites now close their store (and the scratch directory that closing triggers) on every exit path, including mid-pipeline exceptions and `KeyboardInterrupt`. `archex index`'s summary output also no longer reports an index location that the same cleanup path has already deleted.
- **Delta-benchmark scratch cache directory never cleaned up.** `run_delta_benchmark()`'s `archex-delta-cache-` working directory, created via `tempfile.mkdtemp()` for isolated benchmark corpus copies, was never removed on any exit path. Now wrapped in try/finally alongside the existing cache-directory lifecycle.
- **MCP graph-query results could go stale after a re-export.** The `archex mcp` daemon's cached graph-query handles (keyed by artifact path and mtime) were not invalidated when `archex graph export` rewrote an artifact at the same path within the same mtime coarseness window, letting a long-running daemon serve outdated graph data. Cache entries are now invalidated on artifact re-export and no longer accumulate unboundedly.

### Security

- **Git URL scheme allowlist hardened against scp-like and case-variant bypasses.** `archex`'s remote-acquisition path could reach `git clone` with a bare `user@host:path` scp-like address (bypassing the intended http(s)-only allowlist and reaching a real outbound SSH connection to an attacker-chosen host) or with leading/trailing whitespace around an otherwise-disallowed scheme. The real acquisition path (`_acquire()`/`resolve_source()`) now routes through the same hardened, case-insensitive scheme check that already existed but was previously bypassed by a duplicate, laxer gate ahead of it.
- **Artifact decompression is now bounded against decompression bombs.** `lzma.decompress(memlimit=...)` only bounds decoder dictionary memory, not decompressed output size — a crafted low-preset artifact could expand a small compressed payload to hundreds of megabytes before the length check ever ran. Artifact import now enforces a maximum decompressed-size ceiling and rejects an oversized declared header length before attempting to read it.


## [0.19.0] - 2026-07-08

### Added

- **Guided first-run onboarding with `archex setup`.** Added a primary setup command that performs preflight checks, initializes repo-local state, builds or refreshes the first index, checks MCP runtime readiness, offers detected client and agent-guidance registration, configures privacy-aware metrics, optionally installs supported hooks, and prints exact next commands.
- **Repo-ready `archex init` by default.** `archex init` now builds the repository index as part of initialization so a new repo is immediately ready for `archex query`; `--no-index` preserves the state-only escape hatch, and advanced index options can still be routed through explicit indexing flows.
- **Interactive client discovery.** Bare `archex install-client` now discovers supported client config paths, guides TTY users through selected registrations, supports explicit noninteractive all-detected automation, and preserves direct `install-client <client>` compatibility.
- **Privacy-aware metrics setup.** Added `archex metrics setup` and clearer direct metrics command output describing local-only counters, opt-in trace capture, and retention/sensitivity boundaries before enabling metrics.

### Changed

- **MCP is a default runtime dependency.** The `mcp>=1.0` dependency now ships in the core install so `uv tool install archex && archex setup` can complete MCP/client onboarding without requiring a mid-flow optional-extra reinstall.
- **Public onboarding docs now match the product path.** README, the installation trust contract, the client compatibility matrix, and local metrics documentation now describe guided setup, CLI-only initialization, bare client discovery, metrics privacy, and MCP registration versus runtime startability.

### Fixed

- **MCP registration is no longer treated as MCP health.** `doctor` and client setup now distinguish a written MCP client registration from a startable local `archex mcp` runtime, preventing the issue #457 failure mode where `doctor` reported MCP OK while the MCP server could not start. `archex mcp` also prints uv-tool-appropriate remediation guidance when runtime dependencies are missing.

## [0.18.0] - 2026-07-06

### Added

- **Minimal-by-default JSON chunk output, `--full` escape hatch.** `archex query --format json` and `archex scout --format json` now omit `CodeChunk`/`RankedChunk` fields that are unset (`None`) or empty (`""`) — `symbol_name`, `symbol_kind`, `symbol_id`, `qualified_name`, `visibility`, `signature`, `docstring`, `summary`, `imports_context`, `breadcrumbs` — while always keeping `structural_score`/`type_coverage_score`/`cohesion_score`/`relevance_score`/`final_score` regardless of value, since a real `0.0` score is signal, not absence. A new `--full` flag on both commands restores the previous unconditional dump. `render_xml` and `render_markdown` are unchanged — XML was already minimal by hand-curation before this change. This only affects callers who pass `--format json`; the CLI's default format remains `xml`.
- **Optional TOON output format (`archex query --format toon`).** A new token-oriented encoding, gated behind the optional `archex[toon]` extra (`toons>=0.7.0`) so the core install and `smoke-min-install` CI job are unaffected. Reuses the JSON renderer's minimal-by-default field selection, so TOON inherits the same slim/`--full` behavior. Running the command without the extra installed fails with an actionable `uv add 'archex[toon]'` message instead of a raw traceback. Measured on the representative fixture bundle in `tests/serve/test_renderers.py::test_toon_smaller_than_json_for_realistic_bundle`: TOON output is ~17% smaller than the already-slimmed default JSON output for the same bundle — a per-bundle, opt-in reduction that requires explicitly choosing `--format toon`, not a change to any default output path.

## [0.17.0] - 2026-07-06

### Added

- **Opt-in, non-blocking tool-call hook integration across six clients.** `archex install-client <client> --hooks` (`--remove-hooks` to uninstall) now installs a per-client hook wired to the same `python -m archex.integrations.hook` lookup/timeout/freshness engine, so a Grep/Glob-shaped tool call can be augmented with archex symbol-search results without a manual `query`/`scout` call. Every client shares one contract regardless of mechanism: opt-in only (never installed by plain `install-client`), exits/returns non-blocking on every path (a missing or stale index, a timeout, a malformed payload, or any internal error all degrade silently rather than blocking or erroring the calling tool), a hard ~500ms lookup budget (`ARCHEX_HOOK_TIMEOUT_SECONDS`), failures logged to `~/.archex/hook-diagnostics.log` (`ARCHEX_HOOK_DIAGNOSTICS_LOG`) rather than surfaced to the agent, and `Read`/`beforeReadFile` is never a match target on any client. Install/uninstall is idempotent and non-destructive to unrelated config in the same file.
  - Claude Code: a `PreToolUse` hook (`src/archex/integrations/hook.py`) matched on `Glob|Grep`, written to `settings.json`, injecting results as `additionalContext`.
  - oh-my-pi and Pi: an identical shared TypeScript extension module (`archex-hook.ts`, auto-discovered from each host's own extension directory) registering `pi.on("tool_result", ...)` scoped to `grep`/`glob`-equivalent calls (Pi's glob-equivalent is `find`), translating each client's field names onto the subprocess's contract at the edge before shelling out to the same Python engine.
  - OpenCode: a standalone `tool.execute.after` plugin file (auto-loaded from `.opencode/plugins/` or `~/.config/opencode/plugins/`, no `opencode.json` entry needed) scoped to native `grep`/`glob` tool ids only — MCP-routed tool calls are explicitly excluded given a confirmed output-shape inconsistency in OpenCode's own `.after` hook for MCP tools; confirmed reachable for subagent-dispatched calls, not just top-level ones.
  - Codex CLI and Cursor ship a **diagnostics-only** fallback rather than a matching augmentation feature, because neither client exposes a safe Grep/Glob-equivalent hook target: Codex's only `PreToolUse` tool name broad enough to catch search is `Bash` (every shell command, including destructive ones), so `codex_hook.py` matches `^Bash$`, detects search-shaped commands, and only logs what would have been surfaced; Cursor's `beforeSubmitPrompt` hook (the closest thing it has to content-bearing) has no context-injection output field at all, so `cursor_hook.py` fires once per submitted prompt and only logs a withheld match. Both reuse the Claude Code hook's lookup/timeout/diagnostics engine in-process (`hook.py`'s helpers were promoted to a public API for this) rather than duplicating it.
- **`install-client --hooks` help text** now states exactly which clients augment (`claude-code`, `omp`, `pi`, `opencode`) versus ship the diagnostics-only fallback (`codex`, `cursor`), so the CLI's own `--help` output never implies a capability a client can't deliver.

## [0.16.0] - 2026-07-06

### Added

- **Hard-fail benchmark latency gate:** `archex benchmark gate` gains `--max-latency-ms <n>`, wired into `QualityThresholds.max_latency_ms` and `check_latency_violations()`. This is distinct from the existing advisory-only `--warn-latency-ms` / `check_latency_warnings()`, which only ever prints a warning and never fails the gate. When `--max-latency-ms` is set and any result's wall-time exceeds it, the CLI prints `LATENCY GATE FAILED` naming the offending task/strategy/latency and raises `SystemExit(1)` before any other gate check runs, independent of whether `--baseline` is also supplied. Default is `None` (disabled), so existing gate invocations are unaffected until the flag is passed.
- **Three new MCP tools — `get_impact`, `explain_target`, `generate_onboarding`:** Agents connecting to archex only via MCP (no shell/CLI access) can now invoke blast-radius analysis, symbol/file/module explanation, and onboarding-guide generation. All three handlers live in `src/archex/integrations/mcp.py` and follow the existing `_meta`-envelope pattern (`compute_meta`, JSON/Markdown format switch) used by the other tools, wrapping already-shipped CLI backends without changing their underlying logic: `get_impact` wraps the existing `archex impact` CLI command's git-diff and explicit-changed-file blast-radius analysis; `explain_target` wraps the existing `archex explain` CLI command's file/symbol/module context extraction; `generate_onboarding` wraps the existing `archex onboard` CLI command's Markdown-only onboarding guide. Each tool can index a source repo directly or read a previously-exported graph artifact. Parity tests assert byte-for-byte structural equivalence between each new tool's output and its CLI counterpart across every supported mode, plus required-field and mutual-exclusivity error-path coverage. `archex mcp`'s `list_tools()` result count moves from 14 to 17.
- **Pre-promotion regression baseline:** Captured a frozen `Baseline` snapshot of the self-repo dogfood corpus at `.archex/baselines/pre-promotion.json` — recall, precision, F1, MRR, nDCG, MAP, and token efficiency for all 16 self-repo tasks across the `raw_files`, `raw_grepped`, and `archex_query` strategies. Generated via the existing `archex dogfood` + `archex benchmark baseline save` tooling and verified clean (zero regressions) before committing. Added a `.gitignore` carve-out so `.archex/baselines/` stays git-tracked despite the rest of `.archex/` remaining a gitignored local cache/workspace directory.
- **Ranking-stability regression check:** Extended `Baseline` with an optional `ranking: list[RankingSnapshotEntry]` field (`file_path`, PageRank `structural_centrality`, `symbol_count`) and a new `build_ranking_snapshot(repo_root)` helper that indexes a repo and pairs each file's `DependencyGraph` centrality with its chunk-derived symbol count; backward compatible — existing recall-only baseline JSON loads with `ranking` defaulting to `[]`. Added `check_ranking_stability(current, baseline, thresholds)` — a tie-safe Spearman rank correlation (numpy-based, no new dependency) computed independently over `structural_centrality` and `symbol_count`, flagging a `RankingGateViolation` when either metric's correlation drops below its threshold (default `0.8` for both). This catches the case where widening symbol extraction to more languages floods the graph with many low-value symbols and reorders ranking for files that were never touched, without moving recall/precision/F1 at all. `run_dogfood` now builds a live ranking snapshot and runs this check whenever the loaded baseline carries a non-empty `ranking` field; a recall-only baseline skips the check entirely at zero extra indexing cost. `DogfoodRunResult` gains `ranking_violations`, and the dogfood CLI now exits non-zero on either a recall regression or a ranking-stability violation. `archex benchmark baseline save` gains an optional `--ranking-source <path>` flag to attach a ranking snapshot when capturing a baseline. Added `docs/LANGUAGE_PROMOTION_GATE.md`, documenting what the gate checks, where the baseline artifact lives and why it is a git-tracked exception under `.archex/`, and how a language-tier promotion re-runs the gate.
- **Full-tier language promotions: PHP, Ruby, Scala, C, C++.** `.php`, `.rb`, `.scala`/`.sc`, `.c`/`.h`, and `.cc`/`.cpp`/`.cxx`/`.hpp`/`.hh`/`.hxx` files move from `LanguageTier.CHUNK_ONLY` to `LanguageTier.FULL`, each gated on the language-promotion regression gate against the committed pre-promotion baseline. See **BREAKING CHANGE** below.
  - PHP: added `src/archex/parse/adapters/php.py` plus a `tests/fixtures/php_simple/` corpus and 49 adapter tests. `extract_symbols` walks `namespace_definition` (both semicolon- and brace-style — a namespace contributes only a `.`-joined qualified-name prefix to its contents, never a standalone symbol), then `class_declaration` → `SymbolKind.CLASS`, `interface_declaration` → `SymbolKind.INTERFACE`, `trait_declaration` → `SymbolKind.CLASS` (no dedicated trait kind exists in the model), `enum_declaration` → `SymbolKind.ENUM` with each case → `SymbolKind.CONSTANT`, methods → `SymbolKind.METHOD`, properties and PHP 8.0+ constructor-promoted parameters → `SymbolKind.VARIABLE`, and class constants → `SymbolKind.CONSTANT`. Visibility defaults to `Visibility.PUBLIC` when no modifier is present. `parse_imports` handles simple, aliased, grouped (`use A\{B, C};`), and `use function`/`use const` forms. `resolve_import` scores candidate files by PSR-4-style trailing-namespace-segment-to-directory-segment overlap, falling back to a basename-only match for unnamespaced (global) imports. `detect_entry_points` matches `index.php`/front-controller basenames plus shebang and `php_sapi_name()` textual markers.
  - Ruby: added `src/archex/parse/adapters/ruby.py` plus `tests/fixtures/ruby_simple/` (including a dedicated `mixins/` subdirectory) and adapter tests covering symbol/import extraction, `require` resolution, and entry points/visibility. `module`/`class` declarations extract as `SymbolKind.MODULE`/`SymbolKind.CLASS`; instance and singleton (`def self.x`) methods both extract as `SymbolKind.METHOD`, distinguished by signature and, for a singleton method on an explicit named receiver, a qualified parent derived from that receiver; top-level constant assignments extract as `SymbolKind.CONSTANT`. `private`/`protected`/`public` visibility calls are tracked both as an ambient mode and retroactively by method name when passed explicit arguments; `class << self` singleton-class bodies recurse with every contained method forced to singleton form. `resolve_import` resolves `require_relative` relative to the requiring file and plain `require` via a repo-wide load-path search. The fixture corpus specifically exercises `include`/`extend`/`self.included(base)` mixin idioms to prove the bundled grammar does not corrupt declaration boundaries around them. `detect_entry_points` matches `app.rb`/`main.rb`/`server.rb`/`config.ru`/`Rakefile` basenames plus Ruby shebang lines.
  - Scala: added `src/archex/parse/adapters/scala.py` plus `tests/fixtures/scala_simple/` and 46 adapter tests. `class_definition`/`object_definition` both extract as `SymbolKind.CLASS` (matching the existing Kotlin `object_declaration` → `CLASS` precedent), so a `class Foo`/`object Foo` companion pair intentionally shares one `qualified_name` and `kind`, with the resulting `symbol_id` collision resolved by the existing overload-disambiguation mechanism. `trait_definition` extracts as `SymbolKind.INTERFACE`. Member `val`/`var` definitions extract as `CONSTANT`/`VARIABLE`; functions extract as `FUNCTION` (top-level) or `METHOD` (member). Both semicolon-style and brace-style `package` declarations are handled. `parse_imports` expands `import a.b.{C, D => E, _}` selector groups into one `ImportStatement` per name. `resolve_import` reuses the shared JVM package-to-directory-segment resolver already backing Java/Kotlin. `detect_entry_points` matches `Main.scala`/`App.scala`/`Boot.scala` basenames plus `extends App` and explicit `def main(args: Array[String]` markers.
  - C: added `src/archex/parse/adapters/c.py` plus `tests/fixtures/c_simple/` and 39 adapter tests. Function definitions and prototypes share one extraction path and both report `SymbolKind.FUNCTION`. Visibility derives from the `static` storage-class specifier. Struct extraction covers bare, K&R combined, typedef-anonymous, and typedef-named forms, each reporting `SymbolKind.TYPE` (never `CLASS`); forward declarations are excluded. Symbol extraction and `#include` parsing both walk through a shared helper that flattens declarations nested inside `#ifdef`/`#if`/`#elif`/`#else` preprocessor conditionals and `extern "C" { ... }` linkage blocks. `resolve_import` treats a quoted `#include "..."` as potentially local and an angle-bracket `#include <...>` as always external. `detect_entry_points` regex-matches a `main` function definition (not a prototype) in `.c` files only.
  - C++: added `src/archex/parse/adapters/cpp.py` plus `tests/fixtures/cpp_simple/` and 71 adapter tests. `class_specifier` extracts as `SymbolKind.CLASS` and `struct_specifier` as `SymbolKind.TYPE`, each with a different default member visibility matching real C++ semantics. Namespaces — classic, C++17 nested, and anonymous — extract as `SymbolKind.MODULE`. Free/member functions, constructors, destructors, and operators extract as `FUNCTION`/`METHOD`; data members extract as `VARIABLE`, or `CONSTANT` for `static const`. `qualified_name` deliberately omits parameter-type signatures to avoid corrupting the shared parent-qualified-name lookup every language adapter's outline nesting depends on; overloads share one `qualified_name` and are disambiguated by the existing `symbol_id` mechanism, while template specializations get a genuinely distinct `qualified_name` by construction. Header/impl-split out-of-class member definitions resolve via a full-text namespace-qualifier split, correct at any nesting depth, so a header prototype and its `.cpp` definition resolve to the same `qualified_name` in different files with no collision and no orphan. `resolve_import` is extension-agnostic across the whole-repo file map, so a `.cpp` file including a C-tier `.h` header resolves the same way a C file's own include would.
- **New `LanguageTier.STRUCTURED` tier.** Added `LanguageTier.STRUCTURED` between `FULL` and `CHUNK_ONLY`, and audited every call site that branches on language tier — `languages.py` gains a `STRUCTURED_LANGUAGE_IDS` export; `archex doctor`'s grammar-availability report prints a distinct "structured grammars: N/M available" line and treats a missing STRUCTURED grammar as non-fatal, the same as chunk-only. Added the shared `StructuredAdapter` base: `extract_symbols` is declared `@final` and unconditionally returns `[]`, architecturally enforcing that no STRUCTURED adapter can ever claim a programming symbol — confirmed with adversarial tests that parse XML/CSS content literally named like symbols and still get zero symbols and zero references back. STRUCTURED tier means outline + native cross-file reference edges, explicitly no programming-symbol claim.
- **HTML STRUCTURED adapter with local reference extraction.** `HtmlAdapter` now extracts local, same-repo references from `script[src]`, `link[href]`, `img[src]`, and `a[href]` attributes — external URLs and bare fragments are filtered, and matched targets become `ImportStatement`s that flow through the normal import-resolution path. `archex outline` now prints an `outline:` section (structural chunk ranges) and a `references:` section (the raw extracted local paths). `html` (previously `CHUNK_ONLY`) is registered at `LanguageTier.STRUCTURED`; existing HTML chunk boundaries are unaffected — only reference extraction and tier reporting are new.
- **XML-generic, YAML, Markdown, and CSS STRUCTURED adapters.** Four more previously-`CHUNK_ONLY` languages move to `LanguageTier.STRUCTURED`, each extracting only the cross-reference syntax its format actually defines, confirmed with adversarial negative-case tests (symbol-shaped XML/CSS names, Markdown fenced code, and JSON/TOML left explicitly untouched at `CHUNK_ONLY`): generic `XmlAdapter` is outline-only, since well-formed XML has no universal reference syntax; `YamlAdapter` resolves `&anchor`/`*alias` pairs within a document; `MarkdownAdapter` extracts inline and reference-style link/image destinations plus intra-document `#heading` section-anchor fragments; `CssAdapter` extracts `@import` and property-value `url()` references.
- **Maven POM XML dialect plugin.** `xml_maven.py` layers Maven-specific dependency-edge extraction on top of the generic XML adapter for `pom.xml` files specifically — not generic XML. `is_maven_pom()` requires both the `pom.xml` filename and a `<project>` root element, guarding against Apache Ant's `build.xml` and any file merely named `pom.xml`. `extract_maven_dependencies()` walks only the project's direct `<dependencies>` block, deliberately excluding `<parent>` inheritance and `<dependencyManagement>`/`<profiles>`-nested lists, and emits each `<dependency>`'s `groupId:artifactId[:version]` coordinate as a reference. `resolve_maven_dependency()` resolves a coordinate to another module's `pom.xml` only when exactly one `pom.xml` in the repo lives in a directory named after the artifact ID, leaving anything ambiguous or external (e.g. Maven Central coordinates) unresolved. Verified end-to-end against a real multi-module fixture through `GraphQuery.neighbors`/`shortest_path`.
- **Local-fixture benchmark repos and a polyglot fixture.** `archex benchmark run --tasks-dir` task specs can now point `repo:` at an existing local fixture directory in addition to `"."` or a GitHub `owner/repo` slug. Added a four-repo HTML+JS+CSS polyglot fixture and paired required-file-recall tasks, with two of the four repos deliberately giving their JS/CSS assets the same basename to exercise same-basename reference collisions and the other two serving as non-colliding controls.
- **Portable index artifact export/import (`archex index --export-artifact <path>`, `archex init --from-artifact <path>`).** `archex index --export-artifact` writes a `VACUUM INTO`-compacted, stdlib-`lzma`-compressed copy of the repo-local index database with the FTS5 derived tables stripped before compression (fully re-derivable, would otherwise duplicate corpus text). The artifact is a custom framed container — a magic header, a length-prefixed UTF-8 JSON header (format/compat version range, index revision, schema version, chunk/file counts), then the compressed payload — with the header validated before the payload is ever decompressed. `archex init --from-artifact` imports the header, rejects a version outside the artifact's declared compat range with a loud error (never a silent partial import), rebuilds the FTS5 tables locally, then diffs the artifact's recorded file states against the current working tree and applies a targeted delta sync when the change ratio is below a configurable threshold, or falls back to an ordinary full re-index with a loud warning above it. Measured on this repository: roughly 8x compression and roughly 2.5x faster than a full re-index, with byte-identical file/chunk counts confirming correctness. Documented in full in `docs/PORTABLE_INDEX_ARTIFACT.md`.
- **`.gitattributes` auto-management for exported artifacts.** Export calls `ensure_artifact_gitattributes()`, which writes (or idempotently updates) a `merge=ours -diff` entry for the artifact's repo-relative path so committing the binary artifact never produces a binary merge conflict. Because `.gitattributes` alone cannot activate the `ours` merge strategy, export also best-effort registers `git config merge.ours.driver true` on the exporting machine; every other clone must run that one-line command once. The edit is written to disk only, never staged or committed automatically.
- **Identifier-aware BM25 tokenization, opt-in (`IndexConfig.identifier_fragment_tokenization`, default `False`).** The BM25 index can now optionally split identifiers on camelCase/PascalCase and snake_case boundaries for the `symbol_name` and `breadcrumbs` FTS columns, lowercasing and deduplicating fragments and appending them alongside the original text — additive, never replacing the original searchable token. A composite schema-version value is checked on every index construction so a stale store or a flipped flag transparently triggers a rebuild rather than silently serving stale content. **Measured, not shipped enabled:** a dedicated identifier-fragment benchmark corpus showed mean recall dropping from 0.556 to 0.333 and mean MRR from 0.444 to 0.278 with the flag on, root-caused to fragment collisions among related PascalCase symbols that used to tokenize as distinct opaque tokens. Per the project's measure-then-claim discipline, this ships **disabled by default** — the tokenization implementation and schema-versioning machinery are merged and tested as reusable infrastructure, but general-purpose fragment expansion is not turned on.
- **Diff-scoped impact analysis with per-symbol risk classification (`archex impact --diff [<ref>]`, MCP `get_impact(diff=...)`).** `archex impact` gains an optional-value `--diff` option (bare `--diff` defaults to `HEAD`; `--diff <ref>` diffs that ref against the working tree). A new diff-hunk parser maps changed line ranges onto the symbols (chunks) they overlap, and each affected file is classified into a risk level (LOW/MEDIUM/HIGH) from three deterministic, file-scoped graph signals: structural centrality well above the graph's mean, at least three distinct direct importers, and reachability from an entry point within two upstream import hops. A file that is itself an entry point is always HIGH; otherwise HIGH requires two or more signals, MEDIUM exactly one, LOW none — every signal's fired/not-fired detail and threshold is recorded on the result, so the tier is auditable rather than an opaque label. The output is purely additive: existing `archex impact` consumers see byte-for-byte unchanged output when `--diff` is not passed. The MCP `get_impact` tool gets a matching optional `diff` parameter producing output byte-identical to the CLI's `--diff` mode for the same target and edit.

**BREAKING CHANGE:** Previously PHP, Ruby, Scala, C, and C++ were chunked only by a fixed set of declaration-boundary node types per language, producing whole-declaration chunks with no `symbol_name`/`symbol_kind` attached. Every chunk from these five languages' files now carries real `symbol_name`/`symbol_kind`/`qualified_name` data, and files gain resolved import edges in the dependency graph where none existed before. Concretely, for any existing caller of these five languages: (1) chunk boundaries differ from any previously cached index or hardcoded line-range expectation, because chunks are now anchored to extracted symbol start/end lines instead of the old fixed boundary-node set; (2) `outline`/`query`/`graph`/`symbol` results for these files, previously symbol-less, now return named entries; (3) each language maps its own constructs onto the pre-existing `SymbolKind` enum in language-specific, non-uniform ways a caller cannot infer from source syntax alone — PHP `interface` → `SymbolKind.INTERFACE` but PHP `trait` → `CLASS`, while Scala `trait` → `INTERFACE`; C `struct` → `TYPE` (never `CLASS`); C++ `class` → `CLASS` but `struct` → `TYPE`, each with a different default member visibility; (4) `archex doctor`'s full-tier grammar count increases by one per language as each lands. No API is removed and no migration is required to keep using archex, but a caller that depended on the old fixed chunk-boundary shape or on these five languages being symbol-less must update any such assumption; there is no compatibility flag to opt back into the old chunk-only behavior for these languages.

### Changed

- **Parallel parsing by default:** `Config.parallel` now defaults to `True` instead of `False`. Every default `archex index`/`archex analyze` run on a batch of more than 10 files now routes symbol/import extraction through `ProcessPoolExecutor` out of the box, without a manual config flag.
- **Streaming chunk reads:** Added `IndexStore.iter_chunks(batch_size=500)`, a generator that fetches rows via batched cursor calls and yields `CodeChunk` objects incrementally, as an alternative to `get_chunks()`'s single materialized list. A caller that processes rather than retains each chunk keeps peak memory bounded to the batch size regardless of total repo chunk count. `BM25Index.build()`/`update()` now accept an iterable rather than requiring a materialized list.
- **`archex benchmark delta` gains a default action:** The command was previously a bare Click group with no default action. A bare `archex benchmark delta` now runs every delta benchmark task followed by the quality gate in one pass, exiting non-zero on any correctness or speedup violation, and is wired into CI as a weekly/manual job.
- **Bounded co-directory edge growth:** The Go/Rust import-resolution fallback that links files sharing a directory previously built a full pairwise edge set for every directory regardless of size — a 500-file flat directory added roughly 250k edges. Directories at or below a 50-file threshold keep the dense pairwise behavior; above it, each file links only to a bounded 20-file forward window in both directions, bounding growth while preserving local same-directory connectivity.
- **Automatic cache eviction:** The global (non-project-layout) cache previously grew only through the manual `archex cache clean` CLI command. `put()` now calls `clean()` opportunistically on every write and additionally caps total entry count at 500 (evicting oldest-first) as a backstop for bursts faster than the existing 24-hour age window catches. Project-layout caches are exempt, since they never accumulate.
- **README status badges:** Added Downloads, Tests, Coverage, Languages, MCP tools, Ruff, and typing status badges alongside the existing CI/PyPI/Python/license set, and corrected the proof-bar declared-language count, which had gone stale mid-promotion.

### Fixed

- **Per-file parse fault isolation:** The sequential (non-parallel) parse loop previously propagated a `ParseError` from a single oversized or malformed file, aborting symbol/import extraction for the entire batch. The per-file adapter call is now wrapped in try/except, collecting failures and only raising in strict mode — matching the error-collection shape already used by the parallel path.
- **Targeted BM25 delta re-indexing:** `apply_delta()` previously ended every delta apply by reading every chunk in the store and rebuilding the full FTS5 index — an O(total repo chunks) operation regardless of how small the change was. Added an insert-only update path with no table drop, verified to produce results identical to a full rebuild over the same final corpus.
- **HTML/CSS local references collided across same-basename sibling files:** Reference resolution fell back to an extension-stripped, dotted module-style lookup key when a reference didn't resolve directly — but two files sharing a directory and basename across extensions (e.g. `app.js`/`app.css`, a common static-asset pattern) collapsed onto the same key, so one of the two silently resolved to nothing or to the wrong sibling file. File lookup now also registers each file's literal repo-relative path (extension preserved), tried first since path-based references always carry their extension explicitly. Measured on the polyglot fixture: mean required-file recall rose from 0.750 to 0.917, with both colliding-basename tasks improving from 0.667 to 1.000 and zero regression on non-colliding controls.
- **`archex dogfood`/`archex benchmark` hanging on repos with a vendored virtualenv:** one benchmark strategy shelled out to plain `grep -r` with no directory excludes, so every keyword search walked the entire `.venv` and `.git` history instead of respecting `.gitignore`. On archex's own repository this made a full dogfood run take 20+ minutes. The strategy now passes the same directory-exclude list already used elsewhere, dropping the affected task from a per-keyword risk of tens of seconds to about 1.5 seconds total.


## [0.15.2] - 2026-07-04

### Added

- **XML chunk-only language support:** Registered `.xml` in `LANGUAGE_SUPPORT` at `chunk-only` tier, reusing the already-bundled, mature `tree-sitter-grammars/tree-sitter-xml` grammar via `tree_sitter_language_pack` (no new dependency). Chunks on top-level `element` nodes, mirroring HTML's existing pattern; no symbol claim is made, since XML element semantics are dialect-specific (entity vs. bean vs. build-script definitions all share the same grammar but mean different things). Fixes the whole-file/line-window fallback previously reported for `.xml` files (#371) — entity, service, and screen definition files in Java/Groovy codebases now chunk per top-level element. Grammar coverage in `archex doctor` moves from 15/15 to 16/16 chunk-only grammars available.

## [0.15.1] - 2026-07-01

### Added

- **Intent-adaptive scout file limit:** `scout`'s ranked-file selection previously used a hardcoded 12-file cap regardless of query intent, unlike `query`'s existing score-separation-adaptive file count. Added `INTENT_SCOUT_FILE_LIMITS`, computed via the query's classified intent (`DEFINITION_LOOKUP`: 6, `ARCHITECTURE_BROAD`: 16, `USAGE_SEARCH`: 12, `DEBUGGING`: 10, `CLI`: 8, `GENERAL`: 12 unchanged) when the caller does not pass an explicit `file_limit`. Narrow lookups skip ranking/scoring files they'll never need; architecture-broad queries on larger or fast-growing repos are no longer capped below what similar intent-based limits already use elsewhere in the same table. Explicit `file_limit` overrides still bypass intent classification entirely; unclassified (`GENERAL`) queries keep the historical default of 12.

## [0.15.0] - 2026-06-23

### Added

- **Realistic targeted-read savings baseline:** Added a second per-event token-savings baseline, `targeted_read`, recorded alongside the full-file baseline. It models reading the matched line ranges plus a small context window (the union of `[start_line-K, end_line+K]` spans, K=5) rather than pasting whole files, derived deterministically from the returned chunks' indexed content — no file read and no model call on the metrics path — and clamped to the invariant `returned <= targeted_read <= full_file`. Scout events omit it (no chunk bodies to cost). The metrics ledger schema is bumped 1 → 2 with a forward, idempotent, per-column migration; new columns `tokens_targeted_read`, `tokens_saved_vs_targeted_read`, and `savings_pct_vs_targeted_read`.
- **Cross-tool token-efficiency benchmark:** Added `archex benchmark cross-tool`, an offline, benchmark-only comparison of the tokens archex spends to localize a task's required files versus a naive grep/read agent (whole grep-hit files, or `+/-K` context windows around grep hits), measured at a fixed required-file recall so no figure compares unequal recall. Both paths are tokenized with the existing cl100k_base encoder; the naive model is a pure, deterministic function of the gitignore-aware corpus, the task keywords, and `K`. It is not in `DEFAULT_STRATEGIES` and touches no product code path. Checked in the per-corpus reference artifact at `benchmarks/cross-tool-efficiency/cross-tool-comparison.json` and documented the per-corpus reductions (95.4%–99.8% at 100% required-file recall, conditioned on archex fully localizing the task) in `docs/LOCAL_METRICS.md`.

### Changed

- **Exact full-file token-savings baseline:** The full-file (raw-equivalent) baseline now sums the true per-file token cost (`count_tokens(full_file_text)`) instead of `SUM(chunk.token_count)`, which double-counted synthetic, per-chunk-duplicated `imports_context` and inflated the headline savings percentage. Added a nullable `token_count` column to the index `file_states` table (index schema 4 → 5, forward `ALTER` plus fresh-DB create), populated from the bytes already read for sha256 with the existing cl100k_base encoder. A legacy or unpopulated index returns `None` and the baseline is silently omitted (non-fatal, never latched as a warning); a reindex repopulates it. Chunk `token_count`, dynamic budgeting (`repo_total_tokens` metadata stays on the chunk-sum source), retrieval ranking, and the returned set are unchanged.
- **Honest two-baseline metrics reporting:** `archex metrics summary` now reports two labeled savings numbers — "vs full-file paste" (compression versus a naive paste) and "vs realistic targeted read" (the conservative counterfactual) — and demotes the whole-repo line below them, tagged `(upper bound, not savings)`. JSON `totals` keeps `savings_pct` meaning the full-file value (unchanged for existing consumers) and adds `savings_pct_vs_targeted_read` and `tokens_targeted_read`; the `status` repo savings figure carries the targeted figures consistently. Updated `docs/LOCAL_METRICS.md`, `docs/INSTALLATION_TRUST_CONTRACT.md`, and the README so no surface claims a number the ledger does not produce.

## [0.14.0] - 2026-06-22

### Added

- **`omp` install-client target:** Added oh-my-pi (`omp`) as a first-class `archex install-client` client. `archex install-client omp` writes `~/.omp/agent/mcp.json` (user scope) with the standard `mcpServers.archex = {command: "archex", args: ["mcp"]}` payload plus the oh-my-pi `$schema`, merged non-destructively and idempotently.
- **Agent-file MCP guidance prompt:** Added a ready-to-paste prompt that points an agent at the archex MCP tools (`scout_repo`, `query_repo`, `analyze_repo`, `search_symbols`, `get_symbol`) and the discovery/activation step for tool-gated harnesses. `archex install-client --agent-file <path>` appends it to a global or repo-specific agent file (`CLAUDE.md`, `AGENTS.md`, ...) inside a delimited block — non-destructive and idempotent — and `--dry-run` previews it without writing.
- **CLI-vs-MCP surface mix in metrics:** `archex metrics` and `archex metrics summary` now report a per-surface event split (`cli`, `mcp`, `python_api`) so near-zero MCP adoption is observable at a glance. The split renders in the text summary and is exposed as `totals.by_surface` in the JSON summary; no event recording or token-savings math changed.

### Changed

- **README adoption refresh:** Reframed the opening around context-window burn, kept the quickstart as the one always-visible happy path, collapsed only the secondary Docker/extras blocks, and added a metric-anchored workflow translation under Measured results without changing product behavior or benchmark numbers.
- **install-client default-global + `--dry-run`:** `archex install-client <client>` now installs at global (user) scope by default; pass a `[SOURCE]` repo path or `--scope project` for a repo-local install. Removed the `--write` flag — configs are written by default and `--dry-run` previews the exact target and config without touching the filesystem. Writes stay non-destructive (merge into existing config, never clobber unrelated sections) and are idempotent: re-running with an identical `archex` entry is a no-op, while a different existing entry is left untouched and refused. Updated the installation trust contract, compatibility matrix, and README to match.
- **MCP adoption docs:** Documented the omp install target, the agent-file MCP guidance prompt and `--agent-file`, discovery-gated harnesses, the registration → surfacing → invocation distinction, and the CLI-vs-MCP surface split across the compatibility matrix, installation trust contract, README, and LOCAL_METRICS.

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
