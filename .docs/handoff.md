# Handoff — archex

**Last touched:** 2026-05-31T00:19:09+05:30 · **branch:** `feat/module-summary-prefilter` · **HEAD:** `current leaf` · **session:** codex-gpt-5

> Authority: this file owns *transient session state*. Persistent facts live in Codex memory. Static setup lives in `AGENTS.md` / repo instructions. Strategic roadmap lives in `.docs/2026-05-29-retrieval-recall-enhancement-plan.md`. Committed history lives in `git log`.

## Status
- Working tree: `.docs/handoff.md` records the completed operator benchmark verdict and is force-added because `.docs/` is ignored.
- Active Tier 2 candidate-pool stack:
  1. `feat/splade-leg` on `main` at `f5580cd`; PR #151, base `main`.
  2. `feat/module-summary-prefilter` on `feat/splade-leg`; PR #152, base `feat/splade-leg`.
- Operator benchmark completed from `.docs/benchmark-log.md` using the requested commands. This session did not run `archex benchmark run` or `archex dogfood`.
- Tier 2 pass criterion was not met for the requested final strategy `archex_query_fusion_rerank`: vocabulary-mismatch / external-large did not improve versus `archex_query`, and self / architecture-broad regressed.
- Important measurement caveat: the operator commands did not pass `--splade` or `--module-prefilter`, and both new Tier 2 legs default off. The run measured existing query/fusion/rerank behavior on this branch, not the opt-in SPLADE/module-prefilter legs. No SPLADE/module-prefilter per-leg deltas are available from this run.
- Validation completed on each implementation slice:
  - `uv run ruff check && uv run ruff format --check . && uv run pyright` -> pass on both branches.
  - `uv run pytest tests/index/test_splade.py tests/analyze/test_modules.py -q --no-cov` -> pass; latest leaf run `44 passed, 2 deselected`.
  - Requested exact scoped pytest command was run: `uv run pytest tests/index/test_splade.py tests/analyze/test_modules.py -q`. Tests passed functionally but command exits nonzero because repo-wide coverage fail-under still applies to scoped pytest; latest leaf run `44 passed, 2 deselected`, coverage `23.71% < 85%`.
- `pr-review` ran after each committed PR slice:
  - `feat/splade-leg`: fixed SPLADE fusion seed/cutoff consistency before publishing.
  - `feat/module-summary-prefilter`: fixed invalid `module_prefilter=True, bm25=False` no-op by enforcing a config invariant before publishing.

## Benchmark verdict
- Commands recorded in `.docs/benchmark-log.md`:
  ```bash
  uv run archex benchmark run --query-fusion --rerank --tasks-dir benchmarks/tasks --output .archex/e2e-results
  uv run archex benchmark triage --input .archex/e2e-results --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
  ```
- Aggregate recall / F1 by bucket:
  | Bucket | n | query recall | fusion recall | fusion+rerank recall | rerank delta vs query | query F1 | fusion+rerank F1 | F1 delta |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | architecture-broad | 3 | 0.556 | 0.444 | 0.222 | -0.333 | 0.444 | 0.167 | -0.278 |
  | external-framework | 9 | 0.685 | 0.667 | 0.537 | -0.148 | 0.522 | 0.402 | -0.120 |
  | external-large | 5 | 0.367 | 0.500 | 0.333 | -0.033 | 0.279 | 0.228 | -0.051 |
  | framework-semantic | 2 | 0.333 | 0.333 | 0.500 | +0.167 | 0.325 | 0.393 | +0.068 |
  | self | 16 | 0.590 | 0.618 | 0.252 | -0.338 | 0.495 | 0.177 | -0.317 |
- Zero-recall count by bucket:
  | Bucket | query | fusion | fusion+rerank |
  |---|---:|---:|---:|
  | architecture-broad | 0/3 | 0/3 | 2/3 |
  | external-framework | 0/9 | 0/9 | 0/9 |
  | external-large | 1/5 | 0/5 | 2/5 |
  | framework-semantic | 0/2 | 0/2 | 0/2 |
  | self | 0/16 | 0/16 | 3/16 |
- Latency: fusion+rerank is ~107-168s average per bucket versus query at ~1.4-13.8s. The quality regression plus latency makes `archex_query_fusion_rerank` non-mergeable as a default decision.
- Fusion alone is more promising than rerank: external-large recall improves from 0.367 to 0.500 and self improves from 0.590 to 0.618, but architecture-broad and external-framework still regress.
- Graph expansion T2.3:
  - `archex_graph_expansion` under `archex_query`: final recall 1.0, seed recall 0.5, expansion contribution +0.5, 8 expanded files. Expansion helped the non-rerank query path.
  - `archex_graph_expansion` under fusion: final recall 1.0, seed recall 1.0, expansion contribution 0.0, 8 expanded files. Seeds already covered expected files.
  - `archex_graph_expansion` under fusion+rerank: final recall 0.0, seed recall 1.0, expansion contribution -1.0, 8 expanded files. Rerank dropped the relevant graph-expanded/seed candidates.
  - `MAX_EXPANSION_FILES=8` was hit often, but widening it is not the next move while rerank is eliminating already-present relevant candidates. Fix or disable rerank before tuning expansion width.

## What changed this session
- Synced local `main` with `origin/main` before starting Tier 2.
- PR #151 (`feat/splade-leg`) adds opt-in SPLADE retrieval:
  - `IndexConfig.splade`, `.archex/settings.toml` default `splade = false`, and CLI flags `archex index --splade` / `archex query --splade`.
  - Cache-miss and explicit indexing build SPLADE rows only when opted in.
  - Query-time SPLADE fails fast against cached indexes without SPLADE rows instead of silently dropping the leg.
  - `assemble_context(...)` accepts `splade_results`, gates SPLADE via existing fusion confidence logic, contributes SPLADE candidates only when fused, and records SPLADE metadata.
  - Tests verify SPLADE search contract, deterministic build/query, and SPLADE candidate assembly without loading the real model.
- PR #152 (`feat/module-summary-prefilter`) adds opt-in module responsibility prefiltering:
  - `analyze/modules.py` now populates `Module.responsibility` deterministically from module names, paths, exports, and external dependencies.
  - `IndexStore` persists module summaries as JSON in a `modules` table.
  - `IndexConfig.module_prefilter`, `.archex/settings.toml` default `module_prefilter = false`, and CLI flags `archex index --module-prefilter` / `archex query --module-prefilter` keep behavior opt-in.
  - Cached query fails fast when module prefiltering is requested without persisted module summaries.
  - Query path runs BM25 over module responsibility strings and contributes capped chunks from matched modules into the candidate pool.
  - Tests verify responsibility population and candidate boosting for lifecycle-style queries.

## Decisions
1. **Opt-in only** (2026-05-30) — SPLADE and module prefiltering are both gated by explicit config/CLI flags. Product-default BM25 behavior remains unchanged until measured.
2. **Fail fast on missing opt-in artifacts** (2026-05-30) — Querying with `--splade` or `--module-prefilter` against an old cached index raises `ArchexIndexError` with the refresh command. Silent fallback would corrupt benchmark interpretation.
3. **Module prefilter requires BM25** (2026-05-30) — The prefilter is a BM25 responsibility pass that biases BM25 candidate assembly. `module_prefilter=True` with `bm25=False` is invalid instead of being accepted as a no-op.

## Blockers / open questions
- [ ] GitHub checks for PR #151 and PR #152 still need to complete if CI is enabled.
- [ ] Decide whether to merge the opt-in Tier 2 plumbing despite the benchmark caveat. The run did not exercise `--splade` or `--module-prefilter`, so it does not validate those legs.
- [ ] Do not start Tier 3. The default/rerank decision is not justified by this benchmark.

## Resume checklist
1. Confirm branch and tree: `git status --short --branch`.
2. Confirm PR topology: `gh pr view 151 --json headRefName,baseRefName,state,url` and `gh pr view 152 --json headRefName,baseRefName,state,url`.
3. Watch or inspect CI if available. Do not delete `feat/splade-leg` while PR #152 still targets it.
4. If measuring the actual new opt-in Tier 2 legs, add benchmark support or a controlled command path that passes `splade=True` and `module_prefilter=True`; the recorded operator command does not enable them.
5. Keep product defaults unchanged. `archex_query_fusion_rerank` failed the Tier 2 pass criterion.
6. Do not widen `MAX_EXPANSION_FILES` yet. Rerank candidate elimination, not graph expansion width, is the measured failure mode for `archex_graph_expansion`.
7. Do not start Tier 3.

## Refs
- Plan: `.docs/2026-05-29-retrieval-recall-enhancement-plan.md` Tier 2.
- Related PRs: #151 (`feat/splade-leg`), #152 (`feat/module-summary-prefilter`).
- Memory: `MEMORY.md` archex stacked-PR workflow notes used for stack discipline and scoped pytest coverage caveat.
- Conversation: current `/goal Implement Tier 2 (candidate pool expansion)` thread.
