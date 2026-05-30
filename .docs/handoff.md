# Handoff — archex

**Last touched:** 2026-05-30T18:54:39+05:30 · **branch:** `feat/module-summary-prefilter` · **HEAD:** `2d17349` · **session:** codex-gpt-5

> Authority: this file owns *transient session state*. Persistent facts live in Codex memory. Static setup lives in `AGENTS.md` / repo instructions. Strategic roadmap lives in `.docs/2026-05-29-retrieval-recall-enhancement-plan.md`. Committed history lives in `git log`.

## Status
- Working tree: clean before this handoff rewrite; `.docs/handoff.md` is the only expected uncommitted edit and must be force-added because `.docs/` is ignored.
- Active Tier 2 candidate-pool stack:
  1. `feat/splade-leg` on `main` at `f5580cd`; PR #151, base `main`.
  2. `feat/module-summary-prefilter` on `feat/splade-leg` at `2d17349`; PR #152, base `feat/splade-leg`.
- No `archex benchmark run` and no `archex dogfood` command was run in this session.
- T2.3 is measurement only. Graph expansion instrumentation was left untouched; the operator block below includes expansion contribution triage and `MAX_EXPANSION_FILES` review.
- Validation completed on each implementation slice:
  - `uv run ruff check && uv run ruff format --check . && uv run pyright` -> pass on both branches.
  - `uv run pytest tests/index/test_splade.py tests/analyze/test_modules.py -q --no-cov` -> pass; latest leaf run `44 passed, 2 deselected`.
  - Requested exact scoped pytest command was run: `uv run pytest tests/index/test_splade.py tests/analyze/test_modules.py -q`. Tests passed functionally but command exits nonzero because repo-wide coverage fail-under still applies to scoped pytest; latest leaf run `44 passed, 2 deselected`, coverage `23.71% < 85%`.
- `pr-review` ran after each committed PR slice:
  - `feat/splade-leg`: fixed SPLADE fusion seed/cutoff consistency before publishing.
  - `feat/module-summary-prefilter`: fixed invalid `module_prefilter=True, bm25=False` no-op by enforcing a config invariant before publishing.

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
- [ ] The decisive Tier 2 benchmark has not been run. The operator must run it outside this session.
- [ ] After benchmark results are pasted back, record per-leg deltas and graph-expansion contribution in this file.

## Resume checklist
1. Confirm branch and tree: `git status --short --branch`.
2. Confirm PR topology: `gh pr view 151 --json headRefName,baseRefName,state,url` and `gh pr view 152 --json headRefName,baseRefName,state,url`.
3. Watch or inspect CI if available. Do not delete `feat/splade-leg` while PR #152 still targets it.
4. Operator run, separate terminal:
   ```bash
   uv run archex benchmark run --query-fusion --rerank --tasks-dir benchmarks/tasks --output .archex/e2e-results
   uv run archex benchmark triage --input .archex/e2e-results --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
   ```
5. Pass criterion: vocabulary-mismatch / external-large recall improves; no regression on self or framework-semantic buckets.
6. T2.3 measurement triage: quantify graph expansion recall contribution using `meta.seed_file_paths` / `meta.expanded_file_paths`, then decide whether `MAX_EXPANSION_FILES` should widen beyond 8. Do not change graph expansion constants unless measurement justifies it.
7. If operator results are pasted back, update `.docs/handoff.md` with per-leg deltas and the expansion verdict. Do not start Tier 3.

## Refs
- Plan: `.docs/2026-05-29-retrieval-recall-enhancement-plan.md` Tier 2.
- Related PRs: #151 (`feat/splade-leg`), #152 (`feat/module-summary-prefilter`).
- Memory: `MEMORY.md` archex stacked-PR workflow notes used for stack discipline and scoped pytest coverage caveat.
- Conversation: current `/goal Implement Tier 2 (candidate pool expansion)` thread.
