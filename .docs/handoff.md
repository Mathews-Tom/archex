# Handoff — archex

**Last touched:** 2026-06-11T22:36:41Z · **branch:** `feat/headtohead-report` · **HEAD:** `5a667c6` · **session:** claude-fable-5

> Authority: this file owns *transient session state*. Persistent facts live in `~/.claude/projects/<project>/memory/`. Static setup lives in `CLAUDE.md`. Strategic roadmap lives in `~/.claude/plans/<plan>.md`. Committed history lives in `git log`.

## Status
- Working tree: 1 modified, 0 untracked, 0 staged (`README.md`)
- Tests: last green leaf gate `uv run pytest tests/benchmark/ -q` → pass (290 passed, 2 deselected) on `feat/headtohead-report`
- Lint/type: `uv run ruff check && uv run ruff format --check . && uv run pyright` → pass on `feat/headtohead-report`
- Last verified: operator head-to-head run 2026-06-11 (19/19 external tasks, report rendered); code gates green before that run

## What changed this session
- Roadmap sessions 1 (G5 strict benchmark validation) and 2 (C1 head-to-head harness) complete. C1 stack: #185 `feat/headtohead-adapter`, #186 `feat/headtohead-manifest`, #187 `feat/headtohead-token-parity`, #188 `feat/headtohead-report`.
- Operator ran the public 19-task head-to-head (manifest `archex-vs-ccc-c1-public`, M1 Pro). **Verdict: archex wins every quality cell** — recall 0.95 vs ccc 0.32, F1 0.66 vs 0.31, token efficiency 0.76 vs 0.48, completion penalty 922 vs 11188 tokens, warm latency 408 vs 521 ms. Full table with provenance: `.docs/head-to-head.md`.
- Detailed result table removed from this handoff; `.docs/head-to-head.md` is the result surface, summary also in memory `benchmark_results.md`. Raw run JSON: `.archex/headtohead/*.json`.
- Local `README.md` still carries the uncommitted pointer to the head-to-head harness.

## Decisions
1. **C1 uses explicit tool bootstrap steps** (2026-06-11) — External lanes cannot assume the cloned task repo is pre-initialized. The manifest owns ordered bootstrap commands (`ccc init -f`, then `ccc index`).
2. **README remains results-light** (2026-06-11) — README points to the harness and result location; no inlined tables or cherry-picked cells.
3. **Publish even if losing stayed intact** (2026-06-11) — All three lanes recorded with full provenance, including the raw-read lane's definitional recall 1.00 / token efficiency 0.00.
4. **Results are not publication-ready until two credibility holes close** (2026-06-11) — See blockers; C4 (comparison page) must not consume the current table as-is.

## Blockers / open questions
- [ ] **Credibility fix 1 — asymmetric cold-start instrumentation.** archex lane reports `cold_start_ms=0` while ccc's bootstrap was timed (4721 ms); archex index build (vector pre-compute is tens of seconds on a 10K-chunk repo) ran outside the timer. Fix: time archex indexing inside the same boundary the ccc lane uses, or replace the cell with `n/m` + footnote. Honest expectation: archex cold start is likely worse than ccc's — publish that; C2 addresses it and the roadmap re-runs freshness cells after C2 merges.
- [ ] **Credibility fix 2 — ccc embedder sensitivity row.** ccc ran its default `Snowflake/snowflake-arctic-embed-xs` (22M, general-purpose) vs archex's code-specific jina-v2. ccc supports CodeRankEmbed locally. Add a ccc+CodeRankEmbed lane to the manifest before public claims; state defaults-vs-defaults framing explicitly either way.
- [ ] PR #188 close-out: commit the local `README.md` pointer (amend leaf or docs-only follow-up), push, merge the C1 stack to `main`.
- [ ] Operator decision: long-term home of the published table (`benchmarks/headtohead/` report output vs `docs/`) — feeds C4.4.

## Resume checklist
1. `git status --porcelain` — confirm only `README.md` is modified on `feat/headtohead-report`.
2. Commit/push the README pointer onto #188 and merge the C1 stack to `main`.
3. Next roadmap session: paste the **Session 3 — C2** block from `.docs/2026-06-12-unified-roadmap-session-prompts.md` (working-tree delta + warm MCP; independent of the C1 credibility fixes).
4. Schedule the two credibility fixes as a small harness follow-up (timer boundary + manifest sensitivity lane) any time before Session 9 (C4) consumes the results. Do not re-run the benchmark otherwise; `.archex/headtohead/*.json` holds the current sample.

## Refs
- Plan: `.docs/2026-06-12-unified-roadmap-session-prompts.md` (sequencing authority); `.docs/2026-06-12-competitive-enhancement-plan.md` C1/C4; `docs/RETRIEVAL_DEFAULT_DECISIONS.md` token-efficiency definition
- Related PRs: #185, #186, #187, #188 (C1 stack); G5 stack merged prior
- Memory: `enhancement_status.md`, `benchmark_results.md`
- Conversation: `C1 public head-to-head benchmark archex ccc credibility cold start embedder`
