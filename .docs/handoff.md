# Handoff — archex

**Last touched:** 2026-06-10T21:45:00Z · **branch:** `chore/retrieval-blocker-triage` · **HEAD:** pending · **session:** openai-codex/gpt-5.5

> Authority: this file owns *transient session state*. Persistent facts live in `~/.claude/projects/<project>/memory/`. Static setup lives in `CLAUDE.md`. Strategic roadmap lives in `.docs/2026-06-09-system-improvements-enhancement-plan.md`. Committed history lives in `git log`.

## Status
- Working tree: active G2 precision/F1 stack work on `chore/retrieval-blocker-triage`.
- Baseline evidence: `.archex/benchmark-current` readiness for `archex_query` reports recall `0.819`, precision `0.480`, F1 `0.589`, token efficiency `0.704`, p95 `2059 ms`, zero-recall tasks `0`.
- Tests: pending for G2.1.

## What changed this session
- G2.1 blocker classification recorded in `.docs/retrieval-blocker-triage.md` using existing `.archex/benchmark-current` readiness and triage outputs.

## Decisions
1. **No model-default change** (2026-06-10) — Keep `archex_query` as the product default; improve ranking, normalization, path alignment, and expansion selectivity only.
2. **No hosted/generative inference** (2026-06-10) — G2 remains local-only and deterministic.
3. **Top blockers are ranking/packing problems, not oracle defects** (2026-06-10) — The top ten all return oracle files; precision/F1 bottlenecks come from semantic gaps, path-alignment misses, expansion noise, and large-repo ambiguity.

## Blockers / open questions
- [ ] Operator verdict still required after the stack: full retrieval gate plus dogfood command block from Goal G2.

## Resume checklist
1. Continue stack order: `chore/retrieval-blocker-triage` → `feat/framework-semantic-normalization` → `feat/self-lifecycle-ranking` → `feat/expansion-diagnostics` → `fix/expansion-selectivity`.
2. For each PR slice, run `uv run ruff check && uv run ruff format --check . && uv run pyright` and `uv run pytest tests/serve/ tests/benchmark/ -q`.
3. Do not run `archex benchmark run` or `archex dogfood` in-session.
4. Print the operator verdict block at final handoff.

## Refs
- Plan: `.docs/2026-06-09-system-improvements-enhancement-plan.md` Goal G2
- Baseline report: `.archex/benchmark-current`
- Triage report: `.docs/retrieval-blocker-triage.md`
