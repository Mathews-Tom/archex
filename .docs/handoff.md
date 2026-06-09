# Handoff — archex

**Last touched:** 2026-06-09T00:00:00Z · **branch:** `feat/arch-extraction-improve` · **HEAD:** `95819f9` · **session:** openai-codex/gpt-5.5

> Authority: this file owns *transient session state*. Persistent facts live in `~/.claude/projects/<project>/memory/`. Static setup lives in `CLAUDE.md`. Strategic roadmap lives in `~/.claude/plans/<plan>.md`. Committed history lives in `git log`.

## Status
- Working tree: 0 modified, 0 untracked, 0 staged.
- Tests: `uv run pytest tests/analyze/ tests/benchmark/ -q` → 342 passed, 2 deselected, then failed on existing global coverage fail-under (`45% < 85%`). Test bodies pass; coverage config measures all `archex` source for this scoped subset.
- Lint/type: `uv run ruff check && uv run ruff format --check . && uv run pyright` → passed; pyright reported `0 errors, 0 warnings, 0 informations`.
- Last verified: `uv run ruff check && uv run ruff format --check . && uv run pyright && uv run pytest tests/analyze/ tests/benchmark/ -q` on 2026-06-09.

## What changed this session
- Synced `main`, created and pushed Tier 4 stack with `Stack-Id: arch-quality-20260609` trailers.
- PR #168 `feat/arch-quality-tasks` adds architecture oracle schema/loaders and `benchmarks/arch_tasks/{python_false_positives,python_patterns,python_strategy_sorting}.yaml`.
- PR #169 `feat/arch-quality-scorer` adds `src/archex/benchmark/arch_quality.py`, `benchmark arch run/report/gate`, advisory absolute floors, and advisory baseline-regression warnings via `--baseline`.
- PR #170 `feat/arch-extraction-improve` rewrites Strategy pattern detection to aggregate protocol/concrete/context evidence across files while rejecting unrelated shared-method classes.
- Harness snapshot before extraction fix: `python_strategy_sorting_architecture` scored pattern precision/recall `0.0`, decision recall `0.0`, overall `0.5`.
- Harness snapshot after extraction fix: all three labeled architecture tasks scored `1.0` for boundary F1, pattern precision/recall, interface completeness, decision recall, and overall. This was a local smoke snapshot, not the final operator benchmark.
- Each PR was reviewed with the `pr-review` skill; all reported findings were fixed and re-reviewed before advancing.

## Decisions
1. **Architecture gate is advisory** (2026-06-09) — Open Decision 8 defaults to non-blocking until the labeled set proves stable; `benchmark arch gate` exits zero and prints `ARCHITECTURE QUALITY ADVISORY` warnings.
2. **Architecture tasks are local-only** (2026-06-09) — The harness rejects `repo != "."` to avoid network and keep the oracle set held-out-clean in checked-in fixtures.
3. **Strategy detection requires protocol evidence** (2026-06-09) — Cross-file concretes must match protocol methods and meet a minimum protocol+concrete evidence count; this prevents repo-wide method-name collisions from becoming false Strategy detections.

## Blockers / open questions
- [ ] Implementation gate command is not green because scoped pytest is combined with repository-wide coverage fail-under. The actual tests pass; the coverage gate configuration is incompatible with the requested scoped command.
- [ ] Operator must run the architecture-quality benchmark block below and paste per-dimension scores back. Record those scores here when received.
- [ ] `.archex/arch-quality-baseline` must exist for the regression confirmation command. If this is the first accepted arch-quality run, seed that baseline only after the operator accepts the scores.

## Resume checklist
1. Run `git status --porcelain --branch` and confirm branch `feat/arch-extraction-improve` is clean and based on `feat/arch-quality-scorer`.
2. Inspect PR stack: #168 base `main`, #169 base `feat/arch-quality-tasks`, #170 base `feat/arch-quality-scorer`.
3. Re-run `uv run ruff check && uv run ruff format --check . && uv run pyright`; expected pass.
4. Re-run `uv run pytest tests/analyze/ tests/benchmark/ -q`; expect tests pass then coverage fail-under until the scoped coverage configuration is addressed.
5. After operator posts architecture scores, add them to `## Status` or `## What changed this session` and update the baseline note.

## Refs
- Plan: `.docs/2026-05-29-retrieval-recall-enhancement-plan.md` Tier 4
- Related PRs: #168, #169, #170
- Memory: `—`
- Conversation: `Tier 4 architecture extraction codebase intelligence quality`

## Operator architecture-quality benchmark block
Do not run from this agent session. Operator runs in a separate terminal and pastes the report/gate output back.

```bash
rm -rf .archex/arch-quality-current
uv run archex benchmark arch run --tasks-dir benchmarks/arch_tasks --output .archex/arch-quality-current
uv run archex benchmark arch report --input .archex/arch-quality-current
uv run archex benchmark arch gate --input .archex/arch-quality-current --baseline .archex/arch-quality-baseline --min-boundary-f1 0.80 --min-pattern-precision 0.80 --min-pattern-recall 0.80 --min-interface-completeness 0.80
```
