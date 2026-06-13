# Handoff — archex

**Last touched:** 2026-06-13T06:15Z · **branch:** `feat/scout-followups` · **HEAD:** `d84217a` · **session:** openai-codex/gpt-5.4

> Authority: this file owns transient session state. Persistent facts live in project memory. Committed history lives in `git log`.

## Status
- Working tree was clean before this handoff refresh on `feat/scout-followups`.
- Published C5 scout stack is now five PRs, all open and merge-clean after CI:
  - #201 `feat/scout-core` → `main`
  - #202 `feat/scout-handles` → `feat/scout-core`
  - #203 `feat/scout-cli-mcp` → `feat/scout-handles`
  - #204 `feat/scout-benchmark` → `feat/scout-cli-mcp`
  - #205 `feat/scout-followups` → `feat/scout-benchmark`
- `feat/scout-followups` adds:
  - chunk/symbol-first fetch planning
  - adaptive handle caps by intent + score-mass coverage
  - direct-query cost guardrail
  - direct-query precision proxy guardrail
  - weak-coverage fallback
  - hybrid fetch mode
  - per-extra-file and per-missing-file benchmark provenance
- Local follow-up gates passed on `d84217a`:
  - `uv run ruff check`
  - `uv run ruff format --check .`
  - `uv run pyright`
  - `uv run pytest tests/ -q -k "scout or graph_query or mcp"`
  - `uv run pytest tests/benchmark/test_models.py tests/benchmark/test_runner.py tests/benchmark/test_strategy_registry.py tests/benchmark/test_cli.py tests/benchmark/test_strategies.py -q`
- Behavioral smoke on current follow-up branch passed:
  - `uv run archex scout . "how does delta indexing work" --format markdown`
  - Observed result stayed within cap: `Budget: 973/1000 tokens`.
  - Observed recommended fetch plan:
    - `strategy: chunk_first`
    - `estimated_tokens: fetch=780, files=3, total=1753, direct_query=5632/5`
    - `projected_precision: chunk_first=0.321, direct_query=0.200`
    - `projected_coverage: chunk_first=0.962, target=0.820`
    - symbol-level handles emitted for second phase
- Latest operator benchmark run completed from a separate terminal and logged to `.docs/operator-run.log`.
- Latest operator result: C5 now meets both readiness and the original broad/architecture pass criterion:
  - Mean recall `0.814` vs target `>= 0.800` — pass
  - Mean precision `0.651` vs target `>= 0.600` — pass
  - Mean F1 `0.707` vs target `>= 0.700` — pass
  - Zero-recall tasks `0` vs target `<= 1` — pass
- PR merge state snapshot:
  - #201 `CLEAN`
  - #202 `CLEAN`
  - #203 `CLEAN`
  - #204 `CLEAN`
  - #205 `CLEAN` after CI completed on 2026-06-13
## What changed this session
- #201 adds `src/archex/scout.py` and `tests/test_scout.py`.
  - Scout assembly now emits a deterministic structural map: ranked files, module boundaries, top symbols, graph sketch, and omission counts.
  - Strict token-cap enforcement trims graph edges, then symbols, then modules, then files until the rendered scout fits.
- #202 extends exact second-phase fetch.
  - Scout items now carry stable file/symbol/chunk handles.
  - `get_symbol` and `get_symbols_batch` accept symbol and chunk handles.
  - `query(..., handles=[...])` accepts file/symbol/chunk handles and fetches exact chunks instead of rerunning search.
- #203 exposes the protocol.
  - New CLI command: `archex scout`.
  - New MCP tool: `scout_repo`.
  - MCP tool list size increased from 13 to 14.
- #204 adds benchmark coverage.
  - New strategy enum: `archex_scout_fetch`.
  - New runner: `run_archex_scout_fetch`.
  - New benchmark CLI flag: `--scout`.
  - Benchmark tests updated for enum, registry, runner availability, CLI wiring, and strategy behavior.


## Operator findings
- Readiness output says `Ready: yes` for `archex_scout_fetch`.
- Fifth operator run vs fourth operator run:
  - recall: `0.710 → 0.814`
  - precision: `0.736 → 0.651`
  - F1: `0.707 → 0.707`
  - tokens total: `59,888 → 89,013`
  - zero-recall tasks: `0 → 0`
- Current comparison vs `archex_query`:
  - `archex_query`: recall `0.907`, precision `0.575`, F1 `0.687`, tokens `191,051`, median latency `717 ms`
  - `archex_scout_fetch`: recall `0.814`, precision `0.651`, F1 `0.707`, tokens `89,013`, median latency `2069 ms`
  - Scout wins on tokens in `23/35` tasks and ties in `12/35`; no token losses observed.
  - Scout loses recall in `10/35` tasks and ties in `25/35`; no recall wins observed.
  - Scout wins F1 in `13/35`, ties in `12/35`, loses in `10/35`.
  - Scout wins precision in `17/35`, ties in `15/35`, loses in `3/35`.
- Architecture-broad now satisfies the original C5 criterion:
  - mean delta vs `archex_query`: `-1047` tokens, `0.000` recall delta, `+0.036` F1 delta
- Guardrails fired in `12/35` tasks (`direct_query`); `chunk_first` remained active in `23/35`.
- Diagnostics in latest run:
  - `missing_from_scout_map != none` in `8` tasks
  - `missing_from_fetch != none` in `20` tasks
  - `missing_from_fetch_reasons == none` in `15` tasks
  - `extra_fetch_file_reasons == none` in `4` tasks
- Remaining weak spots are concentrated in large framework repos:
  - `django_orm_queries`
  - `sqlalchemy_sessions`
  - `celery_task_dispatch`
  - `fastapi_dependency_injection`
## Decisions
1. **Scout cap is fixed at 1000 tokens by default** (2026-06-12) — not intent-routed. Reason: the map phase must be predictable, easy to benchmark across repos, and easy for agents/operators to budget mentally.
2. **Handle contract is explicit and exact** (2026-06-12) — scout emits three prefixes:
   - `file:<repo-relative-path>`
   - `symbol:<stable-symbol-id>`
   - `chunk:<chunk-id>`
   Second-phase consumers accept these directly. No re-search required.
3. **Scout smoke output omits code bodies** (2026-06-12) — markdown shows paths, kinds, line spans, scores, and handles only. Multiline signatures were removed from rendered scout output.
4. **Module boundaries fall back to `analyze()` when the index lacks persisted modules** (2026-06-12) — current query indexing only persists modules when module-prefilter is enabled; scout now recovers module boundaries without changing `archex query` behavior.

## Review notes
- Manual PR review completed on each branch before advancing.
- Attempting the `code-reviewer` task agent failed in this environment with a strict-tools incompatibility on the configured model; no automated reviewer output was available. Local review still completed before publishing.

## Blockers / open questions
- [ ] `missing_from_fetch_reasons` is still absent in part of the run output, so diagnostics are improved but not complete.
- [ ] Scout module names/responsibility quality still depends on current `analyze.modules` heuristics. Functional, but some repo-wide module labels remain coarse.
## Resume checklist
1. Run `git status --porcelain`; only `.docs/handoff.md` should be dirty if this refresh is not committed yet.
2. Merge the stack root-to-leaf: #201 → #202 → #203 → #204 → #205.
3. Do not change default `archex query` behavior; C5 shipped as an additional protocol and surfaces.
4. If docs are updated to recommend scout, scope that change to broad/architecture exploration use cases because that is the measured win.
## Refs
- Plan: `.docs/2026-06-12-competitive-enhancement-plan.md` sections `2.3 Research synthesis` and `C5`
- Related PRs: #201, #202, #203, #204, #205
- Follow-up branch / PR: `feat/scout-followups` @ `d84217a` / #205
- Latest smoke command output: observed in-session on `2026-06-13T06:15Z`
- Latest local validation output: `artifact://191`
- Operator log: `.docs/operator-run.log`
