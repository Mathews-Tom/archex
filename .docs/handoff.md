# Handoff — archex

**Last touched:** 2026-06-13T01:58Z · **branch:** `feat/scout-followups` · **HEAD:** `1797154` · **session:** openai-codex/gpt-5.4

> Authority: this file owns transient session state. Persistent facts live in project memory. Committed history lives in `git log`.

## Status
- Working tree is clean on `feat/scout-followups`.
- Published C5 scout stack remains open:
  - #201 `feat/scout-core` → `main`
  - #202 `feat/scout-handles` → `feat/scout-core`
  - #203 `feat/scout-cli-mcp` → `feat/scout-handles`
  - #204 `feat/scout-benchmark` → `feat/scout-cli-mcp`
- Follow-up branch `feat/scout-followups` is local-only in this session and contains post-operator fixes:
  - chunk/symbol-first fetch planning
  - direct-query guardrail
  - improved scout file ranking from query bundle + seed/expansion hints
  - benchmark diagnostics for files missing from scout map vs final fetch
- Local follow-up gates passed:
  - `uv run ruff check`
  - `uv run ruff format --check .`
  - `uv run pyright`
  - `uv run pytest tests/ -q -k "scout or graph_query or mcp"`
  - `uv run pytest tests/benchmark/test_models.py tests/benchmark/test_runner.py tests/benchmark/test_strategy_registry.py tests/benchmark/test_cli.py tests/benchmark/test_strategies.py -q`
- Behavioral smoke on follow-up branch passed:
  - `uv run archex scout . "how does delta indexing work" --format markdown`
  - Observed result stayed within cap: `Budget: 978/1000 tokens`.
  - Observed recommended fetch plan:
    - `strategy: chunk_first`
    - `estimated_tokens: fetch=2692, total=3670, direct_query=5632`
    - symbol-level handles emitted for second phase
- Latest operator benchmark run completed from a separate terminal and logged to `.docs/operator-run.log`.
- Latest operator result: scout cap held on all 35 tasks and the original C5 token-cost/recall criterion for architecture-broad intents is now satisfied, but readiness still fails on precision/F1:
  - Mean recall `0.926` vs target `>= 0.800` — pass
  - Mean precision `0.380` vs target `>= 0.600` — fail
  - Mean F1 `0.505` vs target `>= 0.700` — fail
  - Zero-recall tasks `0` vs target `<= 1` — pass
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
- Readiness output still says `Ready: no` for `archex_scout_fetch`.
- Second operator run vs first operator run:
  - recall: `0.579 → 0.926`
  - F1: `0.604 → 0.505`
  - tokens total: `310,236 → 107,419`
  - zero-recall tasks: `2 → 0`
- Current comparison vs `archex_query`:
  - `archex_query`: recall `0.907`, precision `0.575`, F1 `0.687`, tokens `191,677`, median latency `709 ms`
  - `archex_scout_fetch`: recall `0.926`, precision `0.380`, F1 `0.505`, tokens `107,419`, median latency `2089 ms`
  - Scout wins on tokens in `28/35` tasks and ties in `7/35`; no token losses observed.
  - Scout wins recall in `2/35` tasks and ties in `33/35`; no recall losses observed.
  - Scout loses precision/F1 in `23/35` tasks.
- Architecture-broad category now meets the original pass criterion:
  - mean delta vs `archex_query`: `-1636` tokens, `0.000` recall delta, `-0.157` F1 delta
- Guardrails fired in `9/35` tasks (`direct_query`), chunk-first fetch stayed active in `26/35`.
- Diagnostics now identify remaining misses:
  - `missing_from_scout_map != none` in `8` tasks
  - `missing_from_fetch != none` in `8` tasks
- Top failure pattern in the new run: first-phase coverage is mostly fixed, but final fetch still keeps too many ranked files alive. That preserves recall and cuts token cost, but precision collapses because one precise handle per ranked file still returns too many irrelevant files at the file-level evaluation layer.
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
- [ ] Provider CI / merge state for #201–#204 still has not been re-audited from this branch.
- [ ] Scout is closer, but still not ready for promotion. Latest operator data satisfies the original broad/architecture token-cost + recall criterion, but the benchmark readiness gate still fails badly on precision/F1.
- [ ] Main remaining problem moved: no longer missing the right region entirely; now over-returning broad file sets after scout. Next work should reduce false-positive files without giving back the recall win.
- [ ] Scout module names/responsibility quality still depends on current `analyze.modules` heuristics. Functional, but some repo-wide module labels remain coarse.

## Resume checklist
1. Run `git status --porcelain`; expect a clean tree on `feat/scout-followups`.
2. Decide whether to publish `feat/scout-followups` as a new PR on top of #204 or fold it into `feat/scout-benchmark`.
3. Do not promote scout in docs/README yet; readiness is still `no`.
4. Next enhancement work should target precision, not recall:
   - rerank or prune scout fetch handles instead of taking one handle per ranked file
   - consider task-intent-aware cap on number of fetched files/handles
   - consider evaluating final fetch against direct-query precision before choosing chunk-first
   - keep the new `missing_from_scout_map` / `missing_from_fetch` diagnostics and extend them to explain why each extra file survived

## Refs
- Plan: `.docs/2026-06-12-competitive-enhancement-plan.md` sections `2.3 Research synthesis` and `C5`
- Related PRs: #201, #202, #203, #204
- Follow-up branch: `feat/scout-followups` @ `1797154`
- Latest smoke output: `artifact://110`
- Latest local validation output: `artifact://114`
- Operator log: `.docs/operator-run.log`
