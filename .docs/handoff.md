# Handoff — archex

**Last touched:** 2026-06-13T03:55Z · **branch:** `feat/scout-followups` · **HEAD:** `3e81bf7` · **session:** openai-codex/gpt-5.4

> Authority: this file owns transient session state. Persistent facts live in project memory. Committed history lives in `git log`.

## Status
- Working tree is clean on `feat/scout-followups`.
- Published C5 scout stack remains open:
  - #201 `feat/scout-core` → `main`
  - #202 `feat/scout-handles` → `feat/scout-core`
  - #203 `feat/scout-cli-mcp` → `feat/scout-handles`
  - #204 `feat/scout-benchmark` → `feat/scout-cli-mcp`
- Follow-up branch `feat/scout-followups` is local-only in this session and now contains:
  - chunk/symbol-first fetch planning
  - direct-query cost guardrail
  - intent-capped handle selection
  - direct-query precision proxy guardrail
  - per-extra-file fetch survival reasons in benchmark provenance
- Local follow-up gates passed on `3e81bf7`:
  - `uv run ruff check`
  - `uv run ruff format --check .`
  - `uv run pyright`
  - `uv run pytest tests/ -q -k "scout or graph_query or mcp"`
  - `uv run pytest tests/benchmark/test_models.py tests/benchmark/test_runner.py tests/benchmark/test_strategy_registry.py tests/benchmark/test_cli.py tests/benchmark/test_strategies.py -q`
- Behavioral smoke on follow-up branch passed:
  - `uv run archex scout . "how does delta indexing work" --format markdown`
  - Observed result stayed within cap: `Budget: 992/1000 tokens`.
  - Observed recommended fetch plan:
    - `strategy: chunk_first`
    - `estimated_tokens: fetch=717, files=2, total=1709, direct_query=5632/5`
    - `projected_precision: chunk_first=0.449, direct_query=0.200`
    - symbol-level handles emitted for second phase
- Latest operator benchmark run completed from a separate terminal and logged to `.docs/operator-run.log`.
- Latest operator result: scout cap held on all 35 tasks and token usage improved sharply, but the new precision-oriented pruning overcorrected and C5 still fails:
  - Mean recall `0.589` vs target `>= 0.800` — fail
  - Mean precision `0.781` vs target `>= 0.600` — pass
  - Mean F1 `0.652` vs target `>= 0.700` — fail
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
- Third operator run vs second operator run:
  - recall: `0.926 → 0.589`
  - precision: `0.380 → 0.781`
  - F1: `0.505 → 0.652`
  - tokens total: `107,419 → 52,820`
  - zero-recall tasks: `0 → 0`
- Current comparison vs `archex_query`:
  - `archex_query`: recall `0.907`, precision `0.575`, F1 `0.687`, tokens `191,605`, median latency `712 ms`
  - `archex_scout_fetch`: recall `0.589`, precision `0.781`, F1 `0.652`, tokens `52,820`, median latency `2156 ms`
  - Scout wins on tokens in `33/35` tasks and ties in `2/35`; no token losses observed.
  - Scout loses recall in `26/35` tasks and ties in `9/35`; no recall wins observed.
  - Scout wins precision in `24/35` tasks but still loses F1 in `18/35`.
- Architecture-broad regressed again and no longer satisfies the original pass criterion:
  - mean delta vs `archex_query`: `-3136` tokens, `-0.444` recall, `-0.222` F1
- Guardrails fired in only `2/35` tasks (`direct_query`); chunk-first fetch still stayed active in `33/35`.
- Diagnostics in latest run:
  - `missing_from_scout_map != none` in `8` tasks
  - `missing_from_fetch != none` in `30` tasks
  - `extra_fetch_file_reasons == none` in `18` tasks, so provenance is still incomplete for many misses
- Top failure pattern in the new run: we over-pruned fetch breadth. Token cost and precision proxies improved, but recall collapsed because the chunk-first plan now keeps too few handles on broad and framework tasks.
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
- [ ] C5 is still not complete. Latest operator data regressed below the core readiness bar and below the original broad/architecture “beat query on token cost without recall loss” criterion.
- [ ] Main failure mode moved again: broad/file discovery is now good enough to seed the map, but fetch-plan pruning is too aggressive and drops required files before final bundle assembly.
- [ ] Provenance is only partially diagnostic. `extra_fetch_file_reasons` is missing in many failing tasks because missing files dominate instead of surviving extras.
- [ ] Scout module names/responsibility quality still depends on current `analyze.modules` heuristics. Functional, but some repo-wide module labels remain coarse.

## Resume checklist
1. Run `git status --porcelain`; expect a clean tree on `feat/scout-followups`.
2. Decide whether to publish `feat/scout-followups` as a new PR on top of #204 or fold it into `feat/scout-benchmark`.
3. Do not promote scout in docs/README yet; readiness is still `no`.
4. Next enhancement work should rebalance precision and recall instead of pushing either extreme:
   - relax intent handle caps for architecture/framework tasks or make them score-adaptive
   - use direct-query fallback more aggressively when the scout fetch plan projects missing-file risk
   - add a minimum coverage heuristic before allowing chunk-first (for example, require enough independent ranked files or enough score mass)
   - improve diagnostics for `missing_from_fetch` so dropped expected files carry explicit “why excluded” reasons

## Refs
- Plan: `.docs/2026-06-12-competitive-enhancement-plan.md` sections `2.3 Research synthesis` and `C5`
- Related PRs: #201, #202, #203, #204
- Follow-up branch: `feat/scout-followups` @ `3e81bf7`
- Latest smoke output: `artifact://140`
- Latest local validation output: `artifact://143`
- Operator log: `.docs/operator-run.log`
