# Handoff — archex

**Last touched:** 2026-06-13T05:05Z · **branch:** `feat/scout-followups` · **HEAD:** `f6fdadd` · **session:** openai-codex/gpt-5.4

> Authority: this file owns transient session state. Persistent facts live in project memory. Committed history lives in `git log`.

## Status
- Working tree was clean before this handoff refresh on `feat/scout-followups`.
- Published C5 scout stack remains open:
  - #201 `feat/scout-core` → `main`
  - #202 `feat/scout-handles` → `feat/scout-core`
  - #203 `feat/scout-cli-mcp` → `feat/scout-handles`
  - #204 `feat/scout-benchmark` → `feat/scout-cli-mcp`
- Follow-up branch `feat/scout-followups` is local-only in this session and now contains:
  - chunk/symbol-first fetch planning
  - adaptive handle caps by intent + score-mass coverage
  - direct-query cost guardrail
  - direct-query precision proxy guardrail
  - weak-coverage fallback
  - per-extra-file and per-missing-file benchmark provenance
- Local follow-up gates passed on `f6fdadd`:
  - `uv run ruff check`
  - `uv run ruff format --check .`
  - `uv run pyright`
  - `uv run pytest tests/ -q -k "scout or graph_query or mcp"`
  - `uv run pytest tests/benchmark/test_models.py tests/benchmark/test_runner.py tests/benchmark/test_strategy_registry.py tests/benchmark/test_cli.py tests/benchmark/test_strategies.py -q`
- Behavioral smoke on current follow-up branch passed:
  - `uv run archex scout . "how does delta indexing work" --format markdown`
  - Observed result stayed within cap: `Budget: 956/1000 tokens`.
  - Observed recommended fetch plan:
    - `strategy: chunk_first`
    - `estimated_tokens: fetch=717, files=2, total=1673, direct_query=5632/5`
    - `projected_precision: chunk_first=0.449, direct_query=0.200`
    - `projected_coverage: chunk_first=0.898, target=0.700`
    - symbol-level handles emitted for second phase
- Latest operator benchmark run completed from a separate terminal and logged to `.docs/operator-run.log`.
- Latest operator result: scout cap still holds, token cost remains strongly better than `archex_query`, F1 now clears the readiness bar, but recall still misses the bar so C5 is still incomplete:
  - Mean recall `0.710` vs target `>= 0.800` — fail
  - Mean precision `0.736` vs target `>= 0.600` — pass
  - Mean F1 `0.707` vs target `>= 0.700` — pass
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
- Fourth operator run vs third operator run:
  - recall: `0.589 → 0.710`
  - precision: `0.781 → 0.736`
  - F1: `0.652 → 0.707`
  - tokens total: `52,820 → 59,888`
  - zero-recall tasks: `0 → 0`
- Current comparison vs `archex_query`:
  - `archex_query`: recall `0.907`, precision `0.575`, F1 `0.687`, tokens `191,051`, median latency `712 ms`
  - `archex_scout_fetch`: recall `0.710`, precision `0.736`, F1 `0.707`, tokens `59,888`, median latency `2197 ms`
  - Scout wins on tokens in `31/35` tasks and ties in `4/35`; no token losses observed.
  - Scout loses recall in `19/35` tasks and ties in `16/35`; no recall wins observed.
  - Scout wins F1 in `15/35`, ties in `6/35`, loses in `14/35`.
  - Scout wins precision in `23/35`, ties in `7/35`, loses in `5/35`.
- Architecture-broad is improved from the last run but still fails the original C5 criterion:
  - mean delta vs `archex_query`: `-2784` tokens, `-0.222` recall, `-0.063` F1
- Guardrails fired in `4/35` tasks (`direct_query`); chunk-first still stayed active in `31/35`.
- Diagnostics in latest run:
  - `missing_from_scout_map != none` in `8` tasks
  - `missing_from_fetch != none` in `26` tasks
  - `missing_from_fetch_reasons == none` in `9` tasks
  - `extra_fetch_file_reasons == none` in `12` tasks
- Top failure pattern in the new run: adaptive caps repaired the worst recall collapse, but chunk-first is still active too often on broad/external tasks. Required files are still being pruned by cap with score-mass around `0.70–0.78`, especially large framework repos.
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
- [ ] C5 is still not complete. Latest operator data passes precision/F1 but still misses the readiness recall bar and still fails the original broad/architecture “beat query on token cost without recall loss” criterion.
- [ ] Main remaining failure mode: fetch-plan coverage heuristics are still too conservative for broad/framework tasks; required files are pruned before final bundle assembly.
- [ ] Provenance improved, but `missing_from_fetch_reasons` and `extra_fetch_file_reasons` are still absent in a non-trivial subset of runs, so diagnostics remain incomplete.
- [ ] Scout module names/responsibility quality still depends on current `analyze.modules` heuristics. Functional, but some repo-wide module labels remain coarse.

## Resume checklist
1. Run `git status --porcelain`; only `.docs/handoff.md` should be dirty if this refresh is not committed yet.
2. Decide whether to publish `feat/scout-followups` as a new PR on top of #204 or fold it into `feat/scout-benchmark`.
3. Do not promote scout in docs/README yet; readiness is still `no`.
4. Next enhancement work should push recall upward without losing the new precision/F1 gains:
   - make chunk-first fallback trigger earlier for architecture-broad and external-large tasks
   - relax score-mass coverage targets or max caps for broad/framework intents only
   - add missing-file exclusion reasons for all pruned expected files, not only files present in the ranked set
   - consider hybrid fetch plans: chunk-first for top files plus direct-query fallback when expected coverage looks thin

## Refs
- Plan: `.docs/2026-06-12-competitive-enhancement-plan.md` sections `2.3 Research synthesis` and `C5`
- Related PRs: #201, #202, #203, #204
- Follow-up branch: `feat/scout-followups` @ `f6fdadd`
- Latest smoke command output: observed in-session on `2026-06-13T05:05Z`
- Latest local validation output: `artifact://172`
- Operator log: `.docs/operator-run.log`
