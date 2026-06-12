# Handoff — archex

**Last touched:** 2026-06-12T22:30Z · **branch:** `feat/scout-benchmark` · **HEAD:** `ffc1fb0` · **session:** openai-codex/gpt-5.4

> Authority: this file owns transient session state. Persistent facts live in project memory. Committed history lives in `git log`.

## Status
- Working tree was clean before this handoff refresh.
- C5 scout stack is published and open:
  - #201 `feat/scout-core` → `main`
  - #202 `feat/scout-handles` → `feat/scout-core`
  - #203 `feat/scout-cli-mcp` → `feat/scout-handles`
  - #204 `feat/scout-benchmark` → `feat/scout-cli-mcp`
- Root-to-leaf implementation gates passed locally on the leaf:
  - `uv run ruff check`
  - `uv run ruff format --check .`
  - `uv run pyright`
  - `uv run pytest tests/ -q -k "scout or graph_query or mcp"`
  - `uv run pytest tests/benchmark/test_models.py tests/benchmark/test_runner.py tests/benchmark/test_strategy_registry.py tests/benchmark/test_cli.py tests/benchmark/test_strategies.py -q`
- Behavioral smoke passed:
  - `uv run archex scout . "how does delta indexing work" --format markdown`
  - Observed result stayed within cap: `Budget: 979/1000 tokens`.
  - Observed stable handles in output: `file:src/archex/index/delta.py`, `chunk:src/archex/index/delta.py:apply_delta:457`, `symbol:src/archex/index/delta.py::apply_delta#function@3`.
- Operator benchmark run completed from a separate terminal and logged to `.docs/operator-run.log`.
- Operator result: scout map cap held on all 35 tasks (`max scout_tokens = 998`, zero cap violations), but readiness failed for `archex_scout_fetch`:
  - Mean recall `0.579` vs target `>= 0.800`
  - Mean F1 `0.604` vs target `>= 0.700`
  - Zero-recall tasks `2` vs target `<= 1`
- Operator comparison vs `archex_query` went the wrong direction:
  - `archex_query`: recall `0.907`, F1 `0.687`, tokens `191,677`, median latency `751 ms`
  - `archex_scout_fetch`: recall `0.579`, F1 `0.604`, tokens `310,236`, median latency `2058 ms`
  - Scout used more tokens on all 35 tasks and lost recall on 24/35 tasks.
  - Broad/architecture intents also regressed: `architecture-broad` mean delta vs `archex_query` was `+4459` tokens, `-0.333` recall, `-0.194` F1.

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
- Readiness output says `Ready: no` for `archex_scout_fetch`.
- Category breakdown from `.archex/e2e-scout`:
  - `self`: recall `0.673`, precision `0.776`, F1 `0.697`
  - `architecture-broad`: recall `0.556`, precision `0.422`, F1 `0.472`
  - `external-framework`: recall `0.463`, precision `0.676`, F1 `0.517`
  - `external-large`: recall `0.467`, precision `0.733`, F1 `0.533`
- Zero-recall tasks:
  - `django_middleware` (`architecture-broad`)
  - `react_hooks` (`external-large`)
- Top failure pattern from the run: scout retrieved a bounded exact bundle from selected handles, but the first-phase map often failed to surface all needed files. The second phase then fetched the wrong narrow set precisely, so precision stayed decent while recall collapsed.
- Secondary issue: scout+fetch is currently not token-light relative to `archex_query`. The fixed 1000-token map overhead plus whole-file handle expansion increased total tokens on every task in the operator run.
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
- [ ] Provider CI for #201–#204 has still not been observed in this session.
- [ ] Scout is not ready for promotion. Operator data failed the stated pass criterion: it did not beat `archex_query` on token cost for broad/architecture intents and did lose recall.
- [ ] Main follow-up target: improve first-phase file selection before any docs/default rollout. Current exact-fetch handle path is working as designed; ranking quality is not.
- [ ] Scout module names/responsibility quality still depends on current `analyze.modules` heuristics. Functional, but some repo-wide module labels remain coarse.

## Resume checklist
1. Run `git status --porcelain`; only `.docs/handoff.md` may be dirty if this refresh is not committed yet.
2. Monitor #201 → #202 → #203 → #204 root-to-leaf and land the stack if CI is green.
3. Do not promote scout in docs/README yet.
4. Next enhancement work should focus on file-selection quality and map overhead reduction:
   - improve scout file ranking for broad/architecture intents
   - avoid paying scout-map overhead when the selected fetch bundle is not materially narrower than `archex_query`
   - consider chunk-level or symbol-level second-phase handles instead of expanding whole files by default
   - add task-level diagnostics explaining why expected files were absent from the scout map

## Refs
- Plan: `.docs/2026-06-12-competitive-enhancement-plan.md` sections `2.3 Research synthesis` and `C5`
- Related PRs: #201, #202, #203, #204
- Smoke output: `artifact://73`
- Local validation output: `artifact://75`
- Operator log: `.docs/operator-run.log`
