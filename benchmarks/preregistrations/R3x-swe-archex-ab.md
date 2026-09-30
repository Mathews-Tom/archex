# R3x — SWE-task A/B of archex surfaces under omp, four models

**DRAFT — not frozen; freeze after Stage 1.** Fields marked *(set at freeze)* are fixed from Stage 0 and Stage 1 measurements before the first confirmatory cell. No campaign cell exists at the time of writing; nothing here is post-hoc. Stage 1 pilot data is never pooled into the confirmatory analysis.

Design source: the approved design `SWE-task A/B of archex surfaces under omp` (2026-09-29). Harness: `src/archex/benchmark/swe_ab.py`, `scripts/run_swe_ab_cell.py`, `scripts/run_swe_ab_suite.py`, `scripts/swe_ab_stage0.py`, `scripts/swe_ab_stub_provider.py`; operator steps in `benchmarks/swe_ab/RUNBOOK.md`.

## Study identity

- **Spike ID and title:** R3x — SWE-task A/B of archex surfaces (annotation hook, CLI) under omp
- **Evidence class:** `original`
- **Decision owner and date:** archex maintainer, 2026-09-29 (draft)
- **First-run commit:** Leave blank until this pre-registration is frozen and merged; then record the first commit allowed to generate confirmatory data.

### Pinned identities

| | Value |
| --- | --- |
| Agent | omp `18.4.4` (`OMP_VERSION`), auto-update off, isolated `--profile swebench` with no login store |
| archex | the release that ships the annotation hook; wheel SHA-256 recorded per cell *(set at freeze)* |
| Hook module | `render_annotation_hook_module("/opt/archex/venv/bin/python")`, SHA-256 `3ab3ae16cfc52c113c83ce135947f74ac3fec3b9e468f2680e1c79adb0982f46` at archex 0.34.0 *(re-pin at freeze)* |
| CLI guide | `benchmarks/swe_ab/cli-guide.md`, SHA-256 `2741251d17a923fbd703eb5adea494fbfa0245f5d842ab38083d6e5a60ee0597` |
| Models | `anthropic/claude-sonnet-5-5`, `anthropic/claude-opus-5-5`, `openai-codex/gpt-6-sol`, `openai-codex/gpt-6-luna`, as omp selectors; each resolves in the omp 18.4.4 catalog to its own provider (`anthropic` or `openai-codex`) and that provider has an enabled login in omp's `agent.db` (Stage 0 `model_routes_and_logins`, 2026-09-30) |
| Billing and auth | operator subscriptions via `omp auth-broker`; the container gets `OMP_AUTH_BROKER_URL` and `OMP_AUTH_BROKER_TOKEN` only (*Billing mode and route*) |
| Thinking | `high` in every arm |
| Tools | `read,bash,edit,write,grep,glob,todo` in every arm |
| Isolation | `--no-lsp --no-skills --no-rules --no-extensions`; `-e <hook module>` only in H and HC; `--no-title` |
| Time cap | `--max-time 60m` |
| Channel rule table | `CHANNELS` and `omp_channel` in `swe_ab.py`, tokenizer `cl100k_base` |
| Task set | SWE-bench Pro V2 public split (642 tasks, 11 repositories), images `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` |

## Hypothesis

With the same omp build, model, thinking level, task prompt, tool set, and time cap, and SWE-bench Pro V2 tasks as the population:

- **H1 (efficiency, per model):** the annotation hook alone (H) reduces total billed tokens per task relative to control (A0) by at least the SESOI.
- **H2 (efficiency, per model):** the hook plus the archex CLI and its guide (HC) reduces total billed tokens per task relative to A0 by at least the SESOI.
- **H3 (quality guardrail, pooled):** H and HC are each non-inferior to A0 in solve rate within the NIM.

Primary comparison family: H vs A0 and HC vs A0, **Holm-adjusted within each model** for H1/H2. Arms:

| Arm | archex surface | Extra system prompt | Stage |
| --- | --- | --- | --- |
| A0 | none; archex not installed | none | pilot + confirmatory |
| H | annotation hook; archex not on the agent's `PATH` | none | pilot + confirmatory |
| HC | annotation hook + CLI on `PATH`, index pre-built | CLI guide | pilot + confirmatory |
| C | CLI on `PATH`, index pre-built | CLI guide | pilot only (adoption without the hook) |

## Primary metric

1. **Efficiency, per model.** Per task, the ratio of total billed tokens (provider-reported input + cache-write + cache-read + output, summed over every request of the cell) in the treatment arm to A0; aggregated as the geometric mean over tasks. Lower is better. A failed cell counts at the tokens it consumed.
2. **Quality guardrail, pooled.** Paired solve-rate difference, treatment − A0, pooled over the four models with the model as a stratum, for H and HC separately. A cell is resolved when the task's own V2 verifier (`tests/test.sh`) passes on its `git diff` applied in a fresh container of the same image. Every failed cell scores unresolved.

Everything else is exploratory: hook activity from the extension ledger (annotated ÷ eligible calls, annotation tokens compounded, fail-open rate by reason, annotated share after the first edit), CLI adoption, channel decomposition (compounded: search, read, edit, test, archex-annotation, archex-CLI, other), the displacement ratio, localization (first gold read/edit request, all gold read, non-gold edits), turns, wall time, timeouts, quota blocks and omp's in-run rate-limit retries, and money as omp's list-price model of the tokens (modelled, not billed).

## Billing mode and route

- **Billing mode: subscription.** Every model runs on the operator's Claude (`anthropic`) or ChatGPT (`openai-codex`) subscription login held by omp. Nothing is billed per token. The efficiency metric is provider-reported tokens, which subscription billing does not change; **money is omp's list-price model of those tokens** (`usage.cost.total`), reported as a modelled figure and used only as the runaway-cost ceiling, never as spend.
- **Route.** The logins live in omp's `agent.db` on the host. `omp auth-broker serve` runs on the host over that file and is the only process that refreshes them. An agent container receives exactly two variables, `OMP_AUTH_BROKER_URL` and `OMP_AUTH_BROKER_TOKEN`, through an explicit allow-list; each cell records those names, never the values; `agent.db` is never copied into a container and a profile carrying a login store is refused. The container's omp calls the provider directly with the access token the broker hands it.
- **Exposure, disclosed.** The agent runs with `--approval-mode yolo` and a `bash` tool, so it can read the broker token from its environment and use the broker's API (read access tokens; refresh tokens are never sent to clients; disable or block a login). Egress from the container is not restricted to the broker and the provider endpoints; the network is recorded per cell and fixed for a stage.
- **Quota.** Subscriptions have rolling windows. Before each cell the suite reads the broker's usage report and pauses while the cell's provider has under 10% headroom. A cell that ends in a subscription rate-limit or quota block (recorded before its first tool call, or mid-run) is filed under `quota-blocked/`, counted in cost, never scored, and re-run once the quota clears (up to 2 re-runs per invocation); omp's own in-run retries and waits are recorded per cell. The primary metric uses the cell that completed; the tokens of blocked attempts are reported separately. Blocked attempts are reported per arm and model; because a heavier cell is more likely to be blocked, an imbalance of blocked attempts between arms is reported as a threat to the token comparison, and the efficiency ratio is also computed excluding tasks that had any blocked attempt, as a sensitivity check.
- **Comparability.** Subscription plans can apply plan-specific rate limits and service tiers that API access does not; token counts are comparable across arms within this study, not with API-billed studies.

## Emulation disclosure

Task images are `linux/amd64`. On Apple silicon they run under x86 emulation (Docker Desktop with Rosetta). Every cell records `emulated`; a stage is all-emulated or all-native, and the validator refuses a mix. Wall times, the 60-minute omp cap, and the 3000-second verifier cap are wall-clock, so an emulated stage is comparable only with itself; emulated timeouts are recorded failures like any other. Instances whose gold patch does not resolve, or whose empty patch does not fail, under the host's emulation leave the pool at Stage 0. Whether Bun's default x86-64 build runs under Rosetta, or the baseline build is needed, is a Stage 0 check (`bun_runs_under_emulation`); the annotate hook is pre-warmed once per hook-arm container so the first call is not a cold start.

## Open items

- **Terms of service — open, blocking.** Not yet checked: whether automated, high-volume, containerised use of Claude and ChatGPT subscription logins through omp's auth broker is permitted by each provider's current terms and usage policies. Owner: archex maintainer. No real-model cell runs until this is closed and the outcome recorded here.
- **Stage 0 on the operator's machine — open.** The container checks (gold/empty validity, omp and archex in the image, broker reachable from a container, Bun under emulation, emulated wall times) were not run at the time of writing because Docker Desktop was not running; the local checks (model routes and logins, broker health and usage, stub rehearsal per arm) passed.

## SESOI

A **10%** reduction in the per-task billed-token ratio (geometric mean ≤ 0.90). *(Confirm or revise at freeze from the Stage 1 A0-vs-A0 token CV; the value is set from the operator decision, not from observed variance.)*

## Decision margins

- **Minimum worthwhile gain (MWG):** 10% billed-token reduction per task, per model — the smallest saving that pays for a default-on hook's maintenance and review surface.
- **Non-inferiority margin (NIM):** **−5 percentage points** of solve rate, pooled, per treatment arm; a larger loss is not acceptable for any token saving.
- **Equivalence margin (EQM):** ±5% on the billed-token ratio for reading a flat result as "no effect" *(set at freeze)*.

## Clustering unit

The **repository** (11 in V2). Tasks from one repository share code, conventions, and test infrastructure, so their outcomes are correlated; repetitions of a task stay within its repository. Inference: repository-clustered percentile bootstrap, 10,000 resamples, seed 20260909, for the efficiency ratio and the guardrail difference; a stratified McNemar test cross-checks the guardrail.

## Kill criterion

Evaluated in order; each is binding.

1. **Stage −1 headroom gate — passed.** `benchmarks/evidence/annotation-headroom.json`: 49.2% of 12,220 eligible omp search calls hit two or more code units (repository-clustered 95% CI 42.8–53.3%), above the 15% gate.
2. **Stage 0 feasibility gate.** Every check in `scripts/swe_ab_stage0.py --host` passes, or the campaign stops with the blocking check named. This includes the one-real-cell-per-model check and the broker-reachability and emulation checks, and it requires the terms-of-service open item below to be closed.
3. **Headroom gate (per model, from Stage 1 A0).** For H the lever is the out-of-patch **read** share of compounded billed tokens in A0; for HC it is search plus out-of-patch reads. If a model's A0 lever share is below 2 × SESOI (20%), that model's efficiency hypothesis is declared mis-specified and reported as "no attainable headroom", not as "archex does not help".
4. **Adoption gate (HC, from Stage 1).** If fewer than 25% of HC cells make at least one archex CLI call, HC reduces to H plus instruction tokens and the confirmatory run keeps only A0 vs H.
5. **Hook-activity check.** A cell with zero annotated calls despite eligible calls is flagged; if the H arm's annotated ÷ eligible share is below 50% in Stage 1 for reasons other than index staleness after edits, stop and localize before Stage 2.
6. **Cost.** Each stage runs under a hard cumulative ceiling checked before every cell; reaching it aborts the stage. Stage 1: *(set at freeze from Stage 0 cost per cell)*. Stage 2: set from the measured Stage 1 cost per cell.

A null result on H1/H2 with a non-inferior guardrail is a valid outcome: the hook stays opt-in for that model.

## Run and analysis boundary

- **Population freeze.** Candidate pool: every V2 instance whose image passes the Stage 0 validity check on our infrastructure (gold patch resolves, empty patch fails); failures are excluded before any agent run and listed. Stage 1: 24 tasks (≈2 per repository); Stage 2: 100 new tasks. Stratified by repository with seed 20260909; pilot and confirmatory samples are disjoint. `scripts/swe_ab_sample.py` draws both samples deterministically (module constant `SEED = 20260909`, not a flag): within each repository, instances are ranked by `sha256(f"{SEED}:{instance_id}")`. Stage 1 takes 2 per repository plus 1 extra for each of the two repositories with the largest V2 pool (ties by repository name); Stage 2 allocates 100 tasks proportionally to each repository's full V2 pool by the largest-remainder method. An instance excluded for one of the two validity failures (`gold_not_resolved`, `empty_not_failing`) is replaced by the next valid instance in the same repository's rank order, and a repository that runs out has its shortfall reassigned, one task at a time, to the repository with the most remaining valid instances. Stage 2 skips every Stage 1 instance and refuses a Stage 1 plan that differs from the sampler's own recompute.
- **Cells.** Stage 1: per model, A0 twice, H, HC, and C once each (120 cells per model, 480 total). Stage 2: 100 tasks × 4 models × {A0, H, HC}, one run each (1,200 cells; 800 if the adoption gate drops HC).
- **Power.** The pooled non-inferiority test at 15% discordance and a 5-point margin needs about 370 pairs per comparison at one-sided α = 0.05 and 80% power; 400 pairs meets that. Efficiency power is recomputed from the Stage 1 token CV before freezing.
- **Harness.** `scripts/run_swe_ab_suite.py run --runtime docker` per `benchmarks/swe_ab/RUNBOOK.md`; `scripts/run_swe_ab_suite.py validate` must accept the result directory before analysis. The validator refuses any cell run against an overridden provider endpoint, any cell that did not authenticate through the broker or that ran on a provider other than its model's subscription route, any cell left as a quota block, any undeclared cell, missing declared cells, mixed identities within an arm, and a stage that mixes emulated and native cells.
- **Exclusions.** None after the population freeze. Timeouts, provider errors, and harness errors are recorded failures (unresolved; tokens as consumed). A provider failure that is not a quota block, before the first tool call, is retried once and the retry is recorded. A quota block is handled as in *Billing mode and route*, not as a failure.
- **Harness defects found after data exists:** discard and restart the affected stage; never repair cells in place.

## Deviations from the design, recorded before any data

- **Task set is SWE-bench Pro V2.** The design names "the SWE-bench Pro public set". The public set is now V2 (642 validated tasks, same 11 repositories) with per-instance GHCR images and task-owned verifiers; the v1 pipeline and its Docker Hub images are kept upstream only to reproduce v1 numbers.
- **Time cap stays 60 minutes.** V2's locked protocol uses a 50-minute agent budget; this study keeps the design's 60 minutes in every arm and discloses that its solve rates are not directly comparable to V2 leaderboard numbers.
- **Agent-phase network.** V2's locked protocol runs the agent offline except for the model endpoint. The harness runs agent containers on a configurable Docker network (`--network`); whether the campaign restricts egress is fixed at freeze and applied identically to every arm.
- **Annotation degree is importers only.** The design's annotation line carries "callers / importers"; the archex index stores no call edges, so lines carry importer counts only.
- **`--no-title`.** Added to the design's command line so omp spends no model call on a session title.
- **System-prompt isolation is checked on the stub.** omp does not persist a top-level session's system prompt; Stage 0 checks the rendered prompt and tool list from the request omp sends to the local stub under the same build and flags, per arm. Per-model prompt variants are not observable without a recording proxy.
- **Models run on operator subscriptions through omp's auth broker, not on API billing.** The design's Bedrock inference-profile ids (`global.anthropic.…`, `global.openai.…`) are replaced by `anthropic/claude-sonnet-5-5`, `anthropic/claude-opus-5-5`, `openai-codex/gpt-6-sol`, `openai-codex/gpt-6-luna`. See *Billing mode and route* below; the validator refuses any cell that ran through another provider.
- **omp `18.4.4`, not `18.4.2`.** The build installed when the subscription route was verified (2026-09-30); the whole campaign is pinned to it.
- **Quota blocks are re-run, not scored.** A cell stopped by a subscription rate limit or quota block is not a task outcome; it is re-run from scratch once the quota clears and the blocked attempt is kept, unscored (*Billing mode and route*).
- **Emulation may be used.** Stages may run on Apple silicon under x86 emulation, disclosed and never mixed with native cells (*Emulation disclosure*).

## Post-hoc changes

None.
