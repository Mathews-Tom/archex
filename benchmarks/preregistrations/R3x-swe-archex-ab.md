# R3x — SWE-task A/B of archex surfaces under omp, three configurations on Muna-hosted open models

**DRAFT — not frozen; freeze after Stage 1.** Fields marked *(set at freeze)* are fixed from Stage 0 and Stage 1 measurements before the first confirmatory cell. No campaign cell exists at the time of writing; nothing here is post-hoc. Stage 1 pilot data is never pooled into the confirmatory analysis.

Design source: the approved design `SWE-task A/B of archex surfaces under omp` (2026-09-29). Harness: `src/archex/benchmark/swe_ab.py`, `scripts/run_swe_ab_cell.py`, `scripts/run_swe_ab_suite.py`, `scripts/swe_ab_stage0.py`, `scripts/swe_ab_stub_provider.py`, `scripts/swe_ab_sample.py`, `scripts/swe_ab_analysis.py`, `benchmarks/swe_ab/muna-models.yml`, `benchmarks/swe_ab/omp-campaign.yml`; operator steps in `benchmarks/swe_ab/RUNBOOK.md`.

## Study identity

- **Spike ID and title:** R3x — SWE-task A/B of archex surfaces (annotation hook, CLI) under omp
- **Evidence class:** `original`
- **Decision owner and date:** archex maintainer, 2026-09-29 (draft)
- **First-run commit:** Leave blank until this pre-registration is frozen and merged; then record the first commit allowed to generate confirmatory data.

### Pinned identities

| | Value |
| --- | --- |
| Agent | omp `18.4.4` (`OMP_VERSION`), auto-update off, isolated `--profile swebench` with no login store |
| Provider config | `benchmarks/swe_ab/muna-models.yml` installed as the profile's `models.yml`, SHA-256 *(set at freeze)*; recorded per cell |
| omp settings overlay | `benchmarks/swe_ab/omp-campaign.yml` passed with `--config`, SHA-256 *(set at freeze)*; recorded per cell |
| archex | the release that ships the annotation hook; wheel SHA-256 recorded per cell *(set at freeze)* |
| Hook module | `render_annotation_hook_module("/opt/archex/venv/bin/python")`, SHA-256 `3ab3ae16cfc52c113c83ce135947f74ac3fec3b9e468f2680e1c79adb0982f46` at archex 0.34.0 *(re-pin at freeze)* |
| CLI guide | `benchmarks/swe_ab/cli-guide.md`, SHA-256 `2741251d17a923fbd703eb5adea494fbfa0245f5d842ab38083d6e5a60ee0597` |
| Configurations | three, below; the label is the `model` of every plan, cell, and stratum |
| Route and credentials | provider `muna`, `https://inference.muna.ai/v1`, `api: openai-completions`; the container gets exactly `MUNA_ACCESS_KEY` (bare `-e NAME`); cells record the name, never the value (*Billing mode and route*) |
| Thinking | per configuration (below), identical across arms |
| Tools | `read,bash,edit,write,grep,glob,todo` in every arm |
| Isolation | `--no-lsp --no-skills --no-rules --no-extensions`; `-e <hook module>` only in H and HC; `--no-title` |
| Time cap | `--max-time 60m` |
| Hook budget | `ARCHEX_HOOK_TIMEOUT_SECONDS=5` in every arm's agent environment (acts only where the hook loads); recorded per cell as `hook_timeout_seconds`, one value per stage |
| Channel rule table | `CHANNELS` and `omp_channel` in `swe_ab.py`, tokenizer `cl100k_base` |
| Task set | SWE-bench Pro V2 public split (642 tasks, 11 repositories), images `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` |

Configurations (effort is a configuration factor, not an arm factor):

| Label | omp selector | Thinking |
| --- | --- | --- |
| `qwen-3.8-27b@low` | `muna/@qwen/qwen-3.8-27b` | `low` |
| `qwen-3.8-27b@high` | `muna/@qwen/qwen-3.8-27b` | `high` |
| `gemma-4-26b-a4b-it@high` | `muna/@google/gemma-4-26b-a4b-it` | `high` |

omp sends the effort as `reasoning_effort` only when the model's provider config sets `compat.thinkingFormat: openai`; Qwen requires it, Gemma sends it by default. Stage 0 `effort_request_shape` checks the first request body of each configuration. The two Qwen configurations share weights and are not independent models: no claim is made across them as independent replications.

## Hypothesis

With the same omp build, configuration, task prompt, tool set, and time cap, and SWE-bench Pro V2 tasks as the population (two model families, three configurations):

- **H1 (efficiency, per configuration):** the annotation hook alone (H) reduces total billed tokens per task relative to control (A0) by at least the SESOI.
- **H2 (efficiency, per configuration):** the hook plus the archex CLI and its guide (HC) reduces total billed tokens per task relative to A0 by at least the SESOI.
- **H3 (quality guardrail, pooled):** H and HC are each non-inferior to A0 in solve rate within the NIM, pooled over the configurations that clear the solve-rate floor (*Kill criterion* 4). The pool takes the configurations as strata; because the two Qwen configurations are not independent, the pooled interval is not a three-model replication.

Primary comparison family: H vs A0 and HC vs A0, **Holm-adjusted within each configuration** for H1/H2. Arms:

| Arm | archex surface | Extra system prompt | Stage |
| --- | --- | --- | --- |
| A0 | none; archex not installed | none | pilot + confirmatory |
| H | annotation hook; archex not on the agent's `PATH` | none | pilot + confirmatory |
| HC | annotation hook + CLI on `PATH`, index pre-built | CLI guide | pilot + confirmatory |
| C | CLI on `PATH`, index pre-built | CLI guide | pilot only (adoption without the hook) |

## Primary metric

1. **Efficiency, per configuration.** Per task, the ratio of total billed tokens (provider-reported input + cache-write + cache-read + output, summed over every request of the cell) in the treatment arm to A0; aggregated as the geometric mean over tasks. Lower is better. A failed cell counts at the tokens it consumed.
2. **Quality guardrail, pooled.** Paired solve-rate difference, treatment − A0, pooled over the configurations that clear the solve-rate floor with the configuration as a stratum, for H and HC separately. A cell is resolved when the task's own V2 verifier (`tests/test.sh`) passes on its `git diff` applied in a fresh container of the same image. Every failed cell scores unresolved.

Everything else is exploratory: hook activity from the extension ledger (annotated ÷ eligible calls, annotation tokens compounded, fail-open rate by reason, annotated share after the first edit), CLI adoption, channel decomposition (compounded: search, read, edit, test, archex-annotation, archex-CLI, other), the displacement ratio, localization (first gold read/edit request, all gold read, non-gold edits), turns, wall time, timeouts, rate-limit and capacity blocks and omp's in-run retries, and money as omp's model of the tokens from the prices in `muna-models.yml`.

## Billing mode and route

- **Billing mode: Muna API key, per-token credits.** Cost is real money. omp models it per request from the prices in `benchmarks/swe_ab/muna-models.yml` (`usage.cost.total`): USD per million tokens, input / cached input / output, Qwen 3.8 27B $0.25 / $0.02 / $1.75 and Gemma 4 26B $0.065 / $0.02 / $0.30 (Muna's price list as given by the operator, 2026-10-02). The suite refuses to start a docker run while any price of input, output, or cacheRead is 0, so the cost ceiling is never inert (cacheWrite is exempt: `openai-completions` reports no cache-write tokens). The ceiling is a runaway guard on that model, not an invoice; Muna's own billing is authoritative.
- **Route.** The agent container's omp calls `https://inference.muna.ai/v1` (`api: openai-completions`) directly with `MUNA_ACCESS_KEY`. The container receives exactly that one variable, through an explicit allow-list (bare `-e NAME`, so the value never appears in a command line); each cell records the name, never the value. The profile carries no login store; the key is read by the suite from `--env-file` (default repo-root `.env`, only the `MUNA_ACCESS_KEY=` line).
- **Exposure, disclosed.** The agent runs with `--approval-mode yolo` and a `bash` tool, so it can read `MUNA_ACCESS_KEY` from its environment and spend credits or use the account. Egress from the container is not restricted; the network is recorded per cell and fixed for a stage.
- **Rate limits and capacity.** Muna swaps models in and out of shared GPU capacity. A request can return HTTP 429, including `model_loading` and `model_capacity_exhausted`; waits of about 6 minutes were observed when switching models. omp retries in-run under the committed overlay `omp-campaign.yml` (`retry.maxRetries: 20`, `retry.waitForUsageReset: false`). Measured against omp 18.4.4 and the local stub answering a Muna-shaped 429 with no retry-after (pre-data, no hosted call): the default budget (10 retries) gave up after 398–400 s wall-clock, the overlay after 785–789 s (20 retries; before the first tool call and after one; `model_capacity_exhausted` and `model_loading` alike), so omp keeps retrying for more than 10 minutes; a usage-limit 429 with a 4-hour reset still fails fast. The stub answers instantly, so real Muna round-trip time only lengthens the wait; Stage 0 re-checks it against Muna's real swap behaviour *(set at freeze)*. A cell that still ends in a rate-limit or capacity block is filed under `quota-blocked/`, counted in cost, never scored, and re-run by the suite after `--block-cooldown-seconds` (default 300), up to `--quota-retries` (default 2) per invocation, then left for resume; the validator refuses blocked cells. The primary metric uses the cell that completed; the tokens of blocked attempts are reported separately. Blocked attempts are reported per arm and configuration; because a heavier cell is more likely to be blocked, an imbalance of blocked attempts between arms is reported as a threat to the token comparison, and the efficiency ratio is also computed excluding tasks that had any blocked attempt, as a sensitivity check.
- **Credit exhaustion.** Credit exhaustion (HTTP 402 or the insufficient-credits family; Muna's response shape is unverified) is never retried and never scored: the attempt is filed under `quota-blocked/`, the suite waits for running cells and exits with status 5, and the run resumes after a top-up.
- **Scheduling for capacity.** Cells run ordered by (omp selector, i.e. model family; task; configuration label; arm; repetition), so one family stays loaded. `--prune-images` removes a task's image when its cells for the current family are done; docker re-pulls it for the next family.
- **Comparability.** Token counts are comparable within this study only: Muna's serving stack, tokenizer yield, and caching behaviour are not those of other providers.

## Emulation disclosure

Task images are `linux/amd64`. On Apple silicon they run under x86 emulation (Docker Desktop with Rosetta). Every cell records `emulated`; a stage is all-emulated or all-native, and the validator refuses a mix. Wall times, the 60-minute omp cap, and the 3000-second verifier cap are wall-clock, so an emulated stage is comparable only with itself; emulated timeouts are recorded failures like any other. Instances whose gold patch does not resolve, or whose empty patch does not fail, under the host's emulation leave the pool at Stage 0. Whether Bun's default x86-64 build runs under Rosetta, or the baseline build is needed, is a Stage 0 check (`bun_runs_under_emulation`); the annotate hook is pre-warmed once per hook-arm container so the first call is not a cold start.

## Open items

- **Terms of service — closed.** Operator statement, verbatim, 2026-10-02: "I have verified Muna documentation and we are clear to use the endpoint".
- **Stage 0 on the operator's machine — open.** Not yet run: the container checks (gold/empty validity, omp and archex in the image, Muna reachable from a container, Bun under emulation, emulated wall times) and the one-real-cell-per-configuration check.
- **Muna prices — closed.** Entered in `benchmarks/swe_ab/muna-models.yml` from Muna's price list as given by the operator on 2026-10-02 (see *Billing mode and route*). `usage.cost_usd` being non-zero in the first real cells confirms the prices reached omp (RUNBOOK §5).
- **Credit-exhaustion response shape — unverified.** Detection follows the 402 / insufficient-credits family; the real shape is confirmed only if it occurs.

## SESOI

A **10%** reduction in the per-task billed-token ratio (geometric mean ≤ 0.90). *(Confirm or revise at freeze from the Stage 1 A0-vs-A0 token CV; the value is set from the operator decision, not from observed variance.)*

## Decision margins

- **Minimum worthwhile gain (MWG):** 10% billed-token reduction per task, per configuration — the smallest saving that pays for a default-on hook's maintenance and review surface.
- **Non-inferiority margin (NIM):** **−5 percentage points** of solve rate, pooled, per treatment arm; a larger loss is not acceptable for any token saving.
- **Equivalence margin (EQM):** ±5% on the billed-token ratio for reading a flat result as "no effect" *(set at freeze)*.

## Clustering unit

The **repository** (11 in V2). Tasks from one repository share code, conventions, and test infrastructure, so their outcomes are correlated; repetitions of a task stay within its repository. Inference: repository-clustered percentile bootstrap, 10,000 resamples, seed 20260909, for the efficiency ratio and the guardrail difference; a stratified McNemar test cross-checks the guardrail.

## Kill criterion

Evaluated in order; each is binding.

1. **Stage −1 headroom gate — passed.** `benchmarks/evidence/annotation-headroom.json`: 49.2% of 12,220 eligible omp search calls hit two or more code units (repository-clustered 95% CI 42.8–53.3%), above the 15% gate. Scope: this evidence (and `TOKEN_HEADROOM.md`) came from Sonnet 4.5 / GPT-5 trajectories and does not transfer to these models.
2. **Stage 0 feasibility gate.** Every check in `scripts/swe_ab_stage0.py --host` passes, or the campaign stops with the blocking check named. This includes the one-real-cell-per-configuration check, Muna reachability, key acceptance, the effort request shape, and the emulation checks. The terms-of-service item is closed (*Open items*).
3. **A0 feasibility stop (pre-data rule).** Before Stage 1, run the first 6 tasks of the Stage 1 plan × 3 configurations × A0 × 1 repetition (18 cells; command in RUNBOOK §8). Results are kept outside `benchmarks/swe_ab/results/` and are never analysis data. They decide only whether SWE-bench Pro is solvable enough to proceed: if every configuration resolves 0 of 6, stop and revisit the benchmark.
4. **Solve-rate floor (Stage 1 gate).** Per configuration, the A0 solve rate over all A0 cells (both repetitions) must be ≥ 15%. A configuration below it is uninformative for the guardrail: it is reported per configuration labelled `uninformative: below the 15% solve-rate floor`, excluded from the pooled guardrail in Stage 2 (`--pilot-analysis`), and the guardrail reads `not_estimable` if none pass. Its efficiency result is still reported per configuration.
5. **Headroom gate (per configuration, from Stage 1 A0).** For H the lever is the out-of-patch **read** share of compounded billed tokens in A0; for HC it is search plus out-of-patch reads. If a configuration's A0 lever share is below 2 × SESOI (20%), that configuration's efficiency hypothesis is declared mis-specified and reported as "no attainable headroom", not as "archex does not help".
6. **Adoption gate (HC, from Stage 1).** If fewer than 25% of HC cells make at least one archex CLI call, HC reduces to H plus instruction tokens and the confirmatory run keeps only A0 vs H.
7. **Hook-activity check.** A cell with zero annotated calls despite eligible calls is flagged; if the H arm's annotated ÷ eligible share is below 50% in Stage 1 for reasons other than index staleness after edits, stop and localize before Stage 2.
8. **Cost.** Each stage runs under a hard cumulative ceiling checked before every cell; reaching it aborts the stage. Stage 1: *(set at freeze from Stage 0 cost per cell)*. Stage 2: set from the measured Stage 1 cost per cell.

A null result on H1/H2 with a non-inferior guardrail is a valid outcome: the hook stays opt-in for that configuration.

## Run and analysis boundary

- **Population freeze.** Candidate pool: every V2 instance whose image passes the Stage 0 validity check on our infrastructure (gold patch resolves, empty patch fails); failures are excluded before any agent run and listed. Stage 1: 24 tasks (≈2 per repository); Stage 2: N new tasks, N = max(100, ⌈370 / k⌉) with k the number of configurations passing the solve-rate floor in Stage 1 (k = 3 → 124; the guardrail needs ≈370 pairs). Stratified by repository with seed 20260909; pilot and confirmatory samples are disjoint. `scripts/swe_ab_sample.py` draws both samples deterministically (module constant `SEED = 20260909`, not a flag): within each repository, instances are ranked by `sha256(f"{SEED}:{instance_id}")`. Stage 1 takes 2 per repository plus 1 extra for each of the two repositories with the largest V2 pool (ties by repository name); Stage 2 allocates N tasks (`--stage2-tasks N`, N ≥ 100, recorded in the manifest) proportionally to each repository's full V2 pool by the largest-remainder method. An instance excluded for one of the two validity failures (`gold_not_resolved`, `empty_not_failing`) is replaced by the next valid instance in the same repository's rank order, and a repository that runs out has its shortfall reassigned, one task at a time, to the repository with the most remaining valid instances. Stage 2 skips every Stage 1 instance and refuses a Stage 1 plan that differs from the sampler's own recompute.
- **Cells.** Stage 1: 24 tasks × 3 configurations × {A0 twice, H, HC, C once each} = 360 cells. Stage 2: N tasks × 3 configurations × {A0, H, HC}, one run each (N = 124 → 1,116 cells; two thirds of that if the adoption gate drops HC).
- **Power.** The pooled non-inferiority test at 15% discordance and a 5-point margin needs about 370 pairs per comparison at one-sided α = 0.05 and 80% power; N tasks × k passing configurations ≥ 370 meets that (k = 3, N = 124: 372 pairs). Efficiency power is recomputed from the Stage 1 token CV before freezing.
- **Harness.** `scripts/run_swe_ab_suite.py run --runtime docker` per `benchmarks/swe_ab/RUNBOOK.md`; `scripts/run_swe_ab_suite.py validate` must accept the result directory before analysis. The validator refuses any cell whose resolved provider base URL is not Muna's (a local stub or another endpoint), any cell on a provider other than `muna`, any cell with credential variables other than `MUNA_ACCESS_KEY`, any cell left as a blocked attempt, any undeclared cell, missing declared cells, mixed identities within an arm (including the provider-config and omp-config SHA-256, once per stage), and a stage that mixes emulated and native cells. Cells are scheduled by model family as in *Billing mode and route*.
- **Exclusions.** None after the population freeze. Timeouts, provider errors, and harness errors are recorded failures (unresolved; tokens as consumed). A provider failure that is not a rate-limit, capacity, or credit block, before the first tool call, is retried once and the retry is recorded. A blocked attempt is handled as in *Billing mode and route*, not as a failure.
- **Harness defects found after data exists:** discard and restart the affected stage; never repair cells in place.

## Deviations from the design, recorded before any data

- **Task set is SWE-bench Pro V2.** The design names "the SWE-bench Pro public set". The public set is now V2 (642 validated tasks, same 11 repositories) with per-instance GHCR images and task-owned verifiers; the v1 pipeline and its Docker Hub images are kept upstream only to reproduce v1 numbers.
- **Time cap stays 60 minutes.** V2's locked protocol uses a 50-minute agent budget; this study keeps the design's 60 minutes in every arm and discloses that its solve rates are not directly comparable to V2 leaderboard numbers.
- **Agent-phase network.** V2's locked protocol runs the agent offline except for the model endpoint. The harness runs agent containers on a configurable Docker network (`--network`); whether the campaign restricts egress is fixed at freeze and applied identically to every arm.
- **Annotation degree is importers only.** The design's annotation line carries "callers / importers"; the archex index stores no call edges, so lines carry importer counts only.
- **`--no-title`.** Added to the design's command line so omp spends no model call on a session title.
- **System-prompt isolation is checked on the stub.** omp does not persist a top-level session's system prompt; Stage 0 checks the rendered prompt and tool list from the request omp sends to the local stub under the same build and flags, per arm. Per-model prompt variants are not observable without a recording proxy.
- **Models replaced (operator decision, 2026-10-02).** The design's Bedrock inference-profile ids (`global.anthropic.…`, `global.openai.…`) were first replaced by four subscription models (`anthropic/claude-sonnet-5-5`, `anthropic/claude-opus-5-5`, `openai-codex/gpt-6-sol`, `openai-codex/gpt-6-luna`, run through omp's auth broker); that is withdrawn before any data, and the study runs the three Muna configurations in *Pinned identities*. Scope condition: the Stage −1 headroom evidence and `TOKEN_HEADROOM.md` came from Sonnet 4.5 / GPT-5 trajectories and do not transfer to these models.
- **Effort is a configuration factor.** A pre-declared rule decides whether a model honours the effort setting: mean reasoning tokens at `high` ≥ 1.5× mean at `low` and min(high) > max(low) ⇒ honoured. Qwen 3.8-27B: 82 vs 271 reasoning tokens at low/high ⇒ honoured, so it runs at both. Gemma 4-26B-A4B: 3 + 3 requests, mean 1,033 vs 1,100, ratio 1.06, ranges overlapping ⇒ not honoured, so it runs at `high` only.
- **Qwen needs `compat.thinkingFormat: openai`.** Stub evidence: without it `--thinking low` and `--thinking high` send byte-identical requests (`enable_thinking: true`, no effort); with it omp sends `reasoning_effort`. The provider config sets it for Qwen and Stage 0 `effort_request_shape` checks it per configuration.
- **omp auth broker removed.** The omp auth broker, the `agent.db` login store, and the subscription quota guard are deleted from the harness; the key is a plain environment variable, and prompt-time quota polling is replaced by the block handling in *Billing mode and route*.
- **Capacity scheduling.** Cells are ordered by model family so one family stays loaded, and `--prune-images` frees a task's image per family (*Billing mode and route*).
- **Solve-rate floor.** A per-configuration floor of 15% A0 solve rate gates the pooled guardrail (*Kill criterion* 4), because a configuration that rarely solves gives the guardrail no information.
- **Stage 2 size rule.** N = max(100, ⌈370 / k⌉) tasks instead of a fixed 100, so the guardrail keeps ≈370 pairs when fewer configurations pass the floor (*Run and analysis boundary*).
- **A0 feasibility check.** An 18-cell A0 check before Stage 1 (*Kill criterion* 3) decides only whether the benchmark is solvable by these models; its cells are not analysis data.
- **omp retry overlay.** `omp-campaign.yml` raises omp's in-run retry budget from 10 to 20 retries so a capacity 429 without a retry-after is retried for 785–789 s instead of 398–400 s (measured on the stub, pre-data), and sets `retry.waitForUsageReset: false` explicitly; Stage 0 re-checks the budget against Muna *(set at freeze)*.
- **omp `18.4.4`, not `18.4.2`.** The design named 18.4.2; the whole campaign is pinned to 18.4.4 instead (verified 2026-09-30).
- **Emulation may be used.** Stages may run on Apple silicon under x86 emulation, disclosed and never mixed with native cells (*Emulation disclosure*).
- **The analysis is fixed in code: `scripts/swe_ab_analysis.py`.** It refuses a directory the validator refuses or any unscored cell. Choices the text above left open, fixed before any data: the control of every pair is A0 repetition 1 (Stage 1's second A0 repetition feeds only the noise estimate); H1/H2 test H0 "ratio ≥ 0.90" with the one-sided bootstrap p (1 + #resampled ratios ≥ 0.90) / 10,001, Holm-adjusted over H and HC within the configuration at α = 0.05; each comparison reads `superior` (adjusted p < 0.05), else `equivalent` (the 90% interval inside 1 ± EQM), else `inconclusive`. A pair in which either cell billed zero tokens (a failure before any model request) has no log ratio: it is listed and left out of that configuration's efficiency estimate only, and stays in the guardrail as unresolved. The guardrail estimate is the mean of per-configuration mean differences over the configurations that clear the solve-rate floor; it reads `non_inferior` when the 5th bootstrap percentile is above −5 pp and `inferior` when the 95th is below it; the McNemar cross-check pools discordant pairs across the configuration strata (exact two-sided binomial p). The sensitivity check drops the (configuration, task) pairs in which either cell had a blocked attempt.
- **Lever share (kill criterion 5) defined.** As in `scripts/swebench_pro_token_headroom.py`: out-of-patch reads are `read`-channel results of files absent from both the gold patch and the test patch, or whose target cannot be resolved, compounded; each cell records them as `out_of_patch_read_tokens_compounded`. H's lever share is that figure over the cell's billed input tokens (input + cache read + cache write); HC's adds the compounded search channel. The per-configuration share is the mean over A0 repetitions within a task, then over tasks, compared by its point estimate with 20% (repository-clustered interval reported). Channel tokens are `cl100k_base` counts and billed input is provider-tokenized; the ratio is not rescaled, so where a provider's tokenizer yields more tokens than `cl100k_base` on code the share is understated, which errs toward declaring no headroom.
- **Hook activity (kill criterion 7) defined.** The H arm's annotated calls over its eligible calls less those declined as `index_not_fresh` after the cell's first `edit`/`write` tool call (ledger field `not_fresh_after_first_edit`), pooled over H cells. Stale declines before that call count against the hook, including ones caused by edits made through `bash`. HC's share is reported, not gated. HC adoption (criterion 6) is pooled over every HC cell, failed ones included.
- **Efficiency power from the A0 replicate.** Per configuration, the spread `sd` of per-task ln(A0 rep 2 / A0 rep 1) stands for the paired log-ratio spread under no effect; tasks needed = ⌈((z₀.₉₇₅ + z₀.₈₀)·sd / ln(1/0.90))²⌉ (the Holm first step at one-sided α/2), with power and the minimum detectable ratio at the Stage 2 size (max(100, ⌈370 / k⌉) from the Stage 1 floor result; not estimable when no configuration passes it), the 90% half-width against ln(1.05) for the equivalence margin, and the same at the one-sided 80/90/95% chi-square upper limits of `sd`. Tasks are treated as independent: about two tasks per repository cannot separate a between-repository component.
- **A verifier run past its 3,000-second cap scores unresolved.** The harness previously raised, which recorded a harness error and zeroed the cell's tokens.
- **Stage 0 measures steady-state annotate latency.** `annotate_latency_in_container` times 10 calls of the annotate entry, end to end as the hook spawns it, on 20 hits of `git grep -w return` in the image's Python, Go, JavaScript, and TypeScript sources, inside each Pro image after setup, and fails when the median exceeds the campaign's hook budget: over budget the hook drops the annotation, and the H arm would measure an inert hook. (The probe first searched every file; in qutebrowser its first 20 hits fell in `.pylintrc`, `LICENSE`, docs, and shell userscripts, so it timed the "no code units" decline instead of an annotation.)
- **omp starts through an entry wrapper that also runs on musl images.** Three images of the Stage 1 draw are Alpine (musl): both `protonmail/webclients` images and one of the two `gravitational/teleport` images. Bun's glibc build cannot start there (`failed to open elf at /lib64/ld-linux-x86-64.so.2`), so every cell on such an image would have been a harness error. The bundle now carries glibc's loader and the five libraries Bun and omp's native addon link (copied from `oven/bun:1-debian`), and `benchmarks/swe_ab/omp-entry.sh` (bundle `/opt/omp/bin/omp-entry`, the container command for every cell) runs Bun through that loader only where the image has no glibc loader; on every other image it starts Bun directly, as before. Verified with omp 18.4.4: `--version` on the protonmail and teleport Alpine images and on a glibc image, and on a protonmail image against the stub a `bash` call and a native `grep` call. The tools omp runs are the image's own.
- **The hook's budget is 5 seconds in the campaign, not the shipped 0.5 s (operator decision, 2026-10-03).** Under this host's Rosetta emulation every one of 290 timed annotate calls across 29 Stage 0 instances exceeded 0.5 s (medians 1.06–2.15 s, slowest 3.1 s; the same path is about 0.4 s natively), so at the shipped default the omp extension would kill the call and the H and HC arms would add almost no annotation (kill criterion 5). Every arm's agent runs with `ARCHEX_HOOK_TIMEOUT_SECONDS=5`, which covers the slowest call measured; the variable acts only where the hook loads, so A0 and C are unchanged, and each search in H and HC waits about 1–2 s longer in wall-clock time (tokens are unaffected). Verified on the stub: at 5 s both hook arms annotate; forced to 0.001 s both fail open with `timeout`. What H measures is therefore the effect of the annotations the hook produces, not of the shipped 0.5 s budget on an emulated host; annotations still dropped past 5 s are counted in the ledger's fail-open reasons and in criterion 5. Stage 0's latency check compares against 5 s.
- **Task images are pulled with retries.** An anonymous ghcr.io pull failed once with `401 Unauthorized` and once with a TLS handshake timeout during Stage 0; both succeed when repeated. The cell runner pulls a missing image explicitly before `docker run` and retries three times (30, 60, 120 s) before raising, so a transient registry failure is not recorded as a failed task. Stage 0 now records an instance whose container cannot start as `container_setup:<instance>` and continues; it previously ended the run.

## Post-hoc changes

None.
