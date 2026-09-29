# R3x — SWE-task A/B of archex surfaces under omp, four models

**DRAFT — not frozen; freeze after Stage 1.** Fields marked *(set at freeze)* are fixed from Stage 0 and Stage 1 measurements before the first confirmatory cell. No campaign cell exists at the time of writing; nothing here is post-hoc. Stage 1 pilot data is never pooled into the confirmatory analysis.

Design source: the approved design `SWE-task A/B of archex surfaces under omp` (2026-09-29). Harness: `src/archex/benchmark/swe_ab.py`, `scripts/run_swe_ab_cell.py`, `scripts/run_swe_ab_suite.py`, `scripts/swe_ab_stage0.py`; operator steps in `benchmarks/swe_ab/RUNBOOK.md`.

## Study identity

- **Spike ID and title:** R3x — SWE-task A/B of archex surfaces (annotation hook, CLI) under omp
- **Evidence class:** `original`
- **Decision owner and date:** archex maintainer, 2026-09-29 (draft)
- **First-run commit:** Leave blank until this pre-registration is frozen and merged; then record the first commit allowed to generate confirmatory data.

### Pinned identities

| | Value |
| --- | --- |
| Agent | omp `18.4.2` (`OMP_VERSION`), auto-update off, isolated `--profile swebench` |
| archex | the release that ships the annotation hook; wheel SHA-256 recorded per cell *(set at freeze)* |
| Hook module | `render_annotation_hook_module("/opt/archex/venv/bin/python")`, SHA-256 `b1d47e16daead490265ecbd5d8bde1099445c7878fd2f8e808ca5b5fc8d6460c` at archex 0.32.0 + this stack *(re-pin at freeze)* |
| CLI guide | `benchmarks/swe_ab/cli-guide.md`, SHA-256 `2741251d17a923fbd703eb5adea494fbfa0245f5d842ab38083d6e5a60ee0597` |
| Models | `global.anthropic.claude-sonnet-5-5`, `global.anthropic.claude-opus-5-5`, `global.openai.gpt-6-sol`, `global.openai.gpt-6-luna`; each resolves to the `amazon-bedrock` route in the omp 18.4.2 catalog (Stage 0, 2026-09-29) |
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

Everything else is exploratory: hook activity from the extension ledger (annotated ÷ eligible calls, annotation tokens compounded, fail-open rate by reason, annotated share after the first edit), CLI adoption, channel decomposition (compounded: search, read, edit, test, archex-annotation, archex-CLI, other), the displacement ratio, localization (first gold read/edit request, all gold read, non-gold edits), turns, wall time, timeouts, and money at each provider's billed tiers.

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
2. **Stage 0 feasibility gate.** Every check in `scripts/swe_ab_stage0.py --host` passes, or the campaign stops with the blocking check named. This includes the one-real-cell-per-model check.
3. **Headroom gate (per model, from Stage 1 A0).** For H the lever is the out-of-patch **read** share of compounded billed tokens in A0; for HC it is search plus out-of-patch reads. If a model's A0 lever share is below 2 × SESOI (20%), that model's efficiency hypothesis is declared mis-specified and reported as "no attainable headroom", not as "archex does not help".
4. **Adoption gate (HC, from Stage 1).** If fewer than 25% of HC cells make at least one archex CLI call, HC reduces to H plus instruction tokens and the confirmatory run keeps only A0 vs H.
5. **Hook-activity check.** A cell with zero annotated calls despite eligible calls is flagged; if the H arm's annotated ÷ eligible share is below 50% in Stage 1 for reasons other than index staleness after edits, stop and localize before Stage 2.
6. **Cost.** Each stage runs under a hard cumulative ceiling checked before every cell; reaching it aborts the stage. Stage 1: *(set at freeze from Stage 0 cost per cell)*. Stage 2: set from the measured Stage 1 cost per cell.

A null result on H1/H2 with a non-inferior guardrail is a valid outcome: the hook stays opt-in for that model.

## Run and analysis boundary

- **Population freeze.** Candidate pool: every V2 instance whose image passes the Stage 0 validity check on our infrastructure (gold patch resolves, empty patch fails); failures are excluded before any agent run and listed. Stage 1: 24 tasks (≈2 per repository); Stage 2: 100 new tasks. Stratified by repository with seed 20260909; pilot and confirmatory samples are disjoint.
- **Cells.** Stage 1: per model, A0 twice, H, HC, and C once each (120 cells per model, 480 total). Stage 2: 100 tasks × 4 models × {A0, H, HC}, one run each (1,200 cells; 800 if the adoption gate drops HC).
- **Power.** The pooled non-inferiority test at 15% discordance and a 5-point margin needs about 370 pairs per comparison at one-sided α = 0.05 and 80% power; 400 pairs meets that. Efficiency power is recomputed from the Stage 1 token CV before freezing.
- **Harness.** `scripts/run_swe_ab_suite.py run --runtime docker` per `benchmarks/swe_ab/RUNBOOK.md`; `scripts/run_swe_ab_suite.py validate` must accept the result directory before analysis. The validator refuses any cell run against an overridden provider endpoint, any undeclared cell, missing declared cells, and mixed identities within an arm.
- **Exclusions.** None after the population freeze. Timeouts, provider errors, and harness errors are recorded failures (unresolved; tokens as consumed). A provider failure before the first tool call is retried once and the retry is recorded.
- **Harness defects found after data exists:** discard and restart the affected stage; never repair cells in place.

## Deviations from the design, recorded before any data

- **Task set is SWE-bench Pro V2.** The design names "the SWE-bench Pro public set". The public set is now V2 (642 validated tasks, same 11 repositories) with per-instance GHCR images and task-owned verifiers; the v1 pipeline and its Docker Hub images are kept upstream only to reproduce v1 numbers.
- **Time cap stays 60 minutes.** V2's locked protocol uses a 50-minute agent budget; this study keeps the design's 60 minutes in every arm and discloses that its solve rates are not directly comparable to V2 leaderboard numbers.
- **Agent-phase network.** V2's locked protocol runs the agent offline except for the model endpoint. The harness runs agent containers on a configurable Docker network (`--network`); whether the campaign restricts egress is fixed at freeze and applied identically to every arm.
- **Annotation degree is importers only.** The design's annotation line carries "callers / importers"; the archex index stores no call edges, so lines carry importer counts only.
- **`--no-title`.** Added to the design's command line so omp spends no model call on a session title.
- **System-prompt isolation is checked on the stub.** omp does not persist a top-level session's system prompt; Stage 0 checks the rendered prompt and tool list from the request omp sends to the local stub under the same build and flags, per arm. Per-model prompt variants are not observable without a recording proxy.

## Post-hoc changes

None.
