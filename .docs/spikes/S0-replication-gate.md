# S0 — External replication gate (Gate A)

Pre-registered from `TEMPLATE.md`. Completed and merged before the first data-generating run. No field below is revised after data exists; post-hoc changes are recorded in the final section and every affected number is labelled exploratory.

## Study identity

- **Spike ID and title:** S0 — External replication gate (milestone R3, Gate A)
- **Evidence class:** `replication` — both arms. No arm in this spike is `adaptation` or `original`, and no result here licenses any claim about archex's own retrieval.
- **Decision owner and date:** archex maintainer, 2026-07-27.
- **First-run commit:** _blank until this pre-registration has merged; then the first commit allowed to generate data is recorded here._

## Hypothesis

**Primary (RLCoder arm).** Running RLCoder's own released evaluation pipeline, on its own released dataset, with its own released retriever weights and generator, reproduces the paper's reported exact-match gain from swapping the retriever, to within the equivalence band declared below.

- **Treatment:** the `RLCoder` arm — `RLRetriever` (`nov3630/RLRetriever`) as the retrieval model, with the paper's stop-signal and candidate settings.
- **Control:** the `RawRAG` arm — the stock UniXcoder retriever, identical in every other respect.
- **Target population:** all 1 600 tasks of the RepoEval line-level split as distributed in `nov3630/Data4RLCoder`, drawn from 8 source repositories at 200 tasks each.
- **Primary comparison family:** one comparison. Treatment minus control, exact match, one cell.

**Target cell.** RLCoder Table II, RepoEval (Line), backbone DeepSeekCoder-1B: `RawRAG` 39.31 EM against `RLCoder` 44.19 EM, a reported delta of **+4.88 EM points**. This cell is chosen before the run because it is the cheapest cell in Table II that isolates the retriever while holding the generator fixed, and because its generator is the 1.3B model, which the target hardware can run to completion.

The paper's abstract headline of "12.2%" is a *relative* improvement, not an absolute one; it is the CrossCodeEval-Python DeepSeekCoder-7B cell of the same table (26.98 → 30.28, +3.30 absolute). That cell is not the target here: the 7B generator over 8 919 CrossCodeEval-Python tasks is roughly an order of magnitude more compute than the target hardware can complete, and choosing it would risk abandoning the run mid-way. The substitution is declared here, before any data exists, and is not an outcome-informed choice.

**Secondary (cAST arm).** The cAST arm is pre-registered as a *disposition* rather than a hypothesis, because its reference setup is not released. See "Kill criterion".

## Primary metric

**Exact match (EM) delta, in percentage points, treatment minus control, on the RepoEval line-level split.**

- **Numerator:** the count of tasks whose generated completion is exactly equal to the ground-truth completion under RLCoder's own comparison, which strips comments, drops empty lines, strips each remaining line, and tests list equality (`utils/eval_metric.py`), after the Python one-statement truncation in `utils/eval_utils.py`.
- **Denominator:** 1 600 — every task in the split. No task is excluded.
- **Aggregation level:** the task. The reported EM is the pooled per-task mean over all 1 600 tasks; the delta is the difference of the two arms' pooled means on the identical task set.
- **Direction of improvement:** higher EM is better; the hypothesised delta is positive.
- **Measurement procedure:** RLCoder's own metric implementation, unmodified, invoked on the prediction files the two arms emit. Generation is greedy — the released code passes no sampling arguments — so a given arm is deterministic on fixed hardware and fixed library versions.

Everything else is **exploratory**: edit similarity (ES), per-repository EM, retrieval-stage statistics, wall clock, and any CrossCodeEval or RepoEval-API figure.

## SESOI

**+2.88 EM points.** A reproduction that recovers at least this much of the reported +4.88 is treated as having reproduced the published win. The basis is a decision, not a variance estimate: the program's Gate A question is whether this harness can recover a published retrieval effect at all, and the downstream decision that turns on it — whether R7 through R16 proceed — changes only if the recovered effect is large enough to be distinguishable from ordinary reproduction drift. Below +2.88 the recovered effect is inside the range that device, dtype, and library drift alone could plausibly explain, so it does not answer the question.

## Decision margins

Each is derived separately. None is derived from observed standard deviation, and none is substituted for another.

- **Minimum worthwhile gain (MWG): +2.88 EM points.** The smallest reproduced delta that still counts as a reproduction, and therefore the smallest that would authorise R7. Utility basis: authorising R7 commits the program to building a real-agent harness on the premise that published retrieval deltas are recoverable here. That commitment is only worth making if the recovered effect is clearly larger than reproduction drift.
- **Non-inferiority margin (NIM): 2.00 EM points below the reported delta.** The maximum shortfall against the paper's own figure that is still acceptable. Cost basis: at n = 1 600 with EM near 0.40, the paired-delta standard error is bounded above by `sqrt(0.5)/sqrt(1600)` = 1.77 points at worst-case discordance, so a shortfall inside 2.00 points is within what a same-method, different-hardware rerun can produce without any methodological difference. A shortfall larger than that is not attributable to drift and is a genuine failure to reproduce.
- **Equivalence margin (EQM): ±2.00 EM points around the reported delta**, i.e. the strictly positive interval `(−2.00, +2.00)` around +4.88, giving an equivalence band of **[+2.88, +6.88] EM points**. Utility basis: the same worst-case paired standard-error bound above, doubled to two standard errors and rounded to a round number *before* seeing any data. The band is symmetric because overshooting the paper by more than 2 points is as much a sign of a setup mismatch as undershooting it.

The band is fixed here and is never widened. Widening it after data exists is a plan violation, not a revision.

## Clustering unit

**The source repository.** The RepoEval line-level split draws its 1 600 tasks from 8 repositories, exactly 200 tasks each: `huggingface_diffusers`, `nerfstudio-project_nerfstudio`, `awslabs_fortuna`, `huggingface_evaluate`, `google_vizier`, `alibaba_FederatedScope`, `pytorch_rl`, `opendilab_ACE`. Tasks within a repository share a codebase, an API surface, a coding style, and a retrieval candidate pool, so their outcomes are not independent. Every task belongs to exactly one repository and no task is resampled across repositories.

Primary inference is a **cluster bootstrap resampling the 8 repositories with replacement**, 10 000 resamples, recomputing the paired delta within each resample. A paired item-level bootstrap over the 1 600 tasks is reported alongside it as **exploratory** and is never used to decide the gate. Eight clusters is a small number and the cluster interval will be correspondingly wide; that is a real limit of the paper's own corpus and is stated rather than worked around.

## Kill criterion

**Gate A decision rule for the RLCoder arm.** All three outcomes are declared here, before the run.

- **PASS** — the reproduced point delta falls inside `[+2.88, +6.88]` EM points **and** the 95 % cluster-bootstrap interval of that delta excludes zero.
- **FAIL** — the reproduced point delta falls outside `[+2.88, +6.88]`.
- **INCONCLUSIVE AT THIS N** — the point delta falls inside the band but the 95 % cluster-bootstrap interval includes zero. This is **not** a pass: an effect the corpus cannot distinguish from zero has not been reproduced. For Gate A it resolves the same way a fail does, unless another arm passes.

**Feasibility conditions, distinguished from results.** An arm that cannot be executed is recorded as `unrunnable` with its blocking evidence. It is neither a pass nor a fail, and it never counts toward a pass. Gate A is then decided by the remaining arms; a gate with no runnable arm is a design no-go, not a pass.

**cAST arm disposition, declared before the run.** The cAST arm is pre-registered as `unrunnable` on the following already-established evidence, and the run is not attempted:

1. `yilinjz/astchunk` releases the chunker only. There is no evaluation code, no corpus construction, and no implementation of the Appendix A.2 score mapping on which every reported retrieval number depends. The appendix describes that mapping in three sentences and never names the aggregation function.
2. Issues #3, #7, and #8 on that repository all ask for the evaluation script; all are open. The maintainer's answer on #3 points at CodeRAG-Bench and CrossCodeEval but pins neither the metric, the query construction, nor the mapping.
3. The paper's reported RepoEval Recall@5 of 0.707 and 0.750 lies above the mathematical ceiling of `trec_eval` `recall.5` on the released corpus — roughly 0.486 at about 15.3 relevant windows per query — so the reported quantity is not the metric the reference harness computes, and no unambiguous target exists to reproduce.

If the cAST maintainers publish the evaluation and mapping code, the arm becomes runnable and requires its own pre-registration. Reclassifying it on this pre-registration would be a post-hoc change.

**Project-level kill criterion, inherited from the plan and not renegotiated here.** If no arm reproduces a published win in its own setup, all research work stops: R7 through R16 are cancelled and the program reduces to the plan's Section A and Section C outputs plus a root-cause engineering effort.

## Run and analysis boundary

**Treatment matrix.** Two arms, frozen:

| Arm | Retriever | Generator | Reported EM |
| --- | --- | --- | --- |
| `RawRAG` (control) | UniXcoder, BM25 top-50 candidates | `deepseek-ai/deepseek-coder-1.3b-base` | 39.31 |
| `RLCoder` (treatment) | `nov3630/RLRetriever`, BM25 top-100 candidates, stop signal enabled | `deepseek-ai/deepseek-coder-1.3b-base` | 44.19 |

Nothing else differs between the arms. The generator, its weights, its context budgets, the task set, and the metric are identical.

**Immutable input revisions, pinned before the run.**

- Upstream harness: `DeepSoftwareAnalytics/RLCoder` at commit `164d8d88cde324a38f5da70c4f858cc4679ef08e`.
- Dataset: HuggingFace `nov3630/Data4RLCoder`, `repoeval/line_level`, pinned by revision SHA recorded in the evidence artifact at download time.
- Retriever weights: HuggingFace `nov3630/RLRetriever`, pinned by revision SHA recorded in the evidence artifact.
- Generator weights: HuggingFace `deepseek-ai/deepseek-coder-1.3b-base`, pinned by revision SHA recorded in the evidence artifact.
- Generation settings, from the paper's §IV-D and the released defaults: greedy decoding, in-file context 512 tokens, cross-file context 1 536 tokens, total context 2 048 tokens.

**Seeds.** Generation is greedy and the task set is fixed, so there is no sampling seed to vary and one run per arm is the complete measurement. `set_random_seed(123)` from the released code is retained. The 10 000-resample cluster bootstrap uses seed 20260727, fixed here.

**Exclusion rules.** None. All 1 600 tasks enter both arms. A task whose generation fails for any reason is scored as a non-match rather than dropped, and the count of such tasks is reported in the evidence artifact. Dropping tasks after the fact is a post-hoc change.

**Permitted modifications to the upstream harness.** The released eval path hard-codes CUDA placement and computes its batch size as `per_gpu × torch.cuda.device_count()`, which is zero without a GPU. Only the following are permitted, and each is recorded as a diff in `benchmarks/replication/`:

- device placement, from hard-coded `.cuda()` to a selectable device;
- removal of `DataParallel`, which is meaningless on a single device;
- a device-independent, non-zero batch size;
- restricting startup data loading to the split under evaluation.

No change to retrieval logic, prompt construction, generation settings, or the metric is permitted. Any such change voids the arm.

**Analysis procedure, frozen.** Compute per-task EM for both arms over the identical 1 600 tasks. Compute the pooled delta. Compute the 95 % cluster-bootstrap interval over the 8 repositories, 10 000 resamples, seed 20260727, percentile method. Apply the decision rule under "Kill criterion" exactly as written. Record the point delta, the interval, both arms' pooled EM, the per-repository breakdown, and every pin in `benchmarks/evidence/s0-rlcoder-replication.json`. State the verdict in one line in `GATE-A.md`.

**Command.** The exact reproduction command is recorded in `GATE-A.md` and in the evidence artifact, and must rerun to the same figure.

## Post-hoc changes

Each entry is dated, states the affected field and the reason, and marks every number it touches as exploratory.

### 2026-07-27 — the primary metric named the wrong function

**Affected field:** Primary metric.

**What happened.** The pre-registration named "RLCoder's own comparison … in `utils/eval_metric.py`, after the Python one-statement truncation in `utils/eval_utils.py`", measured from `exact_match_idx.jsonl`. That is the harness's *strict* exact match. It is not the quantity RLCoder's Table II reports. The harness prints two numbers per cell, `strict(canonical)`, and the second — `utils/eval_repoeval.py`'s `compute_EM`, which truncates the prediction to the number of ground-truth lines and then compares — is the published one. The control arm makes this unambiguous: its canonical value is 41.63 against a reported 39.31, while its strict value is 18.44.

**Resolution.** Gate A is decided on the paper's own canonical metric, because reproducing a paper means reproducing the paper's quantity. The pre-registered strict metric is reported alongside in `benchmarks/evidence/s0-rlcoder-replication.json`, and it fails the same band. Nothing else moves: the band stays `[+2.88, +6.88]`, the clustering unit stays the repository, the resample count and seed stay as fixed, and no task is excluded.

**Exploratory label.** The canonical figures are the pre-registered comparison applied to the correctly named quantity; the strict figures are the literal pre-registered quantity. Because the choice between them was settled after data existed, treat any *preference* between the two as exploratory. The verdict does not depend on it — both fail.

### 2026-07-27 — silent metric failure in the released harness

**Affected field:** none. Recorded as a finding about the upstream harness, not a change to this pre-registration.

`utils/eval_metric.py` scores through an `mp.Pool`. Its workers never inherit the module-global tree-sitter `parser` under a spawn start method, `postprocess_code_lines` raises, a bare `except` returns the untruncated completion, and the strict exact match collapses. Measured on the control arm: **0.50** as the harness printed it, against **18.44** when the same function is recomputed in a single process with the parser intact. The canonical metric is unaffected, because `compute_EM` re-truncates independently. This is why the strict metric is recomputed rather than read from `exact_match_idx.jsonl`.
