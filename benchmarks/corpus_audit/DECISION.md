# R4 — corpus validity audit

**A literature-sized effect is invisible on this corpus. At 64 tasks over 16 repository clusters, an effect of +4.88 points — the one R3 was asked to reproduce — has a power of 0.108 against a 0.80 target.**

Evidence: `benchmarks/evidence/s2-corpus-validity.json`. Reproduce with:

```bash
uv run python benchmarks/corpus_audit/run_audit.py \
  --output benchmarks/evidence/s2-corpus-validity.json
uv run archex benchmark validate --kind corpus-audit \
  --input benchmarks/evidence/s2-corpus-validity.json
```

Deterministic at seed 20260728. Measurement only: nothing here changes a task, a label, or any retrieval code.

## The finding

| True effect | Power (± MC SE) | Mean 95% interval width |
| --- | --- | --- |
| +2.0 points | 0.077 ± 0.004 | 34.1 |
| **+4.88 points** | **0.108 ± 0.005** | 34.0 |
| +10 | 0.220 ± 0.007 | 33.8 |
| +20 | 0.597 ± 0.008 | 32.7 |
| +25 | 0.789 ± 0.007 | 31.9 |
| +30 | 0.915 ± 0.004 | 30.9 |

The detectable effect is reported as a **bracket, +25 to +30 points**, not a point value. +25 reaches 0.789, which does not clear 0.80 by two of its own standard errors; +30 does. An earlier draft reported "+25" from 400 simulations, where the Monte Carlo standard error was 0.022 — wider than the distance from the estimate to the target. That is reading an estimator at a resolution it does not have, and at 4 000 simulations the honest answer is a range.

Two consequences.

**A literature-sized effect is invisible.** At +4.88 points the corpus has an 11% chance of producing an interval that excludes zero. Nine times in ten a real published-magnitude improvement would be recorded here as "no significant difference". That is not conservatism, it is a coin weighted heavily against detection.

**The interval is wider than the effect space.** A 34-point interval on a metric bounded at 100 cannot distinguish a large win from a large regression. Every archex null measured on this corpus was uninterpretable by construction, before any question about retrieval quality arises.

## How much corpus would be enough

The projection is in **total tasks**, holding the current 16 clusters:

| Tasks | Power at +4.88 |
| --- | --- |
| 64 (today) | 0.112 |
| 256 | 0.235 |
| 512 | 0.399 |
| 1 024 | 0.633 |
| **2 048** | **0.890** |

Roughly **2 048 tasks**, a 32× increase.

An earlier draft framed this as "512 independent repositories" and asserted that repositories, not tasks, are the scarce resource. **That was wrong, and its own simulator refuted it.** Repository count can only limit power through *between-repository variation in the treatment effect*, and with a constant effect the shared cluster term cancels in the paired delta, leaving total task count in control. Estimating that heterogeneity from the only data available — R3's eight measured per-repository deltas, with within-repository binomial noise removed — gives **0.00 points**: their spread (SD 4.52) is *smaller* than the sampling noise at 200 tasks per repository (4.93). So the data show no detectable between-repository heterogeneity, and this audit does not claim repositories are the binding constraint.

`simulate_power` now takes an `effect_sd` parameter so that assumption is explicit and adjustable rather than baked in.

## What the calibration does and does not license

A power projection is only as good as its simulator, so the simulator is checked against a real measured interval before it projects anything. Fed R3's exact structure — 8 balanced clusters of 200, base rate 0.41625, delta +2.8125 — it predicts a mean interval width of **6.0175** against the **5.8125** R3 measured: **3.5% error**. The artifact schema refuses to store a projection whose calibration is missing, failed, or self-contradicting, and recomputes the tolerance arithmetic rather than trusting the recorded verdict.

**This does not uniquely identify the model.** A single-point match on one structure (200 tasks per cluster) does not pin the mechanism, because at that cluster size binomial noise and genuine effect heterogeneity happen to be the same magnitude. Alternative simulators — paired arms, per-cluster random effects — pass the same 20% gate and put the detectable effect between +15 and +30 and the interval width between 16 and 34. Those alternatives fit R3's *raw* per-repository spread as if all of it were heterogeneity, which the variance decomposition above rejects; but with only 8 clusters that decomposition is itself uncertain, so the range is real.

So: treat **+25–30**, **34 points**, and **2 048 tasks** as this model's estimates, not as model-independent facts. What *is* robust is the direction and the order of magnitude — **power at +4.88 lies between 0.11 and 0.21 under every model that passes the calibration gate**, against a 0.80 target. The disposition rests on that, and only on that.

## The verdict does not hinge on `cluster_sd`

`cluster_sd` is the one nuisance parameter the calibration cannot constrain, so its influence is measured rather than assumed. Power at +25 across a twelvefold range:

| `cluster_sd` | 0.02 | 0.05 | 0.08 | 0.15 | 0.25 |
| --- | --- | --- | --- | --- | --- |
| power at +25 | 0.788 | 0.785 | 0.789 | 0.810 | 0.800 |

Power at +25 stays in 0.785–0.810 — below or at the target throughout, so the +25–30 bracket holds across the whole range. The reason is structural, not lucky: the cluster random effect enters both arms and largely cancels in their difference, so this sweep is reassuring by construction rather than by evidence.

## Leakage: 29.7%

| Tier | Signals | Tasks | Rate |
| --- | --- | --- | --- |
| `symbol` — identifier-shaped gold symbol quoted in the query | 25 | 19 | **29.7%** |
| `symbol_word` — a gold symbol that is also the ordinary word for the question | 17 | — | — |
| `path_stem` — a gold file's stem quoted in the query | 64 | — | — |
| any tier | 106 | 49 | 76.6% |

This figure moved twice, and both moves are worth recording because they bracket how easily a detector like this misleads.

A first version counted every gold path fragment and reported **89%**, driven entirely by self-repo questions containing the word "archex". Directory components and the repository's own name are now excluded.

The correction to that over-counted 14.1% — and **that was also wrong, in the opposite direction.** The matcher normalised the query surface (turning `_` and `.` into spaces) but matched the raw symbol, so `default_adapter_registry` was compiled to a literal and tested against "default adapter registry". No snake_case or dotted gold symbol could ever match: the exact identifier shape the strong tier exists to catch. The most blatant leak in the corpus was scored clean — `routing_pl_path_symbol`, whose entire question is *"where is benchmark_repo_source defined in src/archex/benchmark/strategies.py"*.

Identifiers are now matched against a surface that preserves `_` and `.`, while path stems are matched against the normalised one, so `_merge` still does not match the ordinary word "merge" and `block_on` does not match "block on".

**This confirms the plan rather than correcting it.** The plan recorded "8 of 21 confirmed in the `loc_*` family". The strong tier now finds **9 of 21**. An earlier draft of this document claimed the real figure was 2 of 21 and that the plan had over-counted — that claim was an artifact of the broken matcher, and it is withdrawn.

29.7% is a real defect and still the *least* important finding here: a corpus with zero leakage could not detect a literature-sized effect at this cluster count.

## The held-out set is not held out

All five IDs in `benchmarks/held_out.txt` are also top-level tasks in `benchmarks/tasks/`. **100% overlap.**

Not an accident that slipped through: `tests/benchmark/test_generalization.py:29` asserts `set(held_out) <= set(tasks_by_id)`, so the overlap is a *requirement* of the suite rather than something it would catch. And no code under `src/` or `.github/` reads `held_out.txt` at all, so the declaration has no runtime effect — nothing excludes those IDs from a run. The only related enforcement is a CI grep rejecting task-ID-keyed retrieval code, which prevents hardcoding, a different concern.

The held-out set is a labelling convention. Any generalization claim resting on it is unsupported.

## Effective sample size

`N / (1 + (m_A − 1) · ICC)`, where `m_A` is the **size-weighted** mean cluster size, `Σm² / Σm`.

Not the arithmetic mean. One cluster holds 24 tasks and fifteen hold 2 to 4, giving an arithmetic mean of 4.0 against a weighted mean of **10.81**. A large cluster contributes its correlated observations in proportion to its own size, so the arithmetic mean overstates usable N about twofold. An earlier draft did exactly that and reported 33.7 at ICC 0.3.

| ICC | Effective N |
| --- | --- |
| 0.0 | 64.0 |
| 0.1 | 32.3 |
| 0.3 | **16.2** |
| 0.5 | 10.8 |
| 0.8 | 7.2 |

Tasks from one repository share a codebase, an API surface, and a style, so ICC is not plausibly near zero. At a moderate 0.3 the corpus is worth about **16 independent observations** — one per repository, which is the honest reading.

## What this decides

R4 was the pre-declared input to the programme's disposition question, and it answers it.

**Rebuilding the instrument is not a research project this corpus can support.** Reaching literature-detection power needs a ~32× larger corpus with human-annotated gold contexts. That is a dataset-construction programme measured in months, and its output would be a benchmark, not a retrieval result. That was R8's job, and R8 is cancelled with the rest of Section D.

**The negative result is publishable; the retrieval claims are not.** Two findings stand on their own and need neither the cancelled instrument nor a new corpus:

1. A released, peer-reviewed harness silently mis-scores: an `mp.Pool` worker loses the module-global tree-sitter parser under a spawn start method, a bare `except` swallows the exception, and exact match reads 0.50 where the correct value is 18.44.
2. A published paper's reported metric exceeds the mathematical ceiling of the metric its own reference harness computes.

Together with R4's own result — that a corpus of this shape cannot detect the effects the literature reports — that is a measurement paper about the state of code-RAG evaluation.

**A third claim was drafted and is withdrawn.** An earlier version asserted that R3's reference setup was "underpowered for the effect it reports", citing 46.8% power. That is *observed* power at the effect actually observed (+2.81), and presenting it as a property of the design is the post-hoc-power fallacy. At the effect the paper reports (+4.88) the same model puts that setup at **0.825** — adequately powered. R3's narrow Gate A miss is a small-effect outcome, not an underpowered design, and the artifact records the caveat alongside both figures.

**Recommendation.** Wind the research programme down, finish Section C (R5, R6), and publish the negative result plus the two findings above. Do not start a 32×-corpus effort on the strength of a hypothesis the instrument was never able to test.

**What the root-cause effort should now be.** The Gate A fail clause mandates one. R4 narrows it: the delta miss is consistent with a small true effect measured by two underpowered-for-that-effect comparisons, not with an implementation defect. The environment-drift check proposed in the post-Gate-A notes is still worth its ~11 hours, because our control over-performed its published figure by 2.32 points and that remains unexplained. But it is a loose end, not a path back to R7, and it cannot reopen Gate A.

## Limits of this audit

- **The calibration constrains the variance scale, not the mechanism.** See above: alternative paired/heterogeneous models pass the same gate with detectable effects from +15 to +30. Point estimates here are model-dependent; the order of magnitude is not.
- The generative model is a cluster-level Normal random effect on the success rate with independent arms. It reproduces R3's measured interval to 3.5%, which is the strongest validation available, but it is not a model of archex's pipeline.
- Power assumes a binary per-task outcome. Continuous metrics such as F1 differ in detail, though the cluster-count limit dominates either way.
- ICC is a reported sensitivity range, not an estimate. Estimating it needs repeated measurements per task the corpus does not have.
- The leakage detector is lexical: it cannot see a question that paraphrases its gold symbol, so 29.7% is a floor.
- The `base_rate 0.85` arm reports no detectable effect at any searched size. That is a real limit of a high-base-rate metric on 16 small clusters, and effects large enough to saturate the treatment arm are now refused outright rather than silently clamped — an earlier draft read an MDE of +20 off a curve that had plateaued at 0.9275 because clamping made every large effect produce identical data.
