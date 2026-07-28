# Gate A — external replication

**GATE A FAIL — no published win reproduced in its own setup.**

Recorded 2026-07-27 against pre-registration `.docs/spikes/S0-replication-gate.md`, merged at `557c5683e5a2622e0a96370a379365c8498d1dc4` before any data-generating run. Not renegotiable.

## Arms

| Arm | Paper | Class | Verdict |
| --- | --- | --- | --- |
| RepoEval line-level, DeepSeekCoder-1B, RawRAG vs RLCoder | RLCoder, arXiv:2407.19487 (ICSE 2025) | `replication` | **fail** |
| RepoEval Recall@5, GIST-base, fixed-size vs cAST | cAST, arXiv:2506.15655 (Findings of EMNLP 2025) | `replication` | **unrunnable** |

Evidence: `benchmarks/evidence/s0-rlcoder-replication.json`, `benchmarks/evidence/s0-cast-replication.json`. Both validate with `uv run archex benchmark validate --kind replication --input <artifact>`.

## RLCoder arm

Target cell: Table II, RepoEval (Line), DeepSeekCoder-1B — `RawRAG` 39.31 against `RLCoder` 44.19, reported delta **+4.88** EM points. Pre-registered equivalence band **[+2.88, +6.88]**, fixed before the run and not widened.

| Quantity | Reported | Reproduced |
| --- | --- | --- |
| Control (RawRAG) EM | 39.31 | **41.6250** |
| Treatment (RLCoder) EM | 44.19 | **44.4375** |
| Delta | +4.88 | **+2.8125** |
| 95% cluster-bootstrap interval on the delta | not published | **[+0.1250, +5.9375]** |

The reproduced delta falls **below** the pre-registered band by **0.0675** points.

Both absolute values reproduce closely, and the control lands *above* its published figure by 2.32 points while the treatment lands above its own by only 0.25, which is what compresses the delta. The effect is positive and its interval excludes zero, but the pre-registered question is whether the reproduced delta lands inside the band, and it does not. A 0.07-point miss is still a miss; the band was fixed before any data existed and softening it now would make the gate unfalsifiable.

The metric above is the paper's own: `utils/eval_repoeval.py`'s canonical RepoEval exact match, which is the parenthesised value the harness prints and the quantity Table II reports. The pre-registration named the *strict* exact match in `utils/eval_metric.py` instead; that was a naming error, corrected in the pre-registration's dated post-hoc section rather than edited into its body. The strict metric, recomputed correctly, gives control 18.4375, treatment 19.4375, delta **+1.0000**, interval **[−0.1250, +2.1250]** — also below the band. The verdict does not turn on which metric is used.

Per-repository deltas, canonical metric: `alibaba_FederatedScope +2.5`, `awslabs_fortuna +2.0`, `google_vizier −3.5`, `huggingface_diffusers +12.0`, `huggingface_evaluate +3.5`, `nerfstudio-project_nerfstudio −1.0`, `opendilab_ACE +4.5`, `pytorch_rl +2.5`. Two of eight repositories move the wrong way and one carries most of the gain.

## cAST arm

Not run, and recorded as `unrunnable` rather than as a fail: an arm that was never executed has not failed to reproduce anything, and it never counts toward a pass either. The released artifact `yilinjz/astchunk@82029ada` is the chunker only — its tree contains no file matching `eval`, `metric`, `score`, `repoeval`, or `coderag`, so there is no evaluation harness, no corpus construction, and no implementation of the Appendix A.2 score mapping that every reported retrieval number depends on. Issues #3, #7, and #8 each ask for that code and all three are open.

## Reproduction

Environment: Apple M1 Pro, 32 GB, MPS; Python 3.11.11, torch 2.13.0, transformers 4.44.2, tree_sitter 0.20.4, recorded in the artifact's `pins.environment`. Measured wall clock: about 11 hours per arm, 22 hours for the pair.

```bash
python benchmarks/replication/rlcoder/prepare.py --work-dir /tmp/s0-rlcoder
RLCODER_DEVICE=mps benchmarks/replication/rlcoder/run.sh /tmp/s0-rlcoder /tmp/s0-rlcoder/out
# analyze.py imports the harness's own scoring functions, so run it in the harness environment
/tmp/s0-rlcoder/.venv/bin/python benchmarks/replication/rlcoder/analyze.py \
  --work-dir /tmp/s0-rlcoder \
  --run-dir /tmp/s0-rlcoder/out \
  --preregistration-commit 557c5683e5a2622e0a96370a379365c8498d1dc4 \
  --output benchmarks/evidence/s0-rlcoder-replication.json
```

Pins: harness `DeepSoftwareAnalytics/RLCoder@164d8d88cde324a38f5da70c4f858cc4679ef08e`; dataset `nov3630/Data4RLCoder@cb9639f2`, split `repoeval/line_level`; `microsoft/unixcoder-base@5604afdc`; `nov3630/RLRetriever@ec587f5d`; `deepseek-ai/deepseek-coder-1.3b-base@c919139c`. Decoding is greedy and the harness, dataset, and weights are pinned by revision, so a rerun on the same hardware and the recorded library versions reproduces the same figures. The upstream `requirements.txt` pins have no arm64 wheels for recent Pythons, so the recorded environment drifted from it; that drift is recorded rather than hidden, and it applies identically to both arms.

## Consequence

Pre-declared in `.docs/DEVELOPMENT_PLAN.md` §6 R3 and not renegotiable:

- **R7 is not authorized.** All research work stops.
- **R7 through R16 are cancelled.**
- The program reduces to Section A and Section C outputs plus a root-cause engineering effort.

What this result does and does not license. It does **not** show that RLCoder's published gain is wrong: both of its absolute figures reproduced within about 2.3 points, and the treatment beat the control. What it shows is that this harness could not recover a published *delta* to within a band that admits ordinary reproduction drift, on the one external cell it attempted, and that the only other candidate arm could not be run at all because its reference setup is unreleased. Under the pre-declared rule, that is the definition of a Gate A fail, and every archex null recorded to date remains attributable to implementation rather than to the literature.
