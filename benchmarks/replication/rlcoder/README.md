# RLCoder replication (S0 primary arm)

Reproduces one cell of RLCoder (arXiv:2407.19487, ICSE 2025) **in its own released
reference setup**. This is not archex code exercising archex retrieval; it is the
authors' harness, at a pinned commit, on their dataset, with their weights and
their metric.

Pre-registration: [`.docs/spikes/S0-replication-gate.md`](../../../.docs/spikes/S0-replication-gate.md).
It fixes the target cell, the equivalence band, the clustering unit, and the
decision rule, and it merged before this harness ran.

## Target cell

RLCoder Table II, RepoEval line-level, backbone DeepSeekCoder-1B:

| Arm | Retriever | Reported EM |
| --- | --- | --- |
| `rawrag` (control) | UniXcoder, BM25 top-50 candidates | 39.31 |
| `rlcoder` (treatment) | RLRetriever, BM25 top-100 candidates, stop signal | 44.19 |

Reported delta **+4.88 EM points**. Pre-registered equivalence band
**[+2.88, +6.88]**, because the paper publishes no interval, seed count, or
variance for any cell.

The abstract's "12.2%" is a *relative* figure from the 7B CrossCodeEval-Python
cell (26.98 to 30.28, +3.30 absolute), not an absolute EM gain. That cell is
roughly an order of magnitude more compute; the substitution is recorded in the
pre-registration rather than discovered afterwards.

## Layout

| File | Purpose |
| --- | --- |
| `prepare.py` | Clone the harness at its pin, apply `portability.patch`, build the platform parser, download the pinned dataset and weights, write `pins.json`. |
| `portability.patch` | The four permitted edits, and only those. |
| `run.sh` | Run both arms over the identical 1600-task split. |
| `analyze.py` | Cluster-bootstrap the paired delta and emit the evidence artifact. |

Upstream code is never vendored here. A rerun fetches
`DeepSoftwareAnalytics/RLCoder` at `164d8d88cde324a38f5da70c4f858cc4679ef08e`.

## What the patch changes, and what it must not

The released eval path cannot start without a CUDA device: it places tensors
with hard-coded `.cuda()`, wraps models in `DataParallel`, and computes its
batch size as `per_gpu * torch.cuda.device_count()`, which is zero on a machine
with no GPU. It also loads every test split and both training Parquets before
reaching the eval branch, tens of gigabytes for a run that touches one split.

The patch is limited to the four modifications the pre-registration permits:

1. device placement moved behind a single `device.py` selector (CUDA, then MPS, then CPU, with `RLCODER_DEVICE` overriding);
2. `DataParallel` dropped, since on one device it only changes attribute paths;
3. a device-independent, non-zero batch size;
4. `--eval_datasets`, so an eval run loads only the split under evaluation.

It changes no retrieval logic, no prompt construction, no generation setting,
and no metric. Per-task exact match is recomputed from the harness's own
`prediction.jsonl` and `prediction_truncated.jsonl` using the harness's own
scoring functions, imported unmodified. It is deliberately **not** read from
`exact_match_idx.jsonl`: the released scorer runs through an `mp.Pool` whose
workers lose the module-global tree-sitter parser under a spawn start method,
and the resulting exception is swallowed by a bare `except`, which drives the
strict exact match to near zero. See the 2026-07-27 finding in the
pre-registration.

One platform caveat: the repository ships prebuilt **x86-64 ELF** tree-sitter
parsers, which cannot load elsewhere. `prepare.py` rebuilds
`python-lang-parser.so` from `tree-sitter-python` at the same `v0.20.4` tag. The
grammar is identical; only the object file differs.

## Running it

```bash
python benchmarks/replication/rlcoder/prepare.py --work-dir /tmp/s0-rlcoder
RLCODER_DEVICE=mps benchmarks/replication/rlcoder/run.sh /tmp/s0-rlcoder /tmp/s0-rlcoder/out
# analyze.py imports the harness's own scoring functions, so run it with the
# harness's interpreter, not the repo's.
/tmp/s0-rlcoder/.venv/bin/python benchmarks/replication/rlcoder/analyze.py \
  --work-dir /tmp/s0-rlcoder \
  --run-dir /tmp/s0-rlcoder/out \
  --preregistration-commit 557c5683e5a2622e0a96370a379365c8498d1dc4 \
  --output benchmarks/evidence/s0-rlcoder-replication.json
```

`run.sh` needs an environment with the upstream `requirements.txt` dependencies.
The pinned `torch==2.0.1` and friends have no arm64 wheels for recent Pythons,
so the recorded run used newer releases. `analyze.py` captures the versions it
actually ran against into the evidence artifact's `pins.environment`. Decoding is
greedy either way, so the comparison between the two arms is unaffected by that
drift even where absolute values could shift.

For the measured wall clock of the recorded run, see `GATE-A.md`.

## Reading the result

`analyze.py` derives the verdict; it cannot be told what to conclude. Inside the
band with a cluster-bootstrap interval clear of zero is a `pass`; inside the band
with an interval spanning zero is `inconclusive`, which is **not** a pass;
outside the band is a `fail`. `archex benchmark validate --kind replication`
re-derives the same verdict from the artifact and rejects a mismatch.
