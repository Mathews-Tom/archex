# External Quality Frontier — Reproducible Comparison

Status: candidate evidence, self-repo scope. No default-promotion decision is made by this document — see [Verdict](#verdict).

## What this measures

DEVELOPMENT_PLAN.md §4 M3 requires reproducible, task-family-specific evidence comparing archex's default retrieval path against candidate lanes (cAST chunking, the `fast`/`balanced` retrieval profiles, and an only-if-defined symbolic-rerank lane), reported by language, repository size, query intent, and task family rather than collapsed into one cross-family winner.

The M3 stack (`#533`–`#536`) built:

- a pinned-external-corpus policy plus a sealed chronological holdout corpus (`benchmarks/sealed_tasks/`), isolated from CI and from production vocabulary;
- the four-dimension scorecard and raw provenance artifact (`archex benchmark scorecard`);
- the candidate lane matrix (`archex_query_profile_fast`/`_balanced`, cAST via `--chunker`) and a deterministic fixed-agent trajectory signal (`post_bundle_search_turns`);
- the multidimensional promotion gate (`archex benchmark gate --promotion-strategy`), extended with zero-recall, language-family, and fixed-agent non-regression checks;
- `scripts/m3_frontier_pipeline.sh`, the local-operator orchestration script that runs the lane matrix and every gate in one invocation.

## How to reproduce

```bash
# Self-repo scope (no network, ~8 minutes on this machine):
ARCHEX_M3_SELF_ONLY=1 bash scripts/m3_frontier_pipeline.sh

# Full public corpus (64 tasks, network clones for pinned external repos,
# significantly longer -- local-operator only, never run in CI):
bash scripts/m3_frontier_pipeline.sh

# Sealed chronological holdout (2 tasks, network clones, local-operator only):
ARCHEX_M3_TASKS_DIR=benchmarks/sealed_tasks ARCHEX_M3_ALLOW_SEALED_CORPUS=1 \
  bash scripts/m3_frontier_pipeline.sh
```

Every stage's exact command, thresholds, and pass/fail output are logged to `logs/m3_frontier_pipeline.log` (or `$ARCHEX_M3_LOG_FILE`). Evidence directories under `$ARCHEX_M3_OUTPUT_ROOT` (default `.archex/m3-frontier`) are immutable, manifest-backed (`archex.benchmark.evidence`), and pinned to the exact source revision and task-manifest digest that produced them.

## Results: self-repo scope demonstration (24 tasks, 2026-07-23)

This run is **not** the full external quality frontier. It exercises every lane and every gate end to end against this repository's own 24 self-scoped tasks (no network access), as the delivery's functional proof. The full pinned-external (48 tasks) and sealed-holdout (2 tasks) runs require network clones of nine external repositories and take substantially longer; they are documented above as a separate local-operator activity, not executed as part of this PR to keep the delivery reviewable and reproducible without waiting on external-network variance.

Command: `ARCHEX_M3_SELF_ONLY=1 bash scripts/m3_frontier_pipeline.sh`. Source revision `2692a0b`, task-manifest digest `c9b6eb5390…`.

### Scorecard (single slice on this scope: language=python, size=large, intent=self, family=comprehension)

| Lane | Recall | F1 | MRR | Zero-Recall | Dup. Rate | Tok. Eff. | Cold p50 (ms) | Required-File Completeness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default (`archex_query`) | 0.892 | 0.618 | 0.979 | 0 | 0.677 | 0.771 | 4282 | 0.625 |
| fast (`archex_query_profile_fast`) | 0.892 | 0.618 | 0.979 | 0 | 0.677 | 0.771 | 4163 | 0.625 |
| balanced (`archex_query_profile_balanced`) | 0.892 | 0.618 | 0.979 | 0 | 0.677 | 0.772 | 4186 | 0.625 |
| cAST (`archex_query`, `--chunker cast`) | 0.823 | 0.574 | 0.979 | 0 | 0.635 | 0.729 | 5281 | 0.458 |

Every retrieval-quality metric (recall/F1/MRR) is bit-identical between default and both profiles on this scope: `fast`/`balanced` only change vector/rerank/module-prefilter usage, and this repository's self-tasks never exercise those paths under the default configuration either, so the profiles produce the same BM25 ranking with a marginally lower cold p50 (safe to run, no measured required-file-completeness or recall cost here). cAST recall/F1/required-file-completeness are all measurably lower on this scope, consistent with the source analysis's caveat that "a controlled 2026 study found function-only chunks off the cost-quality frontier" — a real, non-cherry-picked signal against promoting cAST as a default, at least at this scope.

The last column is required-file completeness, not downstream task success: it is the fraction of tasks whose required files were all present in the returned bundle, a function of required-file recall with no model in the loop. Its values are unchanged; only the label is corrected.

### Promotion gate: fast/balanced vs. default (same-run control)

```
uv run archex benchmark gate --input .archex/m3-frontier/frontier --tasks-dir benchmarks/tasks \
  --promotion-strategy archex_query_profile_fast --control-strategy archex_query \
  --min-token-efficiency-with-completion 0.08 --max-p95-warm-latency-ms 5000
```

**NO-GO** for both `fast` and `balanced` (27 violations each): 3 absolute-threshold misses shared by both profiles (`routing_pl_scoring` recall/F1, `routing_mixed_chunker` MRR — pre-existing on this scope, not introduced by the profile) plus 24 `warm_latency_unmeasured` violations. The warm-latency violation is a **methodology gap in this specific run**, not a quality finding: `--max-p95-warm-latency-ms` requires a `cache_state == "warm"` sample per task, which needs a `--warm-cache` pre-pass. In this environment, `--warm-cache` combined with the `balanced` profile's `module_prefilter` produced an incomplete per-task strategy set for one task on repeated attempts (see `scripts/m3_frontier_pipeline.sh`'s comment above the frontier-run command) — a pre-existing `archex benchmark run --warm-cache` interaction, not something this stack introduced, and out of scope to root-cause here. Until it is fixed, a genuine warm-latency-gated promotion attempt needs either a fixed `--warm-cache` pass or a relaxed `--max-p95-warm-latency-ms` invocation scoped to cold-only evidence.

### cAST absolute-threshold gate

```
uv run archex benchmark gate --input .archex/m3-frontier/cast --tasks-dir benchmarks/tasks \
  --min-recall 0.60 --min-f1 0.30 --min-mrr 0.55
```

**FAILED** (6 violations: `archex_graph_expansion`/`archex_pattern_detection`/`routing_pl_scoring`/`routing_mixed_chunker` recall/F1/MRR below floor). A true default-vs-cAST *regression* gate via `archex benchmark gate --baseline` is not currently reachable: `validate_baseline_coverage` requires the two evidence directories' `retrieval_options` to match exactly, and `chunker` is part of that object, so two directories that differ only by `--chunker` always fail baseline-coverage validation before the (already-implemented) `format_chunker_frontier_table` render is ever reached. This is a pre-existing `evidence.py`/`gate.py` constraint predating this stack; comparing the two scorecards above (or the `Scorecard: default` / `Scorecard: cast` markdown tables) is the reproducible substitute used here.

## Verdict

**NO-GO for automatic default promotion of any candidate**, on the evidence produced so far:

- `fast`/`balanced`: no measured recall/F1/MRR/required-file-completeness regression, but no genuine warm-latency evidence exists yet (methodology gap above) and the M3 promotion rule requires staying inside profile p95 budgets with *evidence*, not by default.
- cAST: measurable recall/F1/required-file-completeness regression on this scope; already excluded by the absolute-threshold gate.
- symbolic-rerank: not run (only-if-defined; not exercised in this delivery).
- Full pinned-external and sealed-holdout corpus scope: not yet executed (see reproduction commands above).

This is consistent with the M3 constraints: no automatic product-default promotion, no cross-family winner claim, and no result cherry-picking — the retrieval harness's default (`archex_query`, default chunker) remains the shipped product default. The mechanism to reach a GO decision — lane matrix, scorecards, and the extended promotion gate — is complete, tested, and reproducible; reaching a promotion verdict on the full external/sealed corpus, and fixing the `--warm-cache`/`balanced`-profile interaction, are follow-on local-operator activities this stack unblocks but does not itself complete.
