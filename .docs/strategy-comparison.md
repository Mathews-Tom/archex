# Strategy Comparison

## Context

Compare the current product default, `archex_query`, against `archex_query_fusion_rerank` on the corrected benchmark oracle.

## Commands

- Benchmark run: `uv run archex benchmark run --query-fusion --rerank --tasks-dir benchmarks/tasks --output .archex/e2e-results`
- Current-default readiness: `uv run archex benchmark readiness --input .archex/e2e-results --tasks-dir benchmarks/tasks --strategy archex_query --format markdown`
- Fusion-rerank readiness: `uv run archex benchmark readiness --input .archex/e2e-results --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown`

The full benchmark command was run directly twice during this PR. The first run
completed 18 of 35 result files before repeated cross-encoder loads made the run
impractical. After adding process-level reranker model reuse and run-level
external-repo reuse, the command again completed 18 of 35 result files and then
became CPU-bound on the first `django/django` task. The measurements below use
the 18 fresh artifacts produced by the direct run because they are already
decisive against the default-switch rule.

## Measurements

| Strategy | Mean recall | Mean precision | Mean F1 | Mean MRR | Median latency | P95 latency | Zero-recall tasks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `archex_query` | 0.602 | 0.442 | 0.494 | 0.903 | 1876 ms | 2222 ms | 0 |
| `archex_query_fusion_rerank` | 0.282 | 0.392 | 0.309 | 0.593 | 3204 ms | 5525 ms | 6 |

## Decision Rule

Switch the product default to `archex_query_fusion_rerank` only if:

- `archex_query_fusion_rerank` mean F1 is at least `archex_query` mean F1 plus `0.05`.
- `archex_query_fusion_rerank` P95 latency is at most `3000 ms` per task.

## Recommendation

Keep `archex_query` as the product default.

## Chosen Path

Decision B: do not switch the product default in this PR.

`archex_query_fusion_rerank` fails both switch criteria on the measured artifact
set. Its mean F1 is `0.185` below `archex_query`, not `0.05` above it, and its
P95 latency is `5525 ms`, above the `3000 ms` threshold.

## Notes

- Artifact set: 18 tasks from `.archex/e2e-results`.
- Categories covered: 16 self tasks, 1 external-framework task, 1 external-large task.
- `archex_query_fusion_rerank` produced 6 zero-recall tasks in the measured set.
- The full 35-task fusion-rerank run remains too expensive for this PR's local
  validation loop under the current benchmark harness.
