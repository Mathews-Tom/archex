# Strategy Comparison

## Context

Compare the current product default, `archex_query`, against `archex_query_fusion_rerank` on the corrected benchmark oracle.

## Commands

- Benchmark run: `uv run archex benchmark run --query-fusion --rerank --tasks-dir benchmarks/tasks --output .archex/e2e-results`
- Readiness report: `uv run archex benchmark readiness --input .archex/e2e-results --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown`

## Measurements

| Strategy | Mean recall | Mean precision | Mean F1 | Mean MRR | Median latency | P95 latency | Zero-recall tasks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `archex_query` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `archex_query_fusion_rerank` | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Decision Rule

Switch the product default to `archex_query_fusion_rerank` only if:

- `archex_query_fusion_rerank` mean F1 is at least `archex_query` mean F1 plus `0.05`.
- `archex_query_fusion_rerank` P95 latency is at most `3000 ms` per task.

## Recommendation

TBD.

## Chosen Path

TBD.

## Notes

TBD.
