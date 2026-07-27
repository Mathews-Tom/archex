# Cross-tool efficiency artifact

`cross-tool-comparison.json` is the checked-in reference run of `archex benchmark cross-tool`. It compares, at a fixed required-file recall, the tokens archex spends to localize a task's required files against a naive grep/read agent.

Regenerate it from a clean run; never hand-edit metric values:

```bash
uv run archex benchmark cross-tool --tasks-dir benchmarks/tasks \
  --output benchmarks/cross-tool-efficiency
```

## What the baseline models

`tokens_at_recall` (`src/archex/benchmark/cross_tool.py`) walks a path's retrieval units in that path's own fixed ranking and charges every unit consumed until required-file recall reaches the target. For the naive path the ranking is lexical grep relevance and the units are whole grep-hit files (`full_file`) or the merged `+/-K` windows around a file's hits (`grep_window`).

The modelled agent is **blind**: it never inspects a grep hit, judges it irrelevant, and skips reading it, and it never stops early. It pays in full for every false positive that lexical relevance ranked ahead of a required file.

That is a legitimate lower bound on a naive strategy. It is not how a competent agent behaves — a real agent greps, reads the returned line numbers, and opens one or two ranges. Every reduction derived from this artifact is therefore an upper bound on the advantage over a grep/read workflow.

## Measured units-read distribution

`units_consumed` on each `PathTokensAtRecall` records how many units a path read before reaching the target recall. Over the 52 comparable tasks in this artifact, the naive agent reads:

| Scope | Comparable tasks | Median units read | Mean | Max |
| --- | ---: | ---: | ---: | ---: |
| All comparable tasks | 52 | 5.5 | 18.0 | 164 |
| self (withdrawn) | 16 | 32.5 | 48.4 | 164 |
| external-comprehension | 16 | 6.0 | 6.5 | 18 |
| external-localization | 20 | 1.0 | 2.8 | 16 |
| external corpora combined | 36 | 3.0 | 4.4 | 18 |

archex reads a median of 3.0 units (mean 3.1, max 8) over the same tasks. The long right tail is almost entirely self-repo, which supplies the 164-unit maximum. `units_consumed` is identical across the `full_file` and `grep_window` models; they differ only in what a unit costs.

## Published scope

The `self` corpus is withdrawn from every quoted figure. Its 16 comparable tasks remain in the artifact and are not deleted, but they are not published: archex's own generic keywords (`index`, `query`, `config`) match across its entire source, so the reduction there measures keyword density rather than archex.

Published figures come from `external-comprehension` and `external-localization` only, and each names the naive grep/read agent as its baseline. See [`docs/LOCAL_METRICS.md`](../../docs/LOCAL_METRICS.md) for the per-corpus table and [`docs/LOCAL_BENCHMARK_EVIDENCE.md`](../../docs/LOCAL_BENCHMARK_EVIDENCE.md) for the full baseline semantics.
