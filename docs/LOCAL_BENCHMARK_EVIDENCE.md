# Local benchmark evidence

Run the complete corpus only as a local operator. GitHub Actions validates task definitions and bounded evidence contracts; it never executes the 64-task retrieval corpus.

## Procedure

1. Start from a clean committed source tree. The evidence manifest records the resolved source SHA and refuses a dirty tracked tree.
2. Validate the declared corpus:

```text
uv run archex benchmark validate --kind tasks --tasks-dir benchmarks/tasks
```

3. Choose a new, empty evidence directory and run the default control:

```text
uv run archex benchmark run --tasks-dir benchmarks/tasks --output <evidence-dir> --strategy archex_query
```

The command writes exactly one raw report per completed task and `manifest.json`. The manifest records source SHA, task-manifest digest, Archex version, selected strategies, retrieval configuration, report SHA-256 values, generation timestamp, and hardware advisory.

4. Validate retained evidence before inspecting or comparing it:

```text
uv run archex benchmark validate --kind evidence --tasks-dir benchmarks/tasks --input <evidence-dir>
```

5. Record the absolute-gate result without replacing a canonical baseline:

```text
uv run archex benchmark gate --input <evidence-dir> --min-recall 0.60 --min-f1 0.30 --min-mrr 0.55
```

A non-zero gate result is evidence. Preserve the manifest and all 64 reports with the failing result. Do not promote or overwrite a baseline until the declared promotion milestone verifies a passing full-corpus result.

Evidence is invalid when its source revision, task digest, report hashes, task coverage, strategy coverage, or retrieval configuration differs from the manifest.

## Promotion candidate procedure

Promotion evidence uses one named candidate and the retained `archex_query` control from the same manifest. The operator must request a warm-cache run; the runner discards one cache-populating execution for each indexed strategy before it records the result. A promotion gate rejects a candidate row unless it reports `cached: true`, `cache_state: "warm"`, and a nonzero `warm_latency_ms`.

```text
uv run archex benchmark run --tasks-dir benchmarks/tasks --output <evidence-dir> --strategy <candidate-strategy> --warm-cache
uv run archex benchmark validate --kind evidence --tasks-dir benchmarks/tasks --input <evidence-dir>
uv run archex benchmark gate --input <evidence-dir> --promotion-strategy <candidate-strategy> --control-strategy archex_query --min-recall 0.60 --min-precision 0.20 --min-f1 0.30 --min-mrr 0.55 --min-token-efficiency-with-completion 0.08 --max-p95-warm-latency-ms 3000
```

The promotion gate hard-fails every absolute row for the named candidate, including the product token-efficiency floor. It separately rejects required-file, region, or line-recall regressions against the control. The control remains an informational comparator for absolute quality, so its existing failures neither block a candidate nor become a promoted baseline.

## Cross-tool efficiency baseline semantics

`archex benchmark cross-tool` and its checked-in reference artifact
[`benchmarks/cross-tool-efficiency/cross-tool-comparison.json`](../benchmarks/cross-tool-efficiency/cross-tool-comparison.json)
report a token reduction against a naive grep/read agent. Read that number with the
baseline's semantics in hand, because the baseline is deliberately naive and the reduction
is a property of the baseline as much as of archex.

`tokens_at_recall` (`src/archex/benchmark/cross_tool.py`) walks a path's retrieval units in
that path's own fixed ranking and charges every unit it consumes until required-file recall
reaches the target. For the naive path the ranking is lexical grep relevance and the units
are whole grep-hit files (`full_file`) or the merged `+/-K` windows around the hits in a
file (`grep_window`). There is no triage step: the modelled agent never looks at a grep hit,
judges it irrelevant, and skips reading it, and it never stops early. It pays in full for
every false positive that lexical relevance ranked ahead of a required file.

That makes the baseline a **blind-read lower bound on a naive strategy**, not a model of
competent agent behavior. A real agent greps, reads the returned line numbers, and opens one
or two ranges. The published reduction is therefore an upper bound on the advantage over a
grep/read workflow, and it says nothing about the advantage over an agent that triages.

Two further conditions apply to every figure derived from the artifact:

- The reduction is conditioned on archex reaching the target recall. Tasks where archex
  misses are excluded rather than scored at unequal recall, so the number measures how much
  cheaper archex localizes *when it succeeds*.
- The self-repo corpus is withdrawn from every quoted figure (see below); its comparisons
  remain in the artifact and are not deleted.