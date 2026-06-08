# ADR-001: Gate Retrieval Default Changes On Recall, Tokens, And P95

**Status:** Accepted
**Date:** 2026-06-09
**Author:** archex maintainers

## Context

The retrieval-default evaluation settles the deferred default embedder, reranker, and product strategy decisions. Earlier strategy evidence was confounded by mixed embedders and missing token/p95 gates, so a recall-only switch can optimize the benchmark while worsening the product contract: fewer tokens per retrieval/explanation query with interactive p95 latency.

## Decision

Keep `archex_query` as the product default. The 2026-06-09 operator run did not satisfy the default switch rule: `archex_query_fusion_rerank` improved F1 by only `0.005`, regressed token efficiency from `0.701` to `0.612`, and exceeded the p95 budget at `16588 ms`.

## Alternatives Considered

### Switch to `archex_query_fusion_rerank`
- **Pros:** Highest observed MRR on the Jina run (`0.938`) and a small F1 lift (`0.594` vs `0.589`).
- **Cons:** Token efficiency regressed (`0.612` vs `0.701`) and p95 latency was `16588 ms`, far above the `3000 ms` budget.
- **Rejected because:** It failed all non-F1 switch constraints and did not clear the required `+0.05` mean F1 delta.

### Switch the benchmark embedder to CodeRankEmbed
- **Pros:** CodeRankEmbed remains a plausible code-specialized embedder after fixing query-prefix and repeated-load issues.
- **Cons:** The 2026-06-09 CodeRank run completed only `28/35` tasks due clone DNS failures and had extreme partial-run p95 latency (`253658 ms` for fusion rerank).
- **Rejected because:** The evidence is not clean full-run evidence, and the partial frontier was worse than Jina on recall, F1, token efficiency, and p95 latency.

### Select MiniLM as the reranker default
- **Pros:** Much faster than Jina reranker v3 on the same 35-task run (`3924 ms` p95 vs `16522 ms` p95).
- **Cons:** Still misses the `<= 3000 ms` p95 budget and slightly lowers F1 (`0.586` vs `0.594`).
- **Rejected because:** The reranker decision rule requires the selected model to hold p95 at or below `3000 ms` on the operator hardware.

## Consequences

### Positive

- Default changes require evidence across quality, token economy, and latency.
- Baseline refresh cannot bless regressions because it remains approval-gated.
- Jina v2 and `archex_query` stay reachable as rollback paths.

### Negative

- Rerank remains optional and not product-defaulted until a local reranker holds p95 `<= 3000 ms`.
- CodeRankEmbed needs a separate clean re-run after the query-prefix and model-reuse fixes before it can be reconsidered.

### Neutral

- Benchmark-only knobs, such as `--embedder` and `--rerank-model`, do not change product behavior by themselves.

## References

- `docs/RETRIEVAL_DEFAULT_DECISIONS.md`
- 2026-06-09 retrieval-default benchmark run and readiness summaries
