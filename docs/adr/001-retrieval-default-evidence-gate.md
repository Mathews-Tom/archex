# ADR-001: Gate Retrieval Default Changes On Recall, Tokens, And P95

**Status:** Proposed
**Date:** 2026-06-08
**Author:** archex maintainers

## Context

Tier 3 settles the deferred default embedder, reranker, and product strategy decisions. Earlier strategy evidence was confounded by mixed embedders and missing token/p95 gates, so a recall-only switch can optimize the benchmark while worsening the product contract: fewer tokens per retrieval/explanation query with interactive p95 latency.

## Decision

Keep the current product default until clean warm-cache operator evidence satisfies the documented recall/token/p95 switch rules. Do not refresh `benchmarks/dogfood_baseline.json` or flip defaults without explicit approval after the evidence is reviewed.

## Alternatives Considered

### Switch to `archex_query_fusion_rerank` immediately
- **Pros:** Uses the highest-precision candidate path and may reduce returned context when rerank is effective.
- **Cons:** Current accepted evidence does not prove p95 <= 3000 ms on the clean single-embedder frontier.
- **Rejected because:** The Tier 3 rule requires operator-run median and p95 latency plus token efficiency before changing the product default.

### Keep `archex_query` permanently
- **Pros:** Preserves the known product default and rollback path.
- **Cons:** Could reject a better frontier point if CodeRankEmbed or a tuned reranker improves quality without token or p95 regression.
- **Rejected because:** Tier 3 exists to decide from clean evidence, not freeze the old default without measuring the candidates.

## Consequences

### Positive

- Default changes require evidence across quality, token economy, and latency.
- Baseline refresh cannot bless regressions because it remains approval-gated.
- Jina v2 and `archex_query` stay reachable as rollback paths.

### Negative

- The final default remains pending until the operator runs the long benchmark block.
- Reviewers must inspect the evidence table before accepting any default flip.

### Neutral

- Benchmark-only knobs, such as `--embedder` and `--rerank-model`, do not change product behavior by themselves.

## References

- `docs/RETRIEVAL_DEFAULT_DECISIONS.md`
- `.docs/2026-05-29-retrieval-recall-enhancement-plan.md` Tier 3
