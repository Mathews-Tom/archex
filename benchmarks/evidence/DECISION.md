# S7 Determinism Economics Decision

## Decision

**Keep the economic framing for the frozen S7 session fixture. Do not change archex retrieval ordering or claim retrieval-quality improvement.**

The pre-registered 5% SESOI was exceeded for both comparators, so the pre-declared retirement rule does not apply:

| Arm | Cache-hit rate, 95% repository-cluster interval | Input cost / resolved task, 95% repository-cluster interval |
| --- | ---: | ---: |
| Deterministic ordering | 66.67% [66.67%, 66.67%] | $0.00095846 [$0.00071594, $0.00144350] |
| Perturbed ordering | 33.26% [20.94%, 45.13%] | $0.00136863 [$0.00110406, $0.00189775] |
| Seeded ANN-ordering comparator | 20.92% [16.59%, 29.17%] | $0.00152004 [$0.00118025, $0.00219963] |

Relative to the comparators, deterministic ordering reduced modeled input cost by 29.97% [23.94%, 35.15%] against perturbed ordering and 36.95% [34.38%, 39.34%] against the seeded ANN-ordering comparator.

## Evidence boundary

`benchmarks/evidence/s7-determinism-economics.json` records input-side prefix-cache accounting for eight frozen, repeated multi-turn coding sessions across four repository clusters. It uses byte-identical rendered prefixes and `cl100k_base` token counts. It does not call a hosted model, observe a provider cache ledger, alter archex's ordering, or evaluate retrieval quality. The ANN comparator is seeded for reproducibility while representing unstable per-turn context ordering; it is not a claim about any vendor ANN implementation.

The dollar figures use the recorded public `claude-opus-5` input schedule: $5.00 per million base input tokens, 1.25x five-minute cache writes, and 0.1x cache reads. They are pricing-model estimates, not billed usage. The pricing source was rechecked at run time: <https://platform.claude.com/docs/en/about-claude/pricing>.

## Reproduction

Start from a clean checkout at source revision `2224d0d41c293702d6450a182899f48576252e2d` after pre-registration commit `0c8c8483aab09c8b4d9e6c4e2f80408de7e2ed6e` is reachable:

```text
uv run archex benchmark determinism-economics --output benchmarks/evidence/s7-determinism-economics.json --preregistration-commit 0c8c8483aab09c8b4d9e6c4e2f80408de7e2ed6e
uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s7-determinism-economics.json
```

The evidence fixes the session-fixture digest, pricing schedule, 10,000 repository-cluster bootstrap resamples, and seed `20260729`. Reproducing the command produces the same values except `generated_at`.

## Limits and next action

Treat this as proof that stable context order materially changes the modeled prefix-cache ledger for the frozen fixture. Do not extrapolate it to production spend until a real model-backed run records provider-reported cache tokens under an explicitly pinned model and TTL. Archex's default ordering remains unchanged.
