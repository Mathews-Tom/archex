# S7 Determinism Economics Decision

## Decision

**Retire the economic framing. Determinism remains a reproducibility property only. Archex retrieval ordering is unchanged.**

The frozen fixture's rendered prefixes contain fewer than the 512-token minimum for Claude Opus 5 prompt caching. The pricing model therefore classifies every prefix as uncached. All three arms have a 0.00% cache-hit rate and the same modeled input cost per resolved task: $0.00151000. Each deterministic-versus-comparator cost reduction is 0.00% with a 95% repository-cluster bootstrap interval of [0.00%, 0.00%].

This fires the pre-registered kill criterion: each comparator is below the +5% SESOI. The null is acceptable and is not a retrieval-quality result.

## Evidence boundary

The artifact records eight frozen, repeated three-turn sessions in four repository clusters. Each session deliberately repeats the same selected context order over its turns; the deterministic arm's counterfactual cache hit rate would therefore be at its two-of-three-turn ceiling if every prefix were cacheable. It is not cacheable under the recorded provider rule, so the degenerate zero cache-hit and cost-reduction intervals are construction plus eligibility facts, not an estimate of production spend.

The harness stores the canonical rendered prefix strings, SHA-256 prefix identities, session-fixture digest, exact command, token-counting method, pricing URL, and pricing retrieval timestamp. `cl100k_base` is a proxy rather than Claude Opus 5's tokenizer; Claude documents that its newer tokenizer can produce approximately 30% more tokens for the same text. That mismatch affects absolute dollar estimates but not the zero cost-reduction result because every arm uses identical input-token accounting.

The price ledger uses the recorded Claude Opus 5 schedule: $5.00 per million base input tokens, 1.25x five-minute cache writes, 0.1x cache reads, and the 512-token minimum cacheable prefix. Pricing source retrieved at `2026-07-28T22:19:00Z`: <https://platform.claude.com/docs/en/about-claude/pricing>. Cache eligibility source: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>.

## Reproduction

Run from the repository root in a clean checkout at source revision `f1f1671873189c881bf29fe945547a0fb7e8af8c`:

```text
uv run archex benchmark determinism-economics --sessions benchmarks/determinism_economics/sessions.json --output benchmarks/evidence/s7-determinism-economics.json --preregistration-commit 501636abb09cbcf6edee5783305af6bcb313606a --pricing-retrieved-at 2026-07-28T22:19:00Z --resamples 10000 --seed 20260729
uv run archex benchmark validate --kind determinism-economics --input benchmarks/evidence/s7-determinism-economics.json
uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s7-determinism-economics.json
```

## Limits and next action

Do not extrapolate this fixture-only null to a deployment. A future study needs real model-backed cache usage and cache-eligible context lengths, pre-registered before data generation. No retrieval-quality claim follows from this study.
