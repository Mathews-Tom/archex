# R30 — Frozen scope-aware benchmark execution

## Scope and terminal boundary

This report is the terminal benchmark evidence for R26 Candidate A, scope-aware monorepo ranking. It makes no product registration, default-change, MCP, language, release, or promotion claim. A continuation-eligible result remains benchmark-only.

**Terminal disposition:** EVIDENCE NO-GO — RELEASE: none — REASON: binding gate: no_new_zero_recall

## Coverage and immutable identity

- Raw evidence: `4,128/4,128` unique declared cells; `4,128` successes and `0` recorded failures.
- Campaign: `r27-scope-aware-monorepo-ranking`; canonical manifest SHA-256: `20caed725c4091a4ad675ead0d09f8bbd8ee6fcd1844b631480658a862175ca2`.
- Candidate identity SHA-256: `80729d35addc6af8679279cd0bcf78fa31c97a308f384b214665e5a1f2e6e938`.
- Every declared cell remains in the denominator. Recorded failures have required-file recall `0.0`; no cell is excluded, repaired, or rerun.

## Primary inference

- **Primary selector:** exactly `kind=treatment`, 2,048 paired tasks. The 16 `kind=single_scope_control` pairs are invariant-only and never enter the primary estimate.
- **Aggregation:** task-level treatment-minus-control required-file recall is averaged within each repository; the 16 repository means receive equal weight.
- Point estimate: **0.029053**.
- Bootstrap: 10,000 whole-repository percentile resamples at seed `20260913`.
- **95% beneficial/non-inferiority interval:** [-0.002930, 0.059326].
- **90% TOST equivalence interval:** [0.002686, 0.055176].
- MWG `+0.05`: FAIL; beneficial (95% lower bound > 0): FAIL; NIM `-0.02`: PASS; EQM `±0.02`: FAIL.

## Binding invariant gates

- `complete_unique_cells`: PASS.
- `single_scope_payload`: PASS.
- `multi_scope_receipts`: PASS.
- `no_subgroup_regression`: PASS.
- `no_new_zero_recall`: FAIL.
- `treatment_warm_p95`: PASS.

### Single-scope payloads and receipts

- Single-scope payload mismatches: `0`.
- Multi-scope receipt failures: `0`.
- New zero-recall treatment tasks: `110`.
- Treatment warm p95: `1536.526` ms; frozen limit: `3000` ms.

### Declared subgroups

- `javascript`: 0.027604 repository-weighted required-file-recall difference across `15` repositories; regression: PASS. Region and line metrics are not applicable because R27 supplies no region or line labels.
- `rust`: 0.021484 repository-weighted required-file-recall difference across `2` repositories; regression: PASS. Region and line metrics are not applicable because R27 supplies no region or line labels.
- `typescript`: 0.027604 repository-weighted required-file-recall difference across `15` repositories; regression: PASS. Region and line metrics are not applicable because R27 supplies no region or line labels.
- `single_scope_control`: 0.000000 paired required-file-recall difference; regression: PASS.

## Interpretation

binding gate: no_new_zero_recall.

A result compatible only with NIM `−0.02` or EQM `±0.02` is `EVIDENCE NO-GO`; it does not authorize promotion. The frozen continuation rule requires a point estimate at least `+0.05`, a 95% beneficial lower bound above zero, and every binding gate to pass.
