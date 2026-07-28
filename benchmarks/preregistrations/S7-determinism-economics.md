# S7 — Determinism as Prefix-Cache Economics

## Study identity

- **Spike ID and title:** S7 — Determinism as Prefix-Cache Economics
- **Evidence class:** `original`
- **Decision owner and date:** archex R6 milestone owner, 2026-07-29
- **First-run commit:** A commit descended from the merge commit that introduces this pre-registration. The exact source revision and session-fixture digest are recorded by the run artifact before its first data-generating invocation.

## Hypothesis

For the same repeated multi-turn coding sessions, deterministic retrieval ordering reduces input-side prefix-cache cost per resolved task by at least 5% against both a per-turn perturbed ordering and a non-deterministic ANN ordering baseline. The control is archex's existing deterministic ordering; neither comparator is a retrieval-quality treatment.

## Primary metric

The primary metric is the relative reduction in input-side USD cost per resolved task for deterministic ordering against each comparator:

\[
100 \times \frac{C_{\mathrm{comparator}} - C_{\mathrm{deterministic}}}{C_{\mathrm{comparator}}}.
\]

For one arm, input-side USD cost is the sum over all turns of the current published base-input price times `(1.25 × 5-minute cache-write tokens + 0.1 × cache-read tokens + uncached input tokens) / 1,000,000`. Each resolved task contributes one denominator unit; an unresolved task contributes its input cost but no denominator unit. The point estimate pools costs and resolved counts over sessions. Larger positive values favor deterministic ordering.

A task is resolved only when its fixed session fixture marks it resolved. Resolution labels are held identical across all arms and are not an outcome of this study; this prevents a cost measurement from becoming a retrieval-quality claim. Cache-hit rate, total input tokens, per-turn prefix identity, and the deterministic-versus-each-comparator cost interval are reported as exploratory diagnostics.

## SESOI

The smallest effect size of interest is a 5% reduction in input-side USD cost per resolved task. This is the source plan's pre-declared threshold for retaining the economic framing; a smaller difference leaves determinism as a reproducibility property only.

## Decision margins

- **Minimum worthwhile gain (MWG):** +5 percentage points of relative cost reduction for deterministic ordering. Below this, the economic framing is retired because the stated operator decision does not change.
- **Non-inferiority margin (NIM):** −5 percentage points. Deterministic ordering may not cost more than 5% per resolved task than a comparator and still be described as no worse on this economic measure.
- **Equivalence margin (EQM):** ±5 percentage points. Effects wholly inside this interval are practically negligible for the economic-framing decision.

## Clustering unit

The independent resampling unit is the repository. Multiple tasks and their repeated multi-turn sessions from one repository remain together in every bootstrap resample because their prompts share code, task structure, and retrieval context. The analysis uses a paired percentile cluster bootstrap over repositories, with 10,000 resamples, seed `20260729`, and a 95% interval.

## Kill criterion

Retire the economic framing and state that determinism remains a reproducibility property only when either comparator's point estimate is below +5% or its 95% cluster-bootstrap interval does not exclude zero. This is an acceptable null or inconclusive result, not a retrieval-quality finding. The decision document must state the criterion and each comparator's point estimate and interval.

## Run and analysis boundary

The frozen matrix contains exactly three arms on the identical ordered session list:

| Arm | Ordering rule | Role |
| --- | --- | --- |
| `deterministic` | archex's shipped stable ordering, unchanged | Control |
| `perturbed` | a deterministic, seed-recorded per-turn permutation of the same selected context | Ordering perturbation |
| `ann_baseline` | a seed-recorded non-deterministic ANN result ordering over the same candidate context | Comparator baseline |

The run must not change archex ordering, retrieval ranking, task resolution labels, sessions, model pricing, cache TTL, or the output-token treatment between arms. Each arm uses the same model-price schedule and five-minute cache TTL. The harness records the canonical rendered prompt bytes, SHA-256 prefix identities, token-counting method, input source revision, session-fixture digest, price schedule URL and retrieval timestamp, all bootstrap settings, and the exact command in `benchmarks/evidence/s7-determinism-economics.json`.

The first run is permitted only after this file has merged. The recorded command is:

```text
uv run python -m archex.benchmark.determinism_economics --sessions <committed-session-fixture> --output benchmarks/evidence/s7-determinism-economics.json --bootstrap-resamples 10000 --bootstrap-seed 20260729
```

The validator is:

```text
uv run archex benchmark validate --kind determinism-economics --input benchmarks/evidence/s7-determinism-economics.json
```

## Post-hoc changes

- **2026-07-28T21:53:33Z — cache eligibility and evidence-command correction.** The original pre-registration omitted Claude Opus 5's 512-token minimum cacheable prefix and named command surfaces that were not implemented. The implemented run records that eligibility floor, canonical prefixes and hashes, pricing retrieval timestamp, exact command, and a `determinism-economics` validator kind. The frozen arms, session fixture, pricing multipliers, TTL, SESOI, and kill criterion remain unchanged. This completed run is exploratory with respect to the corrected cache ledger because the eligibility correction occurred after the first data-generating run; its null result retires the economic framing rather than supporting a positive economic claim.
