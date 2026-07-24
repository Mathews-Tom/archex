# M9 — Documentation-graph evidence: measurement and promotion decision

Milestone: link ADR, documentation, ownership, and release-status evidence
as relations distinct from code dependency; publish a dimensioned
evidence-linked status card; use immutable read-only Action/example pins;
include compatibility/benchmark release artifacts. Promoted to the shipped
default only if a declared discoverability/trust downstream gate improves
without M3 regressions. This records the `archex_query` vs
`archex_query_documentation_evidence` comparison on the self-repo
benchmark family, the resulting promotion decision, and confirmation that
this run does not reproduce M8's pre-existing benchmark-health finding as
a new regression.

## What was implemented

- `archex.integrations.docs`: a `DocLinkEvidenceProvider`/
  `AdrEvidenceProvider`/`OwnershipEvidenceProvider` contract
  (`probe()`/`collect()`) with `DocLinkProvider` (scans README.md plus
  `docs/`/`.docs/` markdown already on disk for links resolving to a real
  file under the repository root -- never a remote fetch, never a record
  for a nonexistent target), `AdrProvider` (reads a conventional ADR
  directory when present; `UNAVAILABLE` with an explicit reason otherwise),
  and `OwnershipProvider` (reads a conventional CODEOWNERS-style manifest)
  implementations.
- `archex.index.documentation_evidence.collect_documentation_evidence()`:
  the gate flag dispatcher. `IndexConfig.documentation_evidence_providers:
  list[str] = []` -- empty by default, zero provider runs, zero bytes of
  retrieval behavior change.
- `IndexStore`: `get/set_documentation_provider_receipts()`,
  `get/set_documentation_links()`, `get/set_documentation_adr_records()`,
  `get/set_documentation_ownership_records()`; `ContextReceipt.
  documentation_providers` surfaces the receipts on the query cold path.
  Structurally distinct from code-dependency evidence throughout: no
  documentation/ADR/ownership relation is ever added to `DependencyGraph`
  as an `Edge`/`EdgeKind` -- proven by a dedicated graph-distinctness test
  (`tests/index/test_documentation_evidence.py`) that builds a real
  self-repo index with `documentation_evidence_providers=["doc_link"]`
  enabled and asserts byte-identical graph edges vs the channel disabled.
- `archex report status-card SOURCE --format json|markdown`: a
  dimensioned, evidence-linked status card (Documentation linkage, ADR
  provenance, Ownership coverage, Release & CI evidence) that is
  structurally never scored -- no field anywhere on `StatusCard`/
  `StatusDimension` aggregates dimensions into a composite grade (enforced
  by a type-design test that enumerates `model_fields`). Every dimension
  is `UNKNOWN` unless its provider is configured and produced real
  evidence; there is no default-enabled dimension.
- `archex report release-artifact SOURCE`: bundles archex's own installed
  version, supported Python range, report/index schema versions, a
  pointer to any checked-in benchmark manifest, and SOURCE's status card
  into one read-only `CompatibilityArtifact`, suitable for attaching to a
  GitHub release.
- `.github/workflows/status-card.yml`: a new immutable-pinned, read-only
  CI example (mirrors `report-diff.yml`'s pin discipline exactly) that
  runs both new commands and uploads their outputs.
- `archex_query_documentation_evidence` benchmark strategy
  (`Strategy.ARCHEX_QUERY_DOCUMENTATION_EVIDENCE`): `archex_query`'s exact
  configuration plus `documentation_evidence_providers=["doc_link"]`.

Like M8's `git_log`, `doc_link` needs no operator-staged evidence file: it
scans real markdown documentation automatically and unconditionally from
the corpus's own working tree.

## Method

Self-repo benchmark family (`benchmarks/tasks/*.yaml`, `repo: "."`,
24 tasks) — the same family M6/M7/M8 used. Same token budget, same
chunker, same cache policy (`--warm-cache`) as the product default
`archex_query` strategy; the only `IndexConfig` variable is
`documentation_evidence_providers`.

```
uv run archex benchmark run --tasks-dir benchmarks/tasks --self-only \
  --strategy archex_query_documentation_evidence --warm-cache --output <dir> --no-progress
uv run archex benchmark gate --input <dir> --tasks-dir benchmarks/tasks \
  --min-recall 0.50 --min-precision 0.10 --min-f1 0.25 --min-mrr 0.50 \
  --min-token-efficiency-with-completion 0.20 --max-p95-warm-latency-ms 15000 \
  --promotion-strategy archex_query_documentation_evidence --control-strategy archex_query
```

`--strategy` is additive to the runner's always-included baseline set
(`raw_files`, `raw_ripgrep`, `archex_query`), so `archex_query` and
`archex_query_documentation_evidence` are measured in the same run against
the same cold index build per task — the same-run control the gate command
consumes. Raw per-task result JSON and the run's evidence manifest are
committed alongside this file under `raw/` (evidence manifest
`source_revision`: `1925fd798edc2062a3b1db12f3ead4ac8867c410`, the tip of
this PR's own stack — a real, verified-clean git revision, per
`archex.benchmark.evidence.source_revision()`'s dirty-tree guard).

## Results

`doc_link` scanned this repository's own README/`docs`/`.docs` markdown
for every task (real local documentation, no external tool dependency).
Every metric is nonetheless identical to full floating-point precision on
every one of the 24 self-repo tasks — recall, precision, F1, MRR, and
`tokens_output` all match exactly, because documentation evidence is a
pure side channel in this milestone: it populates the store and the
receipt/status-card surfaces but is never read by search, ranking, or
candidate expansion.

| task | base recall | cand recall | base f1 | cand f1 | base mrr | cand mrr |
|---|---:|---:|---:|---:|---:|---:|
| archex_adapter_registry | 1.000 | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 |
| archex_benchmark_gate_lifecycle | 0.800 | 0.800 | 0.800 | 0.800 | 1.000 | 1.000 |
| archex_delta_index_lifecycle | 0.667 | 0.667 | 0.500 | 0.500 | 0.333 | 0.333 |
| archex_delta_indexing | 1.000 | 1.000 | 0.571 | 0.571 | 1.000 | 1.000 |
| archex_graph_expansion | 1.000 | 1.000 | 0.667 | 0.667 | 1.000 | 1.000 |
| archex_mcp_query_lifecycle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| archex_pattern_detection | 1.000 | 1.000 | 0.571 | 0.571 | 1.000 | 1.000 |
| archex_project_config_resolution | 0.750 | 0.750 | 0.750 | 0.750 | 1.000 | 1.000 |
| archex_project_index | 0.800 | 0.800 | 0.800 | 0.800 | 1.000 | 1.000 |
| archex_project_init | 0.750 | 0.750 | 0.750 | 0.750 | 1.000 | 1.000 |
| archex_project_reset | 1.000 | 1.000 | 0.857 | 0.857 | 1.000 | 1.000 |
| archex_project_status | 0.750 | 0.750 | 0.750 | 0.750 | 1.000 | 1.000 |
| archex_query_cache_lifecycle | 0.800 | 0.800 | 0.800 | 0.800 | 1.000 | 1.000 |
| archex_query_pipeline | 0.667 | 0.667 | 0.500 | 0.500 | 1.000 | 1.000 |
| archex_scoring | 1.000 | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 |
| archex_vector_cache_lifecycle | 0.600 | 0.600 | 0.600 | 0.600 | 1.000 | 1.000 |
| routing_mixed_cache | 1.000 | 1.000 | 0.571 | 0.571 | 1.000 | 1.000 |
| routing_mixed_chunker | 1.000 | 1.000 | 0.333 | 0.333 | 0.500 | 0.500 |
| routing_pl_intent | 1.000 | 1.000 | 0.333 | 0.333 | 1.000 | 1.000 |
| routing_pl_large | 1.000 | 1.000 | 0.400 | 0.400 | 1.000 | 1.000 |
| routing_pl_path_symbol | 1.000 | 1.000 | 0.333 | 0.333 | 1.000 | 1.000 |
| routing_pl_scoring | 0.500 | 0.500 | 0.286 | 0.286 | 1.000 | 1.000 |
| routing_pl_tight | 1.000 | 1.000 | 0.333 | 0.333 | 1.000 | 1.000 |
| routing_trace_api | 0.500 | 0.500 | 0.286 | 0.286 | 0.500 | 0.500 |

`archex benchmark gate` with `--promotion-strategy
archex_query_documentation_evidence --control-strategy archex_query`
reports **`PROMOTION GATE FAILED: 1 violation(s)`**:
`archex_delta_index_lifecycle/archex_query_documentation_evidence
mrr: 0.333 < 0.500`.

### Root cause: the same pre-existing self-repo corpus drift M8 already root-caused, not an M9 regression

This is the identical absolute-floor violation M8's own gate run recorded
(`benchmarks/results/m8_repository_memory/DECISION.md`): the candidate's
MRR on `archex_delta_index_lifecycle` (`0.3333333333333333`) is
bit-for-bit identical to the control's own MRR on the same task, and to
M8's previously recorded value. `archex_query` **itself** still misses
this task's declared 0.500 MRR floor because the self-repo corpus
contains two files both literally named `delta.py`
(`src/archex/report/delta.py`, added by the already-merged M4 milestone,
and `src/archex/index/delta.py`, the task's actual expected target) —
unrelated to M6/M7/M8/M9 and unchanged since M8's own run confirmed it.
M9's documentation evidence is architecturally inert on retrieval (see
"Results" above), so it cannot have caused or worsened this finding; it
is re-noted here only to confirm this run did not introduce a new
regression on top of the already-tracked one.

## Decision

**Do not enable by default.** M9's stated acceptance is that "the feature
is enabled only when its declared discoverability/trust evaluation
improves without violating M3 gates." The measured candidate shows zero
improvement on any metric (bit-for-bit identical to the control) across
all 24 self-repo tasks, so promotion does not apply.
`IndexConfig.documentation_evidence_providers` ships **defaulting to `[]`**
(disabled): the provider contracts, store/receipt integration, dimensioned
status card, per-release compatibility artifact, and gated benchmark lane
are real, tested, reusable infrastructure that reads genuine local
markdown/ADR/CODEOWNERS evidence — but nothing in this stack changes a
single byte of default retrieval behavior. Its `Unreleased` CHANGELOG
entry stays staged for a future minor, per `DEVELOPMENT_PLAN.md`'s
"no standalone release by default" for this section — to be revisited once
a downstream retrieval or discoverability surface actually consumes this
evidence and can be measured for improvement.
