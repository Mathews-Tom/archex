# M7 — Runtime and coverage evidence: measurement and promotion decision

Milestone: add revision-bounded coverage mappings and folded-stack runtime
evidence as read-only, typed inputs, promoted to the shipped default only
if a declared performance-localization/downstream gate improves without M3
regressions. This records the `archex_query` vs
`archex_query_runtime_evidence` comparison on the self-repo benchmark
family and the resulting promotion decision.

## What was implemented

- `archex.integrations.runtime`: a `CoverageEvidenceProvider`/
  `RuntimeProfileEvidenceProvider` contract (`probe()`/`collect()`),
  `CoverageXmlProvider` (reads a previously generated Cobertura
  `coverage.xml` report), and `RuntimeProfileProvider` (reads a previously
  collected folded-stack profile, `frame;frame;...;frame count`). Every
  provider is revision-bound: it validates its evidence's declared
  collection revision against the caller's current revision and reports an
  explicit `STALE` receipt with no records on mismatch, never applying
  stale evidence.
- `archex.index.runtime_evidence.collect_runtime_evidence()`: dispatches
  the two providers; zero cost when no provider is requested.
- `IndexConfig.runtime_evidence_providers: list[str] = []` — the gate
  flag. Empty by default: no provider runs, zero bytes of retrieval
  behavior change.
- `IndexStore`: `get/set_runtime_provider_receipts()`,
  `get/set_runtime_coverage_evidence()`, `get/set_runtime_profile_evidence()`;
  `ContextReceipt.runtime_providers` surfaces the receipts on the query
  cold path, mirroring M6's `semantic_providers`.
- `AnalysisArtifactV1`: `SymbolCandidate` gains
  `runtime_sample_count`/`runtime_revision`/`runtime_stale`; `TestCandidate`
  gains `coverage_line_rate`/`coverage_revision`/`coverage_stale`. Both
  renderers (`report/render_html.py`, `report/render_markdown.py`) display
  the evidence source and staleness alongside every candidate.
- `archex_query_runtime_evidence` benchmark strategy
  (`Strategy.ARCHEX_QUERY_RUNTIME_EVIDENCE`): `archex_query`'s exact
  configuration plus
  `runtime_evidence_providers=["coverage", "runtime_profile"]`.

Unlike M6's `EdgeKind.SEMANTIC_*` edges (which fold into the dependency
graph traversed by candidate expansion), M7's evidence is deliberately kept
**separate from retrieval**, per the milestone's in/out-of-scope split
("Out: ... causal claims from static structure"): it is stored, revision-
validated, and surfaced through receipts and the diff-review artifact, but
no ranking, candidate-selection, or graph-expansion code path reads it in
this milestone. This is a read-only-evidence-collection milestone, not a
ranking milestone.

## Method

Self-repo benchmark family (`benchmarks/tasks/*.yaml`, `repo: "."`,
24 tasks) — the same family M6 used, and the only family where evidence for
the corpus under test can be collected against a known, verifiable
revision. Same token budget, same chunker, same cache policy
(`--warm-cache`) as the product default `archex_query` strategy; the only
`IndexConfig` variable is `runtime_evidence_providers`.

Unlike M6 (which could not obtain a working real-world SCIP index in this
environment), this run used **real, revision-bound evidence**, collected
entirely from tools already present in this repository/environment — no
external tool dependency, no fixture data:

- `coverage.xml`: `uv run pytest --cov-report=xml` (the exact command this
  milestone's own VERIFICATION section runs), 4270 tests, 91.40% line
  coverage, at commit `605984f445822963fe4c9983ea9001de6a91d7b8`.
- `profile.folded`: `cProfile` over a real `index_repository()` self-index
  run of this repository, converted to the folded-stack format
  (caller→callee edges with call counts) at the same commit.

Both were written to `.archex/runtime-evidence/{coverage,profile}/` with a
`manifest.json` recording that revision, then verified `AVAILABLE` (not
`UNAVAILABLE`/`STALE`) via each provider's `collect()` before the benchmark
ran — `188` coverage records, `121` profile records.

```
uv run pytest --cov-report=xml
# -> .archex/runtime-evidence/coverage/{manifest.json,coverage.xml}
# -> .archex/runtime-evidence/profile/{manifest.json,profile.folded}
#    (cProfile capture + folded-stack conversion of index_repository())

uv run archex benchmark run --tasks-dir benchmarks/tasks --self-only \
  --strategy archex_query_runtime_evidence --warm-cache --output <dir> --no-progress
uv run archex benchmark gate --input <dir> --tasks-dir benchmarks/tasks \
  --min-recall 0.50 --min-precision 0.10 --min-f1 0.25 --min-mrr 0.50 \
  --min-token-efficiency-with-completion 0.20 --max-p95-warm-latency-ms 15000 \
  --promotion-strategy archex_query_runtime_evidence --control-strategy archex_query
```

`--strategy` is additive to the runner's always-included baseline set
(`raw_files`, `raw_ripgrep`, `archex_query`), so `archex_query` and
`archex_query_runtime_evidence` are measured in the same run against the
same cold index build per task — the same-run control the gate command
consumes. Raw per-task result JSON and the run's evidence manifest are
committed alongside this file under `raw/`.

## Results

Both providers reported `AVAILABLE` for every task (`188` coverage
records, `121` profile records collected once per task's index build).
Every metric is nonetheless identical to 3 decimal places on every one of
the 24 self-repo tasks — Δ = 0.000 everywhere, including `tokens_output`
and warm latency shape — because evidence collection is a pure side
channel in this milestone: it populates the store and the receipt/artifact
surfaces but is never read by search, ranking, or candidate expansion.

| task | base recall | cand recall | base f1 | cand f1 | base mrr | cand mrr |
|---|---:|---:|---:|---:|---:|---:|
| archex_adapter_registry | 1.000 | 1.000 | 0.750 | 0.750 | 1.000 | 1.000 |
| archex_benchmark_gate_lifecycle | 0.800 | 0.800 | 0.800 | 0.800 | 1.000 | 1.000 |
| archex_delta_index_lifecycle | 0.667 | 0.667 | 0.500 | 0.500 | 0.500 | 0.500 |
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
| **mean** | **0.858** | **0.858** | **0.596** | **0.596** | **0.938** | **0.938** |

`archex benchmark gate` with `--promotion-strategy archex_query_runtime_evidence
--control-strategy archex_query` reports **`Quality gate passed`**: the
absolute floors and the same-run control non-regression checks all pass,
because the candidate is byte-for-byte identical to the control.

## Decision

**Do not enable by default.** Non-regression is not the milestone's bar —
M7's stated acceptance is that "the channel is enabled only after a
predeclared performance-localization/downstream gate **improves** with no
M3 regression." The measured candidate shows zero improvement on any
metric. This is not an infrastructure gap or an external-tool failure (both
providers ingested real, revision-verified evidence successfully): it is
the correct, honest result of a milestone scoped to *collect and surface*
runtime/coverage evidence without yet wiring it into ranking or candidate
selection. A future milestone that consumes `SymbolCandidate
.runtime_sample_count` / `TestCandidate.coverage_line_rate` (or the
underlying `IndexStore` evidence) in a performance-localization or
test-risk-aware retrieval strategy is the natural next step to actually
move this gate — that is out of scope here per M7's own "no ... causal
claims from static structure" / "no auto-fixing" boundary.

`IndexConfig.runtime_evidence_providers` ships **defaulting to `[]`**
(disabled): the provider contracts, revision validation, store/receipt/
artifact integration, and gated benchmark lane are real, tested, reusable
infrastructure, but nothing in this stack changes a single byte of default
retrieval behavior. Its `Unreleased` CHANGELOG entry stays staged for a
future minor, per `DEVELOPMENT_PLAN.md`'s "no standalone release by
default" for this section — to be revisited once a downstream retrieval
strategy actually consumes this evidence and can be measured for
improvement.
