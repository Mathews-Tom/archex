# M6 — Compiler-grade semantic evidence: measurement and promotion decision

Milestone: conditionally enrich the syntax graph with SCIP and LSAP/LSP
definition/reference/implementation evidence, promoted to the shipped default
only if it improves M3's declared benchmark without regression. This records
the `archex_query` vs `archex_query_semantic` comparison on the self-repo
benchmark family and the resulting promotion decision.

## What was implemented

- `archex.integrations.semantic`: `SemanticEvidenceProvider` contract
  (`probe()`/`collect()`), `ScipEvidenceProvider` (reads a pre-built SCIP
  index off disk via a vendored, protoc-generated `scip_pb2`), and
  `LspEvidenceProvider` (reuses the existing `archex.integrations.lsap`
  wrapper against a caller-supplied `lsp_client.Client`). Every provider
  returns an explicit `AVAILABLE`/`PARTIAL`/`UNAVAILABLE`/`STALE` receipt
  instead of raising or inventing edges.
- `DependencyGraph.add_semantic_edges()`: folds evidence into the file graph
  as new `EdgeKind.SEMANTIC_*` edges, distinct from every syntax kind, only
  connecting files already known to the graph and never overwriting an
  existing edge.
- `IndexConfig.semantic_evidence_providers: list[str] = []` — the gate flag.
  Empty by default: no provider runs, zero bytes of the index change.
- `IndexStore`: nullable `provider`/`provider_version` edge columns and
  `get/set_semantic_provider_receipts()`; `ContextReceipt.semantic_providers`
  surfaces the receipts on the query cold path.
- `archex_query_semantic` benchmark strategy (`Strategy.ARCHEX_QUERY_SEMANTIC`):
  `archex_query`'s exact configuration plus
  `semantic_evidence_providers=["scip"]`.

## Method

Self-repo benchmark family (`benchmarks/tasks/*.yaml`, `repo: "."`,
`category: self`, 24 tasks) — the only family where a SCIP index for the
corpus under test can plausibly be generated locally without cloning and
type-checking an external repository. Same token budget, same chunker, same
cache policy (`cache=False`, fresh cold build per task) as the product
default `archex_query` strategy; the only variable is
`semantic_evidence_providers=["scip"]`.

```
uv run archex benchmark run --tasks-dir benchmarks/tasks --self-only \
  --strategy archex_query_semantic --warm-cache --output <dir> --no-progress
uv run archex benchmark gate --input <dir> --tasks-dir benchmarks/tasks \
  --min-recall 0.50 --min-precision 0.10 --min-f1 0.25 --min-mrr 0.50 \
  --min-token-efficiency-with-completion 0.20 --max-p95-warm-latency-ms 15000 \
  --promotion-strategy archex_query_semantic --control-strategy archex_query
```

`--strategy` is additive to the runner's always-included baseline set
(`raw_files`, `raw_ripgrep`, `archex_query`), so `archex_query` and
`archex_query_semantic` are measured in the same run against the same cold
index build per task — the same-run control the gate command consumes.

### Why no SCIP index was present for the measured run

The provider is designed to read a pre-built SCIP index at
`<repo_root>/index.scip`. The only real-world SCIP producer readily
available in this environment, `@sourcegraph/scip-python` (run via `npx`),
was exercised directly: it completes without error and reports "Total
Project Files 413" for this repository (and even a trivial hand-written
2-file smoke-test project with zero dependencies), but in every invocation
tried — default, `--target-only`, `--project-name`, an explicit
`pyrightconfig.json`, and an explicit empty `--environment` JSON array —
emits an `Index` message with **zero `documents`**. This reproduces
identically on the minimal smoke project, so it is a defect or environment
incompatibility in `scip-python` itself in this sandbox, not a defect in
this milestone's provider or graph-integration code, which is independently
verified correct against hand-constructed SCIP protobuf fixtures
(`tests/integrations/semantic/test_scip_provider.py`,
`tests/index/test_semantic_evidence.py::TestIndexRepositoryWiring`,
`tests/benchmark/test_semantic_strategy.py`) and produces real
`SEMANTIC_DEFINITION`/`SEMANTIC_REFERENCE`/`SEMANTIC_IMPLEMENTATION` edges,
correct provider/version/confidence, and an `AVAILABLE` receipt end to end
through the public `index_repository()` API.

The self-repo benchmark below therefore measures the provider's **honest,
common real-world case: no pre-built SCIP index present.**

## Results

No `index.scip` was present at the repository root for the measured run.
`ScipEvidenceProvider.probe()` correctly reported `UNAVAILABLE` (reason:
`no SCIP index found at index.scip`) for every task; zero semantic edges
were added; the two strategies' `IndexConfig` differ only in the unused
flag.

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

Every metric is identical to 3 decimal places on every one of the 24
self-repo tasks — Δ = 0.000 everywhere. `archex benchmark gate` with
`--promotion-strategy archex_query_semantic --control-strategy archex_query`
reports **`Quality gate passed`**: the absolute floors and the same-run
control non-regression checks all pass, because the candidate is byte-for-byte
identical to the control.

## Decision

**Do not enable by default.** Non-regression is not the milestone's bar —
M6's stated acceptance is "production promotion occurs only if M3's declared
benchmark **improves**." The measured candidate shows zero improvement on
any metric, because no pre-built SCIP index was available to activate the
provider (an external-tool defect in `scip-python` documented above, not an
implementation gap in this milestone — see the passing unit and integration
tests proving the provider produces real, correct edges when given real SCIP
data). `IndexConfig.semantic_evidence_providers` ships **defaulting to
`[]`** (disabled): the provider contract, graph/receipt integration, and
gated benchmark lane are real, tested, reusable infrastructure, but nothing
in this stack changes a single byte of default retrieval behavior. Its
`Unreleased` CHANGELOG entry stays staged for a future minor, per
`DEVELOPMENT_PLAN.md`'s "no standalone release by default" for this section
— to be revisited once a working real-world SCIP (or a connected LSP server)
source is available to measure an actual improvement against.
