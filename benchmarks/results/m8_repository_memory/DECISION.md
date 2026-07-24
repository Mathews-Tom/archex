# M8 — Repository-memory evidence: measurement and promotion decision

Milestone: surface linked commit/issue-reference, temporal-coupling, and
operator-rationale history evidence only for history-rich, relevant
queries with explicit validity bounds, promoted to the shipped default
only if a declared history-rich downstream gate improves without M3
regressions. This records the `archex_query` vs
`archex_query_history_evidence` comparison on the self-repo benchmark
family, the resulting promotion decision, and a root-caused pre-existing
benchmark-health finding surfaced while running the gate.

## What was implemented

- `archex.integrations.history`: a `GitLogEvidenceProvider`/
  `OperatorRationaleEvidenceProvider` contract (`probe()`/`collect()`) with
  `GitLogHistoryProvider` (reads local `git log`/`git rev-parse` bound to a
  200-commit window, never a remote call; extracts issue/PR references from
  commit subjects via local regex) and `OperatorRationaleProvider` (reads
  previously supplied, revision-bound operator rationale) implementations.
- `archex.integrations.history.eligibility.evaluate_history_eligibility()`:
  the density/linkage/relevance gate. History evidence is collected
  unconditionally once `git_log` is enabled, but is surfaced only when the
  collected window clears three predeclared thresholds (density ≥ 0.30,
  linkage ≥ 0.10, relevance ≥ 0.20) against the operation's own candidate
  file set.
- `IndexConfig.history_evidence_providers: list[str] = []` — the gate flag.
  Empty by default: no provider runs, zero bytes of retrieval behavior
  change.
- `IndexStore`: `get/set_history_provider_receipts()`,
  `get/set_history_change_cards()`, `get/set_history_coupling_observations()`,
  `get/set_history_operator_rationale()`; `ContextReceipt.history_providers`/
  `.history_eligibility` surface the receipts and decision on the query cold
  path. `AnalysisArtifactV1`'s `DiffFileChange`/`DiffAnalysis` display the
  same for the diff-review artifact.
- `archex_query_history_evidence` benchmark strategy
  (`Strategy.ARCHEX_QUERY_HISTORY_EVIDENCE`): `archex_query`'s exact
  configuration plus `history_evidence_providers=["git_log"]`.

Unlike M6/M7's providers, `git_log` needs no operator-staged evidence file:
it collects real commit history automatically and unconditionally from the
corpus's own `.git` directory, subject only to the eligibility gate deciding
whether to *surface* it.

## Method

Self-repo benchmark family (`benchmarks/tasks/*.yaml`, `repo: "."`,
24 tasks) — the same family M6/M7 used. Same token budget, same chunker,
same cache policy (`--warm-cache`) as the product default `archex_query`
strategy; the only `IndexConfig` variable is `history_evidence_providers`.

```
uv run archex benchmark run --tasks-dir benchmarks/tasks --self-only \
  --strategy archex_query_history_evidence --warm-cache --output <dir> --no-progress
uv run archex benchmark gate --input <dir> --tasks-dir benchmarks/tasks \
  --min-recall 0.50 --min-precision 0.10 --min-f1 0.25 --min-mrr 0.50 \
  --min-token-efficiency-with-completion 0.20 --max-p95-warm-latency-ms 15000 \
  --promotion-strategy archex_query_history_evidence --control-strategy archex_query
```

`--strategy` is additive to the runner's always-included baseline set
(`raw_files`, `raw_ripgrep`, `archex_query`), so `archex_query` and
`archex_query_history_evidence` are measured in the same run against the
same cold index build per task — the same-run control the gate command
consumes. Raw per-task result JSON and the run's evidence manifest are
committed alongside this file under `raw/`.

## Results

`git_log` collected real, live commit history for every task (200-commit
window ending at the resolved revision). Every metric is nonetheless
identical to full floating-point precision on every one of the 24 self-repo
tasks — recall, precision, F1, MRR, and `tokens_output` all match exactly,
because history evidence is a pure side channel in this milestone: it
populates the store and the receipt/artifact surfaces but is never read by
search, ranking, or candidate expansion.

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

`archex benchmark gate` with `--promotion-strategy archex_query_history_evidence
--control-strategy archex_query` reports **`PROMOTION GATE FAILED: 1
violation(s)`**: `archex_delta_index_lifecycle/archex_query_history_evidence
mrr: 0.333 < 0.500`.

### Root cause: pre-existing self-repo corpus drift, not an M8 regression

This is an absolute-floor violation, not a same-run regression: the
candidate's MRR on this task (`0.3333333333333333`) is bit-for-bit identical
to the control's own MRR on the same task. `archex_query` **itself**
currently misses this task's declared 0.500 MRR floor, independent of any
M8 code -- M8's history evidence is architecturally inert on retrieval (see
"Results" above), so it cannot have caused this.

Inspecting the task: `archex_delta_index_lifecycle` expects
`src/archex/index/delta.py`, `src/archex/cache.py`, `src/archex/status.py`
for the question "How does archex compute delta indexing changes...".
`archex_query`'s actual top-5 result for this task is
`['src/archex/report/delta.py', 'src/archex/project.py',
'src/archex/index/delta.py', 'src/archex/cache.py',
'src/archex/index/store.py']` -- `src/archex/report/delta.py` (added by the
already-merged M4 milestone, unrelated to M6/M7/M8) now lexically
out-ranks the expected `src/archex/index/delta.py` for this query, because
the self-repo corpus now contains two files both literally named
`delta.py`. This is corpus drift in the shared self-repo benchmark family
from **prior, unrelated milestone work**, not something introduced by this
PR stack; the same drift would reproduce running this exact gate against
`archex_query` alone, with no M8 code present at all.

This is a genuine finding worth tracking (self-repo benchmark tasks need
periodic re-validation as the corpus they benchmark against -- archex's own
source tree -- keeps growing), but it is out of scope for M8's own
deliverable (a conditional, currently-inert evidence-collection
infrastructure) to fix, and does not change M8's own promotion outcome:
zero measured improvement either way.

## Decision

**Do not enable by default.** M8's stated acceptance is that "the channel
is promoted only after the predeclared history-rich downstream gate
**improves** while M3 gates remain satisfied." The measured candidate shows
zero improvement on any metric (bit-for-bit identical to the control), so
promotion does not apply regardless of the one pre-existing, root-caused
absolute-floor finding above. `IndexConfig.history_evidence_providers`
ships **defaulting to `[]`** (disabled): the provider contracts,
eligibility policy, store/receipt/artifact integration, and gated
benchmark lane are real, tested, reusable infrastructure that collects
genuine local commit history and correctly gates its exposure by
density/linkage/relevance -- but nothing in this stack changes a single
byte of default retrieval behavior. Its `Unreleased` CHANGELOG entry stays
staged for a future minor, per `DEVELOPMENT_PLAN.md`'s "no standalone
release by default" for this section -- to be revisited once a downstream
retrieval strategy actually consumes this evidence and can be measured for
improvement.
