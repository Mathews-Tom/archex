# M17 — Identifier-aware BM25 tokenization: pre-fragmentation baseline

Milestone: index-time camelCase/PascalCase/snake_case identifier splitting for the
BM25 `symbol_name`/`breadcrumbs` FTS columns, additive to the existing
`content`-column identifier expansion. This records the `archex_query`
recall/MRR baseline on the identifier-fragment task corpus **before** any
tokenization change, per the "measure then claim" gate.

## Corpus and method

`benchmarks/tasks/identifier_fragments/*.yaml` — 9 self-repo tasks whose
`question` is literally the space-separated lowercase fragments of a known
compound identifier defined in this repository (e.g. `"context bundle"` for
`ContextBundle`). Each task's `expected_files`/`expected_symbols` point at the
identifier's actual definition site.

Task selection avoided two known confounders, found by direct inspection of
the BM25 index and the `archex_query` pipeline before committing any task:

- **Existing hardcoded query-expansion terms.** `archex.api._expand_retrieval_question`
  injects extra terms when the query contains `query`, `pipeline`, `retrieval`,
  or `index` — verified absent from every task's fragment set so results
  reflect BM25 tokenization, not this unrelated expansion heuristic.
- **File-path term overlap.** `BM25Index._apply_path_bonus` gives an exact-term
  path-segment bonus independent of identifier tokenization; targets were
  chosen so query fragments do not already appear as literal path segments
  (e.g. `ContextBundle` → `models.py`, not `context_bundle.py`).

Six control/reference tasks (`graph_schema_version`, `scoring_weights` variant
dropped for redundancy, `tree_sitter_engine`, `benchmark_task`, `file_tree`,
`structural_context`) were included alongside three harder cases with no
baseline recall (`context_bundle`, `ranked_chunk`, `task_category`,
`symbol_source`) to give the corpus headroom in both directions.

Each task was run via the product's own `archex_query` strategy (BM25-only,
`IndexConfig(vector=False)`), matching the milestone's own acceptance
criterion — not an isolated `BM25Index.search()` call, since a raw-BM25-only
measurement was found (during investigation) to disagree with the full
retrieval pipeline's ranking on some queries.

Reproduce:

```
uv run archex benchmark run --tasks-dir benchmarks/tasks/identifier_fragments \
  --strategy archex_query --output <dir> --no-progress
```

## Baseline (no identifier-fragment tokenization) — `archex_query`

| task | symbol | recall | mrr | precision |
|---|---|---:|---:|---:|
| idfrag_context_bundle | `ContextBundle` | 0.000 | 0.000 | 0.000 |
| idfrag_ranked_chunk | `RankedChunk` | 0.000 | 0.000 | 0.000 |
| idfrag_symbol_source | `SymbolSource` | 0.000 | 0.000 | 0.000 |
| idfrag_task_category | `TaskCategory` | 0.000 | 0.000 | 0.000 |
| idfrag_benchmark_task | `BenchmarkTask` | 1.000 | 1.000 | 0.200 |
| idfrag_file_tree | `FileTree` | 1.000 | 0.500 | 0.200 |
| idfrag_structural_context | `StructuralContext` | 1.000 | 0.500 | 0.200 |
| idfrag_graph_schema_version | `GraphSchemaVersion` | 1.000 | 1.000 | 0.250 |
| idfrag_tree_sitter_engine | `TreeSitterEngine` | 1.000 | 1.000 | 0.200 |
| **mean** | | **0.556** | **0.444** | |

Four tasks (the "harder" PascalCase multi-word symbols with no already-strong
content-level signal) score zero — real, demonstrated headroom for an
identifier-splitting change to close, consistent with the milestone's premise.
The other five already pass, giving control coverage against regression.

## Unrelated infrastructure fix bundled in this PR

`archex dogfood --all --baseline <path>` (this milestone's required
verification command) was found to take 20+ minutes / effectively hang on
this repository. Root cause: the `raw_grepped` diagnostic benchmark strategy
(`run_raw_grepped` in `src/archex/benchmark/strategies.py`) shells out to
plain `grep -r` with no `--exclude-dir`, so every keyword search walks the
entire `.venv` (tens of thousands of vendored `.py` files) and `.git` history
instead of respecting `.gitignore` the way the `raw_ripgrep` strategy already
does. Fixed by adding the same exclude list `discover_files` already uses
(`.git`, `.venv`, `node_modules`, caches, build output, `.archex`) as
`--exclude-dir` flags. Verified: `run_raw_grepped` on a single task dropped
from a ~30s-per-keyword timeout risk to ~1.5s total; `archex dogfood --all`
now completes in ~3 minutes. This is a pre-existing bug, unrelated to BM25
tokenization, discovered only because it blocked running M17's required gate
command — included here as measurement infrastructure, not part of the
tokenization change itself.
