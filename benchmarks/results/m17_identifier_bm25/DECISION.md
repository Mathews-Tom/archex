# M17 — Identifier-aware BM25 tokenization: measurement and merge decision

Continues `BASELINE.md`. Records `archex_query` recall/MRR on the same
identifier-fragment corpus **with** `identifier_fragment_tokenization`
enabled (index-time camelCase/PascalCase splitting of the `symbol_name` and
`breadcrumbs` FTS columns, additive — the original token is preserved
alongside the lowercase fragments), and the resulting merge decision.

## What was implemented

`src/archex/index/bm25.py`: `BM25Index._insert_rows` now additionally runs
the existing `expand_identifiers()` helper (already used for the `content`
column since well before this milestone) over `symbol_name` and
`breadcrumbs` when `identifier_fragment_tokenization` is enabled. An FTS
content-schema version (`bm25_content_version`, composed with the flag value)
is stamped in store metadata and checked on every `BM25Index` construction;
a mismatch — stale schema, a version bump, or a flipped flag — drops and
rebuilds `chunks_fts`, and the existing `has_data`-driven delta path
(`src/archex/index/delta.py`) then performs one full `build()` instead of a
targeted `update()`, so re-indexing existing stores transparently picks up
the new tokenization. `build()` and `update()` share the same
`_insert_rows`, so the delta path and full-build path always tokenize
identically by construction (`test_update_path_expands_symbol_name_fragments_same_as_build`).

The flag (`IndexConfig.identifier_fragment_tokenization`) is threaded through
every `BM25Index` construction site that has an `IndexConfig` in scope
(`api.py` query-time cache-hit and cache-miss paths, `delta.py`
`apply_delta`). It defaults to **`False`** — see "Decision" below.

## Method

Same corpus, same `archex_query` (BM25-only) strategy, same token budget, as
`BASELINE.md`. The only variable changed is
`identifier_fragment_tokenization` (`False` → `True`); reproduced with a
`git stash` toggle of the implementation commits against an otherwise
identical working tree, so no other confound (cache state, task set, code
version elsewhere) differs between the two runs.

```
uv run archex benchmark run --tasks-dir benchmarks/tasks/identifier_fragments \
  --strategy archex_query --output <dir> --no-progress
```

## Results

| task | symbol | baseline recall | with-tokenization recall | Δ | baseline mrr | with-tokenization mrr | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| idfrag_context_bundle | `ContextBundle` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| idfrag_ranked_chunk | `RankedChunk` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| idfrag_symbol_source | `SymbolSource` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| idfrag_task_category | `TaskCategory` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| idfrag_benchmark_task | `BenchmarkTask` | 1.000 | 0.000 | **−1.000** | 1.000 | 0.000 | **−1.000** |
| idfrag_file_tree | `FileTree` | 1.000 | 0.000 | **−1.000** | 0.500 | 0.000 | **−0.500** |
| idfrag_structural_context | `StructuralContext` | 1.000 | 1.000 | 0.000 | 0.500 | 0.500 | 0.000 |
| idfrag_graph_schema_version | `GraphSchemaVersion` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| idfrag_tree_sitter_engine | `TreeSitterEngine` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| **mean** | | **0.556** | **0.333** | **−0.222** | **0.444** | **0.278** | **−0.167** |

None of the four zero-recall tasks improved. Two previously-passing tasks
regressed to zero. Net: **recall −0.222, MRR −0.167** on the corpus this
milestone built specifically to demonstrate the intended improvement.

### Root cause of the regression

Traced (`archex query . "benchmark task" --strategy bm25 --format json`,
`retrieval_metadata.seed_file_paths`) to **fragment collision**: splitting
`symbol_name` on case boundaries makes multiple *related but distinct*
PascalCase symbols expose overlapping fragment sets that used to be
disjoint, unsplit tokens:

- `BenchmarkTask`, `_BenchmarkTaskLike` (a structural-typing `Protocol`),
  `ArchitectureBenchmarkTask`, and `DeltaBenchmarkTask` all decompose to
  include `{"benchmark", "task"}`. Pre-fragmentation, FTS5's native
  `unicode61` tokenizer already splits on underscore (verified empirically:
  `parse_imports` tokenizes to `parse`+`imports` today, no code change
  needed) but never splits camelCase, so `_BenchmarkTaskLike` indexed as one
  opaque token (`benchmarktasklike`) and never competed with `BenchmarkTask`.
  Splitting it newly exposes it — and its siblings — as equally strong
  10×-weighted `symbol_name` matches for the query "benchmark task",
  outranking the one file (`benchmark/models.py`) that used to win outright.
- The same mechanism applies to `FileTree` inside a module with several
  other tree/file-related symbols.

This is not a tuning bug in this implementation — it is the identifier
tokenization technique's own well-known precision/recall tradeoff for a
codebase with a family of derived/related type names (`X`, `XLike`,
`FooX`, `BarX`), which is common in this repository given its dataclass and
protocol-heavy style.
