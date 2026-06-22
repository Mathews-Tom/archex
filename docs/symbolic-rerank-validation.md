# Symbolic-blend reranker — per-corpus validation

Validated evidence for the benchmark-only `archex_query_symbolic_rerank` lane (Workstream L2 of `.docs/2026-06-21-localization-rerank-enhancement-plan.md`). This is the checked-in aggregated result that backs the disposition in `docs/RETRIEVAL_DEFAULT_DECISIONS.md`. The raw per-task benchmark JSONs are local-only (the run directory is gitignored); this aggregated table is the committed evidence.

## Run

- Corpus: `benchmarks/tasks` — 64 tasks (self=24, external-comprehension=19, external-localization=21).
- Cross-encoder pinned to `cross-encoder/ms-marco-MiniLM-L-6-v2`, warmed once in-process; BM25-only clean driver (no vector fusion lanes).
- Lanes: `archex_query` (base), `archex_query_conditional_rerank` (pure cross-encoder A/B baseline), `archex_query_symbolic_rerank` (blend mode, the lane default, alpha=0.5).
- The ambiguity gate fired and the model applied on every task (`cross_encoder_status=applied`). Reranking only reorders the returned set, so recall, required-file recall, and F1 are identical to `archex_query` in every corpus.

Reproduce: warm `CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", allow_remote_code=True).warm()`, assert `loaded_reranker_model_names() == [model]`, then `run_all(Path("benchmarks/tasks"), <local-out-dir>, strategies=[ARCHEX_QUERY, ARCHEX_QUERY_CONDITIONAL_RERANK, ARCHEX_QUERY_SYMBOLIC_RERANK], retrieval_options=BenchmarkRetrievalOptions(rerank_model=model))`.

## Order-metric comparison (base = `archex_query`)

### Self-repo (24 tasks)

| metric | base | conditional (pure CE) | symbolic-blend |
|---|---:|---:|---:|
| recall | 0.8708 | 0.8708 | 0.8708 |
| required_file_recall | 0.8708 | 0.8708 | 0.8708 |
| f1_score | 0.6033 | 0.6033 | 0.6033 |
| mrr | 0.9792 | 0.9062 | 0.9375 |
| ndcg | 0.8792 | 0.8218 | 0.8507 |
| map_score | 0.8217 | 0.7390 | 0.7789 |
| ranked_region_mrr | 0.4745 | 0.3740 (−0.1005) | 0.5229 (+0.0484) |
| ranked_region_ndcg | 0.3567 | 0.2906 | 0.3562 |
| p95_wall_ms | 2446 | 3063 | 2685 |
| guard_fired (total) | — | 0 | 53 |

### External-comprehension (19 tasks)

| metric | base | conditional (pure CE) | symbolic-blend |
|---|---:|---:|---:|
| recall | 0.9474 | 0.9474 | 0.9474 |
| required_file_recall | 0.9474 | 0.9474 | 0.9474 |
| f1_score | 0.6610 | 0.6610 | 0.6610 |
| mrr | 0.8246 | 0.9474 | 0.9386 |
| ndcg | 0.7965 | 0.8717 | 0.8636 |
| map_score | 0.6791 | 0.7746 | 0.7617 |
| ranked_region_mrr | 0.4585 | 0.5037 (+0.0452) | 0.6889 (+0.2304) |
| ranked_region_ndcg | 0.5144 | 0.5776 | 0.6440 |
| p95_wall_ms | 610 | 889 | 786 |
| guard_fired (total) | — | 0 | 16 |

### External-localization (21 tasks)

| metric | base | conditional (pure CE) | symbolic-blend |
|---|---:|---:|---:|
| recall | 0.9524 | 0.9524 | 0.9524 |
| required_file_recall | 0.9524 | 0.9524 | 0.9524 |
| f1_score | 0.3567 | 0.3567 | 0.3567 |
| mrr | 0.6671 | 0.8492 | 0.8333 |
| ndcg | 0.7372 | 0.8758 | 0.8634 |
| map_score | 0.6671 | 0.8492 | 0.8333 |
| ranked_region_mrr | 0.4838 | 0.6782 (+0.1945) | 0.7391 (+0.2554) |
| ranked_region_ndcg | 0.5220 | 0.6703 | 0.6994 |
| p95_wall_ms | 1130 | 1353 | 1373 |
| guard_fired (total) | — | 0 | 68 |

## Reading

- The symbolic-blend lane preserves and exceeds the pure-cross-encoder localization win: external-localization ranked-region MRR `0.7391` ≥ pure-CE `0.6782` (and far above base `0.4838`).
- It reverses the self-repo regression the pure cross-encoder suffers: self ranked-region MRR `0.5229` ≥ base `0.4745`, where pure CE drops to `0.3740` (−0.1005).
- p95 stays at or below `3000 ms` warm in every corpus where the lane fires (self `2685`, external-comprehension `786`, external-localization `1373`).
- The lane never changes the returned set (recall/required-file recall/F1 identical to `archex_query`), so the win lives entirely in ranking quality.
