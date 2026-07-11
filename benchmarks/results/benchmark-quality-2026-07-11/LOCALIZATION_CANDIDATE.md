# Localization scope-query candidate

## Candidate and control

The benchmark-only `archex_query_localization_candidate` preserves the product-default `archex_query` code path. For localization tasks only, it issues a deterministic BM25 query built from the task's existing `keywords` and the final component of each declared `include_paths` entry. It does not inspect `expected_files` or `expected_symbols`.

Control and candidate were measured from the same working tree with explicit warmed-cache reuse:

```text
uv run archex benchmark run --tasks-dir benchmarks/tasks --output .archex/benchmark-warm-control-cli --strategy archex_query --warm-cache --no-progress
uv run archex benchmark run --tasks-dir benchmarks/tasks --output .archex/benchmark-warm-candidate-cli --strategy archex_query_localization_candidate --warm-cache --no-progress
```

Each command wrote 64 reports. `--warm-cache` is an explicit benchmark-only retrieval option; product defaults remain unchanged. The candidate is registered only in `AVAILABLE_STRATEGIES`, never `DEFAULT_STRATEGIES`.

## Results

| Measure | Control | Candidate |
| --- | ---: | ---: |
| Django required-file recall | 0.000 | 1.000 |
| Django MRR | 0.000 | 1.000 |
| Django completion-adjusted token efficiency | 0.558 | 0.407 |
| Localization mean required-file recall | 0.952 | 1.000 |
| Localization mean precision | 0.242 | 0.246 |
| Localization mean MRR | 0.667 | 0.849 |
| Localization mean completion-adjusted token efficiency | 0.653 | 0.646 |
| Full-corpus p95 wall time (ms), warmed cache | 423.3 | 292.0 |

The candidate has no required-file-recall regression in any of the 43 non-localization tasks and no negative completion-adjusted token result. Returned-file ordering recovers `django/contrib/auth/validators.py` first for `loc_django_username_validator`.

## Disposition

**Eligible for default-promotion evaluation; not promoted.** The candidate recovers the total miss, improves localization precision/MRR, has no required-file-recall regression, has no negative completion-adjusted token result, and stays within the `3000 ms` warmed product-default p95 budget. It remains benchmark-only until PR 5 establishes an all-green 64-task absolute gate after resolving the remaining independent retrieval failures. No `archex_query` ranking or scoring behavior changes in this PR.