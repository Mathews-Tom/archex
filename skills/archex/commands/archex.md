# /archex

Use archex to gather local codebase context before reading files.

## Arguments

`/archex <question>` uses the current repository. `/archex <path> -- <question>` uses an explicit repository path.

## Procedure

1. Run `archex doctor <path> --format json`.
2. If `index_health` is `error` for an uninitialized or missing index, run `archex init <path>` and `archex index <path>`.
3. If `index_staleness` is `warning`, run `archex index <path>`.
4. Run `archex scout <path> "<question>" --budget 1000 --format json`.
5. Follow `fetch_plan.recommended_strategy`:
   - `chunk_first`: fetch listed `symbol:` and `chunk:` handles with `archex symbol`.
   - `hybrid_fetch`: fetch top handles, then run `archex query` if the fetched context is insufficient.
   - `direct_query`: run `archex query <path> "<question>" --format xml`.
6. Stop when the returned bundle answers the code-context need. Do not run long benchmarks.

## Output

Return the exact archex command outputs or summarize the fetched files, handles, and bundle provenance. Report any failing `archex doctor` check verbatim.
