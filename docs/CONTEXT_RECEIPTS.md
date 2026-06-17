# Context Receipts

Context receipts are the machine-readable provenance block attached to `query`, `scout`, and MCP query/scout responses.

## What a receipt contains

Every receipt includes:

- `query`
- `token_budget.requested` and `token_budget.consumed`; scout receipts use the scout map budget and rendered scout token count
- `index_revision`
- `freshness`
- `freshness_checked_at`, `index_fresh_at`, and `watch_fresh_at` when available; non-watch paths keep unavailable timestamps as `null`
- bounded `returned_context`, `included_edges`, `omitted_edges`, and `skipped_candidates` lists
- `returned_total`, `included_edges_total`, `omitted_edges_total`, and `skipped_total` counts before receipt or renderer caps
- stable `content_hash` values for returned context; scout rows use indexed file hashes when present and do not read files solely to compute hashes
- `context_complete`
- `context_complete_reason`
- `recommended_next_action`

## Freshness

`freshness` is one of:

- `clean`
- `dirty`
- `watch_active`
- `watch_unavailable`
- `unknown`

Current query/scout receipts emit `clean` for the normal refresh path and `unknown` when inline refresh is skipped.

## Completeness

`context_complete` is one of:

- `complete`
- `incomplete`
- `unknown`

`context_complete_reason` is a machine-readable explanation such as:

- `complete`
- `budget_exhausted`
- `dependency_frontier_cut`
- `duplicate_suppressed`
- `no_candidates`
- `stale_index`
- `unsupported_grammar`
- `unknown`

`recommended_next_action` tells the caller what to do next:

- `use_bundle`
- `narrow_query`
- `raise_budget`
- `refresh_index`
- `fetch_skipped_candidate`
- `manual_review`

## Output surfaces

- `archex query --format json` includes `receipt` in the serialized bundle.
- `archex query --format xml` includes a compact `<receipt>` block with shown/total counts.
- `archex query --format markdown` includes a `## Receipt` block with freshness, budget, completeness, shown/total counts, skipped candidates, and omitted dependency edges.
- `archex scout --format json` includes `receipt` in the scout result.
- `archex scout` markdown includes the same compact receipt summary and actionable skipped/frontier details before ranked files.
- MCP `query_repo` and `scout_repo` return `receipt` as a top-level envelope field next to `content` and `_meta`.

## Determinism

Receipt construction is deterministic for the same repository state and query configuration.

- Returned context rows are sorted by file path and line range.
- Included and omitted dependency edges are sorted.
- Skipped candidates are sorted by actionable priority, then by stable path/handle fields.
- `index_revision` is derived from persisted file-state hashes when available.

## Benchmark relationship

The benchmark harness compares receipt completeness claims against required-file ground truth through `receipt_accuracy`. It also records `required_file_recall`, `missed_required_file_rate`, `all_required_files_present`, `task_completion_result`, and `completion_preserved` in benchmark outputs.
