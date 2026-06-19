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

## Compression metadata (benchmark-only)

Returned context rows carry an optional `compression` block, populated only by the benchmark-only `archex_query_compressed` lane after bundle assembly. It is absent (neutral) on uncompressed rows. When present it contains:

- `compression_mode`: `passthrough_required`, `structural_code_elision`, `comment_and_whitespace_slimming`, `large_literal_summarization`, or `json_log_smart_crushing`
- `original_tokens` and `compressed_tokens`
- `compression_ratio`: compressed/original token fraction (1.0 means nothing was removed)
- `original_content_hash`: equals the row's original `content_hash`
- `compressed_content_hash`: hash of the displayed (possibly compressed) content
- `fetch_original_handle`: the exact handle to retrieve the uncompressed region
- `compression_loss_risk`: `none`, `low`, `medium`, or `high`

Compression preserves provenance: the file path, original line range, original content hash, and fetch-original handle stay intact, so a downstream agent can always fetch the exact original source. Compressed regions are clearly marked in markdown output and surfaced in JSON.

Compression is orthogonal to completeness. It never upgrades `context_complete`, and **compression cannot make incomplete context complete**: if retrieval missed required context, compression metadata does not hide or repair that miss. Required, direct, and high-confidence code passes through uncompressed by default.

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

The benchmark harness compares receipt completeness claims against required-file ground truth through `receipt_accuracy`. It also records `required_file_recall`, per-file `missed_required_file_rate`, task-level `missed_required_task_rate`, `all_required_files_present`, `task_completion_result`, `completion_preserved`, `bundle_completion_tokens`, and `token_efficiency_with_completion` in benchmark outputs. The optional `archex benchmark bundle-eval` lane is separate from core gates: it runs only when the operator supplies a local evaluator command, passes that command the rendered bundle and receipt JSON, and reports bundle-only success plus files needed outside returned context/frontier/top candidates.
