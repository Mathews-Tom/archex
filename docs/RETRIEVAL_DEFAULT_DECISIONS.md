# Retrieval Default Decision Protocol

Operator evidence from the 2026-06-09 retrieval-default benchmark keeps `archex_query` as the product default. CodeRankEmbed and reranker default changes remain blocked until a clean full run clears the recall/F1, token-efficiency, and p95 latency rules.

## Invariants

- Run core retrieval benchmarks locally only; no network or generative LLM inference. Bundle-only eval is separate and runs only when an operator supplies a local evaluator command.
- Pin exactly one embedder per benchmark run with `--embedder`.
- Compare candidates on recall, required-file recall, missed-required-task rate, receipt accuracy when available, token efficiency after completion, F1, median latency, and p95 latency. Recall/F1 alone is not sufficient, and raw token efficiency is not sufficient when completion penalty cancels the savings.
- Keep `archex_query` as the product default; the 2026-06-09 run did not satisfy the strategy switch rule.
- Do not refresh `benchmarks/dogfood_baseline.json` without explicit approval after a proven improvement.


## Candidate region and context-efficiency gate inputs

The benchmark eval frontier adds optional retrieval-quality signals that are available as **candidate** gate inputs for tasks that declare `expected_regions`: region recall, line recall, ranked-region MRR/nDCG, context noise ratio, and relevance per 1k tokens. The gate exposes optional thresholds (`min_region_recall`, `min_line_recall`, `max_context_noise_ratio`, `min_relevance_per_1k_tokens`).

These inputs do not change the current switch rule. They are optional and ignored when a task has no region labels, and they enforce a threshold only when the label exists. No region/context-efficiency values are claimed here; any future numeric claim must come from checked-in benchmark artifacts that contain the region fields. Until a region-labelled run is published, treat these signals as descriptive, not as a passing or failing default-switch criterion.

## Task-aware candidate lane

`archex_query_task_aware` is a benchmark-only retrieval lane (Workstream 2 of `.docs/2026-06-18-retrieval-quality-token-efficiency-enhancement-design.md`). It classifies query modality (`pl_to_pl`, `nl_to_pl`, `mixed`) and budget tier (`tight`, `standard`, `large`) and routes sparse-vs-dense retrieval accordingly: BM25-first for code-heavy and mixed queries with a single confidence-gated dense expansion, and a bounded hybrid pass for natural-language queries. Confidence-aware fusion runs only on the hybrid/dense pass, the cross-encoder reranker is never run, and total work is bounded to at most two retrieval passes.

Decision rule: this lane is **not eligible for product-default promotion** unless a clean warm run satisfies the full `Product strategy decision` rule below — mean F1 at least `0.05` higher than `archex_query`, token efficiency after completion at least as high, and p95 latency at or below the `3000 ms` budget — **and** does not regress required-file recall or the region/context-efficiency signals where region labels exist. Recall or token-efficiency gains alone do not qualify, and missing the p95 budget disqualifies it regardless of quality gains. It is evaluated on the metrics and thresholds the default-switch rule already uses, plus the region and context-efficiency signals above.

No numeric results are claimed here. Any future promotion claim must come from checked-in benchmark artifacts that contain the file-level, region, token-efficiency, and p95 fields needed to verify it.

Warm-run note: the lane runs vector retrieval only on its conditional hybrid/dense pass, so it is not a member of the benchmark vector warm sets. A bare `--strategy archex_query archex_query_task_aware` run would therefore time the lane's first vector pass cold (embedder load plus vector-index build) while `archex_query` stays BM25-only, inflating the lane's p95. Before comparing p95, warm the shared vector store the lane reuses — for example run the comparison once and discard it, or include `--query-fusion` so the existing fusion warm-up populates the store — then rerun against the warmed cache.

Disposition (local self-repo evaluation): a routing-exercising self-repo task set was added (`benchmarks/tasks/routing_*.yaml`: identifier-dense `pl_to_pl` lookups, a path/symbol query, a stack trace, mixed issues, and `tight`/`large` budget variants) and run warm against `archex_query` and `archex_query_fusion`. Strategy provenance confirms the routing is enacted: code-heavy queries with a confident top BM25 hit stay BM25-only (`dense_expansion=skipped:confident_sparse`), while diffuse or low-confidence queries escalate to one bounded dense pass (`dense_expansion=ran`), so the lane is genuinely distinct from `archex_query_fusion` rather than a duplicate of it. On that set the lane tied `archex_query` on file recall and F1 (no regression, no improvement), showed a small token-efficiency edge over always-fusion but stayed below the BM25 default, and its conditional second pass can cost more latency than single-pass fusion when it escalates (both well under the p95 budget). Per the rule above it is therefore **not eligible for promotion** and remains benchmark-only. No measured wins are claimed; the numbers depend on the local working tree and are not checked in. Reproduce warm with: `uv run archex benchmark run --tasks-dir benchmarks/tasks --self-only --query-fusion --strategy archex_query_task_aware --allow-remote-code --output .archex/ta-measured --no-progress` followed by `uv run archex benchmark report --input .archex/ta-measured --format markdown`.

## Compression candidate lane

`archex_query_compressed` is a benchmark-only lane (Workstream 3 of `.docs/2026-06-18-retrieval-quality-token-efficiency-enhancement-design.md`). It runs the exact `archex_query` retrieval and packing path, then applies deterministic, low-risk compression after bundle assembly. Retrieval is untouched, so retrieval metrics stay attributable to the uncompressed retrieval set and `archex_query` behavior is unchanged. Compression modes are deterministic text transforms only (passthrough of required/direct/high-confidence regions, structural code elision, comment/whitespace slimming, large-literal summarization, JSON/log crushing); there are no hosted calls, local model calls, or semantic summarization.

Required, direct, and high-confidence code passes through uncompressed by default, and `protect_code` disables code elision for fix/debug/review intents. Every compressed region exposes its original line range, original content hash, and an exact fetch-original handle, so the original source is always retrievable.

Decision rule: this lane is **not eligible for product-default promotion** and does not change the switch rule. Compression can improve token efficiency only if required-file/region metrics and receipt accuracy do not regress. Crucially, **compression cannot make incomplete context complete**: it never upgrades `context_complete`, and `compression_hidden_required_region_count` (required regions hidden by compression) is reported separately from retrieval misses and must stay `0`.

No numeric compression results are claimed here. Any future compression-win claim must come from checked-in benchmark artifacts that contain the `bundle_tokens_uncompressed`/`bundle_tokens_compressed`, `bundle_compression_ratio`, required-context passthrough, and completion-adjusted token-efficiency fields needed to verify it.

Disposition (local evaluation): the lane was run warm against both the self-repo task set and the external-repo task set in `benchmarks/tasks`, comparing `archex_query` and `archex_query_compressed`. Across every task the two strategies produced identical retrieval metrics (recall, required-file recall, F1, result files, and token efficiency after completion), confirming retrieval is unaffected. The safety invariants held on every task: `compression_hidden_required_region_count` and `required_context_compressed_tokens` were `0` — no required/high-confidence region was ever compressed or hidden. Measured compression was modest, because most returned regions are high-confidence seed content that passes through and only the lower-confidence frontier tail compresses, so it provided no token-efficiency improvement large enough to justify promotion. Per the rule above the lane stays **benchmark-only and not promotion-eligible**. No measured wins are claimed; the numbers depend on the local working tree and external clones and are not checked in. Reproduce with `uv run archex benchmark run --tasks-dir benchmarks/tasks --strategy archex_query --strategy archex_query_compressed --output .archex/compression-full --no-progress` followed by `uv run archex benchmark report --input .archex/compression-full --format markdown`.

## Retrieval-aware packing candidate lane

`archex_query_efficiency_packed` is a benchmark-only lane (Workstream 4 of `.docs/2026-06-18-retrieval-quality-token-efficiency-enhancement-design.md`). It runs the exact `archex_query` retrieval path, then re-packs the assembled bundle with a deterministic, relevance-per-token packer: each retrieved region is scored from intrinsic signals only (retrieval score, direct path/symbol match, graph distance and edge confidence, token count, compression eligibility and loss risk, scout/fetch handle priority, and remaining budget — never benchmark ground truth). Direct/high-confidence targets are preserved before optional context, smaller enclosing evidence is preferred over whole-file context except at large budgets or files with multiple high-confidence regions, low-risk regions may be compressed, and large low-score graph-distant context is anchored or skipped. The packed bundle is the lane's returned context; the product default and `DEFAULT_STRATEGIES` are unchanged.

Packing provenance records the include/compress/elide/skip decision counts, the budget tier, and the delivered relevance-per-token; compressed and elided regions keep their original content hash and an exact fetch-original handle so the source stays retrievable. Reports compare normal packing (`archex_query`), compressed packing (`archex_query_compressed`), and efficiency-aware packing on token efficiency after completion, region quality where labels exist, compression ratio, and p95 latency.

Decision rule: this lane is **not eligible for product-default promotion** and does not change the switch rule. It becomes eligible for default consideration only if a clean run shows that required-file and region/line metrics (where labels exist) do not regress, token efficiency after completion improves, and p95 latency does not regress versus `archex_query` — in addition to the full `Product strategy decision` rule below. Because packing changes which regions earn budget, `compression_hidden_required_region_count` (required regions hidden by compression or elision) is reported and must stay `0`; packing can never make incomplete context complete.

No numeric packing results are claimed here. Any future packing-win claim must come from checked-in benchmark artifacts that contain the packing decision counts, `packed_relevance_per_1k_tokens`, `bundle_compression_ratio`, required-file/region, and completion-adjusted token-efficiency and p95 fields needed to verify it.

## Decision rationale

### Why this gate exists

Earlier retrieval-default evaluation mixed embedders and lacked token-efficiency and p95 latency gates. That made recall-only wins unsafe to promote into the product default. The default path has to preserve quality, token economy, and interactive latency together.

### Alternatives considered

#### Switch the product default to `archex_query_fusion_rerank`

- Pros: highest observed MRR on the clean Jina run (`0.938`) and a small F1 lift (`0.594` vs `0.589`).
- Cons: token efficiency regressed (`0.612` vs `0.701`) and p95 latency rose to `16588 ms`, far above the `3000 ms` budget.
- Disposition: rejected. It failed every non-F1 switch constraint and did not clear the required `+0.05` mean F1 delta.

#### Switch the benchmark embedder to CodeRankEmbed

- Pros: remains a plausible code-specialized candidate after the query-prefix and repeated-load fixes.
- Cons: the 2026-06-09 CodeRank run completed only `28/35` tasks due to clone DNS failures and showed an extreme partial-run p95 (`253658 ms` for fusion rerank).
- Disposition: rejected as decision evidence. The run was not clean full-frontier evidence and underperformed Jina on recall, F1, token efficiency, and p95 even on the partial set.

#### Select MiniLM as the reranker default

- Pros: materially faster than Jina reranker v3 on the same run (`3924 ms` p95 vs `16522 ms` p95).
- Cons: still misses the `<= 3000 ms` p95 budget and slightly lowers F1 (`0.586` vs `0.594`).
- Disposition: rejected. The reranker decision rule requires the selected model to stay at or below the p95 budget on the operator hardware.

### Consequences

- Default changes remain evidence-gated across quality, token economy, and latency.
- `archex_query` stays the product default until a clean warm run clears the whole rule set.
- Benchmark-only knobs such as `--embedder` and `--rerank-model` do not change shipped product behavior by themselves.
- CodeRankEmbed and lighter reranker candidates remain valid future experiments, but only after clean reruns.
## Embedder decision

Candidate embedders:

| Candidate | Benchmark flag | Rollback path |
| --- | --- | --- |
| Jina v2 code embeddings | `--embedder jina-v2` | Current benchmark default |
| CodeRankEmbed | `--embedder coderank` | Registered as `coderank`; no product default flip before approval |

Switch from Jina v2 to CodeRankEmbed only when a warm full run shows `archex_query_fusion_rerank` recall and F1 improve, token efficiency is not worse, and p95 latency does not regress.

2026-06-09 result: do not switch. The CodeRank run completed only `28/35` tasks because seven GitHub clones failed DNS resolution, so it is invalid as final A/B evidence. Even on the partial set, CodeRank fusion rerank underperformed Jina fusion rerank on recall (`0.774` vs `0.818`), F1 (`0.590` vs `0.594`), token efficiency (`0.597` vs `0.612`), and p95 latency (`253658 ms` vs `16588 ms`).

Operator commands:

```bash
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --tasks-dir benchmarks/tasks --output .archex/e2e-jina
uv run archex benchmark run --query-fusion --rerank --embedder coderank --tasks-dir benchmarks/tasks --output .archex/e2e-coderank
uv run archex benchmark readiness --input .archex/e2e-jina --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark readiness --input .archex/e2e-coderank --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
```

## Reranker decision

Candidate rerankers:

| Candidate | Benchmark flag | Decision role |
| --- | --- | --- |
| Jina reranker v3 on detected device | omit `--rerank-model` | Current default reranker |
| MiniLM cross-encoder | `--rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2` | Lighter fallback if Jina exceeds the p95 budget |
| TinyBERT cross-encoder | `--rerank-model cross-encoder/ms-marco-TinyBERT-L-2-v2` | Lighter candidate to test after MiniLM missed p95 |

Keep the highest-quality reranker that holds p95 latency at or below `3000 ms` on the operator's hardware. Evaluate Jina first with MPS device selection, then MiniLM, then TinyBERT only if heavier candidates remain over budget.

2026-06-09 result: do not change the reranker default. Jina reranker p95 was `16522 ms`, and MiniLM p95 was `3924 ms`; both miss the `<= 3000 ms` budget. MiniLM is the better latency candidate observed so far, but it still needs further tuning or a lighter local reranker before it can be selected. The next local candidate is TinyBERT.

Before comparing reranker p95, run the Jina command once as a cache warm-up and discard that output, then rerun both candidates against the warmed index cache.

Operator commands:

```bash
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --tasks-dir benchmarks/tasks --output .archex/e2e-rerank-jina
uv run archex benchmark readiness --input .archex/e2e-rerank-jina --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark gate --input .archex/e2e-rerank-jina --warn-latency-ms 3000
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 --tasks-dir benchmarks/tasks --output .archex/e2e-rerank-minilm
uv run archex benchmark readiness --input .archex/e2e-rerank-minilm --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark gate --input .archex/e2e-rerank-minilm --warn-latency-ms 3000
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --rerank-model cross-encoder/ms-marco-TinyBERT-L-2-v2 --tasks-dir benchmarks/tasks --output .archex/e2e-rerank-tinybert
uv run archex benchmark readiness --input .archex/e2e-rerank-tinybert --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark gate --input .archex/e2e-rerank-tinybert --warn-latency-ms 3000
```

## Product strategy decision

Candidate product defaults:

| Candidate | Meaning | Decision role |
| --- | --- | --- |
| `archex_query` | Current BM25 + graph product path | Default to keep unless beaten on the full frontier |
| `archex_query_fusion_rerank` | BM25 + vector fusion + cross-encoder rerank | Candidate only if quality and latency both clear the rule |

Switch to `archex_query_fusion_rerank` only if the clean warm run shows mean F1 at least `0.05` higher than `archex_query`, token efficiency at least as high as `archex_query`, and p95 latency at or below `3000 ms`. If the rule does not pass, keep `archex_query` as the product default.

2026-06-09 result: keep `archex_query`. On the clean Jina run, `archex_query` had F1 `0.589`, token efficiency `0.701`, and p95 `2186 ms`; `archex_query_fusion_rerank` had F1 `0.594`, token efficiency `0.612`, and p95 `16588 ms`. The F1 delta was only `+0.005`, token efficiency regressed, and p95 exceeded the `3000 ms` limit.

Operator commands:

```bash
uv run archex benchmark readiness --input .archex/e2e-jina --tasks-dir benchmarks/tasks --strategy archex_query --format markdown
uv run archex benchmark readiness --input .archex/e2e-jina --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark gate --input .archex/e2e-jina --baseline .archex/e2e-tier2 --warn-latency-ms 3000
```

## C6 chunking and retrieval experiment archive

### 2026-06-15 — stack disposition after C6 validation

Stable benchmark spine:

- `feat/cast-chunker` / PR `#212`
- `feat/chunker-benchmark-arm` / PR `#213`

Archived benchmark-only experiment branches:

| Branch / PR | Experiment | Result |
| --- | --- | --- |
| `feat/adaptive-rerank-benchmark-policy` / `#214` | adaptive rerank candidate limit | Safe benchmark knob. No recall recovery. |
| `feat/adaptive-fusion-benchmark-policy` / `#215` | adaptive fusion policy | No targeted recall win. Latency regressed. |
| `feat/dual-leg-benchmark-orchestration` / `#216` | BM25 default + vector cast dual-leg retrieval | Efficiency improved. Recall did not. |
| `feat/file-first-benchmark-ranking` / `#217` | file-first ranking contract | No recall recovery. |
| `feat/file-stage-benchmark-orchestration` / `#218` | file-stage orchestration | Strong targeted wins. Broader frontier regressed. |
| `feat/expansion-controls-benchmark-policy` / `#219` | strict expansion controls | Helped `fastapi_dependency_injection`. Did not fix the main self-repo failures. |
| `feat/direct-file-preservation-benchmark-policy` / `#220` | direct file preservation | Best targeted recovery so far. Still lost to the stable spine. |
| `feat/delta-vector-cache-preservation-policy` / `#221` | delta/vector-cache preservation | Beat its parent branch. Not enough to beat the stable spine after consolidation. |
| `feat/retrieval-policy-consolidation` / `#222` | consolidated retrieval policy line | Improved recall and token efficiency. Failed the stable-spine gate on F1 and latency. |

Final disposition:

- Merge only the durable infrastructure line: `#212` then `#213`.
- Keep `archex_query` as the product default.
- Keep the current chunker default unchanged.
- Close or park `#214` through `#222` as archived benchmark evidence.

Future work starts from the stable benchmark spine, not the draft ladder. The next narrow investigation set is:

- `archex_project_index`
- `archex_project_init`
- `express_error_handling`
- `fastapi_dependency_injection` rerank
