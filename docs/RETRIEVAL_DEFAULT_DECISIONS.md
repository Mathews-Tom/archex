# Retrieval Default Decision Protocol

Operator evidence from the 2026-06-09 retrieval-default benchmark keeps `archex_query` as the product default. CodeRankEmbed and reranker default changes remain blocked until a clean full run clears the recall/F1, token-efficiency, and p95 latency rules.

## Invariants

- Run benchmarks locally only; no network or generative LLM inference.
- Pin exactly one embedder per benchmark run with `--embedder`.
- Compare candidates on recall, F1, token efficiency, median latency, and p95 latency. Recall/F1 alone is not sufficient.
- Keep `archex_query` as the product default; the 2026-06-09 run did not satisfy the strategy switch rule.
- Do not refresh `benchmarks/dogfood_baseline.json` without explicit approval after a proven improvement.

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


## Chunker decision

Candidate chunkers:

| Candidate | Benchmark flag | Decision role |
| --- | --- | --- |
| `default` | `--chunker default` | Current index chunker; keep unless the candidate wins the full frontier |
| `cast` | `--chunker cast` | Recursive AST split + greedy sibling merge candidate |

Switch from `default` to `cast` only when the clean local run shows the same retrieval strategy improving F1, holding recall flat or better, keeping token efficiency at least flat, and staying within the `3000 ms` p95 budget. Chunk count and mean chunk tokens are diagnostic only; they explain the granularity shift but do not override the quality, efficiency, and latency rules.

2026-06-13 rerun after the benchmark metadata fix and cast merge-budget retune: keep `default`. On `archex_query_fusion_rerank`, `default` scored recall `0.867`, precision `0.570`, F1 `0.672`, token efficiency `0.679`, and p95 `3029 ms`; `cast` scored recall `0.877`, precision `0.581`, F1 `0.684`, token efficiency `0.693`, and p95 `3213 ms`. The candidate now wins on quality and token efficiency, and its granularity is materially tighter than the first cast attempt (`1015` mean index chunks at `302.8` tokens vs the prior `769` at `385.7`), but it still misses the `<= 3000 ms` latency budget and remains slower than `default` at p95.

The rerun fixed the earlier measurement defect: `archex_query` rows now report `chunker=cast` with non-zero chunk metadata, and stale cached stores are invalidated on `chunker_revision` mismatches before reuse. Even with that fix, `archex_query` itself still regresses versus `default` on recall (`0.872` vs `0.907`), F1 (`0.669` vs `0.687`), token efficiency (`0.728` vs `0.759`), and p95 (`2231 ms` vs `1723 ms`), so the current BM25-only default path should not change.

The same gate still failed on `11` recall regressions against the default baseline, concentrated in `archex_benchmark_gate_lifecycle`, `archex_mcp_query_lifecycle`, `archex_pattern_detection`, `django_middleware`, `mini_redis_async`, and `rust_tokio_runtime`. `cast` is closer to the default-switch bar than the first run, but it is still not a full-frontier winner.


Follow-up experiment, 2026-06-14: a tighter cast merge budget (`3755` total cast chunks at `279.0` mean tokens on the local repo before benchmark packing, versus the prior `3236` at `323.7`) improved targeted smoke runs on the fusion paths, but failed the full rerun and was not kept. The full cast rerun moved `archex_query_fusion_rerank` to recall `0.883`, precision `0.587`, F1 `0.689`, token efficiency `0.708`, and p95 `3681 ms`; `archex_query` fell further to recall `0.851`, precision `0.555`, F1 `0.656`, token efficiency `0.747`, and p95 `2543 ms`. The gate still failed on `11` baseline regressions, including new `archex_query` losses on `archex_adapter_registry` and `archex_benchmark_gate_lifecycle`.

Interpretation: one global cast granularity is still trading BM25 recall against fusion quality/efficiency. The tighter merge budget helped the vector-assisted paths but degraded the BM25-only path and pushed rerank p95 farther above budget. Keep the earlier cast tuning as the best candidate state on this branch; a materially better result likely needs a different design, such as strategy-specific chunking or a hybrid chunk inventory, not another blind global budget tweak.
Operator commands:

```bash
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --chunker default --tasks-dir benchmarks/tasks --output .archex/e2e-chunk-default
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --chunker cast --tasks-dir benchmarks/tasks --output .archex/e2e-chunk-cast
uv run archex benchmark gate --input .archex/e2e-chunk-cast --baseline .archex/e2e-chunk-default --warn-latency-ms 3000
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
