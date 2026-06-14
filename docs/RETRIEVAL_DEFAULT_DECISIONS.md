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
