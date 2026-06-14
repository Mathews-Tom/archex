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

## Targeted benchmark experiments

### 2026-06-14 — dual-leg file-stage orchestration

Candidate:

| Candidate | Benchmark flags | Decision role |
| --- | --- | --- |
| Dual-leg control | `--bm25-chunker default --vector-chunker cast --dual-leg-orchestration` | Current benchmark-only control |
| Dual-leg file-stage orchestration | `--bm25-chunker default --vector-chunker cast --dual-leg-orchestration --file-stage-orchestration` | Richer within-query file selection before final chunk packing |

Decision rule:

- Run only the targeted failing families first: `archex_benchmark_gate_lifecycle`, `archex_mcp_query_lifecycle`, `archex_pattern_detection`.
- Keep the candidate only if it improves at least one failing family with no regression on the others.
- Run a wider frontier only after the targeted set clears that bar.

2026-06-14 targeted result: keep the candidate and promote it to the next wider frontier rerun. It preserved or improved recall on every targeted family, recovered `archex_pattern_detection` from `0.5` to `1.0` recall on both fusion paths, and reduced latency materially on all three tasks. Token efficiency improved on `archex_benchmark_gate_lifecycle` and `archex_mcp_query_lifecycle`, but regressed on `archex_pattern_detection`, so this is still a benchmark candidate rather than a default decision.

Targeted smoke comparison:

| Task | Strategy | Recall | F1 | Token efficiency | Latency |
| --- | --- | --- | --- | --- | --- |
| `archex_benchmark_gate_lifecycle` | fusion | `1.0 -> 1.0` | `1.000 -> 1.000` | `0.591 -> 0.698` | `5329 ms -> 797 ms` |
| `archex_benchmark_gate_lifecycle` | fusion+rereank | `1.0 -> 1.0` | `1.000 -> 1.000` | `0.591 -> 0.698` | `9127 ms -> 2992 ms` |
| `archex_mcp_query_lifecycle` | fusion | `0.8 -> 0.8` | `0.800 -> 0.800` | `0.842 -> 0.848` | `325 ms -> 272 ms` |
| `archex_mcp_query_lifecycle` | fusion+rereank | `0.8 -> 0.8` | `0.800 -> 0.800` | `0.842 -> 0.848` | `1159 ms -> 592 ms` |
| `archex_pattern_detection` | fusion | `0.5 -> 1.0` | `0.286 -> 0.571` | `0.873 -> 0.716` | `287 ms -> 256 ms` |
| `archex_pattern_detection` | fusion+rereank | `0.5 -> 1.0` | `0.286 -> 0.571` | `0.873 -> 0.716` | `1133 ms -> 695 ms` |

Operator commands:

```bash
for task in archex_benchmark_gate_lifecycle archex_mcp_query_lifecycle archex_pattern_detection; do
  uv run archex benchmark run --task "$task" --query-fusion --rerank --embedder jina-v2 --bm25-chunker default --vector-chunker cast --dual-leg-orchestration --tasks-dir benchmarks/tasks --output .archex/file-stage-control-smoke
done

for task in archex_benchmark_gate_lifecycle archex_mcp_query_lifecycle archex_pattern_detection; do
  uv run archex benchmark run --task "$task" --query-fusion --rerank --embedder jina-v2 --bm25-chunker default --vector-chunker cast --dual-leg-orchestration --file-stage-orchestration --tasks-dir benchmarks/tasks --output .archex/file-stage-candidate-smoke-v2
done
```

### 2026-06-14 — dual-leg file-stage orchestration full frontier rerun

Follow-up after the targeted pass:

- control: `--bm25-chunker default --vector-chunker cast --dual-leg-orchestration`
- candidate: `--bm25-chunker default --vector-chunker cast --dual-leg-orchestration --file-stage-orchestration`

2026-06-14 full-frontier result: do not adopt the candidate yet. The targeted pattern-detection win did not generalize. The broader rerun improved token efficiency and p95 latency, but it reduced mean F1 on both fusion paths and failed the baseline gate with nine recall regressions.

Frontier summary:

| Strategy | Recall | F1 | Token efficiency | p95 latency |
| --- | --- | --- | --- | --- |
| Control `archex_query_fusion` | `0.834` | `0.648` | `0.708` | `1207 ms` |
| Candidate `archex_query_fusion` | `0.838` | `0.626` | `0.739` | `1099 ms` |
| Control `archex_query_fusion_rerank` | `0.834` | `0.648` | `0.709` | `2341 ms` |
| Candidate `archex_query_fusion_rerank` | `0.828` | `0.620` | `0.738` | `2094 ms` |

Baseline-gate failures:

- `archex_adapter_registry/archex_query_fusion`
- `archex_adapter_registry/archex_query_fusion_rerank`
- `archex_project_init/archex_query_fusion`
- `archex_project_init/archex_query_fusion_rerank`
- `archex_project_reset/archex_query_fusion`
- `archex_project_reset/archex_query_fusion_rerank`
- `archex_query_cache_lifecycle/archex_query_fusion`
- `archex_query_cache_lifecycle/archex_query_fusion_rerank`
- `fastapi_dependency_injection/archex_query_fusion_rerank`

Operator commands:

```bash
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --bm25-chunker default --vector-chunker cast --dual-leg-orchestration --tasks-dir benchmarks/tasks --output .archex/file-stage-control-full
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --bm25-chunker default --vector-chunker cast --dual-leg-orchestration --file-stage-orchestration --tasks-dir benchmarks/tasks --output .archex/file-stage-candidate-full
uv run archex benchmark gate --input .archex/file-stage-candidate-full --baseline .archex/file-stage-control-full --warn-latency-ms 3000
```
