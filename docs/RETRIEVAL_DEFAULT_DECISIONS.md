# Retrieval Default Decision Protocol

Tier 3 decisions stay pending until the operator provides clean warm-cache benchmark evidence. Product defaults remain unchanged until the switch rules in this document pass and the operator approves the change.

## Invariants

- Run benchmarks locally only; no network or generative LLM inference.
- Pin exactly one embedder per benchmark run with `--embedder`.
- Compare candidates on recall, F1, token efficiency, median latency, and p95 latency. Recall/F1 alone is not sufficient.
- Keep `archex_query` as the product default until the final strategy switch rule passes.
- Do not refresh `benchmarks/dogfood_baseline.json` without explicit approval after a proven improvement.

## Embedder decision

Candidate embedders:

| Candidate | Benchmark flag | Rollback path |
| --- | --- | --- |
| Jina v2 code embeddings | `--embedder jina-v2` | Current benchmark default |
| CodeRankEmbed | `--embedder coderank` | Registered as `coderank`; no product default flip before approval |

Switch from Jina v2 to CodeRankEmbed only when a warm full run shows `archex_query_fusion_rerank` recall and F1 improve, token efficiency is not worse, and p95 latency does not regress.

Operator commands:

```bash
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --tasks-dir benchmarks/tasks --output .archex/e2e-jina
uv run archex benchmark run --query-fusion --rerank --embedder coderank --tasks-dir benchmarks/tasks --output .archex/e2e-coderank
uv run archex benchmark readiness --input .archex/e2e-jina --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark readiness --input .archex/e2e-coderank --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
```
