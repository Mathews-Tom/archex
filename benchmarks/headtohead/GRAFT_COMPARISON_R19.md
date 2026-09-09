# R19 — Pinned Graft graph-memory comparison: result and terminal decision

Protocol: [`benchmarks/preregistrations/R19-graft-graph-memory-comparison.md`](../preregistrations/R19-graft-graph-memory-comparison.md), frozen and merged before the first Graft cell existed. Every number below is re-derivable from the checked-in artifacts:

```bash
uv run python scripts/r19_graft_comparison_analysis.py            # prints the analysis JSON
uv run archex benchmark headtohead competitive --input benchmarks/headtohead/results --format markdown
```

The machine-readable analysis is checked in at [`benchmarks/evidence/r19-graft-graph-memory-comparison.json`](../evidence/r19-graft-graph-memory-comparison.json).

## Identity

| | Value |
| --- | --- |
| Graft released package | `@nanonets/graft@0.16.0` |
| npm integrity (resolved at install) | `sha512-L3E5F1aDYJDCARgfR7O2VaMt8xwO1XNYyHiW2n1WhKnj87gPqoxoZJGNbGXfw6XeA9JSJX3naA36RZ+jDf4AcQ==` |
| Graft source commit | `aa1e2bb0f6326068ac64886da1e67fa25a7804de` (tag `v0.16.0`, equals the package `gitHead`) |
| Graft license | MIT |
| Install | `CI=1 DO_NOT_TRACK=1 npm install @nanonets/graft@0.16.0`; the first attempt died with `ECONNRESET` while a transitive `tree-sitter-cli` install downloaded a platform binary from GitHub release assets, and the retry succeeded in 33s. That is an install-network fact, not a Graft failure. |
| Telemetry | `graft telemetry` reported `off — DO_NOT_TRACK is set in this environment` for the run environment. |
| Runtime | Node v22.23.0, Apple M1 Pro arm64, macOS 25.6.0 (Darwin 25.6.0). |
| Task population | the 19 tasks in `manifest.yaml` `task_subset`, over 15 source repositories. |
| archex control | the `archex_query` artifacts already in `benchmarks/headtohead/results/`. They were **not** regenerated; the graph-memory lane generalization left every pre-existing artifact byte-identical. |

Mode: structural only. `build` ran without `--deep`, so no model was called, no API key was required, and no spend occurred. The graph directory always lived outside the task checkout and `--no-gitignore --no-ignore` were passed, so the checkout the other lanes measure stayed pristine. Every measured query used `ask --no-refresh`; freshness was probed separately with `check --json` and read only from its `graph` section.

## Coverage

38 of 38 planned cells (2 modes × 19 tasks) exist as valid results. There are **0** recorded failures and **0** missing cells. Every cell recorded `status: ok`, and every Graft answer was attributable: each hit named an exact repository path, so no source or path was ever inferred.

Extraction tier, per the frozen protocol: 17 tasks on Graft's native depth tier, 2 tasks (`mini_redis_async`, `rust_tokio_runtime`) on its signature-only WASM breadth tier. Depth-tier and breadth-tier coverage are not presented as equivalent.

## Primary result

Primary metric: **mean required-file recall over the 19 tasks**, treatment `graft_query_warm` minus control `archex_query`. Inference: 10 000-resample bootstrap over the 15 source repositories, seed `20260909`.

| | Value |
| --- | --- |
| `graft_query_warm` mean required-file recall | **0.9298** |
| `archex_query` mean required-file recall | **0.9474** |
| Mean difference (treatment − control) | **−0.0175** |
| 95% cluster-bootstrap CI | **[−0.1228, +0.0635]** |
| Minimum worthwhile gain (+0.05) | **not met** |
| Non-inferiority margin (−0.05) | **not established** (CI lower bound −0.1228 is below the margin) |
| Equivalence margin (±0.03) | **not established** (CI is wider than the margin) |

Only 4 of 19 tasks differ at all: Graft recovers one extra required file on `click_decorators` and `django_middleware` (+0.333 each) and misses one on `pydantic_validators` (−0.333) and two on `celery_task_dispatch` (−0.667). Because inference resamples repositories, the single `celery/celery` cluster carrying the −0.667 delta dominates the interval width.

**The point estimate sits inside the equivalence margin while the interval does not.** With 15 clusters this corpus cannot separate "equivalent" from "worse by more than the non-inferiority margin", so no equivalence claim and no non-inferiority claim is published. This is a precision limit of the frozen population, not a post-hoc reinterpretation: the margins and the clustering unit were fixed before any Graft cell existed and are not widened now.

Pre-declared secondary (exploratory), the same comparison restricted to the 17 depth-tier tasks: treatment 0.9216, control 0.9412, mean difference −0.0196, CI [−0.1333, +0.0714]. Removing breadth-tier coverage does not change the reading.

## Exploratory cells

All exploratory. Both Graft modes answer the same query against the same graph, so their retrieval-quality cells are identical by construction; only the cost columns differ.

| Quantity | `graft_build_plus_query` | `graft_query_warm` |
| --- | --- | --- |
| Recall / required-file recall | 0.9298 | 0.9298 |
| All-required-files-present rate | 0.8421 | 0.8421 |
| Precision | 0.3121 | 0.3121 |
| F1 | 0.4613 | 0.4613 |
| MRR / nDCG / MAP | 0.7939 / 0.7899 / 0.6835 | 0.7939 / 0.7899 / 0.6835 |
| Token efficiency (with completion) | 0.8907 (0.8831) | 0.8907 (0.8831) |
| Mean returned tokens | 2187 | 2187 |
| Cold-start (mean) | 758 ms | 0 ms |
| Warm latency p50 / p95 | 225 / 398 ms | 236 / 392 ms |
| Freshness probe latency (mean, never inside warm latency) | 645 ms | 633 ms |
| Freshness correct | 1.00 | 1.00 |
| Returned units / symbol hits / whole-file hits | 190 / 148 / 42 | 190 / 148 / 42 |
| Returned source units | 148 | 148 |
| Graft query mode | `lexical` in all 19 cells | `lexical` in all 19 cells |

Rank-sensitive cells (MRR, nDCG, MAP) use the emitted `hits` order, never `hits[].score`, which is not monotonically descending in the emitted array.

## Threats to interpretation

- **42 of 190 returned units carry no source.** A whole-file hit is a bare `<path>` pointer with no `code` even under `--source`. It counts toward required-file recall and contributes nothing to the returned-source and token-efficiency cells: the token-efficiency baseline is the content of the files that actually returned source, so a file named only by a whole-file pointer neither adds to the numerator nor inflates the denominator. On `express_error_handling` *all ten* returned units were whole-file hits: that cell's perfect required-file recall is carried entirely by file pointers that returned no source. Graft's token-efficiency cells are within-lane signals, not bundle-for-bundle comparisons against a retrieval lane's returned bundle.
- **`--source` returns crux excerpts, not whole definitions.** The frozen command omits `--full`, so a symbol hit inlines an excerpt (≤8 lines) rather than the full span. The token-efficiency cells describe that pack.
- **The control was measured earlier, on the same machine class but a different run.** The archex artifacts are inputs, not outputs, so cold-start and warm-latency columns are not a same-session latency comparison across tools.
- **Only Graft's lexical query path was exercised.** All 19 cells routed to `mode: lexical`; deep/context-card answers are a paid `--deep` artifact and are out of scope.
- **15 clusters is low precision.** Any future attempt to establish equivalence or non-inferiority on this metric needs a larger clustered population, and would require a new pinned protocol and identity — this one is closed.

## Terminal decision

**R19 is complete with a published null-shaped result: Graft does not earn adoption alongside archex on required-file recall, and this corpus cannot establish equivalence or non-inferiority either.**

- No minimum worthwhile gain: the CI does not clear +0.05, so nothing here justifies a second tool's install, per-repository build, and configuration cost in an orientation workflow.
- No non-inferiority and no equivalence claim: the interval is too wide at 15 clusters.
- Graft is reported as a graph-memory lane, never as a direct retrieval-equivalent winner.
- **No archex retrieval default, ranking signal, strategy, language tier, or MCP tool changes as a result of this comparison, in either direction.** The strategic freeze is untouched.
- The result is descriptive and public. Failed cells would have been retained and counted; none occurred.
