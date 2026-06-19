# archex vs. cocoindex-code

This page compares archex with cocoindex-code for local agent code-context workflows. It uses the accepted C1 head-to-head operator report and checked-in raw result artifacts under `benchmarks/headtohead/results/`. It does not re-measure benchmarks.

## Evidence sources

| Source | What it supports |
| --- | --- |
| `uv run archex benchmark headtohead report --input .archex/headtohead --format markdown` | C1 aggregate cells recorded in the operator report: `archex` vs `ccc` vs `raw-ripgrep/read`, manifest `archex-vs-ccc-c1-public`, 19 external-repo tasks. |
| `benchmarks/headtohead/results/manifest.yaml` | Same-task manifest, local-only archex lane, ccc `0.2.35` lane, and ccc bootstrap commands `ccc init -f` plus `ccc index`. |
| `archex doctor . --format json` | Local trust checks for index health, staleness, local model cache presence, grammar availability, MCP registration, and `.archex/` disk usage. |
| `docker build -f docker/Dockerfile.slim .` and `docker build -f docker/Dockerfile.full .` | Slim BM25-only image and full local-embedding image definitions. |
| `archex scout . "question" --budget 1000 --format json` | Scout map and fetch-plan protocol used by the Claude Code skill. |

## Measured C1 results

Every metric below is copied from the accepted C1 report for manifest `archex-vs-ccc-c1-public` with 19 external-repo tasks. Higher is better for recall, required-file recall, precision, F1, token efficiency, and efficiency after completion. Lower is better for missed task rate, completion penalty tokens, warm latency, and cold-start. Receipt accuracy is `n/a` for this historical run because those artifacts predate receipt capture.

| Lane | Recall | Required-file recall | Missed task rate | Precision | F1 | Token efficiency | Completion penalty tokens | Efficiency after completion | Warm latency ms | Cold-start ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| archex | 0.95 | 0.95 | 0.16 | 0.51 | 0.66 | 0.76 | 922 | 0.74 | 408 | 0 |
| ccc | 0.32 | 0.32 | 0.79 | 0.36 | 0.31 | 0.48 | 11,188 | 0.41 | 521 | 4,721 |
| raw-ripgrep/read | 1.00 | 1.00 | 0.00 | 0.03 | 0.05 | 0.00 | 0 | 0.00 | 773 | 0 |

Citations: each cell above is the corresponding C1 report cell emitted by `uv run archex benchmark headtohead report --input benchmarks/headtohead/results --format markdown`; the raw per-task artifacts are tracked in `benchmarks/headtohead/results/*.json`.

## Losing cells and roadmap coverage

| Losing cell | Current result | Roadmap item that addresses it | Evidence |
| --- | --- | --- | --- |
| Recall vs raw-ripgrep/read | archex `0.95`; raw-ripgrep/read `1.00` | C5 scout protocol and the retrieval evidence gate keep improving recall without copying whole files into context. | C1 recall cells; scout command `archex scout . "question" --budget 1000 --format json`. |
| Required-file misses vs raw-ripgrep/read | archex missed-task rate `0.16`; raw-ripgrep/read `0.00` | Required-file miss gates now keep safe-to-act quality visible beside token efficiency. | C1 `missed_required_task_rate` cells; benchmark gate fields. |
| Warm latency vs raw-ripgrep/read | archex `408 ms`; raw-ripgrep/read `773 ms` | Raw-ripgrep/read is exhaustive but reads very broad matches; C2 freshness/warm-MCP work still narrows indexed warm-path latency. | C1 warm-latency cells; MCP warm command `archex mcp --watch --watch-path .`. |
| Completion penalty vs raw-ripgrep/read | archex `922`; raw-ripgrep/read `0` | Raw-ripgrep/read pays by reading broad context up front; archex tracks completion penalty so missing context remains visible. C5 handle fetch reduces second-pass misses. | C1 completion-penalty cells; scout `fetch_plan` handles. |

## Capability matrix

| Capability | archex | cocoindex-code / ccc | Evidence |
| --- | --- | --- | --- |
| Same-task retrieval quality | Higher precision, F1, token efficiency, and efficiency after completion in the accepted C1 run, with `0.95` required-file recall and `0.16` missed-task rate. | Lower aggregate recall, required-file recall, F1, and efficiency after completion in the same run. | C1 report cells: archex `0.95/0.95/0.16/0.66/0.74`; ccc `0.32/0.32/0.79/0.31/0.41`. |
| Context assembly | Returns a token-budgeted context bundle with provenance and structured renderers. | Returns search hits; the benchmark adds completion penalty tokens and missed-required-file/task rates for missing expected files. | Command `archex query . "question" --format xml`; C1 completion penalty cells: archex `922`, ccc `11,188`. |
| First-run trust | `archex doctor` checks index health, staleness, model cache, grammars, MCP registration, and `.archex/` disk usage. | ccc bootstrap in the C1 manifest uses `ccc init -f` and `ccc index`; no archex-equivalent doctor is measured in C1. | Commands `archex doctor . --format json`, `ccc init -f`, and `ccc index`. |
| Agent onboarding | In-repo Claude Code skill plus `/archex` command teach auto-init, doctor, MCP wiring, and scout→fetch. | Existing onboarding path includes `npx skills add cocoindex-io/cocoindex-code` and plugin-marketplace distribution. | Commands/files: `skills/archex/SKILL.md`, `skills/archex/commands/archex.md`, `npx skills add cocoindex-io/cocoindex-code`. |
| Container distribution | Slim BM25-only image and full local-embedding image; persistent-container MCP pattern documented. | Existing distribution includes Docker slim/full images. | Commands `docker build -f docker/Dockerfile.slim .`, `docker build -f docker/Dockerfile.full .`, and `docker exec -i archex-mcp archex mcp`. |
| Local model posture | Slim path uses BM25 only; full path uses local FastEmbed. Hosted/generative inference is not required. | C1 ccc lane used local Snowflake embeddings; the broader cocoindex-code surface also supports cloud embedding providers. | `docker/Dockerfile.slim`, `docker/Dockerfile.full`, and C1 manifest ccc embedder `Snowflake/snowflake-arctic-embed-xs`. |
| Freshness visibility | Query and MCP paths expose refresh metadata; doctor reports stale, dirty, and missing-index states. | C1 manifest measures ccc bootstrap and warm search latency, not edit-to-correct freshness. | Commands `archex status --strict`, `archex doctor . --format json`; C1 warm/cold cells. |
| Language breadth | archex reports declared grammar availability by tier through doctor. | cocoindex-code advertises broader language coverage in its distribution story. C3 is the archex roadmap item for breadth parity. | Command `archex doctor . --format json` grammar check; roadmap item C3 in the competitive plan. |

## Selection, not compression: where Headroom fits

archex decides what context to retrieve: files, symbols, chunks, dependency neighborhoods, and token-budgeted bundles. Headroom-style layers compress context after it has already been gathered. archex is a retrieval and context-selection system; Headroom is a compression / context-management layer, not a retrieval engine, so this page never reports Headroom as a retrieval competitor. They are composable, not competing: archex selects relevant context first, then a compression layer can reduce the residual payload. Compressed irrelevant context is still irrelevant.

The competitive harness models the layer difference explicitly with two Headroom modes:

- `headroom_only_on_raw_context` — raw or broad context is compressed with no archex selection, testing whether compression alone can rescue broad context.
- `archex_plus_headroom` — archex selects context first, then Headroom compresses supported payloads after selection, testing composability. Source/RAG code is passed through (protected) by default so edit-critical lines are not hidden.

Compression lanes contribute a compression ratio only; retrieval-quality columns (recall, required-file recall, precision) do not apply to them and are reported as `n/a`.

## Broader competitive comparison

`archex benchmark headtohead competitive --input benchmarks/headtohead/results --format markdown` renders a richer comparison than the C1 table above, grouped by repo/task family and aggregate, with no aggregate-only winner claim. Lanes:

- `archex` — the current product default (`archex_query`).
- Benchmark-only archex candidate lanes (`archex_query_compressed`, `archex_query_efficiency_packed`) when their per-task artifacts are present. These stay benchmark-only candidates: per [Retrieval Default Decisions](RETRIEVAL_DEFAULT_DECISIONS.md) they have not cleared the default-switch gate, so they are not the shipped default.
- `ccc` — cocoindex-code, a retrieval engine, version-pinned in the manifest.
- raw-ripgrep/read baseline.
- Headroom compression lanes (`headroom_only_on_raw_context`, `archex_plus_headroom`) — a compression layer, not a retrieval engine, applied locally where reproducible or imported as version-pinned operator artifacts.

Reported dimensions: cold-start, warm p50/p95 latency, recall, required-file recall, region/line recall where labeled, context precision/noise ratio, token efficiency after completion, compression ratio, receipt accuracy, freshness, plus an operational table (local/offline posture, setup steps, embedder). Every numeric value in a published comparison comes from checked-in artifacts under `benchmarks/headtohead/results/`. The current checked-in set covers the `archex`, `ccc`, and raw-ripgrep/read lanes (the C1 cells above); improved-archex-candidate and Headroom-layer cells appear only when an operator supplies the corresponding artifacts.

## Practical choice

Use archex when the task needs a local, inspectable, token-budgeted bundle with provenance and architecture context. Use cocoindex-code when its broader language coverage or existing marketplace distribution matters more than measured bundle quality. Use raw-ripgrep/read when exhaustive recall is worth reading substantially more context by hand.
