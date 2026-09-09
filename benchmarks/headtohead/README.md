# Head-to-head benchmark harness

This directory contains the pinned C1 public comparison manifest for running the same external-repo tasks across archex, cocoindex-code (`ccc`), and the raw-ripgrep/read baseline. The harness records cold-start timing, warm query latency, recall, precision, F1, token efficiency, required-file recall, missed-file/task rates, all-required-present rate, receipt accuracy, and bundle-completion penalty tokens for every lane.

Run only by an operator outside this implementation session:

```bash
uv tool install cocoindex-code  # operator choice: [full] for local embeddings
uv run archex benchmark headtohead run --manifest benchmarks/headtohead/manifest.yaml --output .archex/headtohead
uv run archex benchmark headtohead report --input .archex/headtohead --format markdown
```

Checked-in public artifacts live in `benchmarks/headtohead/results/`. Report them with:

```bash
uv run archex benchmark headtohead report --input benchmarks/headtohead/results --format markdown
```

Publication rule: paste the report back unchanged and copy the result artifacts into `benchmarks/headtohead/results/` regardless of which tool wins each cell. Keep `.archex/` and `.docs/` out of git.

## Competitive comparison report

The competitive report is a richer view of the same artifacts, grouped by repo/task family and aggregate with no aggregate-only winner claim:

```bash
uv run archex benchmark headtohead competitive --input benchmarks/headtohead/results --format markdown
```

It models lanes by `layer_type` (`retrieval`/`graph-memory`/`compression`/`baseline`) so graph/memory and compression layers are never presented as direct retrieval engines. The checked-in public artifact set now includes the benchmark-only archex candidate lanes (`archex_query_compressed`, `archex_query_efficiency_packed`), both Graphify follow-up lanes, and the original `archex` / `ccc` / raw-ripgrep/read lanes. Headroom-style compression lanes appear in the public report when operator artifacts are present.

## Graph-memory lanes

Graphify is modeled under `graphify_lanes`, not `external_tools`. The checked-in public set uses two explicit lanes pinned to `graphifyy 0.8.44`:

- `graphify_build_plus_query` — includes the per-task graph build/setup cost plus the first graph-backed answer.
- `graphify_query_warm` — prebuilds the graph first, then reports only the warm graph-query path.

Both lanes point at checked-in artifact directories (`benchmarks/headtohead/results/graphify_build_plus_query/` and `benchmarks/headtohead/results/graphify_query_warm/`). Each task artifact is `<artifact_dir>/<task_id>.json` and carries the exact numeric fields claimed in docs plus the sanitized Graphify command shape, pinned package/version, and build-vs-warm semantics.

Local reproduction of one lane uses the adapter contract introduced in `scripts/run_graphify_headtohead_lane.py`: it reads the PR2 stdin payload (`task`, `repo_path`, `lane`, `graphify`) and emits one artifact JSON on stdout. The checked-in public artifacts were produced with that script and then copied into the two directories above. No new public claim should be added unless the corresponding Graphify artifact JSON exists in git.

Graphify token-efficiency cells in the public reports count the graph reference listing returned by `graphify query`, not returned source code. They are useful as within-lane efficiency signals, but they are not bundle-for-bundle comparisons against archex or `ccc`.

A graph-memory lane is any tool whose product is a persistent code graph queried after a build step, so the comparison must keep build cost and warm query cost separate instead of collapsing them into one latency number. Graphify is currently the only graph-memory tool with checked-in artifacts, and the lane family is still named after it in code: the manifest key is `graphify_lanes` and the lane names are fixed to `graphify_build_plus_query` and `graphify_query_warm`. The information a lane actually carries is not Graphify-specific — tool identity, pinned released version, whether the cell includes build cost, and the artifact directory holding one JSON per task — so R19 generalizes the naming and reuses that contract for a second tool rather than adding a parallel lane family. Adding a second tool therefore requires that generalization first; it does not slot into `graphify_lanes` unchanged.

### Pinned Graft comparison protocol (pre-registered, not yet run)

Graft is the second graph-memory tool queued for this harness. Its protocol is frozen in [`benchmarks/preregistrations/R19-graft-graph-memory-comparison.md`](../preregistrations/R19-graft-graph-memory-comparison.md), which merges before any Graft cell is generated. **No Graft lane, artifact, or number exists yet**; this section records the frozen protocol so the commit order proves the protocol predates the data.

The pin is the released package `@nanonets/graft@0.16.0` (npm integrity `sha512-L3E5F1aDYJDCARgfR7O2VaMt8xwO1XNYyHiW2n1WhKnj87gPqoxoZJGNbGXfw6XeA9JSJX3naA36RZ+jDf4AcQ==`, MIT), whose `gitHead` `aa1e2bb0f6326068ac64886da1e67fa25a7804de` is the commit tagged `v0.16.0`. A source checkout declaring an unpublished version is not a released artifact and is not pinnable.

The measured protocol, in short: structural mode only (`build` without `--deep`, so no API key and no spend); the graph directory outside the task repository with `--no-gitignore --no-ignore`, so the checkout the other lanes measure stays pristine; `ask --no-refresh` for every measured query, because a default `ask` silently repairs graph drift and would fold synchronization into query latency; freshness read only from `check --json`'s `graph` section, since context cards are a `--deep` artifact and are always absent here; rank taken from the emitted `hits` order rather than `hits[].score`, which is not monotonically descending; `-n 10` to match the external retrieval lane's frozen limit; a per-task extraction tier, because Graft covers the two Rust tasks through a signature-only WASM tier rather than its native tier; and `DO_NOT_TRACK=1` plus `CI=1` on install and every invocation.

Graft, like Graphify, will be reported as a graph-memory lane and never as a direct retrieval-equivalent winner, and no result from it may change an archex retrieval default.

## Headroom is a compression layer, not a retrieval engine

Headroom is modeled under `compression_layers`, not `external_tools`. Each layer pins an exact version and one or both modes: `headroom_only_on_raw_context` (compress raw/broad context with no archex selection) and `archex_plus_headroom` (compress an archex-selected bundle after selection, protecting source/RAG code by default). Compression lanes contribute only a compression ratio; retrieval-quality columns are `n/a`.

Set `compression_layers[].artifact_dir` to import operator-produced Headroom outputs instead of running the binary in-session. Each task's artifact is `<artifact_dir>/<task_id>.json` with a pinned `headroom_version` (must match the manifest) and a `modes` map; the importer stamps the artifact path and SHA-256 into provenance. Example:

```json
{
  "task_id": "httpx_pooling",
  "headroom_version": "0.4.1",
  "modes": {
    "headroom_only_on_raw_context": {
      "source_lane": "raw_files",
      "source_passthrough": false,
      "bundle_tokens_uncompressed": 18481,
      "bundle_tokens_compressed": 9120,
      "command": "headroom compress --profile balanced --no-protect-code",
      "compression_settings": {"profile": "balanced", "protect_code": "false"}
    }
  }
}
```
