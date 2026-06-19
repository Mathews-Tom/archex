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

## Graphify follow-up lanes

Graphify is modeled under `graphify_lanes`, not `external_tools`. The checked-in public set uses two explicit lanes pinned to `graphifyy 0.8.44`:

- `graphify_build_plus_query` — includes the per-task graph build/setup cost plus the first graph-backed answer.
- `graphify_query_warm` — prebuilds the graph first, then reports only the warm graph-query path.

Both lanes point at checked-in artifact directories (`benchmarks/headtohead/results/graphify_build_plus_query/` and `benchmarks/headtohead/results/graphify_query_warm/`). Each task artifact is `<artifact_dir>/<task_id>.json` and carries the exact numeric fields claimed in docs plus the sanitized Graphify command shape, pinned package/version, and build-vs-warm semantics.

Local reproduction of one lane uses the adapter contract introduced in `scripts/run_graphify_headtohead_lane.py`: it reads the PR2 stdin payload (`task`, `repo_path`, `lane`, `graphify`) and emits one artifact JSON on stdout. The checked-in public artifacts were produced with that script and then copied into the two directories above. No new public claim should be added unless the corresponding Graphify artifact JSON exists in git.

Graphify token-efficiency cells in the public reports count the graph reference listing returned by `graphify query`, not returned source code. They are useful as within-lane efficiency signals, but they are not bundle-for-bundle comparisons against archex or `ccc`.

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
