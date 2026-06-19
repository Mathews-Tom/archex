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

It models lanes by `layer_type` (`retrieval`/`compression`/`baseline`) so a compression layer is never presented as a retrieval engine. Beyond the C1 lanes it can include benchmark-only archex candidate lanes (`archex_query_compressed`, `archex_query_efficiency_packed`) declared under `archex.candidate_strategies`, and Headroom-style compression lanes declared under `compression_layers`. Improved-archex-candidate and Headroom cells appear only when the corresponding artifacts are present.

## Headroom is a compression layer, not a retrieval engine

Headroom is modeled under `compression_layers`, not `external_tools`. Each layer pins an exact version and one or both modes: `headroom_only_on_raw_context` (compress raw/broad context with no archex selection) and `archex_plus_headroom` (compress an archex-selected bundle after selection, protecting source/RAG code by default). Compression lanes contribute only a compression ratio; retrieval-quality columns are `n/a` for them.

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
