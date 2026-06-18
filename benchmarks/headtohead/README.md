# Head-to-head benchmark harness

This directory contains the pinned C1 public comparison manifest for running the same external-repo tasks across archex, cocoindex-code (`ccc`), and the raw-ripgrep/read baseline. The harness records cold-start timing, warm query latency, recall, precision, token efficiency, required-file coverage, receipt accuracy, and bundle-completion penalty tokens for every lane.

Run only by an operator outside this implementation session:

```bash
uv tool install cocoindex-code  # operator choice: [full] for local embeddings
uv run archex benchmark headtohead run --manifest benchmarks/headtohead/manifest.yaml --output .archex/headtohead
uv run archex benchmark headtohead report --input .archex/headtohead --format markdown
```

Publication rule: paste the report back unchanged and copy the result artifacts into `benchmarks/headtohead/results/` regardless of which tool wins each cell. Keep `.archex/` and `.docs/` out of git.
