# Local benchmark evidence

Run the complete corpus only as a local operator. GitHub Actions validates task definitions and bounded evidence contracts; it never executes the 64-task retrieval corpus.

## Procedure

1. Start from a clean committed source tree. The evidence manifest records the resolved source SHA and refuses a dirty tracked tree.
2. Validate the declared corpus:

```text
uv run archex benchmark validate --kind tasks --tasks-dir benchmarks/tasks
```

3. Choose a new, empty evidence directory and run the default control:

```text
uv run archex benchmark run --tasks-dir benchmarks/tasks --output <evidence-dir> --strategy archex_query
```

The command writes exactly one raw report per completed task and `manifest.json`. The manifest records source SHA, task-manifest digest, Archex version, selected strategies, retrieval configuration, report SHA-256 values, generation timestamp, and hardware advisory.

4. Validate retained evidence before inspecting or comparing it:

```text
uv run archex benchmark validate --kind evidence --tasks-dir benchmarks/tasks --input <evidence-dir>
```

5. Record the absolute-gate result without replacing a canonical baseline:

```text
uv run archex benchmark gate --input <evidence-dir> --min-recall 0.60 --min-f1 0.30 --min-mrr 0.55
```

A non-zero gate result is evidence. Preserve the manifest and all 64 reports with the failing result. Do not promote or overwrite a baseline until the declared promotion milestone verifies a passing full-corpus result.

Evidence is invalid when its source revision, task digest, report hashes, task coverage, strategy coverage, or retrieval configuration differs from the manifest.