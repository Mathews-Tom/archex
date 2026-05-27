# E2E Verification

Use this contract for release-quality verification in this stack. It separates
implementation health, dogfood regression behavior, and benchmark readiness so a
slow or incomplete benchmark cannot be reported as a product-quality pass.

## Implementation Gate

Run from the repository root:

```bash
uv run ruff check
uv run ruff format --check .
uv run pyright
uv run pytest
grep -rE 'task_id\s*==\s*"' src/archex; test $? -eq 1
```

Pass means the implementation is lint-clean, formatted, typed, covered by the
test suite, and free of direct task-ID-keyed source branches.

## Dogfood Regression Gate

Run product-default dogfood comparisons against an explicit baseline:

```bash
uv run archex dogfood . --all --baseline benchmarks/dogfood_baseline.json --format dogfood-delta
uv run archex dogfood . --all --baseline benchmarks/dogfood_baseline.json --format json
```

Pass means product-default comparisons show no regressions against the baseline.
Diagnostic strategies are excluded from the compact delta summary. Do not refresh
`benchmarks/dogfood_baseline.json` to hide a regression; refresh it only after an
explicit quality improvement and explicit approval.

## Benchmark Readiness

Validate benchmark task definitions before using benchmark outputs:

```bash
uv run archex benchmark validate
```

Run the full benchmark and readiness reports only when runtime is acceptable for
the current slice:

```bash
uv run archex benchmark run --tasks-dir benchmarks/tasks --output .archex/e2e-results
uv run archex benchmark readiness --input .archex/e2e-results --tasks-dir benchmarks/tasks --strategy archex_query --format markdown
uv run archex benchmark triage --input .archex/e2e-results --tasks-dir benchmarks/tasks --strategy archex_query --format markdown
```

Pass means the readiness report meets the documented quality thresholds for the
selected strategy. If a run is interrupted or empirically long-running, record
elapsed time, task progress, and output location, then report benchmark readiness
as unmeasured rather than passed.

## Current Runtime Evidence

- Benchmark task validation passes when `uv run archex benchmark validate`
  reports all 35 task files valid.
- PR8 CodeRankEmbed benchmarking was stopped after more than 6 hours at 19 of 35
  tasks; `django_middleware` was still warming its vector index at 103 of 1338
  batches. Treat those outputs as partial evidence only.
- PR9 reranker smoke benchmarking was stopped during vector warmup before any
  fresh task JSON was written. It does not prove benchmark readiness.
- Dogfood regression status must be reported from the explicit baseline command
  above, with compact deltas and full JSON kept separate.

## Reporting Contract

Every release-quality summary must state these independently:

- Implementation pass/fail: lint, format, type check, unit tests, source grep.
- Dogfood regression pass/fail: product-default deltas against the explicit
  baseline.
- Benchmark readiness: measured pass/fail, or explicitly unmeasured with runtime
  evidence.

For the full local sequence, see `.docs/release-quality.md`.
