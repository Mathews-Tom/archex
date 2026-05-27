# Release Quality

Run this sequence before publishing retrieval-sensitive changes.

## Implementation Gate

```bash
uv run ruff check
uv run ruff format --check .
uv run pyright
uv run pytest
grep -rE 'task_id\s*==\s*"' src/archex; test $? -eq 1
```

Pass means the implementation is formatted, typed, tested, and has no direct
task-ID-keyed source branches.

## Dogfood Regression Gate

```bash
uv run archex dogfood . --all --baseline benchmarks/dogfood_baseline.json --format dogfood-delta
uv run archex dogfood . --all --baseline benchmarks/dogfood_baseline.json --format json
```

Pass means product-default dogfood comparisons show no regressions. Refresh
`benchmarks/dogfood_baseline.json` only after an explicit quality improvement
and explicit approval; never refresh it to bless a regression.

## Benchmark Readiness

```bash
uv run archex benchmark validate
uv run archex benchmark run --tasks-dir benchmarks/tasks --output .archex/e2e-results
uv run archex benchmark readiness --input .archex/e2e-results --tasks-dir benchmarks/tasks --strategy archex_query --format markdown
uv run archex benchmark triage --input .archex/e2e-results --tasks-dir benchmarks/tasks --strategy archex_query --format markdown
```

Run the full benchmark only when runtime is acceptable for the current slice.
When it is empirically long-running, stop it, record elapsed time and task
progress, and treat readiness as unmeasured rather than passed.

## Reporting Contract

Final PR summaries must separate:

- Implementation pass/fail: lint, format, type, unit tests, source grep.
- Dogfood regression pass/fail: product-default deltas against the explicit baseline.
- Benchmark readiness: measured pass/fail or explicitly unmeasured with runtime evidence.
