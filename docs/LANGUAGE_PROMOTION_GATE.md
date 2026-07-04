# Language Promotion Regression Gate

Every language-tier promotion milestone (Section D/E of the enhancement plan: PHP, Ruby, Scala, C, C++ full-tier adapters; the STRUCTURED tier and its per-language adapters) must clear a regression gate against a frozen pre-promotion baseline before it lands. The gate exists because promoting a language changes what the parser, chunker, and dependency graph produce for that language's files, and those outputs feed the same retrieval and ranking machinery every other language shares — a regression there is silent unless something checks for it.

## What the gate checks

The gate reuses the existing `compute_recall`/`compute_required_file_metrics`/F1 machinery in `src/archex/benchmark/strategies.py` through the standard `archex dogfood` baseline-comparison path: recall, precision, F1, MRR, nDCG, MAP, and token efficiency per self-repo task/strategy, compared against the stored baseline with the existing tolerance in `archex.benchmark.baseline.compare_baseline`. A violation is hard-fail: the gate exits non-zero.

## Baseline artifact

The pre-promotion baseline lives at `.archex/baselines/pre-promotion.json`. Unlike the rest of `.archex/` (a gitignored local cache/workspace directory), `.archex/baselines/` is a deliberate, git-tracked exception — see the `.gitignore` carve-out. This is the same `Baseline` JSON schema `archex benchmark baseline save` already produces.

This baseline is a separate artifact from `benchmarks/dogfood_baseline.json` (the ongoing product-default dogfood baseline covered by the "do not refresh without approval" rule in `docs/RETRIEVAL_DEFAULT_DECISIONS.md`). `.archex/baselines/pre-promotion.json` is a one-time-per-promotion-arc checkpoint: it is captured once before Section D/E promotion work begins and is not expected to move again until the arc completes.

## Running the gate

```bash
uv run archex dogfood --all --baseline .archex/baselines/pre-promotion.json --format json
```

A clean run against an unpromoted index passes with zero regressions. Each language milestone (M6–M10, M12, M13) re-runs this exact command as part of its own verification before merging; a non-zero exit blocks the milestone.

## Regenerating the baseline

Only regenerate `.archex/baselines/pre-promotion.json` when the promotion arc itself is restarted or its scope changes — never to make a failing gate pass. Regenerating silently erases the reference point the gate exists to protect.

```bash
uv run archex benchmark run --output .archex/baseline-bootstrap --self-only
uv run archex benchmark baseline save \
  --input .archex/baseline-bootstrap \
  --output .archex/baselines/pre-promotion.json
```
