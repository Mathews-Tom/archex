# R27 scope-aware campaign freeze

This directory freezes the treatment-blind population and run protocol for R26 Candidate A, scope-aware monorepo ranking. It does not implement the candidate, bind candidate source, run a treatment cell, change `archex_query`, or authorize promotion.

## Population

`population.json` contains 16 public repositories pinned to full commits and commit-pinned license files, 2,048 treatment tasks, and 16 single-scope controls. The treatment population contains 688 lexical-collision tasks, 688 cross-scope-dependency tasks, and 672 weak-participation controls. Every treatment task labels its required files and owner scopes and includes a non-dominant required scope with at least a 4:1 dominant-to-required frozen chunk-count ratio.

`control_receipts.json` records two byte-identical measured warm payloads for each single-scope control. `power.json` records the repository-cluster feasibility calculation: 0.9048 power for the fixed +0.05 minimum worthwhile gain, above the required 0.80 threshold.

Validate the PR-1 population contract:

```console
uv run archex benchmark validate --kind scope-aware-population --input benchmarks/campaigns/r27_scope_aware
```

Reproduce scope counts and both warm payloads without importing the campaign validator. The current working directory must be a clean Archex checkout at `1eda0c85de26b4950490062802b41da9f2e00e68`; `CAMPAIGN_CHECKOUT` points to a checkout containing this directory, and `REPO_CACHE` is an optional local cache of the pinned repositories:

```console
cd "$CONTROL_WORKTREE"
PYTHONHASHSEED=0 uv run python "$CAMPAIGN_CHECKOUT/benchmarks/campaigns/r27_scope_aware/reproduce_controls.py" \
  --population "$CAMPAIGN_CHECKOUT/benchmarks/campaigns/r27_scope_aware/population.json" \
  --receipts "$CAMPAIGN_CHECKOUT/benchmarks/campaigns/r27_scope_aware/control_receipts.json" \
  --repo-cache "$REPO_CACHE"
```

The complete immutable manifest and all eligible cells are added in the second R27 pull request. R28 implementation, R29 source binding, and R30 execution remain separate external-merge and authorization boundaries.
