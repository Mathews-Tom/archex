# R20 — paired product-as-shipped agent-loop baseline

This directory holds the raw per-cell artifacts for R20. The protocol is frozen in [`../preregistrations/R20-product-loop-agent-baseline.md`](../preregistrations/R20-product-loop-agent-baseline.md), which merged before the first cell existed; that document, not this one, is authoritative for every measurement decision.

## What is measured

Two arms drive the *same* agent — Claude Code `2.1.266` on `claude-haiku-4-5` — over the same 19 head-to-head tasks, three repetitions each, 114 planned cells. Only the wired context product differs:

| Arm | Wiring | MCP tools advertised | Hooks installed |
| --- | --- | --- | --- |
| `archex_product_loop` | `archex init` + `archex install-client claude-code --scope project` (+ `--hooks`) | `mcp__archex__context`, `mcp__archex__query_repo` | `PreToolUse` on `Glob\|Grep` |
| `graft_product_loop` | `graft init --no-agents` at the R19-pinned `@nanonets/graft@0.16.0` | the six `mcp__graft__*` tools | `SessionStart`, `UserPromptSubmit`, `PostToolUse` ×2, statusline |

Both arms get the identical base tool allowlist (`Read`, `Grep`, `Glob`) and the identical explicit denylist. The primary metric is mean required-file completeness of the agent's final answer, scored against the same `expected_files` labels R19 used.

R20 is a **descriptive baseline**. R19 could not separate equivalence from non-inferiority on this population with a deterministic metric, so an agent loop is not expected to either. No cross-tool superiority, non-inferiority, or equivalence claim is published from it at any observed value. Its load-bearing output is the frozen Archex-arm protocol that later workflow milestones re-run in a paired within-arm design.

## Running it

Install the pinned Graft release once per machine. The install is network-dependent: a transitive `tree-sitter-cli` install script downloads a platform binary from GitHub release assets, and a transient `ECONNRESET` there is retried rather than treated as a Graft failure.

```bash
CI=1 DO_NOT_TRACK=1 npm install --prefix "$GRAFT_PREFIX" @nanonets/graft@0.16.0
```

Exercise the whole harness with **no hosted call** by pointing it at a local stub endpoint. Cells produced this way record `provider_endpoint_overridden: true`, and the registered validator refuses any directory containing one, so a stub run can never be mistaken for evidence:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:<stub-port> ANTHROPIC_AUTH_TOKEN=stub \
uv run python scripts/run_product_loop_suite.py --task celery_task_dispatch --repetitions 1
```

`--dry-run` goes further and performs every setup and check without invoking the agent at all.

The measured run:

```bash
uv run python scripts/run_product_loop_suite.py \
    --output benchmarks/product_loop/results \
    --graft-binary "$GRAFT_PREFIX/node_modules/.bin/graft"
```

The suite refuses to start unless `claude --version` reports the pinned `2.1.266` and the code's frozen prompt still matches the pre-registration's Appendix A byte for byte. It resumes by skipping any cell whose artifact already exists, and aborts before starting a new cell once cumulative modelled cost reaches the `$25` ceiling, retaining everything already written.

Validate a finished directory:

```bash
uv run archex benchmark validate --kind product-loop --input benchmarks/product_loop/results
```

## What each cell records

One artifact per planned cell, success or failure, at `<arm>/<task_id>__rep<n>.json`. Fields are the pre-registration's privacy contract: identity and frozen-prompt hash, the resolved tool list, MCP server and status, tool-call names and counts, token counts, modelled cost, wall and setup timings, freshness state, hook records, the parsed answer paths, scoring flags, and sanitized failure reasons. There is no prompt text, no repository source, no tool input or output, and no absolute machine path — a residual one aborts the run rather than being published.

## Two things the harness deliberately does not hide

**Cost is modelled, never billed.** The run authenticates by OAuth subscription, so no dollars are charged and `total_cost_usd` is Claude Code's list-price arithmetic over the recorded tokens. The ceiling, the abort rule, and every published cost figure inherit that caveat.

**Graft's cards are Grep-visible and Archex's index is not.** `graft build` gitignores `graft/` but also writes an `.ignore` containing `!graft/`, whose own comment says the cards "should stay greppable". Ripgrep reads `.ignore` first, so Graft's generated markdown is deliberately exposed to the agent's `Grep` and `Glob`; Archex writes no comparable agent-readable file. That is part of measuring each product as shipped, it is recorded per cell, and the answer cardinality cap bounds how far breadth alone can carry a completeness score.
