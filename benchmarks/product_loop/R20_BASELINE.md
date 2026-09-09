# R20 — paired product-as-shipped agent-loop baseline

Frozen protocol: [`../preregistrations/R20-product-loop-agent-baseline.md`](../preregistrations/R20-product-loop-agent-baseline.md), merged before the first cell existed. Raw cells: [`results/`](results/). Derived figures: [`../evidence/r20-product-loop-agent-baseline.json`](../evidence/r20-product-loop-agent-baseline.json), regenerable byte-for-byte with `uv run python scripts/r20_product_loop_analysis.py`.

## What ran

114 of 114 planned cells: 2 arms × 19 tasks × 3 repetitions. 113 scored, 1 recorded failure retained. Total modelled cost `$7.74` against the `$25` ceiling; the run used the operator's OAuth subscription, so no dollars were billed and every cost figure here is Claude Code's list-price arithmetic over recorded tokens. Claude Code `2.1.266` on `claude-haiku-4-5`, one ambient tool fingerprint across every scored cell.

## Primary result

Mean required-file completeness of the agent's final answer, treatment minus control, 10 000-resample cluster bootstrap over the 15 source repositories, seed `20260909`:

| | Value |
| --- | ---: |
| `archex_product_loop` (control) | `0.8918` |
| `graft_product_loop` (treatment) | `0.8772` |
| Mean difference | `−0.0146` |
| 95% cluster-bootstrap interval | `[−0.0988, +0.0702]` |

The interval spans the minimum worthwhile gain (`+0.05`) and the non-inferiority margin (`−0.05`), and is far wider than the equivalence margin (`±0.03`). **No cross-tool superiority, non-inferiority, or equivalence claim is drawn.** That was pre-registered as the expected outcome: R19 measured a deterministic retrieval metric on this exact population and returned `[−0.1228, +0.0635]`, so an agent loop, which adds nondeterminism, could not resolve it either. Observed within-task spread across repetitions was `0.219` for the Archex arm and `0.158` for the Graft arm, which is most of the difference the corpus would need to detect.

R20's load-bearing output is the frozen Archex-arm protocol and its per-cell numbers, which later workflow milestones re-run unchanged and compare Archex to Archex.

## The finding that matters for the workflow train

**The Archex arm answered without calling Archex in 84.2% of its cells. The Graft arm called its product in 100% of its cells.**

| | `archex_product_loop` | `graft_product_loop` |
| --- | ---: | ---: |
| Cells with zero product tool calls | `84.2%` | `0.0%` |
| Mean hook invocations per cell | `4.54` | `3.32` |
| Mean tool calls per cell | `11.16` | `6.00` |
| Mean total tokens per cell | `170 394` | `92 997` |
| Mean wall time per cell | `43.4 s` | `29.3 s` |
| Mean setup time per cell | `4.98 s` | `1.26 s` |
| Mean modelled cost per cell | `$0.0839` | `$0.0519` |
| Mean answer precision | `0.550` | `0.505` |
| Answer-unparsed rate | `5.3%` | `7.0%` |
| Answer-over-broad rate | `0.0%` | `0.0%` |

Both products ship an MCP server and Claude Code hooks. Archex's hook fired more often than Graft's, so its integration is live and working — but the agent overwhelmingly reached for `Grep`, `Glob`, and `Read` rather than `mcp__archex__context` or `mcp__archex__query_repo`, and still reached comparable completeness by paying roughly 1.9× the tool calls, 1.8× the tokens, and 1.5× the wall time. Graft's agent consulted `graft_find_code`, `graft_file_api`, `graft_repo_map`, `graft_find_all`, and `graft_trace_calls` in every cell.

This is an adoption result, not a retrieval-quality result. It says the shipped Archex MCP surface is not what an agent reaches for under this prompt and tool policy, and it is the concrete thing R21–R25 have to move.

## Pre-declared secondary, exploratory

Restricted to cells that actually called the arm's product, the difference is `−0.1270` with interval `[−0.2593, −0.0278]`. Read this with care and do not quote it as a head-to-head result: it compares all 114 Graft cells against the 18 Archex cells (15.8%) in which the agent chose to use Archex, which is a self-selected subset, not a randomised contrast. It is reported because the pre-registration required the product-using view, and because the selection effect is itself the point — the Archex cells that used Archex are the hard ones where `Grep` was not enough.

Efficiency over the 43 both-complete comparison units moves in the same direction as over all cells: `10.37` vs `5.91` tool calls, `157 699` vs `89 427` tokens, `41.5 s` vs `28.8 s`, `$0.0817` vs `$0.0506`. Because repetitions are unseeded, a unit is an index alignment, not a matched pair.

## Per-task completeness

| Task | `archex` | `graft` |
| --- | ---: | ---: |
| `celery_task_dispatch` | `0.556` | `0.111` |
| `click_decorators` | `1.000` | `1.000` |
| `django_middleware` | `1.000` | `1.000` |
| `django_orm_queries` | `1.000` | `0.667` |
| `express_error_handling` | `1.000` | `1.000` |
| `express_middleware` | `1.000` | `1.000` |
| `fastapi_dependency_injection` | `1.000` | `1.000` |
| `fastapi_routing` | `1.000` | `1.000` |
| `flask_blueprints` | `0.778` | `0.778` |
| `gin_routing` | `1.000` | `1.000` |
| `go_gin_middleware` | `1.000` | `1.000` |
| `httpx_pooling` | `1.000` | `1.000` |
| `mini_redis_async` | `1.000` | `0.667` |
| `pydantic_validators` | `0.444` | `0.444` |
| `pytest_fixtures` | `0.667` | `1.000` |
| `react_hooks` | `0.833` | `1.000` |
| `requests_sessions` | `0.667` | `1.000` |
| `rust_tokio_runtime` | `1.000` | `1.000` |
| `sqlalchemy_sessions` | `1.000` | `1.000` |

Thirteen of nineteen tasks are tied, most of them at a ceiling of `1.000`. The corpus discriminates on six tasks, which is why the interval is wide.

## Retained failure

One cell, `graft_product_loop/celery_task_dispatch#3`, is a recorded `rate_limited` failure. It is retained, enters the primary mean as `0.0` per the pre-registration, and is not excluded or re-run.

## Metrics recorded but not informative here

`stale_index_event` is `0.0%` on both arms. That is structural, not evidence of freshness parity: write tools are denied on both arms, so the frozen task family performs no edits and neither index can go stale from agent activity. The field exists for the later paired re-measurement, which will need an edit-bearing family to make it mean anything.

## What this does not say

- It does not rank the two products on retrieval quality. The population cannot support that, and the pre-registration forbade the claim before any data existed.
- It does not reproduce or corroborate Graft's published `+12` SWE-bench points or its `23%`/`25%`/`32%` token, call, and time reductions. Those come from a different population, a different model class, and an edit-task harness this milestone does not build. The efficiency figures above happen to move in the same direction on this corpus; that is not a replication.
- It is not portable off this machine. Subscription auth forced the agent to run under the operator's real home, so a shared `~/.archex` and Claude Code's own built-ins were present in every cell. They were identical across arms, so they do not bias the comparison, but a re-measurement reporting a different `ambient_tool_fingerprint` is not comparable to this baseline and must say so.

## Terminal decision

**Baseline frozen and complete.** Every planned cell exists, the population is fully covered, the recorded cost is a third of its ceiling, the client surface was constant throughout, and the derived artifact regenerates byte-for-byte from the retained cells. The cross-tool comparison is inconclusive by design and is published as descriptive only.

R21–R25 may proceed against this baseline. The measurement they most need to move is the `84.2%` no-product-use rate on the Archex arm.
