# What archex's bundle buys per token

Raw cells: [`bundle-budget-records.json`](bundle-budget-records.json) (594 = 66 tasks × 9 budgets). Derived figures: [`../evidence/archex-bundle-budget-curve.json`](../evidence/archex-bundle-budget-curve.json), regenerable with

```bash
uv run python scripts/archex_bundle_budget_curve.py collect \
    --tasks-dir benchmarks/tasks --output benchmarks/swebench_pro/bundle-budget-records.json
uv run python scripts/archex_bundle_budget_curve.py analyze \
    --records benchmarks/swebench_pro/bundle-budget-records.json \
    --headroom benchmarks/evidence/swebench-pro-token-headroom.json \
    --output benchmarks/evidence/archex-bundle-budget-curve.json
```

Only `token_budget` varies across the ladder. Same tasks, same revisions, same `archex_query` product path, same index configuration. The index cache is warm for the sweep, so latency here is not comparable to the cold-index `archex_query` figures elsewhere in `benchmarks/`; retrieval quality and bundle size are unaffected.

## The result

**Required-file recall saturates at a 3 072-token budget. Every token above that is spent buying nothing at the file level.**

| Requested budget | Bundle delivered | Completion tokens | Total | Required-file recall | 95% CI | All-required present |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 1 024 | 997 | 4 683 | 5 680 | 0.780 | [0.679, 0.843] | 0.561 |
| 2 048 | 1 963 | 2 190 | 4 152 | 0.885 | [0.837, 0.944] | 0.727 |
| **3 072** | **2 822** | **1 172** | **3 993** | **0.923** | [0.887, 0.992] | 0.803 |
| 4 096 | 3 583 | 1 172 | 4 755 | 0.923 | [0.887, 0.992] | 0.803 |
| 5 120 | 4 270 | 1 172 | 5 442 | 0.923 | [0.887, 0.992] | 0.803 |
| 6 144 | 4 942 | 1 172 | 6 114 | 0.923 | [0.887, 0.992] | 0.803 |
| **8 192** (caller ceiling) | **6 124** | 1 172 | **7 295** | **0.923** | [0.887, 0.992] | 0.803 |
| 12 288 | 7 952 | 1 172 | 9 124 | 0.923 | [0.887, 0.992] | 0.803 |
| 16 384 | 9 302 | 1 172 | 10 474 | 0.923 | [0.887, 0.992] | 0.847 |

Intervals are a 10 000-resample percentile cluster bootstrap over the 16 source repositories, seed `20260909`.

A bundle requested at the `8192` ceiling delivers a mean **6 124 tokens** for exactly the required-file recall that **2 822 tokens** already reached: `0.9230` either way, identical `all_required_present` (`0.803`). **3 302 tokens per query, 54% of that bundle, buy no additional required file.**

Counting what the agent still has to read afterwards, the total-token optimum is the same rung: **3 993 tokens at budget 3 072 against 7 295 at the 8 192 ceiling, 45% lower**. `bundle_completion_tokens` prices that follow-up at oracle cost — the full contents of every expected file the bundle missed — so it is an upper bound on the recovery, but it is priced identically at every rung and the comparison holds. Below 3 072 it explodes: at 1 024 the bundle is only 997 tokens but the agent then needs 4 683 more, making the tight budget the second-most-expensive rung on the ladder.

The saturation is not an artifact of the self-repo corpus. Splitting the same cells:

| Budget | self (bundle / recall) | external (bundle / recall) | localization family (bundle / recall) |
| ---: | ---: | ---: | ---: |
| 3 072 | 2 601 / 0.869 | 2 965 / **0.958** | 3 011 / 0.952 |
| 8 192 | 4 721 / 0.869 | 7 036 / **0.958** | 7 426 / 0.952 |
| 16 384 | 6 329 / 0.869 | 11 234 / 0.958 | 11 952 / 0.952 |

On the external corpus the ceiling spends **2.4× the tokens for the identical 0.958 required-file recall**.

## The caveat that stops this being free

**Region-level recall does not saturate.** On the 45 region-labelled tasks per rung:

| Budget | 1 024 | 2 048 | 3 072 | 4 096 | 6 144 | 8 192 | 12 288 | 16 384 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Region recall | 0.490 | 0.601 | 0.659 | 0.716 | 0.782 | 0.804 | 0.834 | 0.847 |

Monotone the whole way up. So the extra tokens are not noise — they are additional *lines inside files archex had already found*. Dropping to 3 072 costs **0.145 region recall** (0.804 → 0.659) while costing nothing in required-file recall.

That trade is not uniform, and the product does not apply one budget anyway: `archex.api.query` routes an unnamed budget through `token_budget_for_query`, so `8192` is the caller ceiling, not a typical spend. The actionable question is **which preset sits above its own saturation point**, which `per_intent_presets` in the curve artifact answers.

## Per-intent decision

| Intent | Preset | Bundle | Completion | Total | Required-file recall | Region recall | Disposition |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `architecture_broad` | 8192 → **3072** | 6058 → 2834 | 1600 → 1600 | 7658 → 4434 | 0.911 → 0.911 | 0.472 → 0.472 | **Cut** |
| `cli` | 3072 → **1024** | 1536 → 985 | 2984 → 2984 | 4520 → 3968 | 0.850 → 0.850 | 0.792 → 0.792 | **Cut** |
| `general` | 6144 held | 5466 → 2984 | 966 → 966 | 6433 → 3951 | 0.916 → 0.916 | 0.719 → **0.655** | Held |
| `debugging` | 6144 held | 5604 → 3003 | 915 → 915 | 6519 → 3918 | 0.967 → 0.967 | 0.889 → **0.613** | Held |
| `definition_lookup` | 2048 held | 2015 → 986 | 679 → 679 | 2694 → 1665 | 0.900 → 0.900 | 0.800 → 0.800 | Held — see below |
| `usage_search` | 4096 held | 2802 → 998 | 0 → 0 | 2802 → 998 | 1.000 → 1.000 | 1.000 → **0.583** | Held — at saturation |

The load-bearing column is `Completion`: **it never moves.** Required-file recall is identical at every rung at or above saturation, so the follow-up read cost is unchanged and the shrink is a pure gain in token efficiency after completion — the exact quantity `docs/RETRIEVAL_DEFAULT_DECISIONS.md` requires a default change to improve.

`general` and `debugging` are what a blanket cut would have broken. Their file-level recall saturates at 3 072 like everything else, but region recall keeps climbing past it — debugging goes `0.613 → 0.966` between 3 072 and 8 192. Those tokens buy real within-file coverage for precisely the intents whose consumers need exact lines.

### The two intents the first pass could not grade

`definition_lookup` had no region labels and `usage_search` had no tasks at all, so both were held on absent evidence. The corpus now covers them — all five `routing_pl_*` tasks carry verified `expected_regions`, and two self-repo caller-localization tasks were added — and **both presets are confirmed correct as shipped**:

* `definition_lookup` 2048 → 1024 would hold required-file recall (0.900) *and* region recall (0.800) but regresses **line recall `0.555 → 0.476`**. Only the finer signal catches it. Held.
* `usage_search` 4096 → 1024 collapses region recall `1.000 → 0.583` and line recall `1.000 → 0.383`. Region recall climbs `0.583 → 0.875 → 1.000` across 1024/3072/4096 and is flat above, so 4 096 is the cheapest value reaching full labelled coverage. It was already set there. Held.

A file-level-only reading would have cut both: required-file recall saturates at 1 024 for each. Two further caller-localization candidates were authored and discarded as mis-calibrated — `archex_query` scored `0.000` required-file recall on both — which is why `usage_search` rests on only `n = 2` tasks.

An earlier draft of this file blamed that on archex localizing definitions rather than call sites. **That claim is withdrawn.** Seeding is not the problem: for `Callers of index_config_for_profile` all 3 wanted caller files are among the 79 seeds, and for `Who calls compute_bundle_completion_penalty?` both are among the 83 — yet neither query packs a single one. The cause is now identified: `assemble_context` multiplies the weighted score by `_path_alignment_boost`, a flat **3.0×** when a file's basename stem matches a query term, and `_query_terms` expands the question into terms the user never typed (`cache`, `project`, `store`, `build`). `config.py`, `project.py`, `cache.py`, and `index/store.py` each collect that 3× on a filename coincidence, so a chunk with *perfect* relevance `1.000` and no filename match (final `1.000`) loses to one with relevance `0.450` that has it (final `1.350`). `docs/RETRIEVAL_DEFAULT_DECISIONS.md` carries the score table, the repro snippet, and the gate any fix must clear. **No fix is applied**; `_path_alignment_boost` is on the product path for every query.

Shipped in `src/archex/serve/intent.py`. Verified on the live `query` path with the same question and index: `architecture_broad` 7 949 → 2 907 bundle tokens over an unchanged 5-file set, `cli` 2 972 → 988.

## Why the ceiling matters

[`TOKEN_HEADROOM.md`](TOKEN_HEADROOM.md) measures the other half on 165 real SWE-agent trajectories: in a long-horizon loop a front-loaded bundle is re-sent on every later call, so it only pays below a break-even size set by the displaceable search-and-read spend. `agent_loop_constraint` in the curve artifact joins the two artifacts.

| | Sonnet 4.5 | GPT-5 |
| --- | ---: | ---: |
| Break-even bundle | 6 501 `[4 541, 8 550]` | 18 941 `[14 935, 22 819]` |
| archex at saturation (2 822 tok) | below the interval — a saving whose CI excludes zero | far below — a large saving |
| archex at the `8192` ceiling (6 124 tok) | **inside the interval** — indistinguishable from paying for nothing | comfortably below — a saving |

A request at the `8192` ceiling sits on the Sonnet 4.5 break-even. Landing at the saturation budget instead turns the mechanism from "statistically indistinguishable from zero" into a measurable **+9.9% `[+4.8%, +15.2%]`** reduction in billed input tokens, with no loss of required-file recall. That is what the `architecture_broad` preset cut achieves for the intent that previously requested the ceiling.

That is what makes the constraint survivable: archex does not have to give up retrieval quality to fit under the break-even. It has to stop shipping tokens it was already not being paid for.
