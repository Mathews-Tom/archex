# How much of an agent's token spend a retrieval tool can physically displace

SWE-bench Pro reports resolve rate, not tokens. This measures the thing a token claim actually depends on, from evidence Scale already paid for: **165 real SWE-agent trajectories**, published in a world-readable S3 bucket, decomposed by where the billed input tokens went.

Raw cells: [`records-claude-45sonnet-10132025.json`](records-claude-45sonnet-10132025.json), [`records-gpt-5-250-turns-10132025.json`](records-gpt-5-250-turns-10132025.json). Derived figures: [`../evidence/swebench-pro-token-headroom.json`](../evidence/swebench-pro-token-headroom.json).

```bash
uv run python scripts/swebench_pro_token_headroom.py collect \
    --run claude-45sonnet-10132025 --limit-per-repo 10 --workers 8 \
    --output benchmarks/swebench_pro/records-claude-45sonnet-10132025.json
uv run python scripts/swebench_pro_token_headroom.py analyze \
    --records benchmarks/swebench_pro/records-claude-45sonnet-10132025.json \
    --records benchmarks/swebench_pro/records-gpt-5-250-turns-10132025.json \
    --output benchmarks/evidence/swebench-pro-token-headroom.json
```

No model call, no container, no archex. Arithmetic over someone else's receipts: the Sonnet arm alone represents **298 million billed input tokens and $1 212** of already-spent inference.

## Where the tokens go

An observation produced at step `i` of an `n`-step trajectory is re-sent in every later call, so it costs `T × (n − 1 − i)` billed input tokens. Every share below is that compounded figure over provider-reported `tokens_sent`.

| Channel | Sonnet 4.5: calls | share of billed input | GPT-5: calls | share of billed input |
| --- | ---: | ---: | ---: | ---: |
| `read_file` | 1 872 | **37.2%** | 1 018 | **55.5%** |
| `edit` | 1 444 | 10.1% | 434 | 5.1% |
| `test_build` | 979 | 4.2% | 59 | 0.4% |
| `search` (grep/find/ls) | 2 081 | **2.9%** | 617 | **14.4%** |
| `other` | 1 794 | 4.1% | 659 | 3.5% |

Sonnet 4.5 issues more search calls than any other kind and they account for **2.9%** of its bill. It does not pay to search; it pays to read.

## The displaceable share is a property of the model, not the benchmark

Displaceable = every `search` observation plus `read_file` observations of files absent from both the gold and test patch. Reads of files the accepted patch touched are excluded — the agent has to see their exact lines to edit them. Repository-clustered bootstrap, 10 000 resamples, seed `20260909`:

| | Sonnet 4.5 | GPT-5 |
| --- | ---: | ---: |
| Instances / repositories | 110 / 11 | 55 / 11 |
| Mean turns | 75.1 | 51.1 |
| Mean billed input tokens | 2 713 222 | 2 428 631 |
| Context growth | 877 → 61 025 | 1 110 → 36 122 |
| **Displaceable share** | **17.9%** `[13.0%, 23.1%]` | **49.6%** `[37.4%, 63.1%]` |
| Non-displaceable in-patch reads | 22.1% | 28.4% |
| **Break-even bundle** | **6 501 tok** `[4 541, 8 550]` | **18 941 tok** `[14 935, 22 819]` |

**The intervals do not overlap.** The same benchmark, the same repositories, the same scaffold — and the headroom for a retrieval tool differs by a factor of 2.8 between two frontier models. GPT-5 gropes around the repository (14.4% of its bill is `grep`/`find`/`ls`, 55.5% is reading) and would have a lot handed to it. Sonnet 4.5 navigates efficiently enough that there is far less to take away.

This is the decisive design constraint for any "archex reduces tokens" study: **the effect size is a property of the agent's tool-use policy.** A number measured on one model does not transfer to another, and a number measured on a small local model transfers to nothing.

## What a bundle may cost before it stops paying

A retrieval tool front-loads context, so its bundle takes the maximum compounding multiplier — delivered at step 1 it is charged `n − 2` times. Net effect on billed input at fixed bundle sizes:

| Bundle | Sonnet 4.5 net | 95% CI | GPT-5 net | 95% CI |
| ---: | ---: | :---: | ---: | :---: |
| 2 500 | **+9.9%** | [+4.8%, +15.2%] | **+40.5%** | [+28.4%, +54.0%] |
| 5 000 | +1.9% | [−3.5%, +7.4%] | +31.3% | [+19.2%, +44.9%] |
| 7 500 | −6.2% | [−11.8%, −0.5%] | +22.2% | [+9.9%, +35.9%] |
| 10 000 | −14.2% | [−20.2%, −8.1%] | +13.1% | [+0.6%, +27.1%] |
| 13 247 | −24.6% | [−31.2%, −18.1%] | +1.3% | [−11.8%, +15.7%] |
| 20 000 | −46.2% | [−54.2%, −38.4%] | −23.3% | [−38.0%, −7.6%] |

`13 247` is archex's own checked-in external-localization bundle size from [`../cross-tool-efficiency/cross-tool-comparison.json`](../cross-tool-efficiency/cross-tool-comparison.json). At that size the mechanism is **net negative on Sonnet 4.5 by roughly a quarter of the bill**, and merely break-even on GPT-5.

The measured archex ladder ([`BUNDLE_BUDGET.md`](BUNDLE_BUDGET.md)) is what makes this survivable. archex's required-file recall saturates at a **2 822-token** delivered bundle, below the Sonnet break-even interval's lower bound (`4 541`), where the net effect is a saving whose interval excludes zero. A request at the `8192` caller ceiling delivers **6 124** tokens — inside the break-even interval `[4 541, 8 550]`, i.e. statistically indistinguishable from paying for nothing — for identical required-file recall. The per-intent presets were cut to land at saturation where the labelled evidence permits it.

## Sanity checks and caveats

**The pipeline reproduces the leaderboard.** 49 of 110 Sonnet instances resolved (44.5%) against the published 43.72% over all 730. The GPT-5 subsample resolved 16 of 55 (29.1%) against a published 36.30%, low but within reach of a 55-instance sample.

**Token spend is not a proxy for success, and not its opposite either.** Resolved and unresolved instances cost almost the same (Sonnet: 2 652 057 vs 2 762 354 billed input tokens, 75.9 vs 74.4 turns). Displaceable share is slightly *higher* on failures (19.2% vs 16.3%) — failing agents do more fruitless reading. A study that compares token counts without conditioning on outcome is measuring both effects at once.

**Prompt caching moves the money, not the tokens.** Every figure here is in tokens. The runs' own cost accounting implies very different billing: Sonnet at `$4.06` per million input tokens (above its `$3` list input price, consistent with cache-write premium and output), GPT-5 at `$0.66` per million (below its list input price, consistent with a large cached-input discount) — `$11.02` vs `$1.60` per instance for a comparable token volume. A token reduction on a heavily cached prefix converts to roughly a tenth of the money. Any dollar claim must be computed on the billed tier, not on token counts.

**The break-even model assumes the bundle is never evicted.** These trajectories show monotone context growth to 36k–61k tokens with no truncation, so full-history resend holds here. A scaffold that summarises or drops old turns would cut both the displaceable spend and the bundle's compounding cost.

**Displaceable is an upper bound.** Some out-of-patch reads are load-bearing — a caller read to understand an interface is not waste, and a retrieval tool would have to supply that content at its own token cost. The true attainable share is below the figures above.

**Population.** Deterministic first-10 (Sonnet) and first-5 (GPT-5) instance ids per repository, all 11 public repositories. Fetch order interleaves repositories so partial collection stays balanced. Token counts are estimated at 4 chars/token from the recorded `query` field, then rescaled per instance by that instance's own `sum(len(query)/4) / tokens_sent` (mean bias `1.313` Sonnet, `1.071` GPT-5), so every reported quantity is a fraction of provider-billed tokens and the estimator's scale cancels.

## What this says about benchmarking archex on SWE-bench Pro

The headroom is real but bounded, model-dependent, and smaller than archex's current bundle. A defensible study therefore has to:

1. **Fix the bundle budget at the saturation point** (≈3 072 requested / 2 822 delivered), not the `8192` ceiling. At the ceiling the expected effect on Sonnet 4.5 is approximately zero. Shipped for `architecture_broad` and `cli`; `general`, `debugging`, `definition_lookup`, and `usage_search` are held above saturation because their region or line recall still improves with budget.
2. **Pick the model deliberately and report it as a scope condition.** On GPT-5 the same mechanism has ~2.8× the headroom. Neither number is "the" answer.
3. **Condition on outcome.** Tokens-to-resolution on the both-arms-resolved subset, not tokens per attempt.
4. **Quote tokens, and money separately.** The cached-tier asymmetry above is large enough to invert a cost claim that was correct in tokens.
