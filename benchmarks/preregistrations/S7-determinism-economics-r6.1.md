# S7 — Determinism as Prefix-Cache Economics, R6.1

## Study identity

- **Spike ID and title:** S7 — Determinism as Prefix-Cache Economics, R6.1 replacement
- **Evidence class:** `original`
- **Decision owner and date:** archex R6.1 milestone owner, 2026-07-29
- **First-run commit:** Blank until this pre-registration merges. The first permitted data-generating commit must descend from its merge commit and record that merge SHA, the exact resolved source revisions, and the fixture digest.

## Hypothesis

For the frozen repeated three-turn maintenance sessions below, archex's shipped deterministic context ordering reduces provider-observed input-side USD cost by at least 5% against the seed-recorded ANN-style ordering of the identical frozen context. The control is unchanged deterministic ordering. The per-turn perturbation is a positive control that verifies the provider cache responds to changed prefix order; it is not a decision comparator and cannot license an economic claim. Neither comparator changes retrieval, ranking, context membership, fixed resolution labels, task outcomes, or request content other than the order of the same context chunks. Each session selects its context once with the shipped default retrieval path for Turn 1, then reuses that immutable bundle across all three provider calls; the follow-up turns never invoke retrieval.

## Primary metric

The sole primary metric is the relative reduction in input-side USD cost for deterministic ordering against `ann_baseline`:

\[
100 \times \frac{C_{\mathrm{comparator}} - C_{\mathrm{deterministic}}}{C_{\mathrm{comparator}}}.
\]

All 12 sessions have the fixed label `resolved: true`, so cost per resolved task divides every arm by the same constant and cancels from this comparison. For one arm, input-side USD cost is the sum over all measured turns of the provider-published base-input price times `(1.25 × 5-minute cache-write input tokens + 0.1 × cache-read input tokens + uncached input tokens) / 1,000,000`. Larger positive values favor deterministic ordering. Cache-hit rate, total input tokens, per-turn rendered-prefix identity, and absolute input-side USD cost are exploratory diagnostics. Eligibility prewarm/replay calls are never included in a measured arm's cost.

## SESOI

The smallest effect size of interest is a `+5%` reduction in input-side USD cost. It is the smallest reduction that changes the retain-or-retire decision for the economic framing; it is not derived from observed variance.

## Decision margins

- **Minimum worthwhile gain (MWG):** `+5%`. A smaller reduction does not change the operator's cost decision, so it cannot retain the economic framing.
- **Non-inferiority margin (NIM):** `-5%`. This is the maximum added input-side cost that remains acceptable before deterministic ordering is worse on this measure.
- **Equivalence margin (EQM):** `[-5%, +5%]`. This is the no-decision-change band around zero; its width is derived from the same operator decision, not an observed standard deviation.

## Clustering unit

The independent resampling unit is the repository. One task and its three turns remain together in every paired bootstrap resample because they share source code, a frozen context bundle, and a cache lifecycle. Use a paired percentile repository-cluster bootstrap with 10,000 resamples, seed `20260729`, and 95% intervals.

## Kill criterion

The economics comparison is not run and no result is interpreted when fewer than 12 repository clusters pass fixture validation; any required provider Count Tokens, prewarm, or replay receipt is missing, stale, mismatched, or has a required zero cache-use field; the 84 preflight prefix receipts are incomplete; fewer than four repositories have an `ann_baseline` Turn 2 or Turn 3 rendered-prefix SHA-256 different from the preceding turn in the same lifecycle and a nonzero paired measured input-cost difference from deterministic ordering; any perturbation transition fails its required cache signature; or a measured call lacks conforming provider usage fields. Four is the pre-registered bootstrap feasibility floor: with 12 repository clusters, fewer than four nonzero paired differences leaves at least 2.5% of 10,000 paired bootstrap resamples with an exactly zero estimate. This is an unreached feasibility gate, not a null or an inconclusive comparison.

If the feasibility gate is reached, retire the economic framing and preserve determinism only as a reproducibility property if `ann_baseline` has a point estimate below `+5%` or a 95% repository-clustered interval that includes zero. This is a valid original-study null or inconclusive result, not a retrieval-quality, product, literature, Gate-A, or default-ordering finding.

If `ann_baseline` clears that rule, the only positive license is an original, fixture-bounded observed input-cost result. It authorizes no product, retrieval-quality, literature, Gate-A, README savings headline, or default-ordering claim. The perturbation's cache-use result is a mechanism check only and never contributes to the decision rule.

## Treatment matrix and cell dispositions

The frozen matrix has exactly three arms over the same 12 session IDs, source chunks, labels, model, price schedule, turn text, and five-minute TTL. Arms differ only in rendered context order. The comparator permutations are emitted once during fixture construction and stored in the committed fixture; the measurement command replays them and performs no live ANN retrieval.

| Arm | Ordering rule | Seed | Role |
| --- | --- | --- | --- |
| `deterministic` | archex's shipped stable ordering, unchanged | n/a | Control |
| `perturbed` | For each `(session_id, turn_index)`, apply Fisher–Yates to the same chunk IDs with a PCG64 generator seeded by SHA-256 of `20260729|session_id|turn_index`, interpreted as an unsigned 128-bit integer. The fixture requires at least two chunks per session; if a generated Turn 2 or Turn 3 order equals the preceding turn, rotate that permutation right by one chunk. | `20260729` | Positive control for cache sensitivity to changed order |
| `ann_baseline` | For each `(session_id, turn_index)`, rank the same chunks by descending, fixture-recorded dense retrieval score rounded to four decimal places; break every equal-score group by a PCG64 Fisher–Yates shuffle seeded by SHA-256 of `20260730|session_id|turn_index|score_group`, interpreted as an unsigned 128-bit integer. This fixed, score-quantized tie-break model represents ANN candidate-order instability; it runs only during fixture construction and records one order per turn. A turn with no tied score group can retain the deterministic order; it remains an observed zero-effect session, not an invalid result. | `20260730` | Decision comparator |

The `ann_baseline` comparison cell is `superior` when its point estimate is at least `+5%` and its 95% interval lower bound exceeds zero; `equivalent` when its entire 95% interval lies within `[-5%, +5%]`; otherwise `inconclusive at this N`. A cell whose entire 95% interval is below `-5%` is an observed input-cost increase beyond the NIM, not an N-limited result, and triggers the kill criterion. The study retains the economic framing only when the `ann_baseline` cell is `superior`; every other disposition triggers the kill criterion. The perturbation result is a passed mechanism check only when every Turn 2 and Turn 3 call has a different preceding-prefix SHA-256, nonzero cache-creation input tokens, and zero cache-read input tokens; otherwise the feasibility gate is unreached.

## Frozen source task and session inputs

The fixture builder resolves every source reference below to its immutable commit SHA before data generation. The self-repository `HEAD` reference resolves from this pre-registration's merged source revision. Every session uses the source task's `token_budget: 8192`; Turn 1 is the source task question and is followed by two fixed maintenance turns. The builder invokes shipped default retrieval exactly once per session for Turn 1, freezes the selected chunk identities and dense scores, and reuses that immutable selected-context bundle for every provider call in all arms. No repository may be substituted, duplicated, added, or removed.

| Session ID | Task ID | Repository and pinned task revision | Category | Family | Resolved | Turn 1 | Turn 2 | Turn 3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `archex-query-pipeline` | `archex_query_pipeline` | `.` at `HEAD` | self | comprehension | true | How does archex implement the query pipeline? | Which code path assembles the final context bundle for that query? | Identify the stable ordering point used before final context packing. |
| `celery-task-dispatch` | `celery_task_dispatch` | `celery/celery` at `v5.4.0` | external-large | comprehension | true | How does Celery dispatch and execute distributed tasks? | Which components prepare a task message before a worker receives it? | Where does the worker choose the execution strategy for that task? |
| `click-decorators` | `click_decorators` | `pallets/click` at `8.1.8` | external-framework | comprehension | true | How does click implement command decorators and parameter decoration? | Which decorator factories attach parameters to a command? | Which command and parameter classes consume those attached parameters? |
| `django-middleware` | `django_middleware` | `django/django` at `5.1.4` | architecture-broad | comprehension | true | How do Django's BaseHandler, WSGI entrypoint, and CommonMiddleware implement the request/response middleware chain? | Where is the middleware stack loaded for a request? | Which path applies CommonMiddleware to the request and response? |
| `fastapi-dependency-injection` | `fastapi_dependency_injection` | `tiangolo/fastapi` at `0.115.6` | framework-semantic | comprehension | true | How does FastAPI implement dependency injection for route handlers? | Which function constructs the dependency tree for an endpoint? | Which function resolves nested dependencies before the handler runs? |
| `flask-blueprint-registration` | `loc_flask_blueprint_register` | `pallets/flask` at `3.1.0` | external-framework | localization | true | Issue: registering nested blueprints does not combine the child blueprint's url_prefix and dotted name with the parent correctly. To fix nested blueprint registration, where is the blueprint registration logic that creates the setup state and recurses into nested blueprints applying their url_prefix and name prefix? | Which setup state is created during that registration? | Where does registration recurse into a child blueprint? |
| `gin-routing` | `gin_routing` | `gin-gonic/gin` at `v1.10.0` | external-framework | comprehension | true | How does gin implement HTTP routing with its radix tree? | Which file defines the routing tree nodes? | Which path connects a route group to the routing tree? |
| `httpx-pooling` | `httpx_pooling` | `encode/httpx` at `0.28.1` | external-framework | comprehension | true | How does httpx manage HTTP connection pooling and keep-alive? | Which transport owns the connection pool? | Which client configuration controls keep-alive behavior? |
| `mini-redis-async` | `mini_redis_async` | `tokio-rs/mini-redis` at `e186482ca00f8d884ddcbe20417f3654d03315a4` | external-framework | comprehension | true | How does mini-redis handle async command processing and connection management? | Which server component processes incoming commands? | Which connection component reads and writes protocol frames? |
| `pydantic-validators` | `pydantic_validators` | `pydantic/pydantic` at `v2.10.5` | external-framework | comprehension | true | How does Pydantic chain and apply field validators? | Where are `field_validator` and `model_validator` declared? | Which internal validators support that validation pipeline? |
| `pytest-fixtures` | `pytest_fixtures` | `pytest-dev/pytest` at `8.3.4` | external-framework | comprehension | true | How does pytest discover and inject fixtures? | Which fixture manager discovers available fixture definitions? | Which path resolves fixture arguments for a test item? |
| `react-hooks` | `react_hooks` | `facebook/react` at `v19.0.0` | external-large | comprehension | true | How does React implement the hooks state management system? | Which reconciler module owns hook state during rendering? | Which public hooks module dispatches `useState` calls? |

The selected rows cover every current external `category` represented by the stock task corpus: `external-large`, `external-framework`, `architecture-broad`, and `framework-semantic`; and both `family` values: `comprehension` and `localization`. The self-repository row is separate and is required by the R6.1 scope.

## Provider eligibility and pricing lookup

Pricing source: <https://platform.claude.com/docs/en/about-claude/pricing>, retrieved 2026-07-29T03:08:49Z. Cache semantics source: <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>, retrieved 2026-07-29T02:44:00Z. Selected model: `claude-opus-5` / Claude Opus 5 with global inference routing. Its base input price is `$5 / MTok`; its five-minute cache-write price is `$6.25 / MTok` (1.25×); and its cache-read price is `$0.50 / MTok` (0.1×). The documented minimum cacheable prefix is 512 tokens and cache TTL is five minutes. A changed price invalidates absolute-USD diagnostics only; it cannot change the relative primary metric or cache-hit rate.

The fixture builder must obtain a provider-native Count Tokens receipt showing every rendered prefix reaches the documented floor before data generation. It must prewarm and replay each exact rendered prefix before the five-minute TTL elapses: one deterministic prefix plus three per-turn prefixes for each comparator in each of 12 sessions, or 84 prefixes total. The committed receipt set must carry the rendered-prefix SHA-256, model, Count Tokens result, request usage fields, nonzero cache-creation input tokens for every prewarm, nonzero cache-read input tokens for every replay, source revision, and fixture digest. Missing, stale, mismatched, or required-zero receipts fail fixture validation; no local token estimator is eligibility evidence.

## Run and analysis boundary

Before data generation, PR-2 independently reviews and freezes the fixture, every resolved repository SHA, all rendered-prefix hashes, every provider receipt, the matrix digest, and the recorded comparator permutations. The evidence command may run only after that review merges:

```text
uv run archex benchmark determinism-economics --sessions benchmarks/determinism_economics_r6_1/sessions.json --output benchmarks/evidence/s7-determinism-economics-r6.1.json --preregistration-commit <merged-SHA>
```

The measurement makes three provider calls per `(arm, session)`, in turn order. Each call has exactly one user message with three content blocks: the fixed study instruction; the arm's full rendered selected-context bundle in a document block carrying `cache_control: {"type": "ephemeral"}` on its final chunk; and the current turn's fixed question after that cache breakpoint. Every call sets `model: "claude-opus-5"`, `max_tokens: 0`, no tools, global inference routing, and no assistant message or generated output. The initial request of each isolated `(arm, session)` cache lifecycle is a cache write. Turn 2 starts 60 seconds after Turn 1's response completes and Turn 3 starts 60 seconds after Turn 2's response completes, each within a ±5-second scheduling tolerance; the artifact records every request start, response completion, and inter-turn elapsed time. A retry, provider delay, or rate limit that prevents this schedule invalidates the fixture. Eligibility prewarm/replay calls are a separate preflight lifecycle and are excluded from these three measured calls and from their costs.

The preflight runs all 84 receipt pairs, waits at least 301 seconds after its final replay, then starts measurement. The measurement runs one complete arm at a time and waits at least 301 seconds between arms. This expires every preflight or prior-arm cache key before a byte-identical prefix could occur in a later arm; the arms therefore cannot share cache state without adding arm-specific prompt content.

For `deterministic`, the selected-context bytes are identical on all three turns. For `perturbed` and `ann_baseline`, the frozen per-turn order is used; only the order changes. A provider call failing, rate-limited, refused, or returning missing, stale, or SHA-mismatched usage fields invalidates the entire fixture. A zero cache field invalidates a call only when that field is required by the measured protocol: cache creation for its initial request; cache read for a later request with a matching earlier rendered-prefix SHA-256 whose response completed within the preceding 240 seconds; or, for the perturbation positive control, zero read and nonzero creation on every Turn 2 and Turn 3 call. It is not retried within the measured lifecycle, no repository is removed, and no partial-set analysis is allowed.

The evidence validator is:

```text
uv run archex benchmark validate --kind determinism-economics-r6-1 --input benchmarks/evidence/s7-determinism-economics-r6.1.json
```

## Post-hoc changes

None. Every change after data exists records its UTC timestamp, reason, affected field, and why the affected result is exploratory.