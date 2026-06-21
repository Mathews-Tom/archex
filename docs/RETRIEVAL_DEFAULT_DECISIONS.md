# Retrieval Default Decision Protocol

Operator evidence from the 2026-06-09 retrieval-default benchmark keeps `archex_query` as the product default. CodeRankEmbed and reranker default changes remain blocked until a clean full run clears the recall/F1, token-efficiency, and p95 latency rules.

## Invariants

- Run core retrieval benchmarks locally only; no network or generative LLM inference. Bundle-only eval is separate and runs only when an operator supplies a local evaluator command.
- Pin exactly one embedder per benchmark run with `--embedder`.
- Compare candidates on recall, required-file recall, missed-required-task rate, receipt accuracy when available, token efficiency after completion, F1, median latency, and p95 latency. Recall/F1 alone is not sufficient, and raw token efficiency is not sufficient when completion penalty cancels the savings.
- Keep `archex_query` as the product default; the 2026-06-09 run did not satisfy the strategy switch rule.
- Do not refresh `benchmarks/dogfood_baseline.json` without explicit approval after a proven improvement.


## Candidate region and context-efficiency gate inputs

The benchmark eval frontier adds optional retrieval-quality signals that are available as **candidate** gate inputs for tasks that declare `expected_regions`: region recall, line recall, ranked-region MRR/nDCG, context noise ratio, and relevance per 1k tokens. The gate exposes optional thresholds (`min_region_recall`, `min_line_recall`, `max_context_noise_ratio`, `min_relevance_per_1k_tokens`).

These inputs do not change the current switch rule. They are optional and ignored when a task has no region labels, and they enforce a threshold only when the label exists. A checked-in labeled baseline (`.archex/r0-labeled-baseline`, generated with `uv run archex benchmark run --tasks-dir benchmarks/tasks --output .archex/r0-labeled-baseline`) now populates region recall, region precision, line recall, context noise ratio, and relevance per 1k tokens for the labeled subset. Treat those values as active candidate-gate inputs for future benchmark-only lanes, not as a default switch or an improvement claim.

## Personalized structural-centrality candidate lane

`archex_query_ppr` is a benchmark-only retrieval lane (Workstream R1 of `.docs/2026-06-20-retrieval-ranking-signal-enhancement-design.md`). It keeps the `archex_query` BM25 retrieval, graph expansion, packing, and default scoring weights, but replaces only the `structural` scoring term with a query-personalized, edge-weighted, direction-aware PageRank over a bounded seed neighborhood. The product `archex_query` path continues to call `graph.structural_centrality()` and its cached global PageRank unchanged.

Decision rule: this lane is **not eligible for product-default promotion** unless a clean warm labeled run shows mean F1 at least `+0.05` over `archex_query`, token efficiency after completion at least as high as `archex_query`, p95 latency at or below `3000 ms`, and no regression in required-file recall or region metrics (`region_recall`, `region_precision`, line metrics, `context_noise_ratio`, and `relevance_per_1k_tokens`) where labels exist. The lane must also keep centrality provenance populated (`centrality_variant`, personalization/fallback status, subgraph node/edge counts, and centrality latency) so any latency or quality change is attributable.

No numeric PPR results are claimed here. Any future promotion claim must come from checked-in benchmark artifacts comparing `archex_query_ppr` to `archex_query` on required-file recall, region/line metrics, `context_noise_ratio`, `relevance_per_1k_tokens`, F1, median latency, and p95 latency.

## Recency/churn candidate lane

`archex_query_churn` is a benchmark-only retrieval lane (Workstream R2 of `.docs/2026-06-20-retrieval-ranking-signal-enhancement-design.md`). It keeps the `archex_query` BM25 retrieval, graph expansion, packing, and default scoring weights, and adds a small per-file recency/churn prior as a bounded multiplier on the final score. The prior is **intent-gated**: it applies only to the `DEBUGGING` and `USAGE_SEARCH` intents, where edit locality is the strongest signal, and is off for definition, architecture, CLI, and general queries. The multiplier lies in `[1.0, 1.0 + max_boost]` (`max_boost = 0.05`), so it can only break ties and nudge ordering and can never dominate the relevance signal. The product `archex_query` path never builds a prior and is unchanged.

Determinism / neutral-fallback contract: the prior is **optional and artifact-backed**. Its input comes from either a checked-in `archex.churn.v1` per-file churn fixture or a full local clone's git history. Because benchmark clones may be shallow (`benchmark/runner.py` uses `git clone --depth 1`), git history is often missing or non-representative; whenever history is unavailable (shallow clone, single-commit repo, any git failure) and no fixture is supplied, the prior is **neutral** (multiplier `1.0` for every file) and the lane's output is bit-identical to `archex_query`. Given a fixed commit and a fixed fixture the prior is deterministic and reproducible; a missing-history case never silently changes a deterministic result. This lane is meaningful only on the **gated-intent task subset**; on non-gated tasks it equals `archex_query` by construction.

Decision rule: this lane is **not eligible for product-default promotion** unless a clean warm labeled run on the gated-intent subset shows mean F1 at least `+0.05` over `archex_query`, token efficiency after completion at least as high as `archex_query`, p95 latency at or below `3000 ms`, and no regression in required-file recall or region metrics (`region_recall`, `region_precision`, line metrics, `context_noise_ratio`, and `relevance_per_1k_tokens`) where labels exist. The lane must keep churn provenance populated (`churn_source` of `history`/`fixture`/`neutral_fallback`, the intent-gate decision, and the per-file priors applied) so any ordering change is attributable.

No numeric churn results are claimed here. Any future promotion claim must come from checked-in benchmark artifacts comparing `archex_query_churn` to `archex_query` on the gated-intent subset.

## Issue-to-edit localization family

The benchmark corpus includes a small, maintained issue-to-edit localization family (Workstream R3 of `.docs/2026-06-20-retrieval-ranking-signal-enhancement-design.md`): hand-curated, SWE-bench-style tasks (`benchmarks/tasks/loc_*.yaml`), each an issue-style question plus expected edit locations at file and symbol granularity pinned to a public release tag. It measures the real downstream fault-localization job — given an issue, find the files and functions to edit — rather than the "how does X work" comprehension the rest of the corpus measures. Tasks carry a `family` axis (`comprehension` or `localization`) orthogonal to `category` (repo type). This is corpus and reporting work that reuses the existing region-metric computation; it changes no retrieval code path, no product default, and adds no hosted/LLM/network behavior.

Reporting separation: `archex benchmark report` grades the localization family separately from the comprehension family. `reporter.format_localization_summary` groups tasks by family and surfaces file-level localization (required-file recall, MRR, and nDCG over returned file order) and region-level localization (region recall and ranked-region MRR/nDCG over returned context order) per strategy. Grouping is per task, so every strategy result for a localization task — including the raw baselines — is reported under localization, and the report never publishes a single cross-family aggregate winner that mixes the two families. Region columns read `unknown` for any lane that produces no region metrics (for example the raw baselines).

No numeric localization results are claimed here. Any localization win or regression claim must come from checked-in benchmark artifacts that contain the file-level and region-level fields needed to verify it.

## Task-aware candidate lane

`archex_query_task_aware` is a benchmark-only retrieval lane (Workstream 2 of `.docs/2026-06-18-retrieval-quality-token-efficiency-enhancement-design.md`). It classifies query modality (`pl_to_pl`, `nl_to_pl`, `mixed`) and budget tier (`tight`, `standard`, `large`) and routes sparse-vs-dense retrieval accordingly: BM25-first for code-heavy and mixed queries with a single confidence-gated dense expansion, and a bounded hybrid pass for natural-language queries. Confidence-aware fusion runs only on the hybrid/dense pass, the cross-encoder reranker is never run, and total work is bounded to at most two retrieval passes.

Decision rule: this lane is **not eligible for product-default promotion** unless a clean warm run satisfies the full `Product strategy decision` rule below — mean F1 at least `0.05` higher than `archex_query`, token efficiency after completion at least as high, and p95 latency at or below the `3000 ms` budget — **and** does not regress required-file recall or the region/context-efficiency signals where region labels exist. Recall or token-efficiency gains alone do not qualify, and missing the p95 budget disqualifies it regardless of quality gains. It is evaluated on the metrics and thresholds the default-switch rule already uses, plus the region and context-efficiency signals above.

No numeric results are claimed here. Any future promotion claim must come from checked-in benchmark artifacts that contain the file-level, region, token-efficiency, and p95 fields needed to verify it.

Warm-run note: the lane runs vector retrieval only on its conditional hybrid/dense pass, so it is not a member of the benchmark vector warm sets. A bare `--strategy archex_query archex_query_task_aware` run would therefore time the lane's first vector pass cold (embedder load plus vector-index build) while `archex_query` stays BM25-only, inflating the lane's p95. Before comparing p95, warm the shared vector store the lane reuses — for example run the comparison once and discard it, or include `--query-fusion` so the existing fusion warm-up populates the store — then rerun against the warmed cache.

Disposition (local self-repo evaluation): a routing-exercising self-repo task set was added (`benchmarks/tasks/routing_*.yaml`: identifier-dense `pl_to_pl` lookups, a path/symbol query, a stack trace, mixed issues, and `tight`/`large` budget variants) and run warm against `archex_query` and `archex_query_fusion`. Strategy provenance confirms the routing is enacted: code-heavy queries with a confident top BM25 hit stay BM25-only (`dense_expansion=skipped:confident_sparse`), while diffuse or low-confidence queries escalate to one bounded dense pass (`dense_expansion=ran`), so the lane is genuinely distinct from `archex_query_fusion` rather than a duplicate of it. On that set the lane tied `archex_query` on file recall and F1 (no regression, no improvement), showed a small token-efficiency edge over always-fusion but stayed below the BM25 default, and its conditional second pass can cost more latency than single-pass fusion when it escalates (both well under the p95 budget). Per the rule above it is therefore **not eligible for promotion** and remains benchmark-only. No measured wins are claimed; the numbers depend on the local working tree and are not checked in. Reproduce warm with: `uv run archex benchmark run --tasks-dir benchmarks/tasks --self-only --query-fusion --strategy archex_query_task_aware --allow-remote-code --output .archex/ta-measured --no-progress` followed by `uv run archex benchmark report --input .archex/ta-measured --format markdown`.

## Compression candidate lane

`archex_query_compressed` is a benchmark-only lane (Workstream 3 of `.docs/2026-06-18-retrieval-quality-token-efficiency-enhancement-design.md`). It runs the exact `archex_query` retrieval and packing path, then applies deterministic, low-risk compression after bundle assembly. Retrieval is untouched, so retrieval metrics stay attributable to the uncompressed retrieval set and `archex_query` behavior is unchanged. Compression modes are deterministic text transforms only (passthrough of required/direct/high-confidence regions, structural code elision, comment/whitespace slimming, large-literal summarization, JSON/log crushing); there are no hosted calls, local model calls, or semantic summarization.

Required, direct, and high-confidence code passes through uncompressed by default, and `protect_code` disables code elision for fix/debug/review intents. Every compressed region exposes its original line range, original content hash, and an exact fetch-original handle, so the original source is always retrievable.

Decision rule: this lane is **not eligible for product-default promotion** and does not change the switch rule. Compression can improve token efficiency only if required-file/region metrics and receipt accuracy do not regress. Crucially, **compression cannot make incomplete context complete**: it never upgrades `context_complete`, and `compression_hidden_required_region_count` (required regions hidden by compression) is reported separately from retrieval misses and must stay `0`.

No numeric compression results are claimed here. Any future compression-win claim must come from checked-in benchmark artifacts that contain the `bundle_tokens_uncompressed`/`bundle_tokens_compressed`, `bundle_compression_ratio`, required-context passthrough, and completion-adjusted token-efficiency fields needed to verify it.

Disposition (local evaluation): the lane was run warm against both the self-repo task set and the external-repo task set in `benchmarks/tasks`, comparing `archex_query` and `archex_query_compressed`. Across every task the two strategies produced identical retrieval metrics (recall, required-file recall, F1, result files, and token efficiency after completion), confirming retrieval is unaffected. The safety invariants held on every task: `compression_hidden_required_region_count` and `required_context_compressed_tokens` were `0` — no required/high-confidence region was ever compressed or hidden. Measured compression was modest, because most returned regions are high-confidence seed content that passes through and only the lower-confidence frontier tail compresses, so it provided no token-efficiency improvement large enough to justify promotion. Per the rule above the lane stays **benchmark-only and not promotion-eligible**. No measured wins are claimed; the numbers depend on the local working tree and external clones and are not checked in. Reproduce with `uv run archex benchmark run --tasks-dir benchmarks/tasks --strategy archex_query --strategy archex_query_compressed --output .archex/compression-full --no-progress` followed by `uv run archex benchmark report --input .archex/compression-full --format markdown`.

## Retrieval-aware packing candidate lane

`archex_query_efficiency_packed` is a benchmark-only lane (Workstream 4 of `.docs/2026-06-18-retrieval-quality-token-efficiency-enhancement-design.md`). It runs the exact `archex_query` retrieval path, then re-packs the assembled bundle with a deterministic, relevance-per-token packer: each retrieved region is scored from intrinsic signals only (retrieval score, direct path/symbol match, graph distance and edge confidence, token count, compression eligibility and loss risk, scout/fetch handle priority, and remaining budget — never benchmark ground truth). Direct/high-confidence targets are preserved before optional context, smaller enclosing evidence is preferred over whole-file context except at large budgets or files with multiple high-confidence regions, low-risk regions may be compressed, and large low-score graph-distant context is anchored or skipped. The packed bundle is the lane's returned context; the product default and `DEFAULT_STRATEGIES` are unchanged.

Packing provenance records the include/compress/elide/skip decision counts, the budget tier, and the delivered relevance-per-token; compressed and elided regions keep their original content hash and an exact fetch-original handle so the source stays retrievable. Reports compare normal packing (`archex_query`), compressed packing (`archex_query_compressed`), and efficiency-aware packing on token efficiency after completion, region quality where labels exist, compression ratio, and p95 latency.

Decision rule: this lane is **not eligible for product-default promotion** and does not change the switch rule. It becomes eligible for default consideration only if a clean run shows that required-file and region/line metrics (where labels exist) do not regress, token efficiency after completion improves, and p95 latency does not regress versus `archex_query` — in addition to the full `Product strategy decision` rule below. Because packing changes which regions earn budget, `compression_hidden_required_region_count` (required regions hidden by compression or elision) is reported and must stay `0`; packing can never make incomplete context complete.

No numeric packing results are claimed here. Any future packing-win claim must come from checked-in benchmark artifacts that contain the packing decision counts, `packed_relevance_per_1k_tokens`, `bundle_compression_ratio`, required-file/region, and completion-adjusted token-efficiency and p95 fields needed to verify it.

Disposition (local evaluation): the lane was run against the self-repo task set (24 tasks, `repo: "."`) and the external-repo task set (19 cloned repos) in `benchmarks/tasks`, comparing `archex_query`, `archex_query_compressed`, and `archex_query_efficiency_packed`. The packing safety invariants held on every task: `skip` was `0` and `compression_hidden_required_region_count` was `0` — no region was dropped and no required/high-confidence region was hidden — so required-file recall, recall, and F1 were identical to `archex_query` on every task (no regression at either layer). Packing was modest: it shaped only the lower-confidence tail (self-repo totals across the set: 474 include / 9 compress / 10 elide / 0 skip; external: 408 / 11 / 9 / 0), trimming mean bundle tokens by roughly 1-2% and nudging token efficiency after completion up by a few thousandths. Latency was approximately neutral at the median (external median delta ≈ 0), but the single-run cold p95 tail rose on the largest external bundles (for example `sqlalchemy_sessions`, +462 ms) because `_pack_bundle` probes every chunk through `compress_region` for eligibility. The lane therefore does **not** yet clearly clear the p95-no-regression gate, and per the rule above stays **benchmark-only and not promotion-eligible**. No measured wins are claimed; the numbers depend on the local working tree and external clones and are not checked in. Reproduce with `uv run archex benchmark run --strategy archex_query --strategy archex_query_compressed --strategy archex_query_efficiency_packed --tasks-dir benchmarks/tasks --output .archex/packing-all --no-progress` followed by `uv run archex benchmark report --input .archex/packing-all --format markdown`.

## Advanced quality lanes

Workstream 5 of `.docs/2026-06-18-retrieval-quality-token-efficiency-enhancement-design.md` adds four benchmark-only **advanced quality lanes**. They are exploratory experiments motivated by recent agentic-RAG research (BLAgent, Meta-RAG, graph-aware multi-hop retrieval) and they add cost, latency, or storage that the deterministic default path does not pay. **Every advanced lane is benchmark-only until it clears the full default-switch gate; none changes the product default in this stack.**

- `archex_query_dual_transform` — deterministically rewrites the query into a structural subquery (paths, identifiers, call sites, errors, stack-trace and import signals) and a behavioural subquery (natural-language symptoms and identifier-derived domain nouns), runs the BM25 path once per subquery over one warm index, and reciprocal-rank fuses the two into one budget-bounded bundle. Rule-based only; no hosted LLM calls. Provenance records both subqueries, fallback flags, and fusion counts.
- `archex_query_bounded_rerank` — reranks only a compact candidate set with hard candidate and latency caps. A deterministic symbolic evidence pass (path/symbol/term overlap blended with retrieval rank) always runs; a local cross-encoder is reused only when already loaded in process and only within the latency cap, otherwise the model rerank is skipped or aborted with explicit provenance. No hosted rerankers.
- `archex_query_summary_sidecar` — summary-first selection then original-code retrieval. Summaries come from an explicit, offline-built sidecar (deterministic digests plus source hash, index revision, generation time, granularity, and a fetch handle); they are never auto-generated during a query. Stale entries are excluded, summaries gate file selection only, and the returned bundle is always original code. This lane carries an offline storage/index cost (the sidecar artifact) the other lanes do not.
- `archex_query_graph_multihop` — bounded multi-hop dependency-graph expansion from retrieved seeds under hard caps (edge-confidence threshold, per-hop frontier cap, hop cap, token budget) so dependency expansion cannot flood the bundle. Expanded files contribute original code; every expansion and cut is recorded in receipts (included/omitted edges, skipped candidates) and provenance.

Decision rule: each advanced lane is **not eligible for product-default promotion** and does not change the switch rule. A lane becomes eligible for default consideration only after a clean run satisfies the full `Product strategy decision` rule below — no regression in required-file recall or region/line metrics (where labels exist), token efficiency after completion at least as high, and p95 latency at or below the budget — **and** the lane's extra cost is justified: added latency for the dual-transform/rerank/multihop passes, and the offline storage/index cost plus staleness handling for the summary sidecar. Because these lanes add model dependency, latency, or storage that the deterministic eval and compression layers do not, they are intentionally held behind the gate until the benchmark frontier can attribute their wins and regressions.

No numeric advanced-lane results are claimed here. Reports surface each lane's added latency, token impact, storage/index cost where applicable, and quality metrics in the **Advanced Quality Lanes** table, but any future advanced-lane win claim must come from checked-in benchmark artifacts that contain the per-lane provenance and the required-file/region, completion-adjusted token-efficiency, and p95 fields needed to verify it.

## Low-latency conditional reranker lane

`archex_query_conditional_rerank` is a benchmark-only, **opt-in** retrieval lane (Workstream R4 of `.docs/2026-06-20-retrieval-ranking-signal-enhancement-design.md`). Prior rerank lanes ran a full cross-encoder on every query and were disqualified on p95 (Jina v3 `16.5 s`, MiniLM `3.9 s`; see the Reranker decision below). This lane keeps the `archex_query` BM25 retrieval, graph expansion, packing, and default scoring, and reranks **only when BM25 is ambiguous** — gated on the flat BM25 score CV, the post-retrieval fusion query-performance signal in `src/archex/index/fusion.py` — so the confident-BM25 common case stays model-free. The shared ambiguity gate (`bm25_is_ambiguous`) also accepts a low-AvgIDF signal, but AvgIDF is a retrieval-time input that is not threaded onto the post-assembly bundle, so this lane fires on the score CV alone. It reuses an in-process cross-encoder only (it never downloads a model), respects the remote-code opt-in policy (`src/archex/index/model_policy.py`), and bounds the model stage by a wall-clock latency cap: a model pass slower than the cap is run in a worker thread, the caller is released at the cap, and the original retrieval order is kept. It is designed for a small distilled local cross-encoder (Ettin-class) under ONNX/INT8; the operator supplies the model. The product `archex_query` path never constructs a reranker and is unchanged.

Decision rule: this lane is **not eligible for product-default promotion** unless a clean warm run satisfies the full `Product strategy decision` rule below — mean F1 at least `+0.05` over `archex_query`, token efficiency after completion at least as high, no regression in required-file recall or region metrics (`region_recall`, `region_precision`, line metrics, `context_noise_ratio`, and `relevance_per_1k_tokens`) where labels exist — **and** p95 latency at or below `3000 ms` measured warm on operator hardware. Missing the p95 budget disqualifies it regardless of quality gains, exactly as full cross-encoder rerank was disqualified. **The lane's storage cost must be justified**: for a conditional cross-encoder the storage cost is the rerank model artifact on disk (contrast late interaction, whose cost is precomputed per-token document vectors); the lane reports `rerank_model_storage_bytes` when the model resolves to a local path and `unmeasured` otherwise, and the artifact size must be accounted for before promotion. The lane must keep rerank provenance populated (`bm25_cv`, `bm25_ambiguous`, `cross_encoder_status`, `rerank_ms`, `candidates_reranked`, `rerank_model_storage_bytes`) so any latency or ordering change is attributable.

No numeric conditional-rerank results are claimed here. Any future promotion claim must come from checked-in benchmark artifacts comparing `archex_query_conditional_rerank` to `archex_query` on required-file recall, region/line metrics, `context_noise_ratio`, `relevance_per_1k_tokens`, F1, median latency, p95 latency, and the measured model storage cost.

Disposition (local evaluation, 2026-06-21): the lane was run warm against the full benchmark corpus (48 tasks) with the cross-encoder pinned to `cross-encoder/ms-marco-MiniLM-L-6-v2`, comparing `archex_query` and `archex_query_conditional_rerank`; the artifact is checked in at `.archex/rerank-diversity-validated`. The ambiguity gate fired and the model applied on every task (`cross_encoder_status=applied`). Reranking changes only the order of the returned set — required-file recall, F1, `region_recall`, and `region_precision` are identical to `archex_query` on every task — so the effect appears only in the ranking metrics, and it is opposite-signed by corpus. On the external/real-world repos (24 tasks) ranked-region MRR rose `+0.14` (file MRR `+0.15`, nDCG `+0.10`, MAP `+0.13`) at p95 `1011 ms`; on the self-repo (24 tasks, identifier-dense comprehension queries where BM25 already ranks the right file first, base file MRR `~0.98`) ranked-region MRR fell `-0.10` and p95 was `3226 ms`, above the `3000 ms` budget. No cheap query-side gate signal (BM25 score CV, top-1/top-2 score margin, AvgIDF, or query intent) separates the corpus where the model helps from where it hurts, so the lane is not promotable via a per-query gate and its value is confined to the external/NL-localization distribution. It remains benchmark-only and does not clear the F1-based default-switch gate, since the returned set — and therefore F1 — is unchanged.

## Query-adaptive diversity packing lane

`archex_query_diversity_packed` is a benchmark-only retrieval lane (Workstream R5 of `.docs/2026-06-20-retrieval-ranking-signal-enhancement-design.md`). It runs the exact `archex_query` retrieval path, then re-packs the assembled bundle with a query-adaptive MMR diversity packer derived from the efficiency-aware packer (`src/archex/serve/packing.py`). The MMR lambda is query-adaptive: a narrow single-aspect lookup keeps lambda at `1.0` so diversity is off and the lane is identical to `archex_query_efficiency_packed`, while a multi-aspect query drops redundant low-confidence tail regions. Similarity is a deterministic, local token-signature Jaccard — no model, no network.

Required-region invariant: a required/direct/high-confidence region is **never** de-selected, and the diversity plan only ever flips a redundant non-protected region to a skip while keeping every other region's baseline decision verbatim. A redundant region is dropped only when its file stays represented by another kept region, so the diversity kept-file set is a provable superset of the efficiency packer's and **file recall never regresses**. As with the other packing lanes, `compression_hidden_required_region_count` (required regions hidden by compression or elision) is reported and must stay `0`; diversity can never make incomplete context complete and never compresses, drops, or hides a required region to improve a token metric.

Decision rule: this lane is **not eligible for product-default promotion** and does not change the switch rule. It becomes eligible for default consideration only if a clean run shows required-file and region/line metrics (where labels exist) do not regress, token efficiency after completion improves, and p95 latency does not regress versus `archex_query` — in addition to the full `Product strategy decision` rule below, including p95 latency at or below `3000 ms`. The lane keeps diversity provenance populated (`diversity_applied`, `query_aspects`, `diversity_lambda`, `deselected_for_diversity`, `protected_regions`) so any de-selection is attributable.

No numeric diversity-packing results are claimed here. Any future packing-win claim must come from checked-in benchmark artifacts that contain the packing-decision counts, `deselected_for_diversity`, `relevance_per_1k_tokens`, `context_noise_ratio`, required-file/region, completion-adjusted token-efficiency, and p95 fields needed to verify it.

Disposition (local evaluation, 2026-06-21): run warm against the full benchmark corpus (48 tasks); the artifact is checked in at `.archex/rerank-diversity-validated`. The safety invariants held on every task — `compression_hidden_required_region_count` was `0` and required-file recall, recall, and F1 were identical to `archex_query`, so there is no regression at either layer. Diversity acted on the multi-aspect minority of tasks; where it acted it improved token economy modestly and safely: on the self-repo, token efficiency after completion `+0.008` and relevance per 1k tokens `+0.058` with `context_noise_ratio` `-0.024`; on the external repos, `+0.005`, `+0.004`, and `-0.007` respectively. Latency cost was negligible (self p95 `+102 ms`, external p95 `-196 ms`). The gains are real and safe but modest; the lane stays benchmark-only and does not by itself clear the full default-switch gate.

## Competitive comparison

Workstream 6 of `.docs/2026-06-18-retrieval-quality-token-efficiency-enhancement-design.md` refreshes the **public competitive comparison** after the internal lanes exist. It is comparison infrastructure, not a retrieval change, and it does not alter the switch rule. `archex benchmark headtohead competitive` renders the comparison across these lanes: the `archex_query` product default; optional benchmark-only archex candidate lanes (`archex_query_compressed`, `archex_query_efficiency_packed`); the external retrieval engine `ccc`; the raw-ripgrep/read baseline; Headroom-style compression lanes (`headroom_only_on_raw_context`, `archex_plus_headroom`); and the Graphify follow-up lanes (`graphify_build_plus_query`, `graphify_query_warm`).

The manifest tags each lane's `layer_type` (`retrieval`/`graph-memory`/`compression`/`baseline`). **Headroom is evaluated as a compression / context-management layer, not a retrieval engine**, so its lanes contribute only a compression ratio and the retrieval-quality columns are `n/a` for them. **Graphify is evaluated as a graph / memory layer, not a direct retrieval-equivalent winner**: `graphify_build_plus_query` includes graph construction/setup plus the first graph-backed answer, while `graphify_query_warm` measures only the warm graph-query path against a prebuilt graph. Reports are grouped by repo/task family and aggregate, and the report never publishes a cherry-picked aggregate-only winner.

Disposition: the improved archex candidate lanes remain **benchmark-only candidates**, not shipped defaults — per the disposition notes above, `archex_query_task_aware` tied `archex_query` on file recall and F1, and the compression/packing lanes produced only modest token reduction with the safety invariants intact, so none cleared the default-switch gate. The shipped default is still `archex_query`. The Graphify lanes are public comparison evidence only; they do not participate in default promotion.

No new competitive numbers are claimed here beyond the checked-in public artifact set. Every value in a published competitive report must come from checked-in artifacts under `benchmarks/headtohead/results/`; the current checked-in public set includes `archex`, the benchmark-only archex candidate lanes (`archex_query_compressed`, `archex_query_efficiency_packed`), `ccc`, raw-ripgrep/read, and both Graphify lanes. Headroom-layer cells appear only when an operator supplies the corresponding artifacts.

## Decision rationale

### Why this gate exists

Earlier retrieval-default evaluation mixed embedders and lacked token-efficiency and p95 latency gates. That made recall-only wins unsafe to promote into the product default. The default path has to preserve quality, token economy, and interactive latency together.

### Alternatives considered

#### Switch the product default to `archex_query_fusion_rerank`

- Pros: highest observed MRR on the clean Jina run (`0.938`) and a small F1 lift (`0.594` vs `0.589`).
- Cons: token efficiency regressed (`0.612` vs `0.701`) and p95 latency rose to `16588 ms`, far above the `3000 ms` budget.
- Disposition: rejected. It failed every non-F1 switch constraint and did not clear the required `+0.05` mean F1 delta.

#### Switch the benchmark embedder to CodeRankEmbed

- Pros: remains a plausible code-specialized candidate after the query-prefix and repeated-load fixes.
- Cons: the 2026-06-09 CodeRank run completed only `28/35` tasks due to clone DNS failures and showed an extreme partial-run p95 (`253658 ms` for fusion rerank).
- Disposition: rejected as decision evidence. The run was not clean full-frontier evidence and underperformed Jina on recall, F1, token efficiency, and p95 even on the partial set.

#### Select MiniLM as the reranker default

- Pros: materially faster than Jina reranker v3 on the same run (`3924 ms` p95 vs `16522 ms` p95).
- Cons: still misses the `<= 3000 ms` p95 budget and slightly lowers F1 (`0.586` vs `0.594`).
- Disposition: rejected. The reranker decision rule requires the selected model to stay at or below the p95 budget on the operator hardware.

### Consequences

- Default changes remain evidence-gated across quality, token economy, and latency.
- `archex_query` stays the product default until a clean warm run clears the whole rule set.
- Benchmark-only knobs such as `--embedder` and `--rerank-model` do not change shipped product behavior by themselves.
- CodeRankEmbed and lighter reranker candidates remain valid future experiments, but only after clean reruns.
## Embedder decision

Candidate embedders:

| Candidate | Benchmark flag | Rollback path |
| --- | --- | --- |
| Jina v2 code embeddings | `--embedder jina-v2` | Current benchmark default |
| CodeRankEmbed | `--embedder coderank` | Registered as `coderank`; no product default flip before approval |

Switch from Jina v2 to CodeRankEmbed only when a warm full run shows `archex_query_fusion_rerank` recall and F1 improve, token efficiency is not worse, and p95 latency does not regress.

2026-06-09 result: do not switch. The CodeRank run completed only `28/35` tasks because seven GitHub clones failed DNS resolution, so it is invalid as final A/B evidence. Even on the partial set, CodeRank fusion rerank underperformed Jina fusion rerank on recall (`0.774` vs `0.818`), F1 (`0.590` vs `0.594`), token efficiency (`0.597` vs `0.612`), and p95 latency (`253658 ms` vs `16588 ms`).

Operator commands:

```bash
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --tasks-dir benchmarks/tasks --output .archex/e2e-jina
uv run archex benchmark run --query-fusion --rerank --embedder coderank --tasks-dir benchmarks/tasks --output .archex/e2e-coderank
uv run archex benchmark readiness --input .archex/e2e-jina --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark readiness --input .archex/e2e-coderank --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
```

## Reranker decision

Candidate rerankers:

| Candidate | Benchmark flag | Decision role |
| --- | --- | --- |
| Jina reranker v3 on detected device | omit `--rerank-model` | Current default reranker |
| MiniLM cross-encoder | `--rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2` | Lighter fallback if Jina exceeds the p95 budget |
| TinyBERT cross-encoder | `--rerank-model cross-encoder/ms-marco-TinyBERT-L-2-v2` | Lighter candidate to test after MiniLM missed p95 |

Keep the highest-quality reranker that holds p95 latency at or below `3000 ms` on the operator's hardware. Evaluate Jina first with MPS device selection, then MiniLM, then TinyBERT only if heavier candidates remain over budget.

2026-06-09 result: do not change the reranker default. Jina reranker p95 was `16522 ms`, and MiniLM p95 was `3924 ms`; both miss the `<= 3000 ms` budget. MiniLM is the better latency candidate observed so far, but it still needs further tuning or a lighter local reranker before it can be selected. The next local candidate is TinyBERT.

Before comparing reranker p95, run the Jina command once as a cache warm-up and discard that output, then rerun both candidates against the warmed index cache.

Operator commands:

```bash
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --tasks-dir benchmarks/tasks --output .archex/e2e-rerank-jina
uv run archex benchmark readiness --input .archex/e2e-rerank-jina --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark gate --input .archex/e2e-rerank-jina --warn-latency-ms 3000
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 --tasks-dir benchmarks/tasks --output .archex/e2e-rerank-minilm
uv run archex benchmark readiness --input .archex/e2e-rerank-minilm --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark gate --input .archex/e2e-rerank-minilm --warn-latency-ms 3000
uv run archex benchmark run --query-fusion --rerank --embedder jina-v2 --rerank-model cross-encoder/ms-marco-TinyBERT-L-2-v2 --tasks-dir benchmarks/tasks --output .archex/e2e-rerank-tinybert
uv run archex benchmark readiness --input .archex/e2e-rerank-tinybert --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark gate --input .archex/e2e-rerank-tinybert --warn-latency-ms 3000
```

## Product strategy decision

Candidate product defaults:

| Candidate | Meaning | Decision role |
| --- | --- | --- |
| `archex_query` | Current BM25 + graph product path | Default to keep unless beaten on the full frontier |
| `archex_query_fusion_rerank` | BM25 + vector fusion + cross-encoder rerank | Candidate only if quality and latency both clear the rule |

Switch to `archex_query_fusion_rerank` only if the clean warm run shows mean F1 at least `0.05` higher than `archex_query`, token efficiency at least as high as `archex_query`, and p95 latency at or below `3000 ms`. If the rule does not pass, keep `archex_query` as the product default.

2026-06-09 result: keep `archex_query`. On the clean Jina run, `archex_query` had F1 `0.589`, token efficiency `0.701`, and p95 `2186 ms`; `archex_query_fusion_rerank` had F1 `0.594`, token efficiency `0.612`, and p95 `16588 ms`. The F1 delta was only `+0.005`, token efficiency regressed, and p95 exceeded the `3000 ms` limit.

Operator commands:

```bash
uv run archex benchmark readiness --input .archex/e2e-jina --tasks-dir benchmarks/tasks --strategy archex_query --format markdown
uv run archex benchmark readiness --input .archex/e2e-jina --tasks-dir benchmarks/tasks --strategy archex_query_fusion_rerank --format markdown
uv run archex benchmark gate --input .archex/e2e-jina --baseline .archex/e2e-tier2 --warn-latency-ms 3000
```

## C6 chunking and retrieval experiment archive

### 2026-06-15 — stack disposition after C6 validation

Stable benchmark spine:

- `feat/cast-chunker` / PR `#212`
- `feat/chunker-benchmark-arm` / PR `#213`

Archived benchmark-only experiment branches:

| Branch / PR | Experiment | Result |
| --- | --- | --- |
| `feat/adaptive-rerank-benchmark-policy` / `#214` | adaptive rerank candidate limit | Safe benchmark knob. No recall recovery. |
| `feat/adaptive-fusion-benchmark-policy` / `#215` | adaptive fusion policy | No targeted recall win. Latency regressed. |
| `feat/dual-leg-benchmark-orchestration` / `#216` | BM25 default + vector cast dual-leg retrieval | Efficiency improved. Recall did not. |
| `feat/file-first-benchmark-ranking` / `#217` | file-first ranking contract | No recall recovery. |
| `feat/file-stage-benchmark-orchestration` / `#218` | file-stage orchestration | Strong targeted wins. Broader frontier regressed. |
| `feat/expansion-controls-benchmark-policy` / `#219` | strict expansion controls | Helped `fastapi_dependency_injection`. Did not fix the main self-repo failures. |
| `feat/direct-file-preservation-benchmark-policy` / `#220` | direct file preservation | Best targeted recovery so far. Still lost to the stable spine. |
| `feat/delta-vector-cache-preservation-policy` / `#221` | delta/vector-cache preservation | Beat its parent branch. Not enough to beat the stable spine after consolidation. |
| `feat/retrieval-policy-consolidation` / `#222` | consolidated retrieval policy line | Improved recall and token efficiency. Failed the stable-spine gate on F1 and latency. |

Final disposition:

- Merge only the durable infrastructure line: `#212` then `#213`.
- Keep `archex_query` as the product default.
- Keep the current chunker default unchanged.
- Close or park `#214` through `#222` as archived benchmark evidence.

Future work starts from the stable benchmark spine, not the draft ladder. The next narrow investigation set is:

- `archex_project_index`
- `archex_project_init`
- `express_error_handling`
- `fastapi_dependency_injection` rerank
