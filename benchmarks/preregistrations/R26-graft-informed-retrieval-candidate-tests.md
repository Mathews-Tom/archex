# R26 — Graft-informed retrieval-candidate tests

This pre-registration fixes the eligibility tests for scope-aware monorepo ranking and symbol-aware exhaustive grep before either candidate has code, a registered strategy, tasks, a corpus, or results. R26 authorizes this document only. It does not authorize implementation, data generation, an MCP tool, a default change, or a Gate A reinterpretation.

## Study identity

- **Spike ID and title:** R26 — Graft-informed retrieval-candidate tests
- **Evidence class:** `adaptation`
- **Decision owner and date:** archex maintainer, 2026-09-12
- **First-run commit:** none; R26 authorizes no data-generating run
- **Policy:** [`docs/RETRIEVAL_DEFAULT_DECISIONS.md`](../../docs/RETRIEVAL_DEFAULT_DECISIONS.md#r26-retrieval-candidate-eligibility)
- **Source mechanisms:** Graft `aa1e2bb0f6326068ac64886da1e67fa25a7804de` (tag `v0.16.0`, the R19-pinned released source), specifically `src/graph/scopes.ts`, `src/ask/fuse.ts`, and `src/search/grep.ts`; no Graft result is treated as Archex evidence

A separately authorized future milestone must pass its own design gate and merge an immutable run manifest before candidate implementation. The manifest must name the control revision, task-population digest, exact command contract, seeds, environment, and candidate interface. After implementation, a separate identity binding must name the immutable candidate source revision and merge before data generation; it may bind identity only. Neither artifact may weaken or replace any hypothesis, metric, task-family requirement, invariant, latency gate, receipt gate, or terminal disposition below.

## Shared boundary

The two candidates are independent studies. A result for one cannot qualify the other. Both remain opt-in and benchmark-only. Neither may reuse personalized PageRank, revive file-first or round-robin default ranking, add an MCP capability, modify `archex_query`, change a language tier, or claim that Gate A passed.

The treatment and control in each future run must use the same repository revision, task question, labels, file eligibility rules, token budget, result cardinality, warmed index state, and operator hardware. Every planned cell must exist as either a valid result or a recorded failure. Missing, excluded, or repaired-after-inspection cells make the candidate terminally ineligible under that run identity.

## Clustering unit

The source repository is the independent unit for both studies. Tasks under one repository share layout, language, scope boundaries, symbol indexing, and parse behavior, so inference resamples repositories and keeps all of a selected repository's tasks together. Each run manifest must freeze its resample count and seed before data generation.

## Candidate A — Scope-aware monorepo ranking

### Hypothesis

On tasks constructed to expose monorepo scope imbalance, an opt-in scope-aware ranking candidate improves repository-weighted mean required-file recall over `archex_query` by at least 0.05 while preserving the single-scope context payload exactly and satisfying the receipt and warm-latency gates below.

- **Treatment:** one future scope-aware ranking candidate implementing deterministic scope discovery, shared repository-wide IDF, per-scope normalization, and a participation gate.
- **Control:** the unchanged `archex_query` product default.
- **Target population:** pinned public monorepositories with at least two deterministically discoverable scopes and tasks whose required files include a non-dominant scope.
- **Comparison family:** paired treatment-minus-control required-file recall.

### Primary metric

**Repository-weighted mean required-file recall difference on the scope-imbalance family.**

For each task, required-file recall is the number of labeled required files present in the returned context divided by the number of labeled required files. Task recalls are averaged within each repository, repository means receive equal weight, and the primary estimate is treatment minus control. Higher is better. Inference resamples repositories with every task from a selected repository kept together.

Every other quality measure — task-level recall, all-required-files-present rate, F1, MRR, nDCG, region and line recall where labels exist, context noise, relevance per 1k tokens, token efficiency after completion, median latency, and receipt counts — is secondary. The invariant and latency gates remain binding even though they are not the primary metric.

### SESOI and decision margins

- **Minimum worthwhile gain (MWG):** +0.05 repository-weighted mean required-file recall. Utility basis: a smaller gain does not justify a second ranking path, its scope-discovery complexity, or its additional receipts.
- **Non-inferiority margin (NIM):** −0.02. Cost basis: a candidate designed to recover files from weak scopes cannot accept a material completeness loss on the failure mode it targets.
- **Equivalence margin (EQM):** ±0.02. Utility basis: movement inside this band changes no implementation or operator decision and is reported as practically equivalent, not as a win.

The margins are fixed from utility and risk, not observed variance. They are never widened after any task result exists.

### Required task families

A future population is eligible only when all requirements are met before the first run:

1. Every target repository is pinned to an immutable public revision with provenance and license recorded.
2. Every treatment task identifies at least two scopes under one repository root, records the deterministic scope boundaries, and labels every required file and its scope.
3. Every treatment task places at least one required file in a non-dominant scope. The dominant scope must contain at least four times as many eligible indexed chunks as that required-file scope, measured on the frozen control index.
4. The population includes lexical-collision tasks where the dominant and non-dominant scopes share query terms or symbol names, cross-scope dependency tasks whose answer spans scopes, and weak-participation controls where an irrelevant scope must be rejected.
5. Each source repository contributes a single-scope control produced by the same file-eligibility and indexing rules with exactly one discovered scope. The control records the expected context-payload digest from `archex_query` before the candidate runs.
6. The current 19-task R19 population cannot be substituted: it was not labeled for deterministic scopes or selected to isolate scope imbalance.
7. Before data generation, the run manifest must demonstrate at least 0.80 cluster-aware power for the +0.05 SESOI under its declared variance model. Failure to meet that threshold is a feasibility no-go, not permission to weaken the SESOI.

### Single-scope and receipt gates

On every single-scope control, the candidate's retrieved and packed context bytes must equal the control digest exactly. The candidate may differ only by adding its scope-specific receipt fields. Those fields must report one searched scope, one included scope, zero rejected scopes, the deterministic scope identifier, and the participation decision.

On every multi-scope task, the receipt must list all searched scopes, all included scopes, all rejected scopes with a reason, the shared-IDF identity, per-scope normalization identity, participation threshold, candidate counts before and after participation, and the final contribution count per included scope. Counts must reconcile exactly to the candidate set. A missing field, unknown scope, duplicate scope, or non-reconciling count fails the cell.

### Latency and quality gates

The future run must measure warm latency after an unmeasured warm-up under one pinned index and process lifecycle. Treatment warm p95 must be at or below 3000 ms. Required-file, region, and line metrics must not regress on any declared single-scope or language subgroup. There may be no new zero-recall task. Any later default consideration remains subject to the complete promotion protocol; passing this candidate test does not satisfy or bypass it.

### Kill criterion and terminal disposition

- **Feasibility no-go:** no adequately powered qualifying population, non-deterministic scope discovery, unavailable shared-IDF identity, or an unpinned implementation/control identity.
- **Invariant no-go:** any single-scope context digest mismatch, personalized-PageRank dependency, incomplete/non-reconciling scope receipt, missing cell, hidden subgroup regression, new zero-recall task, or warm p95 above 3000 ms.
- **Evidence no-go:** the point estimate is below +0.05, its repository-cluster interval does not exclude zero in the beneficial direction, or the result is compatible only with the non-inferiority/equivalence margins.
- **Benchmark continuation eligible:** the point estimate reaches +0.05, its interval excludes zero in the beneficial direction, and every invariant, receipt, quality, and latency gate passes. This authorizes a terminal benchmark report only, not product registration or promotion.

## Candidate B — Symbol-aware exhaustive grep

### Hypothesis

On grep-shaped migration and audit tasks, symbol-grouped exhaustive grep reduces the repository-weighted mean tokens required to render the complete exact-hit set by at least 10% versus ungrouped exhaustive grep, while preserving exact-hit recall at 1.0 and satisfying the source-line, truncation-receipt, and warm-latency gates below.

- **Treatment:** one future repository-wide regex/fixed-string operation that groups matches by their innermost enclosing indexed symbol and may order groups by coupling/fan-in metadata.
- **Control:** ungrouped exhaustive grep over the identical eligible-file universe with identical pattern semantics.
- **Target population:** pinned public repositories and grep-shaped migration/audit tasks with human-verified exact match lines.
- **Comparison family:** paired treatment-versus-control complete-result token count.

The operation remains separate from semantic retrieval. A semantic top-k search, symbol-name search, inferred match, or MCP-only surface is not an eligible treatment.

### Primary metric

**Repository-weighted mean percentage reduction in complete-result tokens at exact-hit recall 1.0.**

For each task, tokenize the complete normalized control and treatment payloads with the repository's standard token counter. The task-level reduction is `(control_tokens - treatment_tokens) / control_tokens`. Task reductions are averaged within each repository, and repository means receive equal weight. Higher is better. A task with treatment exact-hit recall below 1.0 is an invariant failure and is not assigned an efficiency value.
The primary token count uses benchmark-owned canonical projections, not each arm's preferred renderer. The control emits one line per hit as `<path>:L<line>:<exact source text>`. The treatment emits one `<path>:<qualified symbol>:L<start>-L<end>` group header followed by one `L<line>:<exact source text>` line per hit; module-level hits use the literal symbol `<module>`. Both append the same searched-file, matching-file, and total-hit receipt line, and the treatment additionally appends its grouped-hit, module-level-hit, and truncation counts. Timestamps and measured latency are the only excluded fields. The complete UTF-8 projections are tokenized with `archex.reporting.count_tokens` (`cl100k_base`). A future runner must produce these projections byte-for-byte before it may compute the primary metric.

Exact-hit recall, region MRR/nDCG, group-order MRR, context noise, searched files, total hits, grouped hits, module-level hits, truncated files/hits, median latency, and p95 latency are secondary. Exact-hit/source/receipt/latency requirements remain binding.

### SESOI and decision margins

- **Minimum worthwhile gain (MWG):** 10% fewer complete-result tokens. Utility basis: a smaller reduction does not justify a second search renderer and symbol-grouping contract.
- **Non-inferiority margin (NIM):** −5% token reduction, meaning up to 5% more tokens. Cost basis: bounded metadata can add small overhead, but larger expansion defeats the candidate's context-economy purpose.
- **Equivalence margin (EQM):** ±5% token reduction. Utility basis: movement inside this band does not change which exhaustive search form an operator chooses.

The margins are fixed before corpus construction and are never changed from observed token counts.

### Required task families

A future population is eligible only when all requirements are met before the first run:

1. Every repository is pinned to an immutable public revision with provenance and license recorded, and the eligible-file universe and ignore rules are frozen by digest.
2. Every task freezes one regex or fixed string, case and multiline semantics, the complete set of expected matching file/line/text triples, and each hit's expected innermost enclosing symbol or explicit module-level disposition.
3. The population includes rename/migration patterns repeated across many symbols, audit patterns with module-level and nested-symbol hits, identical text under distinct symbols, files with no matching symbol span, and high-cardinality patterns that force bounded rendering.
4. At least one task exercises every supported language tier included in the future plan. An unsupported tier receives an explicit disposition before the run; R26 authorizes no language work.
5. The ungrouped control and grouped treatment search the exact same files and use the exact same matching engine and pattern semantics. Only grouping, ordering, receipt metadata, and rendering may differ.
6. The complete-result mode must fit within a separately recorded safety ceiling. Tasks exceeding it are retained as high-cardinality receipt tests and cannot contribute to the primary token metric.
7. Before data generation, the run manifest must demonstrate at least 0.80 cluster-aware power for the 10% SESOI under its declared variance model. Failure to meet that threshold is a feasibility no-go, not permission to weaken the SESOI.

### Exhaustiveness, source, and receipt gates

In complete-result mode, treatment and control must return the identical multiset of repository-relative file, one-based line number, and exact source-line text triples. Exact-hit recall and precision must both equal 1.0. Every treatment group must name an enclosing indexed symbol and exact span, or explicitly classify the hit as module-level. Group membership must reconcile exactly to the complete hit set.

Every cell receipt must report the eligible and searched file counts, matching file count, total hit count, grouped hit count, module-level hit count, rendered file/group/hit counts, truncated file/group/hit counts, matching-engine identity, pattern semantics, graph/coupling identity when ordering uses it, and the output token count. Coupling/fan-in may order groups but cannot filter matches. All counts must reconcile.

Bounded mode may omit rendered groups only after complete matching and counting. It must report every omitted file/group/hit count and a deterministic continuation boundary. Silent truncation, early search termination presented as exhaustive, or an unknown searched-file count fails the cell.

### Latency and quality gates

The future run must measure warm latency after an unmeasured warm-up under one pinned index and process lifecycle. Treatment warm p95 must be at or below 3000 ms. Exact-hit recall/precision, source-line identity, and searched-file accounting are zero-tolerance invariants. Any rank-sensitive benefit from coupling/fan-in is exploratory and cannot rescue a token or exhaustiveness failure.

### Kill criterion and terminal disposition

- **Feasibility no-go:** no adequately powered qualifying population, unmatched control/treatment pattern semantics, unavailable exact-hit labels, or an unpinned implementation/control identity.
- **Invariant no-go:** exact-hit recall or precision below 1.0, any source-line mismatch, non-reconciling group or receipt counts, silent truncation, coupling-based filtering, missing cell, or warm p95 above 3000 ms.
- **Evidence no-go:** mean token reduction is below 10%, its repository-cluster interval does not exclude zero in the beneficial direction, or the result falls only inside the non-inferiority/equivalence margins.
- **Benchmark continuation eligible:** mean token reduction reaches 10%, its interval excludes zero in the beneficial direction, and every exhaustiveness, source, receipt, and latency gate passes. This authorizes a terminal benchmark report only, not an MCP tool, semantic-retrieval integration, product registration, or promotion.

## Run and analysis boundary

R26 freezes the hypotheses, metrics, margins, task-family admission rules, invariants, latency thresholds, receipt schemas, and terminal dispositions above. It creates no candidate, strategy name, task file, corpus manifest, runner, result artifact, or command.

A separately approved future candidate milestone must create and merge a run-specific tracked manifest before implementation. The manifest must bind exactly one candidate contract, its control and immutable source revision, the qualifying population and digest, commands, seeds, token counter, warm-up lifecycle, environment, failure artifacts, and cluster-bootstrap procedure. After implementation, a separate tracked identity binding must name the immutable candidate source revision and merge before data generation. Both artifacts must cite this document and fail validation if any frozen R26 field is missing or weakened; the identity binding cannot alter the protocol or population.

## Post-hoc changes

None. Any change after candidate data exists requires a new protocol identity; it cannot be edited into this document.
