# Execution Prompts — archex Strategic Reassessment Program

One `/goal` block per milestone in `.docs/DEVELOPMENT_PLAN.md` §6. Each block is self-contained and paste-ready. Run order follows §8 Critical Path; R6 is cancelled and R6.1 is its separately authorized, ordering-only replacement.

Milestone IDs are `R`-prefixed. **Never reuse `M1`–`M17`** — that numbering is bound to the prior plan by tracked references in `CHANGELOG.md` and `benchmarks/results/m*/DECISION.md`.

## Global execution rules (apply to every goal)

- Use `stacked-prs`; each implementation PR is based on the preceding stack branch until that base merges.
- Use Conventional Commits, atomic commits, no attribution, and independently reviewable PRs.
- Run the mandatory pre-implementation design gate before creating product-code branches or changing product code.
- `.docs/DEVELOPMENT_PLAN.md`, `.docs/EXECUTION_PROMPTS.md`, and `.docs/spikes/**` are tracked reassessment contracts. `.docs/DEVELOPMENT_PLAN_HISTORY.md` is ignored local evidence; rebuild it from committed artifacts, merged PRs, CI, and current code when absent. All remaining `.docs/` paths stay local by repo convention. Force-add the tracked exceptions when a developer's global ignore also excludes `.docs/`.
- A material plan change must update the current milestone and every affected future milestone before implementation. Rebuild §3, §4, and §8 after the update.
- A docs-only reconciliation PR is required for a material revision. It must be reviewed, green, and externally merged before code begins.
- A shared mismatch in a proposed parallel wave blocks product-code work in every affected lane. Do not continue scaffolding, partial implementation, or isolated ledger writes while reconciliation is pending.
- `GO` only makes the milestone stack merge-eligible. Release preparation stays deferred until every milestone in its train is externally merged.
- **Gates A, B, and C are declared in advance and are not renegotiable after seeing results.** Adjusting a threshold post-hoc is a plan violation.
- **Pre-registration ordering is load-bearing.** Where a milestone requires a pre-registration, its tracked file merges before the first run and commit order is the proof. New pre-registrations belong in `benchmarks/preregistrations/`; only R3's historical, already-tracked record remains under `.docs/spikes/`.
- Never solicit an external/bot code review. The configured reviewer runs automatically.
- The full local gate referenced below is: `uv run ruff check . && uv run ruff format --check . && uv run pyright . && uv run pytest`.

---

### R1 — Planning freeze and pre-registration substrate

```text
/goal Deliver milestone R1 (Planning freeze and pre-registration substrate) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section A R1 + .docs/strategic-reassessment/05-ROADMAP.md §1 and 00-DIAGNOSIS.md §5. Preconditions: none. Repo: Python 3.11+, uv, pytest, ruff, pyright strict, GitHub Actions CI, manual release (tag + uv build/publish + gh release).
OBJECTIVE: Stop mechanism work and establish the pre-registration discipline that makes every later result interpretable. Success contract: prior forward milestones marked SUSPENDED with a pointer to .docs/strategic-reassessment/; a CONTRIBUTING.md freeze clause naming the four prohibited change classes (new retrieval lanes, default-promotion attempts, new language tiers, new MCP tools) and Gate A as the lift condition, with R5 named as the single carve-out; .docs/spikes/TEMPLATE.md enumerating hypothesis, primary metric, SESOI, separately-derived MWG/NIM/EQM (EQM strictly positive and utility-derived), clustering unit, kill criterion, and evidence class; .docs/DEVELOPMENT_PLAN_HISTORY.md seeded and verified ignored.
RELEASE TRAIN: target=unversioned; included milestones=R1; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R2, R3, R4, R5, R6).
4. Append one ledger entry: timestamp, milestone, decision, trigger, evidence, plan/prompt sections changed, downstream impact, and implementation authorization.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R1 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`. This records a completed diagnosis but blocks product-code work until the reconciliation prerequisite merges.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop. After a reconciliation PR merges, repeat this gate and require `DESIGN GO — PLAN REVISION: none` before implementation.

SPECIFIC GATE CHECK: confirm no forward milestone from the prior (deleted) plan was silently resumed after 2026-07-24. Inspect `git log` since that date and `CHANGELOG.md` `[Unreleased]`.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R1 design` as a docs-only prerequisite PR. It contains the authoritative plan/prompt updates and any required `.gitignore` tracking exception, contains no product code, must be reviewed, green, and externally merged before the implementation stack, and must not be folded into an implementation PR.

PLANNED STACK (refine only to keep PRs reviewable):
0. Conditional prerequisite `docs(plan): reconcile R1 design` — scope: authoritative plan/prompt updates and the required `.gitignore` tracking exception only; gate: reviewed, green, and merged before the implementation stack.
1. PR-1 `docs(contributing): freeze mechanism work until Gate A` — scope: CONTRIBUTING.md, suspension markers; commits: freeze clause, suspension markers; verification: `git diff --name-only <base>..HEAD -- src/` prints nothing
2. PR-2 `docs(spikes): add pre-registration template` (on PR-1) — scope: .docs/spikes/TEMPLATE.md, history ledger seed; commits: template, ledger; verification: `git check-ignore -v .docs/DEVELOPMENT_PLAN_HISTORY.md` exits 0

CONSTRAINTS: no scope leakage, no `src/` changes whatsoever, repo style, no version/changelog updates.
VERIFICATION (must pass): `git check-ignore -v .docs/DEVELOPMENT_PLAN_HISTORY.md` exits 0; `git diff --name-only <base>..HEAD -- src/` prints nothing; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; security, data safety, and rollback requirements are addressed where relevant.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; cumulative acceptance and integration hold; CI is green; no regression coverage is removed without replacement.
- The docs-only root, when present, is reviewed and green before dependent code PRs.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
- `GO` requires `DESIGN GO`, every PR correctly based/reviewed/green, local verification, and full milestone acceptance. `NO-GO` applies to pending or failed checks, incomplete review, scope drift, ambiguous readiness, manual gates, or unresolved release target.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R2 — Truth-in-claims correction

```text
/goal Deliver milestone R2 (Truth-in-claims correction) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section A R2 + .docs/strategic-reassessment/04-PRODUCT-AND-ECONOMICS.md §1.1-1.3 and 00-DIAGNOSIS.md §1.1. Preconditions: R1 merged. Repo: as above. Relevant code: src/archex/metrics/math.py (savings_pct_vs_targeted_read), src/archex/metrics/capture.py (_targeted_file_tokens), src/archex/benchmark/scorecard.py, src/archex/benchmark/cross_tool.py (tokens_at_recall).
OBJECTIVE: Stop publishing numbers that cannot survive scrutiny, before anyone challenges them. Success contract: the savings headline in README and docs/LOCAL_METRICS.md re-points at savings_pct_vs_targeted_read; the self-repo row is withdrawn from every quoted figure; benchmarks/cross-tool-efficiency/ and docs/LOCAL_BENCHMARK_EVIDENCE.md are annotated with the blind-read semantics of tokens_at_recall plus the measured units-read distribution (median 6, mean 18, max 164 over the 52 comparable tasks); BenchmarkScorecardRow.downstream_success_rate is renamed required_file_completeness_rate with a docstring stating it is a function of required-file recall and that no model is in the loop; CHANGELOG.md [Unreleased] records all of it.
RELEASE TRAIN: target=> GAP: version not source-traceable — operator selects at preparation time (train claims-and-cost); included milestones=R2, R5; preparation trigger=both externally merged, carrying the pre-existing [Unreleased] prior-M6-M9 entries; required artifacts=both (pyproject.toml version + CHANGELOG.md); release verification=full local gate green on the release commit and `uv run archex mcp-schema-size --format json` reporting the reduced total; publication=git tag then `uv build` then `uv publish` then `gh release create` (manual; no release CI workflow exists).

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R6, R10, R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R2 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: confirm `savings_pct_vs_targeted_read` is actually populated on the live ledger write path before re-pointing the public headline at it. If it is null in practice, that is a material mismatch.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R2 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R2 design`.
1. PR-1 `fix(metrics): publish savings against the targeted-read baseline` — scope: README.md, docs/LOCAL_METRICS.md; commits: headline re-point, self-repo row withdrawal; verification: `grep -rn "445×\|673×\|99\.78\|99\.64" README.md docs benchmarks/*.md` returns nothing
2. PR-2 `docs(benchmarks): annotate the cross-tool blind-read baseline` (on PR-1) — scope: benchmarks/cross-tool-efficiency/, docs/LOCAL_BENCHMARK_EVIDENCE.md; commits: semantics note, units-read distribution
3. PR-3 `refactor(benchmark): rename downstream_success_rate to required_file_completeness_rate` (on PR-2) — scope: src/archex/benchmark/scorecard.py, tests, CHANGELOG.md; commits: rename + docstring, test updates, changelog; verification: `uv run archex benchmark validate --kind evidence` over every artifact in benchmarks/evidence/

CONSTRAINTS: change how numbers are *labelled and published*, never how they are *computed*; delete no historical artifact; no version bump in this stack.
VERIFICATION (must pass): `grep -rn "downstream_success_rate\|445×\|673×\|99\.78\|99\.64" src tests docs README.md benchmarks/*.md` returns nothing; `uv run pytest tests/ -k "scorecard or metrics or cross_tool"` green; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; security, data safety, and rollback requirements are addressed where relevant.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; cumulative acceptance and integration hold; CI is green; no regression coverage is removed without replacement.
- Every remaining savings figure names its baseline in the same sentence.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: claims-and-cost (> GAP: version) — RELEASE PREP: pending` or `NO-GO — RELEASE: claims-and-cost — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R3 — S0 external replication gate (BLOCKING — Gate A)

```text
/goal Deliver milestone R3 (S0 external replication gate) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section B R3 + .docs/strategic-reassessment/03-SPIKES.md S0 and 00-DIAGNOSIS.md §2. Preconditions: R1 merged. Repo: as above.
OBJECTIVE: Establish whether this harness can reproduce any result anyone else has published. Success contract: .docs/spikes/S0-replication-gate.md pre-registered and merged before the first run, fixing the target cell and the equivalence band in advance; a reproduction harness under benchmarks/replication/ pinned to RLCoder's (arXiv:2407.19487) own dataset, retriever, generator, and metric, run in the paper's own reference setup and NOT inside archex's pipeline; the cAST (arXiv:2506.15655) arm recorded with whatever disposition its released artifact supports; benchmarks/evidence/s0-rlcoder-replication.json recording the reproduced delta with a bootstrap interval and benchmarks/evidence/s0-cast-replication.json recording the cAST arm; GATE-A.md stating in one line whether the reproduced delta falls inside the pre-registered equivalence band around the paper's reported point estimate.
RELEASE TRAIN: target=unversioned; included milestones=R3; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R7, R13 — both are void on a Gate A fail).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R3 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: confirm each paper's artifact is still fetchable and its cited figure is still as reported. Neither paper publishes an interval, so the gate compares against a pre-registered equivalence band. An arm whose reference setup is not released is recorded as unrunnable, never scored as a pass; a gate with no runnable arm is `DESIGN NO-GO`.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R3 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R3 design`.
1. PR-1 `docs(spikes): pre-register S0 replication gate` — scope: .docs/spikes/S0-replication-gate.md only; commits: pre-registration; gate: MUST merge before any run — commit order is the proof
2. PR-2 `feat(benchmark): validate external replication artifacts` (on PR-1) — scope: the `--kind replication` validator plus its tests; verification: the validator rejects an artifact missing any pin, class label, or verdict field
3. PR-3 `test(replication): reproduce RLCoder in its own reference setup` (on PR-2) — scope: benchmarks/replication/; commits: pinned harness, run recording; verification: the recorded command reruns to the same figure
4. PR-4 `docs(gate): record Gate A verdict` (on PR-3) — scope: benchmarks/evidence/s0-rlcoder-replication.json, benchmarks/evidence/s0-cast-replication.json, GATE-A.md; commits: evidence artifacts, verdict; verification: `uv run archex benchmark validate --kind replication --input <artifact>` exits 0 for both

CONSTRAINTS: run the mechanism in the PAPER's setup, never inside archex's pipeline; change no archex retrieval code; label every arm `replication` class; do not soften the gate if the reproduction misses.
VERIFICATION (must pass): `uv run archex benchmark validate --kind replication --input benchmarks/evidence/s0-rlcoder-replication.json` and the same command over benchmarks/evidence/s0-cast-replication.json both exit 0; the reproduction command recorded in GATE-A.md reruns to the same figure; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; the pre-registration merged before the run; behavior is meaningfully tested.
- Failures are loud; the reproduction is pinned (dataset, model, commit) and rerunnable.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- GATE-A.md states the verdict in one line and does not editorialize a miss into a pass.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
- Additionally report the Gate A outcome verbatim: `GATE A PASS — <paper>, reproduced delta <x> within pre-registered band <y>` or `GATE A FAIL — no published win reproduced in its own setup`.
- A Gate A FAIL stops all research work: R7-R16 are cancelled, the program reduces to Section A + Section C outputs plus a root-cause engineering effort. This is pre-declared and not renegotiable.
DONE: design verdict, the reviewed stack, the merge verdict, and the Gate A outcome with evidence.
```

---

### R4 — S2 corpus validity audit (BLOCKING)

```text
/goal Deliver milestone R4 (S2 corpus validity audit) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section B R4 + .docs/strategic-reassessment/03-SPIKES.md S2 and 00-DIAGNOSIS.md §1.2-1.4. Preconditions: R1 merged. Runs in parallel with R3. Repo: as above.
OBJECTIVE: Quantify exactly how much the existing corpus can detect, so later effect sizes and equivalence margins are grounded rather than assumed. Success contract: a leakage score for all benchmarks/tasks/*.yaml (gold symbol or path appearing verbatim in question or keywords; 8 of 21 already confirmed in the loc_* family); ICC over repo clusters with items-per-cluster distribution and largest-cluster share (known: 24/64 self-repo = 37.5%); an empirical held-out leak measurement for benchmarks/held_out.txt (note archex_project_status and archex_query_pipeline are self-repo-cluster members that also appear in the tuning set); a power simulation over the measured cluster-size distribution reporting how many INDEPENDENT REPOS make a utility-derived EQM reachable; separately derived MWG, NIM, and EQM per candidate primary metric with EQM strictly positive; benchmarks/evidence/s2-corpus-validity.json.
RELEASE TRAIN: target=unversioned; included milestones=R4; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R9, R10).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R4 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: re-count the corpus before auditing. Task counts have changed across milestones; do not assume 64 or 66.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R4 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R4 design`.
1. PR-1 `test(benchmark): score label leakage across the task corpus` — scope: an audit script plus its tests; commits: leakage scorer, tests; verification: the scorer reruns deterministically
2. PR-2 `test(benchmark): measure clustering, held-out leakage, and ICC` (on PR-1) — scope: clustering statistics, held-out leak measurement; commits: ICC/effective-N, held-out leak
3. PR-3 `docs(benchmark): derive MWG/NIM/EQM and publish the validity audit` (on PR-2) — scope: power simulation, margins, benchmarks/evidence/s2-corpus-validity.json; commits: simulation, margins, artifact; verification: `uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s2-corpus-validity.json` exits 0

CONSTRAINTS: measure only — fix no corpus defect, delete no task, change no retrieval; EQM must be strictly positive and derived from practical utility and cost, NEVER from observed SD; MWG, NIM, and EQM are three distinct quantities and must not be conflated; the existing +0.05 F1 / no-regression / p95<=3000ms product rule is a NIM and must not be reused as an EQM.
VERIFICATION (must pass): `uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s2-corpus-validity.json` exits 0; the audit script reruns deterministically to the same figures; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; every statistic is reproducible from a recorded command; behavior is meaningfully tested.
- Failures are loud; no statistic is reported without its computation being inspectable.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- The report states plainly, per candidate primary metric, whether ANY realistic corpus size makes its EQM reachable. A metric with no reachable EQM is retired here, not after R10.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R5 — MCP retrieval-gated tool disclosure

```text
/goal Deliver milestone R5 (MCP retrieval-gated tool disclosure) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section C R5 + .docs/strategic-reassessment/01-LITERATURE-POSITION.md §3 and 05-ROADMAP.md §2 Lane C. Preconditions: R1 merged. Runs in parallel with R3/R4 and ships regardless of Gate A. Repo: as above. Relevant code: src/archex/integrations/mcp.py; prior work at benchmarks/results/m11_mcp_schema_overhead/DECISION.md.
OBJECTIVE: Cut the fixed per-turn MCP tool-schema cost from a measured 3859 tokens to under 1000 by exposing tools on demand instead of statically. The earlier ~6000 figure was not this repo's baseline and is corrected at the design gate. Success contract: retrieval-gated tool exposure whose advertised default is a minimal retrieval entry point and which expands once the client retrieves, signalled via the MCP notifications/tools/list_changed capability; mcp-schema-size reporting tokens as well as characters, since the acceptance bar is in tokens and the command measured only characters; every tool still reachable (already structural -- call_tool dispatches by name regardless of what list_tools advertised); a documented working configuration for clients that cannot discover on demand, with --no-disclosure restoring the previous surface verbatim (--tools does not disable the gate; it bounds what is advertised once the gate opens); measured schema size at each stack tip; a DECISION.md under benchmarks/results/ following the existing convention; CHANGELOG.md [Unreleased] entry; docs/CLIENT_COMPATIBILITY_MATRIX.md updated for the discovery requirement.
RELEASE TRAIN: target=> GAP: version not source-traceable — operator selects at preparation time (train claims-and-cost); included milestones=R2, R5; preparation trigger=both externally merged, carrying the pre-existing [Unreleased] prior-M6-M9 entries; required artifacts=both (pyproject.toml version + CHANGELOG.md); release verification=full local gate green plus `uv run archex mcp-schema-size --format json` reporting the reduced total; publication=git tag then `uv build` then `uv publish` then `gh release create` (manual).

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R5 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: read benchmarks/results/m11_mcp_schema_overhead/DECISION.md FIRST. Prior-M11 PR-1 (tool scoping) and PR-3 (consolidated graph_query with a deprecation shim) already landed. Do not redo them; build on them.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R5 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R5 design`.
1. PR-1 `feat(mcp): expose tools on demand from a minimal core set` — scope: src/archex/integrations/mcp.py; commits: on-demand exposure, core-set default; verification: `uv run archex mcp-schema-size --format json`
2. PR-2 `feat(mcp): add a documented fallback for clients without on-demand discovery` (on PR-1) — scope: mcp.py, install-client path; commits: fallback, tests; verification: `uv run pytest tests/ -k mcp`
3. PR-3 `docs(mcp): record the schema-overhead decision and client impact` (on PR-2) — scope: benchmarks/results/<dir>/DECISION.md, docs/CLIENT_COMPATIBILITY_MATRIX.md, CHANGELOG.md; commits: decision, matrix, changelog

CONSTRAINTS: add and remove no tool capability; change no tool behavior; every tool stays reachable; no version bump in this stack; this is the single named carve-out from R1's freeze and must not grow beyond cost reduction.
VERIFICATION (must pass): `uv run archex mcp-schema-size --format json` reports a default-scope total below 1000 tokens with the ~6000 baseline recorded alongside it in the decision document; `uv run pytest tests/ -k mcp` green; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; a client losing tool access would fail a test, not degrade silently.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- Per-client verification is recorded in the compatibility matrix.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: claims-and-cost (> GAP: version) — RELEASE PREP: pending` or `NO-GO — RELEASE: claims-and-cost — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R6 — S7 determinism as prefix-cache economics `CANCELLED — INVALID SESSION FIXTURE`

```text
No execution prompt. R6 is cancelled.

PR #592 merged the S7 pre-registration. PR #593 was reviewed and closed without merge after independent replay found every rendered prefix cache-ineligible: 72 prefixes across all three arms (24 per arm) were 50–56 `cl100k_base` tokens, below the recorded 512-token Claude Opus 5 floor. The resulting zero cache hits and zero cost deltas are construction facts, not an economics null, so the 5% kill criterion is not triggered.

Do not revive this run, merge its closed branch, or make an economics claim from its artifact. A replacement requires a new milestone, a fresh pre-registration, and an independently inspected cache-eligible session fixture before any data-generating command.
```

---

### R6.1 — S7 determinism as prefix-cache economics replacement `PENDING — FRESH PRE-REGISTRATION REQUIRED`

```text
/goal Deliver milestone R6.1 (S7 determinism as prefix-cache economics replacement) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section C R6.1 + .docs/strategic-reassessment/03-SPIKES.md S7 + 01-LITERATURE-POSITION.md §G6. Preconditions: R1 merged; R6 remains cancelled and PR #593 remains closed without merge. The unrun first-party protocol in PR #596 is retired. This is an explicitly authorized, benchmark-only replacement after Gate A failed; no successor milestone is reopened. Repo: Python 3.11+, uv, pytest, ruff, pyright strict, GitHub Actions CI. Relevant boundaries: the shipped default retrieval path and `benchmarks/preregistrations/`.
OBJECTIVE: Measure OpenRouter-metered prefix-cache economics over a fresh cache-eligible, frozen, 12-repository, three-turn maintenance-session fixture. The fixture is provider-neutral; its sole registered measurement cell is OpenRouter `anthropic/claude-opus-5` pinned to the Anthropic upstream with fallbacks disabled and explicit cache control. Each session uses `token_budget: 8192`; the 12 repositories comprise the self-repo plus at least one task from every current external corpus family. The same selected chunks, session IDs, resolution labels, model-price schedule, and turn content apply to deterministic, seed-recorded perturbed, and seed-recorded ANN-style ordering arms; only rendered context order differs. The comparator emits a permutation during fixture construction and replays that recorded order at measurement time; it performs no live ANN retrieval. OpenRouter may return enforced response tokens despite `max_tokens: 0`; record them, exclude their cost and content from the metric and all later request history, and use only provider-observed prompt cost. Success contract: a new OpenRouter pre-registration merged before fixture construction or data generation; a committed fixture with exact rendered-prefix, routing, prompt-cost-isolation, and provider-usage receipts; a dedicated validator; and an original, fixture-bounded decision with no product, retrieval-quality, literature, or Gate-A claim.
RELEASE TRAIN: target=unversioned; included milestones=R6.1; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present, R6's cancellation record, and the current pre-registration template.
2. Inspect the current codebase plus merged R1 predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and the current provider pricing/cache documentation.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, the 12-repository matrix, and every listed dependent milestone (none; R16 remains cancelled).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R6.1 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: before PR-1, confirm from official OpenRouter documentation and its live `anthropic/claude-opus-5` model endpoint that explicit cache control, cache-write/read usage fields, and a separable provider-observed prompt-cost field are supported. Fix `provider.only: ["anthropic"]`, `allow_fallbacks: false`, the returned `provider: "Anthropic"` assertion, model price schedule, response-output exclusion, and source retrieval timestamps in the new pre-registration. Before PR-2, confirm `OPENROUTER_API_KEY` and quota without printing a secret. A changed price invalidates dollar interpretation but not hit-rate interpretation; absent, changed, or incompatible cache semantics, routing pinning, usage fields, or prompt-cost isolation is a design no-go that blocks the run. Missing credentials or quota blocks fixture construction and evidence generation. Do not substitute local token counting for provider-accepted cache-write evidence.

RECONCILIATION RULE: R6.1-DG-004 is material. Open `docs(plan): reconcile R6.1 OpenRouter provider receipt field` as a docs-only prerequisite containing the matching plan and prompt update; it must be reviewed, green, and externally merged before the replacement pre-registration, fixture, code, or data PR. After it merges, rerun this gate and require `DESIGN GO — PLAN REVISION: none`.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R6.1 OpenRouter provider receipt field` — scope: authoritative plan and execution prompt only; must externally merge before PR-1.
1. PR-1 `docs(spikes): pre-register OpenRouter S7 output-accounting replacement` — scope: `benchmarks/preregistrations/S7-determinism-economics-r6.1-openrouter-output.md` only; preserves and retires PR #596's unrun direct protocol and PR #598's unrun zero-output protocol; fixes the 12 pinned task/revision rows, three-turn question sequences, `token_budget: 8192`, self-repo-plus-corpus-family selection rule, arm matrix, seeds, OpenRouter `anthropic/claude-opus-5` model/pricing/usage lookup URLs and timestamps, pinned Anthropic upstream/no-fallback policy, explicit cache control, response-output exclusion, distinct `+5%` MWG / `-5%` NIM / `[-5%, +5%]` EQM utility justification, `original` evidence class, per-cell `superior` / `equivalent` / `inconclusive at this N` disposition rules, 10,000-resample paired repository bootstrap, and provider receipt gate; MUST merge before fixture construction or any data-generating command.
2. PR-2 `test(benchmark): freeze and inspect cache-eligible S7 sessions` (on PR-1) — scope: benchmark-only fixture construction, committed 12-repository session fixture, recorded seed comparator permutations, exact OpenRouter prewarm/replay usage receipts, matrix digest, and focused tests; merge only after an independent fixture review recomputes every rendered-prefix SHA, routing assertion, receipt linkage, and prompt-cost isolation. It must fail if any arm has a different chunk identity, label, model, pricing, requested routing policy, resolved upstream provider, or missing, zero, mismatched, fallback, or output-contaminated receipt.
3. PR-3 `test(benchmark): measure S7 ordering cache economics` (on PR-2) — scope: ordering-economics runner, dedicated JSON validator, focused tests, `benchmarks/evidence/s7-determinism-economics-r6.1.json`, and `benchmarks/evidence/S7-R6.1-DECISION.md`; commits: runner and validators, frozen-fixture run, decision. Run only after PR-2 merges and the independent fixture review is recorded.

CONSTRAINTS: change archex's retrieval, ranking, and ordering in no way; do not expose the OpenRouter client on a product path; use no API key from source or artifact; request only `anthropic/claude-opus-5` with `provider.only: ["anthropic"]` and `allow_fallbacks: false`; request `max_tokens: 0`; reject a receipt whose resolved upstream is not Anthropic or that reports fallback activity; record every completion usage field but calculate economics only from the provider-observed prompt-cost field and never carry response content into a later request; compare no retrieval quality, task resolution, or live ANN behavior; do not use the invalid R6 fixture or artifact; model an ANN arm as a seed-recorded reordering of identical chunks; do not batch arms in a way that shares a cache state; reject every undeclared matrix cell or arm lacking a requested prewarm/replay receipt or returned provider usage fields.
VERIFICATION (must pass): the replacement pre-registration commit is an ancestor of fixture and evidence commits; with `OPENROUTER_API_KEY`, `uv run archex benchmark determinism-economics --sessions benchmarks/determinism_economics_r6_1/sessions.json --output benchmarks/evidence/s7-determinism-economics-r6.1.json --preregistration-commit <merged-SHA>` regenerates the artifact; `uv run archex benchmark validate --kind determinism-economics-r6-1 --input benchmarks/evidence/s7-determinism-economics-r6.1.json` exits 0; mutation tests prove matrix-digest tampering, changed routing, fallback metadata, missing receipt linkage, zero cache reads/writes, an arm lacking required usage fields, and response-cost contamination fail; full local gate green.
REVIEW:
Per PR:
- Scope matches its purpose; the replacement pre-registration merges before fixture construction or data generation; no R6 artifact is reused.
- Failures are loud; no missing provider receipt, changed SHA, changed routing, fallback, cache-ineligible prefix, zero cache-use receipt, undeclared cell, or arm lacking required usage fields can record an economic result.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- Independent review verifies OpenRouter usage receipts, requested routing, resolved Anthropic upstream, and prewarm/replay linkage before the measurement command, then verifies every arm made the requested cache-enabled prewarm and replay calls and returned their provider usage fields.
- The decision has no retrieval-quality, product, literature, or Gate-A claim. If the `ann_baseline` comparison fails the `+5%` MWG or includes zero, it explicitly retires the economic framing and preserves determinism only as reproducibility. If it clears the pre-registered rule, it licenses only an original, fixture-bounded OpenRouter-metered input-cost result and authorizes no product, default-ordering, retrieval-quality, literature, or Gate-A claim.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.

```

---

### R7 — Real-agent execution harness

```text
/goal Deliver milestone R7 (Real-agent execution harness) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section D R7 + .docs/strategic-reassessment/03-SPIKES.md S1 and 00-DIAGNOSIS.md §1.1. Preconditions: R3 merged WITH A GATE A PASS, and R4 merged. Repo: as above. Relevant code: src/archex/benchmark/bundle_eval.py (existing opt-in operator-supplied-evaluator lane), src/archex/benchmark/fixed_agent.py (the ten-line stand-in this milestone replaces).
OBJECTIVE: Put a real model in archex's benchmark loop for the first time, with context construction as the only manipulated variable. Success contract: a driver extending bundle_eval.py that holds model, scaffold, prompt template, and temperature fixed and records them in provenance, with N>=3 seeds; the six conditions {ripgrep-agent, BM25-only, dense-only, archex default, archex + symbolic rerank, oracle gold, full-repo dump} as selectable arms; first-class cost instrumentation emitting tokens-to-resolution, tool-calls-to-resolution, wall clock, and $/resolved-task per run; two-level outcome capture (resolve rate AND retrieved-context P/R/F1 against the gold diff).
RELEASE TRAIN: target=unversioned; included milestones=R7; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output. Confirm GATE-A.md records a PASS.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R8, R9, R10, R12, R13, R14, R15).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R7 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: resolve DEVELOPMENT_PLAN.md §2's GAP naming the fixed model and scaffold, and record both in §2, BEFORE writing the driver. Changing either later invalidates every downstream comparison. Also re-read bundle_eval.py's existing operator-supplied-evaluator contract and extend it rather than building a parallel harness. A Gate A FAIL means this milestone does not run at all.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R7 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R7 design` — including the model/scaffold pin.
1. PR-1 `feat(benchmark): add a fixed-agent driver over the bundle-eval lane` — scope: src/archex/benchmark/bundle_eval.py and a new driver module; commits: driver, provenance pin, seeds
2. PR-2 `feat(benchmark): implement the six context conditions as selectable arms` (on PR-1) — scope: condition registry; commits: arms, config serialization
3. PR-3 `feat(benchmark): instrument per-run cost as a first-class output` (on PR-2) — scope: cost fields on the result model; commits: instrumentation, tests asserting non-null
4. PR-4 `feat(benchmark): capture two-level outcomes against the gold diff` (on PR-3) — scope: outcome capture; commits: resolve-rate and context P/R/F1
5. PR-5 `refactor(benchmark): retire the fixed_agent stand-in from the real-agent path` (on PR-4) — scope: src/archex/benchmark/fixed_agent.py call sites; commits: removal from this path, tests

CONSTRAINTS: run no sweep here (that is R10); wire no external corpus here (that is R8); implement no statistics here (that is R9); the real-agent harness is benchmark-only and must never touch the product default path; `compute_fixed_agent_search_turns` must not appear anywhere in the new path.
VERIFICATION (must pass): `uv run pytest tests/ -k "agent_harness or bundle_eval"` green, including a test asserting that two arms' serialized run configurations differ ONLY in the context condition; a two-task, two-condition, one-seed smoke run completes and emits populated (never None) cost fields; re-running one task with the same seed reproduces the same trajectory-level decisions; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; a missing cost field or an unpinned model raises rather than defaults.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- The product default path is untouched; determinism and the no-hosted-inference boundary hold for `archex query`.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R8 — External corpus adapters and decontamination

```text
/goal Deliver milestone R8 (External corpus adapters and decontamination) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section D R8 + .docs/strategic-reassessment/02-STRATEGY.md §C1 elements 1 and 6, and 01-LITERATURE-POSITION.md §3. Preconditions: R7 merged. Repo: as above.
OBJECTIVE: Replace self-authored labels with external, human-annotated gold contexts, and make contamination a shipped artifact rather than a footnote. Success contract: a ContextBench (arXiv:2602.05892 — 1136 tasks, 66 repos, 8 languages) loader mapping its gold contexts onto the harness's two-level outcome; an optional SWE-Explore (arXiv:2606.07297) loader; a MinHash+LSH overlap audit reported per task and checked in as a first-class artifact; corpus provenance recorded on every run.
RELEASE TRAIN: target=unversioned; included milestones=R8; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R9, R10, R11, R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R8 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: confirm the corpus licence permits redistributing derived traces, because R11 and R16 plan to publish them. A licence that forbids it is a material mismatch requiring a plan revision to R11/R16, not a workaround here.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R8 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R8 design`.
1. PR-1 `feat(benchmark): add a pinned ContextBench corpus loader` — scope: corpus adapter; commits: loader, pin, provenance; verification: the loader reports the expected task and repo counts for the pinned release
2. PR-2 `feat(benchmark): map gold contexts onto two-level outcomes` (on PR-1) — scope: outcome mapping; commits: file+symbol mapping, explicit loss record
3. PR-3 `feat(benchmark): add a per-task decontamination audit` (on PR-2) — scope: MinHash+LSH overlap audit; commits: audit, artifact; verification: `uv run archex benchmark validate --kind evidence` exits 0 on the audit artifact
4. PR-4 `feat(benchmark): add the optional SWE-Explore loader` (on PR-3) — scope: second adapter; commits: loader, tests

CONSTRAINTS: author no new task; modify no gold label; every task must carry a repo cluster ID and a decontamination score; if the gold-context schema does not map cleanly onto region-level metrics, map at file+symbol granularity and record the loss EXPLICITLY rather than approximating.
VERIFICATION (must pass): the loader reports the expected task and repo counts for the pinned corpus release; `uv run archex benchmark validate --kind evidence` exits 0 on the decontamination artifact; no task in the evaluation path originates from benchmarks/tasks/ — asserted by a test; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; a corpus-version mismatch raises rather than silently loading a different task set.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- The audit artifact is reproducible from a recorded command.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R9 — Clustered inference and pre-registered analysis

```text
/goal Deliver milestone R9 (Clustered inference and pre-registered analysis) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section D R9 and §7 Statistical discipline + .docs/strategic-reassessment/03-SPIKES.md preamble. Preconditions: R8 merged; R4's margins available. Repo: as above.
OBJECTIVE: Make the analysis defensible before any result exists, using the margins R4 derived. Success contract: a cluster bootstrap resampling REPOSITORIES, not tasks; ICC and effective N on every report; TOST against R4's utility-derived EQM producing three-valued verdicts (superior / equivalent / inconclusive at this N); one declared primary metric and one primary comparison family with everything else labelled exploratory in every rendered table; a predeclared, checked-in, machine-readable eligibility matrix recording per arm the layer, comparison_group, eligible corpora, supported languages, modes, applicable metrics, and a mandatory exclusion_reason for every excluded cell.
RELEASE TRAIN: target=unversioned; included milestones=R9; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R10, R12, R13, R14, R15, R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R9 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: re-read R4's margins. If R4 retired the intended primary metric as having no reachable EQM, choose and record the replacement HERE, before any analysis code is written.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R9 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R9 design`.
1. PR-1 `feat(benchmark): add a repository-clustered bootstrap` — scope: bootstrap over clusters, ICC, effective N; commits: bootstrap, ICC; verification: a test asserting resampling is over repositories, not tasks
2. PR-2 `feat(benchmark): add TOST equivalence with three-valued verdicts` (on PR-1) — scope: TOST against R4's EQM; commits: TOST, verdict rendering
3. PR-3 `feat(benchmark): declare the primary metric and label exploratory rows` (on PR-2) — scope: primary declaration, table rendering; commits: declaration, exploratory labelling
4. PR-4 `feat(benchmark): enforce a predeclared eligibility matrix` (on PR-3) — scope: machine-readable matrix, runner refusal; commits: matrix, refusal; verification: a test asserting the runner raises on an undeclared cell

CONSTRAINTS: run no sweep here; interpret no result here; cluster identity is GLOBAL — a repo appearing in two corpora resolves to one cluster and lands on one side of any split; an unsupported language yields an excluded cell with a reason, never a zero; an inapplicable metric renders `n/a`, never 0.
VERIFICATION (must pass): `uv run pytest tests/ -k "cluster_bootstrap or tost or eligibility"` green, including a test asserting the runner raises on an undeclared cell and a test asserting global cluster identity; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; an undeclared cell raises rather than rendering.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- The eligibility matrix genuinely constrains — the refusal test is the proof, not the matrix's existence.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R10 — S1 cost model, pilot, and full sweep (Gate B)

```text
/goal Deliver milestone R10 (S1 cost model, pilot, and full sweep) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section D R10 + .docs/strategic-reassessment/03-SPIKES.md S1 and 05-ROADMAP.md §3. Preconditions: R9 merged. Repo: as above.
OBJECTIVE: Produce the first measurement in archex's history of whether context quality changes agentic outcomes. Success contract: a cost model computed against current pricing BEFORE any sweep, resolving DEVELOPMENT_PLAN.md §2's budget GAP; a 3-repo x 6-condition x 3-seed pilot used ONLY for plumbing and variance estimation; a cluster-stratified sweep (stratify by repo; never subsample tasks within a repo); .docs/spikes/S1-context-isolation.md pre-registered and merged before the sweep; benchmarks/evidence/s1-context-isolation.json; GATE-B.md.
RELEASE TRAIN: target=unversioned; included milestones=R10; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R11, R12, R13, R14, R15, R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R10 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: compute the real cost FIRST. A figure materially above the operator's ceiling is `DESIGN NO-GO` for R10 — reduce seeds or conditions, NEVER the cluster count. Independent clusters are the scarce resource.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R10 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R10 design`.
1. PR-1 `docs(benchmark): compute the sweep cost model against current pricing` — scope: cost model, budget resolution in DEVELOPMENT_PLAN.md §2; commits: model, GAP resolution
2. PR-2 `test(benchmark): run the plumbing pilot and report variance only` (on PR-1) — scope: pilot run, report; commits: pilot, variance report; verification: the pilot report contains variance and completion statistics ONLY, no per-condition outcome means
3. PR-3 `docs(spikes): pre-register S1 context isolation` (on PR-2) — scope: .docs/spikes/S1-context-isolation.md only; gate: MUST merge before the sweep — commit order is the proof
4. PR-4 `test(benchmark): execute the cluster-stratified sweep and record Gate B` (on PR-3) — scope: sweep, benchmarks/evidence/s1-context-isolation.json, GATE-B.md; commits: sweep, evidence, verdict; verification: `uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s1-context-isolation.json` exits 0

CONSTRAINTS: change no mechanism; run no payload spike; do not inspect the pilot's effect direction; stratify by repository; checkpoint per cluster so a budget overrun yields a run reported as PARTIAL with its cluster count, never aggregated as if complete; every reported figure carries a cluster-bootstrapped interval and a three-valued verdict.
VERIFICATION (must pass): `uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s1-context-isolation.json` exits 0; the recorded sweep command reruns; the pre-registration merged before the sweep, verifiable by commit order; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; a partial sweep is labelled partial, never averaged into a headline.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- GATE-B.md states the oracle-vs-ripgrep spread against R4's pre-declared SESOI in one line, without editorialising.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
- Additionally report the Gate B outcome verbatim: `GATE B PASS — oracle-vs-ripgrep spread <x> exceeds SESOI <y>` or `GATE B FAIL — spread <x> below SESOI <y>`.
- A Gate B FAIL is the program's headline finding, not a failure. R12 and R14 still proceed; R13 and R15 become optional; product positioning shifts to P1/P2/P3 per 04-PRODUCT-AND-ECONOMICS.md §3. Do not pivot the program again on a Gate B fail — write the paper. Pre-declared and not renegotiable.
DONE: design verdict, the reviewed stack, the merge verdict, and the Gate B outcome with evidence.
```

---

### R11 — Harness extraction to a standalone public repository

```text
/goal Deliver milestone R11 (Harness extraction to a standalone public repository) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section D R11 + .docs/strategic-reassessment/04-PRODUCT-AND-ECONOMICS.md §2. Preconditions: R10 merged. Repo: as above.
OBJECTIVE: Make the instrument independently installable, versionable, and citable, and remove the "vendor benchmarks itself" structural conflict. Success contract: a new public repository containing the harness, corpus adapters, eligibility matrix, analysis, and reproduction instructions, depending on archex as ONE ARM AMONG SIX; archex retains only what its own CLI needs; a pointer from docs/ to the new repository.
RELEASE TRAIN: target=none (separate repository; out of scope for archex versioning); included milestones=R11; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=separate-repo release.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R11 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: confirm R8's corpus licence permits redistributing derived traces from the new repository. A negative answer changes R16's scope and must be recorded as a plan revision before extraction.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R11 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R11 design`.
1. PR-1 `feat(bench-repo): scaffold the standalone harness package` — scope: new repository scaffold, packaging, CI; commits: scaffold, CI
2. PR-2 `feat(bench-repo): port harness, adapters, eligibility matrix, and analysis` (on PR-1) — scope: additive port; commits: harness, adapters, analysis
3. PR-3 `docs(bench-repo): add reproduction instructions and the archex arm` (on PR-2) — scope: README, reproduction command; commits: docs, arm registration
4. PR-4 `refactor(benchmark): remove the extracted surface from archex` (on PR-3) — scope: archex removals, docs pointer; commits: removal, pointer; verification: `uv run pytest` green in archex

CONSTRAINTS: extract ADDITIVELY first and remove only in PR-4, so a rollback reverts one PR; move no archex retrieval code; archex must remain one selectable arm, never a privileged default.
VERIFICATION (must pass): in a clean environment, install the extracted package and run its smoke command to completion WITHOUT a working copy of this repository beyond the published archex package; `uv run pytest` green in archex; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; a missing dependency fails at install, not at first run.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green in both repositories; no regression coverage removed without replacement.
- archex's own benchmark CLI still works after the removal PR.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: none — RELEASE PREP: not-required` or `NO-GO — RELEASE: none — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R12 — S6 long context versus retrieval, matched, on code

```text
/goal Deliver milestone R12 (S6 long context versus retrieval) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section E R12 + .docs/strategic-reassessment/03-SPIKES.md S6 and 01-LITERATURE-POSITION.md §2. Preconditions: R10 merged. First payload milestone — reuses R10's harness wholesale. Repo: as above.
OBJECTIVE: Run the matched head-to-head nobody has run: does retrieval still help when the whole repo fits in the window? Success contract: .docs/spikes/S6-longcontext-vs-retrieval.md pre-registered; the same tasks and model across retrieved-budget-packed, full-repo dump, and retrieved-then-dumped, sweeping total context length; benchmarks/evidence/s6-longcontext-vs-retrieval.json reporting resolve rate and $/resolved-task per arm per context length with cluster-bootstrapped intervals.
RELEASE TRAIN: target=unversioned; included milestones=R12; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R12 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: confirm the pinned model's context limit still admits the full-repo arm for the chosen repos. If it does not, report the excluded repos as excluded cells with a reason in R9's eligibility matrix — never shrink the corpus silently.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R12 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R12 design`.
1. PR-1 `docs(spikes): pre-register S6 long-context versus retrieval` — scope: .docs/spikes/S6-longcontext-vs-retrieval.md only; gate: MUST merge before any run
2. PR-2 `feat(benchmark): add full-dump and retrieved-then-dumped arms with a length sweep` (on PR-1) — scope: arms, sweep; commits: arms, sweep config
3. PR-3 `test(benchmark): run the matched comparison and publish evidence` (on PR-2) — scope: run, benchmarks/evidence/s6-longcontext-vs-retrieval.json; commits: run, evidence; verification: `uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s6-longcontext-vs-retrieval.json` exits 0

CONSTRAINTS: introduce no new mechanism; all three arms run on the same task set with the same model; report cost alongside quality at EVERY context length.
VERIFICATION (must pass): `uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s6-longcontext-vs-retrieval.json` exits 0; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; the pre-registration merged before the run; behavior is meaningfully tested.
- Failures are loud; a context-limit overflow raises rather than truncating silently.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- The report states which arm wins per length band with a three-valued verdict. There is no kill criterion — both outcomes are informative, and if full-dump wins, that is the program's most important finding and must be reported as such, not buried.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R13 — S3 strong-baseline replication of published graph-expansion gains

```text
/goal Deliver milestone R13 (S3 strong-baseline replication) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section E R13 + .docs/strategic-reassessment/03-SPIKES.md S3, 00-DIAGNOSIS.md §2.1, and 01-LITERATURE-POSITION.md §1. Preconditions: R10 merged; R3 passed Gate A. Repo: as above.
OBJECTIVE: Test whether the field's reported graph-expansion gains survive a strong baseline — the single result that converts archex's dead nulls into a citable field-level correction. Success contract: .docs/spikes/S3-strong-baseline.md pre-registered; 2-3 of {RepoHyper, GraphCoder, RepoGraph, DraCo} each reproduced against the paper's OWN weak baseline first (replication class, per R3's rule) and then measured against archex's strong BM25F+graph baseline; benchmarks/evidence/s3-strong-baseline.json reporting the DELTA-OF-DELTAS with intervals.
RELEASE TRAIN: target=unversioned; included milestones=R13; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R13 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: confirm each chosen paper's artifact is fetchable and its baseline is specified precisely enough to reproduce. Drop any paper that is not, and record why. Do not substitute an approximation of a paper's baseline.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R13 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R13 design`.
1. PR-1 `docs(spikes): pre-register S3 strong-baseline replication` — scope: .docs/spikes/S3-strong-baseline.md only; gate: MUST merge before any run
2. PR-2 `test(replication): reproduce paper A against its own baseline` (on PR-1) — scope: benchmarks/replication/<paper-a>/; commits: harness, run
3. PR-3 `test(replication): reproduce paper B against its own baseline` (on PR-2) — scope: benchmarks/replication/<paper-b>/; commits: harness, run
4. PR-4 `test(benchmark): measure both mechanisms against the archex strong baseline` (on PR-3) — scope: strong-baseline arms; commits: arms, runs
5. PR-5 `docs(benchmark): publish the delta-of-deltas` (on PR-4) — scope: benchmarks/evidence/s3-strong-baseline.json; commits: evidence; verification: `uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s3-strong-baseline.json` exits 0

CONSTRAINTS: promote no mechanism to the archex default; label every arm `replication` or `adaptation`; a paper whose own baseline does not reproduce is reported `inconclusive` and EXCLUDED from the delta-of-deltas, never silently folded in; the headline quantity is the delta-of-deltas, not either delta alone.
VERIFICATION (must pass): `uv run archex benchmark validate --kind evidence --input benchmarks/evidence/s3-strong-baseline.json` exits 0; every recorded reproduction command reruns; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; every reproduction is pinned and rerunnable; behavior is meaningfully tested.
- Failures are loud; a non-reproducing baseline is reported, not approximated around.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- If NO paper's own baseline reproduces, the milestone reports `inconclusive` and stops — it cannot distinguish "gains are baseline artifacts" from "our reimplementation is wrong." Pre-declared.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R14 — S4 certified context receipt

```text
/goal Deliver milestone R14 (S4 certified context receipt) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section E R14 + .docs/strategic-reassessment/02-STRATEGY.md §C2 and 01-LITERATURE-POSITION.md §G2-G3. Preconditions: R10 merged. Repo: as above. Requires NO change to the retrieval pipeline.
OBJECTIVE: Turn the receipt's asserted completeness verdict into two independently measured, falsifiable properties. Success contract: a monotone submodular coverage function f(S) over code-structural units (referenced symbols, k-hop call/import edges at declining weight, touched type definitions, file breadth) with monotonicity and submodularity PROVEN in the design document; the certificate f(returned)/f(greedy(budget)) emitted into the receipt for whatever the upstream retriever produced; a ReproRAG-style (arXiv:2509.18869) reproducibility score for the hybrid index; correlation of both against R10's resolve rate and gold-context F1; negative controls — random selection, the prior null MMR packer, and greedy-vs-brute-force on small instances; a head-to-head against a learned sufficiency classifier.
RELEASE TRAIN: target=> GAP: version not source-traceable — operator selects at preparation time (train certified-receipt); included milestones=R14; preparation trigger=R14 externally merged AND its promotion verdict is GO; required artifacts=both (pyproject.toml version + CHANGELOG.md); release verification=full local gate green plus the certificate correlation artifact checked in; publication=git tag then `uv build` then `uv publish` then `gh release create` (manual).

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R14 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: confirm the coverage function's units are computable for every language tier, not just `full`. A tier-limited certificate must declare its scope in the receipt rather than silently degrading.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R14 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R14 design`.
1. PR-1 `docs(design): define and prove the structural coverage function` — scope: design document with the monotonicity and submodularity proofs; commits: definition, proofs
2. PR-2 `feat(serve): emit a retriever-agnostic coverage certificate in the receipt` (on PR-1) — scope: certificate computation, receipt field; commits: certificate, receipt; verification: a byte-identity test for retrieval output with certification on and off
3. PR-3 `feat(index): add a reproducibility score for the hybrid index` (on PR-2) — scope: ReproRAG-style scoring; commits: score, tests
4. PR-4 `test(benchmark): add negative controls and the learned-classifier head-to-head` (on PR-3) — scope: random-selection, MMR, brute-force controls, classifier arm; commits: controls, arm; verification: a non-vacuity test asserting random scores materially below greedy
5. PR-5 `docs(benchmark): publish the certificate correlation evidence` (on PR-4) — scope: correlation artifact, CHANGELOG.md; commits: evidence, changelog

CONSTRAINTS: change retrieval, ranking, and packing in no way — retrieval output must be BYTE-IDENTICAL with certification on and off; write the submodularity proof out, never assert it; certify an arbitrary upstream result set, demonstrated by certifying a non-archex arm; the certificate must be computed locally with no hosted inference.
VERIFICATION (must pass): `uv run pytest tests/ -k "coverage_certificate or reproducibility_score"` green, including the byte-identity test and the non-vacuity test; `uv run archex benchmark validate --kind evidence` exits 0 on the correlation artifact; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; an uncomputable coverage unit raises or declares reduced scope, never silently returns a smaller certificate.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- Correlations are reported with cluster-bootstrapped intervals and three-valued verdicts.
- If NEITHER the certificate nor the reproducibility score correlates with anything, publish the null, drop product line P2, and DO NOT fire the certified-receipt release train. Pre-declared.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: certified-receipt (> GAP: version) — RELEASE PREP: pending` or `NO-GO — RELEASE: certified-receipt — REASON: <blocking gate>`.
- A null correlation is a `GO` for merge but explicitly changes the release target to `unversioned` via a plan revision; the train does not fire on a null.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R15 — S5 co-change and ownership ranking fusion

```text
/goal Deliver milestone R15 (S5 co-change and ownership ranking fusion) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section E R15 + .docs/strategic-reassessment/01-LITERATURE-POSITION.md §G4 and 03-SPIKES.md S5. Preconditions: R10 merged. Repo: as above. Existing collectors to REUSE: src/archex/integrations/history (git log, 200-commit window, issue/PR reference extraction) and src/archex/integrations/docs (doc links, ADRs, ownership).
OBJECTIVE: Test the one mechanism direction with a 20-year replicated evidence base that no shipping tool uses, by connecting collectors this repo already has but never wired into ranking. Success contract: a one-day stratum-sizing report measuring what fraction of gold files are UNREACHABLE from query anchors via the static import/call graph, merged BEFORE any mechanism is implemented; if >=10%, a pairwise co-change edge type with independent confidence plus an ownership prior, both fused into ranking as a benchmark-only arm, measured stratified by reachability and pre-registered before any outcome is inspected.
RELEASE TRAIN: target=unversioned; included milestones=R15; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=not requested.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and every listed dependent milestone (R16).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R15 and every affected future milestone, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: read benchmarks/results/m8_repository_memory/DECISION.md and m9_documentation_status/DECISION.md FIRST. Both measured bit-for-bit identical to control BECAUSE the evidence is never read by search, ranking, or expansion. This milestone's entire delta from those is the FUSION step. If the diff touches a collector module, the milestone has drifted.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R15 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R15 design`.
1. PR-1 `test(benchmark): size the graph-unreachable gold-file stratum` — scope: sizing query and report; commits: query, report; gate: MUST merge before any mechanism work; if the stratum is under 10%, STOP HERE and the report is the milestone's deliverable
2. PR-2 `docs(spikes): pre-register S5 co-change fusion` (on PR-1) — scope: .docs/spikes/S5-cochange-fusion.md only; gate: MUST merge before any outcome is inspected
3. PR-3 `feat(index): fuse a pairwise co-change edge into ranking as a benchmark-only arm` (on PR-2) — scope: edge type, confidence, fusion; commits: edge, fusion; verification: an `archex_query` byte-identity test
4. PR-4 `test(benchmark): measure the arm stratified by reachability` (on PR-3) — scope: stratified run, evidence; commits: run, evidence

CONSTRAINTS: rebuild NO collection — the history and docs collectors already exist and the diff must touch no collector module; the arm is benchmark-only and `archex_query` scoring must stay byte-identical; the pre-registered primary is gold-file recall within the UNREACHABLE stratum, with non-inferiority on the reachable stratum as the guard; effect-size expectation is modest (r~0.54, significant only combined with change entropy) — do not oversell.
VERIFICATION (must pass): the sizing query reruns to the same fraction; if implemented, `uv run pytest tests/ -k "cochange or ownership_prior"` green including the `archex_query` byte-identity test; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; a repo without sufficient history yields a neutral fallback that is recorded, not a silent zero.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green; no regression coverage removed without replacement.
- The sizing report merged before any mechanism work, verifiable by commit order.
- A stratum under 10% is a pre-declared stop: the mechanism has no addressable surface and the sizing report is the deliverable. This is not a failure.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: unversioned — RELEASE PREP: not-required` or `NO-GO — RELEASE: unversioned — REASON: <blocking gate>`.
DONE: design verdict with evidence; when authorized, a reviewed stack with a release-aware merge verdict and evidence.
```

---

### R16 — Public artifact release and Gate C disposition

```text
/goal Deliver milestone R16 (Public artifact release and Gate C disposition) from DEVELOPMENT_PLAN.md as a reviewed stack of PRs.

CONTEXT: DEVELOPMENT_PLAN.md §6 Section F R16 + .docs/strategic-reassessment/02-STRATEGY.md §C1 elements 1-7 and 05-ROADMAP.md §4 Gate C. Preconditions: R11, R12, R13, R14, R15 merged. Repo: as above plus the extracted public repository from R11.
OBJECTIVE: Assemble the program's evidence into a public, reproducible artifact set and record the publication decision. Success contract: the extracted repository made public with every context-variant trace, the decontamination audit, the eligibility matrix, and a SINGLE reproduction command; GATE-C.md recording assembly against the seven elements of 02-STRATEGY.md §C1 and the decision between a datasets-and-benchmarks submission alone versus adding the human arm (IRB, recruitment, ~8 additional weeks); docs/WHY_ARCHEX.md and README stating only claims the program measured.
RELEASE TRAIN: target=none (separate repository); included milestones=R16; preparation trigger=n/a; required artifacts=none; release verification=n/a; publication=separate-repo release.

PRE-IMPLEMENTATION DESIGN GATE:
1. Read this milestone, its source-map rows, current prompt, and `.docs/DEVELOPMENT_PLAN_HISTORY.md` when present.
2. Inspect the current codebase plus merged predecessor diffs, merged predecessor PR outcomes, CI/check evidence, and predecessor verification output. Read GATE-A.md and GATE-B.md.
3. Revalidate objective, interfaces, dependencies, acceptance, verification, risks, release train, and dependent milestones (none — terminal).
4. Append one ledger entry with the fields listed in DEVELOPMENT_PLAN_HISTORY.md.
5. If no material mismatch exists, report `DESIGN GO — PLAN REVISION: none`; this authorizes implementation.
6. If a mismatch exists, update both authoritative artifacts for R16, append the revision ID, and report `DESIGN GO — PLAN REVISION: <entry IDs>`.
7. If validity cannot be established, report `DESIGN NO-GO — REASON: <evidence>` and stop.

SPECIFIC GATE CHECK: re-check EVERY corpus and model licence before publishing traces. Assume anything published is permanent.

HUMAN REVIEW GATE: Do not make the repository public until a human reviews the licence check, the trace contents for any leaked private path or credential, and the rollback note. Publication is irreversible in practice.

RECONCILIATION RULE: A material revision opens `docs(plan): reconcile R16 design` as a docs-only prerequisite PR, reviewed, green, and externally merged before any code PR.

PLANNED STACK:
0. Conditional prerequisite `docs(plan): reconcile R16 design`.
1. PR-1 `docs(bench-repo): assemble traces, audit, and matrix for release` — scope: artifact assembly, licence check; commits: assembly, licence record
2. PR-2 `docs(bench-repo): add the single-command reproduction path` (on PR-1) — scope: reproduction command, README; commits: command, docs; verification: a clean-environment reproduction of one headline figure
3. PR-3 `docs(gate): record the Gate C disposition` (on PR-2) — scope: GATE-C.md; commits: seven-element assembly, submission decision
4. PR-4 `docs: restate archex claims against measured evidence only` (on PR-3) — scope: README.md, docs/WHY_ARCHEX.md; commits: claim restatement; verification: every numeric claim greps to an artifact in the index

CONSTRAINTS: write no paper here; run no human-subjects study here — R16 records the DECISION about the human arm only; publish no trace whose licence was not verified.
VERIFICATION (must pass): in a clean environment, clone the public repository and reproduce one headline figure end to end using the documented command and no private input; `grep` every numeric claim in README.md and docs/ against the artifact index and confirm each resolves; the full local gate is green.
REVIEW:
Per PR:
- Scope matches its purpose; contracts match the reconciled plan; behavior is meaningfully tested.
- Failures are loud; a trace with an unverified licence blocks the release rather than shipping with a caveat.
- History is atomic, conventional, attribution-free, and free of unrelated formatting churn.
- PR-specific verification output is captured.
Whole stack:
- Bases form one valid stack; CI is green in both repositories; no regression coverage removed without replacement.
- GATE-C.md addresses all seven elements explicitly, marking each present or absent — an absent element is stated, never omitted.
- No claim in README.md or docs/ lacks a checked-in artifact.
- The human review gate is signed off before the repository is made public.
- Report PR URLs, bases, verification, risks, manual gates, and review completion.
FINAL VERDICTS:
- Report the design verdict before the merge verdict.
- Then report exactly one merge verdict: `GO — RELEASE: none — RELEASE PREP: not-required` or `NO-GO — RELEASE: none — REASON: <blocking gate>`.
- Additionally report the Gate C disposition: `GATE C — D&B ONLY` or `GATE C — D&B PLUS HUMAN ARM (+~8 weeks, IRB required)`.
DONE: design verdict, the reviewed stack, the merge verdict, and the Gate C disposition with evidence.
```
