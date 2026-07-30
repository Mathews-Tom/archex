# Execution Prompts — archex Strategic Reassessment Program

This file preserves execution prompts and disposition records for the strategic-reassessment milestones. R1–R5 are complete; R6/R6.1 and R7–R16 are cancelled. A future authorized root-cause effort must follow the current §8 Critical Path.

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

### R4 — S2 corpus validity audit `COMPLETE — VALIDITY DISPOSITION RECORDED`

```text
No execution prompt. R4 completed after the R4-DG-001 reconciliation PR #581 and merged PRs #582 and #583.

The checked-in corpus-audit evidence records 64 tasks across 16 repository clusters, 29.69% identifier-symbol leakage, 100% held-out overlap with no enforcing code path, and a calibrated finding that +4.88 points has 0.108 power at the current N. The calibrated projection requires about 2,048 total tasks under the current 16-cluster structure. Validate it with:

uv run archex benchmark validate --kind corpus-audit --input benchmarks/evidence/s2-corpus-validity.json
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

### R6.1 — S7 determinism as prefix-cache economics replacement `CANCELLED — PROVIDER FEASIBILITY FAILED`

```text
No execution prompt. R6.1 is cancelled.

Two unchanged protocol runs ended with OpenRouter `503 Overloaded` error payloads and no usable provider receipt. A third unchanged run ended when a required prewarm receipt reported zero `cache_write_tokens`. These are provider-feasibility failures, not economics observations: they establish no cache-hit, dollar, retrieval-quality, product, literature, or Gate-A conclusion. No valid evidence JSON or economics decision document exists.

Do not rerun this protocol, relax receipt validation, change provider routing, or introduce retries. Any future S7 economics work requires a separate milestone and a fresh pre-registration; it must not reuse incomplete provider attempts.
```

---

### R7 — Real-agent execution harness `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed and Section D is cancelled.

R7 required a Gate A pass. It must not be revived without a new authorized program and fresh pre-registration.
```

---

### R8 — External corpus adapters and decontamination `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed and Section D is cancelled.

R8 depended on R7's real-agent harness. It must not be revived without a new authorized program and fresh pre-registration.
```

---

### R9 — Clustered inference and pre-registered analysis `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed and Section D is cancelled.

R9 required R8's external-corpus adapter and R4's now-retired margin derivation. R4 instead established the current corpus's inability to resolve a literature-sized effect; R9 must not be revived without a new authorized program and fresh pre-registration.
```

---

### R10 — S1 cost model, pilot, and full sweep `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed and Section D is cancelled.

R10 depended on cancelled R9 and on the real-agent harness path that Gate A blocked. It must not be revived without a new authorized program and fresh pre-registration.
```

---

### R11 — Harness extraction to a standalone public repository `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed and Section D is cancelled.

R11 depended on R10's cancelled evaluation instrument. It must not be revived without a new authorized program and fresh pre-registration.
```

---

### R12 — S6 long context versus retrieval, matched, on code `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed and Section E is cancelled.

R12 depended on R10's cancelled evaluation instrument. It must not be revived without a new authorized program and fresh pre-registration.
```

---

### R13 — S3 strong-baseline replication of published graph-expansion gains `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed and Section E is cancelled.

R13 depended on R10's cancelled evaluation instrument. It must not be revived without a new authorized program and fresh pre-registration.
```

---

### R14 — S4 certified context receipt `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed and Section E is cancelled.

R14 depended on R10's cancelled evaluation instrument. It must not be revived without a new authorized program and fresh pre-registration.
```

---

### R15 — S5 co-change and ownership ranking fusion `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed and Section E is cancelled.

R15 depended on R10's cancelled evaluation instrument. It must not be revived without a new authorized program and fresh pre-registration.
```

---

### R16 — Public artifact release and Gate C disposition `CANCELLED — GATE A FAIL`

```text
No execution prompt. Gate A failed before Gates B and C could apply.

R16 depended on the cancelled R7–R15 program. It must not be revived without a new authorized program and fresh pre-registration.
```
