# R19 — Pinned Graft graph-memory comparison

This pre-registration fixes the protocol for adding Graft to the public head-to-head harness as a graph-memory lane. It merges before the first Graft cell exists. No Graft artifact, manifest row, or adapter exists at the time of writing, so nothing here is post-hoc.

## Study identity

- **Spike ID and title:** R19 — Pinned Graft graph-memory comparison
- **Evidence class:** `original`
- **Decision owner and date:** archex maintainer, 2026-09-09
- **First-run commit:** Leave blank until this pre-registration has merged; then record the first commit allowed to generate Graft data.

### Pinned identities

| | Value |
| --- | --- |
| archex source revision | `879a2dafd779a1925203c7403bcad034aa2a68bf` |
| Graft released package | `@nanonets/graft@0.16.0` |
| Graft npm integrity | `sha512-L3E5F1aDYJDCARgfR7O2VaMt8xwO1XNYyHiW2n1WhKnj87gPqoxoZJGNbGXfw6XeA9JSJX3naA36RZ+jDf4AcQ==` |
| Graft npm shasum | `49b804eecd7941460dfe33549e9d0ea446fef128` |
| Graft source commit | `aa1e2bb0f6326068ac64886da1e67fa25a7804de` (tag `v0.16.0`, equals the package `gitHead`) |
| Graft license | MIT |
| Task population | the 19 tasks in `benchmarks/headtohead/manifest.yaml` `task_subset`, over 15 source repositories |

`0.16.0` is the pin because it is the newest released package. `@nanonets/graft` publishes no `0.17.0`: the registry version list ends at `0.16.0`, `dist-tags.latest` is `0.16.0`, `trailhq/Graft` publishes no GitHub releases, and no `v0.17.0` tag exists. A source checkout that declares an unpublished version is not a released artifact and has no distribution integrity, so it cannot be pinned here.

## Hypothesis

Over the 19 head-to-head tasks, the warm Graft graph-memory lane (`graft ask` against a prebuilt structural graph) is **equivalent or inferior** to the shipped `archex_query` retrieval default on required-file recall, measured on the same tasks and the same required-file labels.

- Treatment: `graft_query_warm`, the warm Graft lane.
- Control: `archex_query` as already recorded in `benchmarks/headtohead/results/`; those artifacts are not regenerated.
- Population: the 19 tasks named in the manifest, clustered under 15 repositories.
- Comparison family: required-file recall, treatment minus control.

This is a descriptive public comparison. It cannot promote, demote, or modify any archex retrieval default, and no outcome authorizes a new retrieval lane, language tier, or MCP tool.

## Primary metric

**Mean required-file recall over the 19 tasks.**

- Numerator, per task: the number of the task's labeled required files that appear in the lane's returned units.
- Denominator, per task: the number of labeled required files for that task.
- Aggregation: unweighted mean of the per-task ratios across all 19 tasks.
- Direction: higher is better.
- Measurement: for Graft, a returned unit is a `hits[]` entry, and its returned file is the path component of `pointer`. Graft emits two hit shapes and both count as returned files: a symbol hit whose `pointer` is `<path>:L<start>-L<end>` and carries `code`, and a whole-file hit whose `pointer` is a bare `<path>` and carries no `code` even under `--source`. Both name an exact repository path, so neither requires inference. Whole-file hits count toward required-file recall but contribute no returned source, so they are excluded from the exploratory returned-source and token-efficiency counts, and the per-cell split of symbol versus whole-file hits is recorded. For archex, the already-recorded `required_file_recall` per task is used unchanged.

Every other quantity — token efficiency, cold-start time, warm latency, precision, F1, MRR, nDCG, MAP, context noise, freshness — is **exploratory**.

## SESOI

**0.05 mean required-file recall.**

Required-file denominators in this corpus are small (typically 3), so one recovered file moves a single task by about 0.333 and the 19-task mean by about 0.0175. A 0.05 mean difference is therefore about three additional required files recovered across the whole corpus. That is the smallest difference that changes which tool an operator reaches for first when orienting in an unfamiliar repository; anything smaller is invisible in the decision it is supposed to inform.

## Decision margins

- **Minimum worthwhile gain (MWG):** +0.05 mean required-file recall for Graft over `archex_query`. Utility basis: recovering roughly three otherwise-missed required files corpus-wide is the point at which a second tool earns its install, configuration, and per-repository build cost in an orientation workflow.
- **Non-inferiority margin (NIM):** −0.05 mean required-file recall. Cost basis: Graft's workflow properties (continuous post-edit sync, freshness visibility) are only worth adopting alongside archex if they cost less than about three required files of context completeness across the corpus; a larger loss means an agent must re-open files the bundle should have carried, which erases the workflow saving.
- **Equivalence margin (EQM):** ±0.03 mean required-file recall. Utility basis: fewer than two required files corpus-wide changes no operator action and no published recommendation, so a two-sided interval inside ±0.03 is reported as practical equivalence rather than as a win for either tool.

Margins are derived from operator utility and cost, not from any observed spread. They are not widened after data exists.

## Clustering unit

**The source repository (15 clusters).**

Tasks are clustered under repositories because same-repository tasks share layout, language, module conventions, and parse difficulty: `django/django`, `expressjs/express`, `tiangolo/fastapi`, and `gin-gonic/gin` each contribute two of the 19 tasks. Inference resamples repositories, not tasks, so both of a repository's tasks always move together. Cluster bootstrap: 10 000 resamples, seed `20260909`.

## Kill criterion

- **Unavailable released artifact.** If the pinned package cannot be installed from the registry at its recorded integrity, the milestone stops and records the blocker. No source build is substituted for the released artifact.
- **Non-attributable output.** If a Graft hit names no resolvable repository path — neither a `<path>:L<start>-L<end>` symbol pointer nor a bare `<path>` file pointer — the affected cell is a recorded failure. Source and paths are never inferred to fill a cell.
- **Incomplete coverage.** Every one of the 2 modes × 19 tasks = 38 cells must exist as either a valid result or a recorded failure artifact. A missing cell is never dropped, and while any cell is missing the report publishes no cross-tool claim.
- **Inseparable cost.** If per-task build cost cannot be separated from warm query cost, the comparison is reported as build-inclusive only, and no warm-latency claim is published.
- **Null is a valid outcome.** Practical equivalence, or Graft losing on required-file recall, is a complete and publishable result. So is Graft winning; the strategic freeze still forbids any default change either way.

## Run and analysis boundary

Frozen before the first run.

**Install (once per machine, network-dependent):**

```bash
CI=1 DO_NOT_TRACK=1 npm install -g @nanonets/graft@0.16.0
```

The install is not offline-capable: a transitive `tree-sitter-cli` install script downloads a platform binary from GitHub release assets, and a transient `ECONNRESET` there is retried rather than treated as a Graft failure. The recorded provenance is the resolved version plus the integrity hash above.

**Per task, cold lane `graft_build_plus_query` (includes build cost):**

```bash
CI=1 DO_NOT_TRACK=1 graft --dir "$WORKSPACE/graft-index" build --no-gitignore --no-ignore "$REPO"
CI=1 DO_NOT_TRACK=1 graft --dir "$WORKSPACE/graft-index" ask "$TASK_QUESTION" -n 10 --source --no-refresh --json "$REPO"
```

**Per task, warm lane `graft_query_warm` (query only, against the graph the cold lane already built):**

```bash
CI=1 DO_NOT_TRACK=1 graft --dir "$WORKSPACE/graft-index" ask "$TASK_QUESTION" -n 10 --source --no-refresh --json "$REPO"
```

**Freshness, measured separately and never inside warm latency:**

```bash
CI=1 DO_NOT_TRACK=1 graft --dir "$WORKSPACE/graft-index" check --json "$REPO"
```

Frozen protocol decisions, each verified against the pinned release before this document merged:

- **Structural mode only.** `build` runs without `--deep`, so no model is called, no API key is required, and no spend occurs. Paid deep summaries are out of scope because they change the layer, cost model, and privacy posture.
- **Pristine checkout.** The graph directory lives outside the task repository and `--no-gitignore --no-ignore` are passed, so the checkout the other lanes measure is not modified.
- **`--no-refresh` on every measured query.** A default `graft ask` silently repairs graph drift, which would fold synchronization into query latency and let a warm cell answer from a graph the warm lane did not build. Warm latency is query-only by construction.
- **Freshness from the `graph` section only.** `check --json` reports `context` as missing in structural mode, because context cards are a `--deep` artifact. Only the `graph` section is a valid freshness signal here.
- **Rank is the emitted order.** `hits[].score` is not monotonically descending in the emitted array, so rank-sensitive exploratory metrics use the order Graft itself numbers, never a re-sort by score.
- **Result cardinality 10.** `-n 10` matches the `limit: 10` already frozen for the external retrieval lane, so returned-unit counts are comparable.
- **Extraction tier recorded per task.** 17 tasks (Python, Go, TypeScript/JavaScript) use Graft's native depth tier. The 2 Rust tasks (`mini_redis_async`, `rust_tokio_runtime`) use its WASM breadth tier, which is signature-only and emits no receiver typing. The tier is recorded per cell and reported; depth-tier and breadth-tier coverage are never presented as equivalent.
- **Telemetry off.** `DO_NOT_TRACK=1` and `CI=1` are set for install and every invocation; `graft telemetry` must report telemetry off in the run log.

Analysis, frozen:

- Primary: mean required-file recall, treatment minus control, with a 10 000-resample cluster bootstrap over the 15 repositories, seed `20260909`, evaluated against MWG, NIM, and EQM above.
- Pre-declared secondary, labelled exploratory: the same comparison restricted to the 17 depth-tier tasks, reported because breadth-tier coverage is structurally weaker and readers must be able to see the comparison without it.
- No exclusions. Failed cells are retained and counted as recorded failures.
- The archex, `ccc`, Graphify, Headroom, and raw-read artifacts already in `benchmarks/headtohead/results/` are inputs, not outputs; they are not regenerated, and the graph-memory lane generalization must leave them byte-identical.
- Graft is reported as a graph-memory lane. It is never presented as a direct retrieval-equivalent winner, and its token-efficiency cells are within-lane signals, not bundle-for-bundle comparisons.

## Post-hoc changes

None. Record only changes made after data exists, with timestamp, reason, affected field, and why the result is exploratory.
