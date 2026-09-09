# R20 — Paired product-as-shipped agent-loop baseline

This pre-registration fixes the protocol for a paired agent benchmark of the current Archex and Graft product loops. It merges before the first product-loop cell exists. No product-loop runner, artifact, or result exists at the time of writing, so nothing here is post-hoc.

Its purpose is to freeze the *current* behavior of both loops before any Graft-inspired Archex workflow feature lands. The load-bearing output is the Archex-arm baseline that later workflow milestones are re-measured against with this same protocol. The cross-arm comparison is descriptive context; see [Power disposition](#power-disposition) for why no verdict is drawn from it at any observed value.

## Study identity

- **Spike ID and title:** R20 — Paired product-as-shipped agent-loop baseline
- **Evidence class:** `original`, descriptive
- **Decision owner and date:** archex maintainer, 2026-09-09
- **First-run commit:** Leave blank until this pre-registration has merged; then record the first commit allowed to generate product-loop data.

### Pinned identities

| | Value |
| --- | --- |
| archex source revision | recorded per run from a clean tree; the runner refuses a dirty tree |
| Graft released package | `@nanonets/graft@0.16.0` |
| Graft npm integrity | `sha512-L3E5F1aDYJDCARgfR7O2VaMt8xwO1XNYyHiW2n1WhKnj87gPqoxoZJGNbGXfw6XeA9JSJX3naA36RZ+jDf4AcQ==` |
| Graft source commit | `aa1e2bb0f6326068ac64886da1e67fa25a7804de` (tag `v0.16.0`, equals the package `gitHead`) |
| Agent executable | `claude`, Claude Code `2.1.266` |
| Model | `claude-haiku-4-5` |
| Billing mode | the operator's existing OAuth subscription |
| Prompt | task `question` plus Appendix A, SHA-256 `39acce3ce7bff054ee09b52414c9fede8807da77ccc683b211bd16129ee1007d` |
| Task population | the 19 tasks in `benchmarks/headtohead/manifest.yaml` `task_subset`, over 15 source repositories |
| Repetitions | 3 per task and arm |
| Planned cells | 2 arms × 19 tasks × 3 repetitions = 114 |

The Graft pin is inherited unchanged from `R19-graft-graph-memory-comparison.md`, so the two milestones describe the same released artifact.

### Why Claude Code is the agent

Claude Code is the only client both products wire deeply, so it is the only client where "product as shipped" is a fair description of both arms.

- Archex: `archex install-client claude-code <repo> --scope project` writes project `.mcp.json`; `--hooks` writes a `PreToolUse` `Glob|Grep` hook into project `.claude/settings.json`. `src/archex/cli/install_client_cmd.py:74-78` ships codex and cursor a diagnostics-only adapter that injects nothing, and `:175-179` restricts the `SessionStart` primer to claude-code.
- Graft: `graft init` writes project `.claude/settings.json` (`UserPromptSubmit`, `PostToolUse` on `Write|Edit|MultiEdit`, `PostToolUse` on `Bash|mcp__graft__|Read|Grep|Glob`, statusline), `.claude/helpers/graft-statusline.cjs`, `.claude/helpers/graft-hooks.cjs`, `.claude/skills/graft/SKILL.md`, and `.mcp.json`. `graft init --list-agents` at the pinned release prints `agents`, `adal`, `cursor`, `gemini`, `grok`, `hermes`, `antigravity`, `copilot`, `kiro`, `windsurf`, and `claude` — and no omp or pi target.

Running the Graft arm on a host Graft does not wire would measure an arm its vendor never ships, which is an arm-competence failure rather than a product comparison.

### Billing mode is subscription, so every cost figure is modelled

This machine has no `ANTHROPIC_API_KEY`; `claude` authenticates by OAuth. The run therefore charges no dollars and consumes plan quota instead. Claude Code still emits `total_cost_usd` and a per-model `modelUsage.<model>.costUSD` in each cell's terminal `result` object, but under subscription auth those are list-price arithmetic over the recorded token counts, not a charge.

Consequences, frozen:

- Every published cost number is labelled modelled, never billed.
- The ceiling and abort rule below are enforced against the same modelled figure, because it is the only per-cell cost receipt the client emits.
- A mid-run rate limit is a run-integrity event, not a billing event: the affected cell becomes a recorded failure and the run continues.
- Claude Code additionally issues a session-title request per session against its own small model. That request is not attributed to the session receipt, so recorded cost slightly understates true consumption. The understatement is identical in shape on both arms and is disclosed rather than corrected.

## Hypothesis

Over the 19 head-to-head tasks, the Archex-wired loop and the Graft-wired loop are compared descriptively on required-file completeness of the agent's final answer, measured on the same tasks and the same required-file labels. No equivalence, superiority, or non-inferiority verdict is drawn; see [Power disposition](#power-disposition).

- Control: `archex_product_loop` — Claude Code with Archex's shipped project wiring.
- Treatment: `graft_product_loop` — Claude Code with Graft's shipped project wiring.
- Population: the 19 tasks named in the manifest, clustered under 15 repositories.
- Comparison family: required-file completeness, treatment minus control.

This is a descriptive public comparison. It cannot promote, demote, or modify any archex default, and no outcome authorizes a new retrieval lane, language tier, or MCP tool.

## Primary metric

**Mean required-file completeness of the agent's final answer over the 19 tasks.**

- Numerator, per cell: the number of the task's labeled `expected_files` that appear in the answer's file list.
- Denominator, per cell: the number of labeled `expected_files` for that task, unchanged from R19.
- Per task and arm: the unweighted mean of that ratio over the 3 repetitions.
- Aggregation: the unweighted mean of the per-task values across all 19 tasks.
- Direction: higher is better.

**Answer extraction, frozen.** The final assistant text block is scanned for the last line whose stripped content equals `FILES:`; every subsequent non-empty line is split on commas and whitespace into candidate tokens.

- **Cardinality cap.** The block may name at most `K = 6` paths — twice the largest required-file denominator in this corpus. A block naming more than `K` paths scores `0.0` and is flagged `answer_over_broad`. `K` is frozen here and is never widened after data exists. The cap exists because the primary is a pure recall measure: without it, an agent could score `1.0` on every task by enumerating the checkout, and a later workflow feature that merely made the agent list more files would beat this baseline without improving orientation.
- **Normalization, applied identically on both arms,** in this order: strip surrounding backticks and markdown link syntax; strip a trailing `:L<start>-L<end>` or `:<line>` suffix, because Graft's native pointers carry one; convert `\` to `/`; strip a leading `./`; rebase an absolute path under the checkout root to repository-relative; discard any token that does not resolve to a path inside the sliced checkout.
- **Matching** is exact on the normalized path. No prefix, basename, or fuzzy matching is used, on either arm.
- A cell whose final text contains no `FILES:` line scores `0.0` and is flagged `answer_unparsed`. It is a scored cell, not a failure, because an answer the harness cannot read is an answer the operator cannot use.
- **Product-use flag.** A scored cell that issued zero `mcp__archex__*` or `mcp__graft__*` calls is flagged `no_product_use`. The 15 upstream corpora are widely known open-source projects, so an agent can name correct files from pretraining without consulting the wired product at all. The primary mean is therefore reported twice — over all cells, and over product-using cells only — so a pretraining-dominated baseline is visible rather than silent.

Because the denominators and labels are exactly R19's, R19's retrieval figures and R20's agent figures are read on the same scale.

Every other quantity — answer precision, tool calls, tool-call mix, input/output/cache tokens, modelled cost, wall time, setup time, freshness state, hook records, answer-parse and over-broad flags — is **exploratory**.

## SESOI

**0.05 mean required-file completeness.**

Required-file denominators in this corpus are small: 17 tasks label 3 files and 2 label 2, for 55 labeled files in total. One recovered file therefore moves a single task by `1/d` — 0.333 on a 3-file task, 0.5 on a 2-file task — and the 19-task mean by 0.0175 or 0.0263 respectively. A 0.05 mean difference is roughly two to three additional required files recovered across the whole corpus. That is the smallest difference that changes which context product an operator wires into an agent; anything smaller is invisible in the decision it is supposed to inform. The margin is inherited from R19 deliberately so the retrieval-layer and agent-layer results are comparable.

## Decision margins

Recorded so the interval can be read against a fixed scale. Under the power disposition below they are **not** used to declare a verdict.

- **Minimum worthwhile gain (MWG):** +0.05 mean completeness for the Graft loop over the Archex loop. Utility basis: recovering roughly three otherwise-missed required files corpus-wide is the point at which replacing a wired context product earns its install, per-repository build, and in-repo graph-file cost.
- **Non-inferiority margin (NIM):** −0.05 mean completeness. Cost basis: the Archex loop's auditability and receipt discipline are worth keeping only if they cost less than about three required files of answer completeness corpus-wide.
- **Equivalence margin (EQM):** ±0.03 mean completeness. Utility basis: fewer than two required files corpus-wide changes no operator action and no published recommendation.

Margins are derived from operator utility and cost, not from any observed spread. They are not widened after data exists.

## Clustering unit

**The source repository (15 clusters).**

Tasks are clustered under repositories because same-repository tasks share layout, language, module conventions, and parse difficulty: `django/django`, `expressjs/express`, `tiangolo/fastapi`, and `gin-gonic/gin` each contribute two of the 19 tasks. Inference resamples repositories, not tasks or repetitions, so both of a repository's tasks always move together. Repetitions are averaged inside a task before resampling; they estimate agent nondeterminism, not additional independent evidence. Cluster bootstrap: 10 000 resamples, seed `20260909`.

## Power disposition

Declared before any data exists.

R19 measured a fully deterministic retrieval metric on this exact population and returned a mean difference of −0.0175 with a 95% cluster-bootstrap interval of [−0.1228, +0.0635] against a ±0.03 equivalence margin. With zero run-to-run noise the corpus already could not separate practical equivalence from a non-inferiority failure. An agent loop adds nondeterminism on top of that, so R20's interval is expected to be at least as wide; sampling variance means a narrower observed interval is possible but would not indicate a better-powered design.

Therefore:

- R20 is registered as a **descriptive baseline**, not a discriminating comparison.
- No cross-tool superiority, non-inferiority, or equivalence claim is published from R20 at any observed value. The margins above are recorded so the interval can be read against them, not so a verdict can be declared.
- The result that R20 exists to produce is the frozen Archex-arm protocol and per-cell numbers. Later workflow milestones re-run this protocol unchanged against the same population and compare Archex to Archex, which is a paired within-arm design with far more power than this cross-tool one.
- Restating Graft's published `+12` SWE-bench points, `23%` token, `25%` call, or `32%` time figures as an R20 result is forbidden. Those come from a different population, a different model class, and an edit-task harness this milestone does not build.

## Scope exclusion: no edit tasks

R20 measures an orientation loop only. It does not measure multi-file edit success or missed sibling edits.

The reason is corpus availability, recorded here so the exclusion is not mistaken for an oversight. Every one of the 77 standard task YAMLs in `benchmarks/tasks/` is question or localization shaped: 19 external comprehension, 16 `archex_*` self, 8 routing, 21 `loc_*`, 9 identifier-fragment, 4 polyglot. `benchmarks/arch_tasks/` carries an architecture oracle with no file-edit labels, `benchmarks/delta_tasks/` carries a historical changed-file list with no edit prompt or scoring contract, `benchmarks/sealed_tasks/` is localization, and `benchmarks/held_out.txt` is empty. Nothing in the repository labels a set of files that must be edited together.

Producing such a corpus would require per-repository test execution for django, react, tokio, and eleven other upstreams, a patch-validity oracle, and hand-authored must-edit-together labels whose authoring would itself determine the outcome. That is a separate, separately pre-registered milestone.

## Kill criterion

- **Agent unavailable or mismatched.** If `claude --version` does not report `2.1.266`, the run stops and records the blocker. The pinned agent version is not substituted.
- **Tool-parity violation.** A cell whose `system`/`init` event does not report exactly the frozen tool set for its arm is a recorded failure. If any such cell exists, the report publishes no cross-arm efficiency figure.
- **Disconnected context product.** A cell whose `init` event does not report the arm's single MCP server with `status: "connected"`, or whose later transcript shows that server dropping from `connected` or returning an MCP protocol error, is a recorded failure rather than a scored cell.
- **Incomplete coverage.** All 114 cells must exist as either a scored result or a recorded failure artifact. A missing cell is never dropped, and while any cell is missing the report publishes no aggregate.
- **Ceiling reached.** The run aborts before starting the next cell once cumulative recorded cost reaches the ceiling. Everything already completed is retained, and the report states partial coverage and publishes no aggregate.
- **Null is a valid outcome.** A wide interval that resolves nothing is a complete and publishable result for a descriptive baseline. So is either arm leading; the strategic freeze forbids any default change either way.

## Run and analysis boundary

Frozen before the first run.

**Install, once per machine (network-dependent):**

```bash
CI=1 DO_NOT_TRACK=1 npm install --prefix "$GRAFT_PREFIX" @nanonets/graft@0.16.0
```

The install is not offline-capable: a transitive `tree-sitter-cli` install script downloads a platform binary from GitHub release assets, and a transient `ECONNRESET` there is retried rather than treated as a Graft failure. The recorded provenance is the resolved version plus the integrity hash above.

**Per-cell workspace.** Each of the 114 cells gets its own directory containing a private `home/` and a private `repo/`. `repo/` is a pinned checkout of the task's upstream repository sliced to the task's `include_paths`, produced by the same clone-and-slice path R19 used, so both milestones measure identical repository content. Every command in the cell runs under `env -i` with a fixed `PATH`, `HOME` pointed at the cell's `home/`, and `CLAUDE_CONFIG_DIR` pointed inside it. Nothing from the operator's `~/.claude`, `~/.archex`, `~/.omp`, or ambient MCP configuration is visible to any cell. Every `git` invocation inside a cell passes `-c core.excludesFile=/dev/null`, so only the checkout's own `.gitignore` applies and the operator's global excludes cannot change what the commit step below captures.

**Arm `archex_product_loop`, setup (timed as `setup_seconds`):**

```bash
archex init "$REPO"
archex install-client claude-code "$REPO" --scope project -y
archex install-client claude-code "$REPO" --scope project --hooks -y
git -C "$REPO" -c core.excludesFile=/dev/null add -A
git -C "$REPO" -c core.excludesFile=/dev/null commit -m "wire archex"
```

**Arm `graft_product_loop`, setup (timed as `setup_seconds`):**

```bash
CI=1 DO_NOT_TRACK=1 graft init --no-agents "$REPO"
git -C "$REPO" -c core.excludesFile=/dev/null add -A
git -C "$REPO" -c core.excludesFile=/dev/null commit -m "wire graft"
```

**Per cell, the measured invocation (identical on both arms except the MCP config):**

```bash
claude -p "$PROMPT" --model claude-haiku-4-5 \
  --output-format stream-json --verbose \
  --mcp-config "$REPO/.mcp.json" --strict-mcp-config \
  --allowedTools Read Grep Glob "mcp__$ARM_SERVER" \
  --disallowedTools Write Edit MultiEdit NotebookEdit Bash Task TaskCreate TaskGet TaskList TaskOutput TaskStop TaskUpdate WebSearch WebFetch Skill Workflow DesignSync EnterWorktree ExitWorktree ListAgents ReportFindings ScheduleWakeup SendMessage CronCreate CronDelete CronList \
  --permission-mode bypassPermissions --no-session-persistence
```

`$PROMPT` is exactly the task YAML's `question`, then one blank line, then the frozen instruction block in [Appendix A](#appendix-a--frozen-instruction-block) verbatim. No cell deviates from it, and no per-arm or per-task wording exists.

**Freshness, probed after the agent exits and never inside the measured wall time:**

```bash
archex status "$REPO" --format json          # arm archex_product_loop
CI=1 DO_NOT_TRACK=1 graft check --json "$REPO"   # arm graft_product_loop
```

Frozen protocol decisions, each verified against the shipped products and the pinned agent before this document merged:

- **The wiring is committed.** Both products write into the checkout during setup: Archex creates `.archex/` and edits `.gitignore`; Graft creates `graft/`, `.ignore`, `.claude/`, and edits `.gitignore`. Left uncommitted, Archex's index state reports `dirty` and its `PreToolUse` hook no-ops on every call, which would silently run the Archex arm with its hook disabled. Committing each arm's own wiring in its throwaway checkout is what makes both arms competent, and it is applied symmetrically.
- **Both products' graphs are built during setup.** `archex init` builds and embeds the index. `graft init` runs `graft build` itself and reports `built the graph (graft build)`, so no graph construction happens lazily inside the measured invocation on either arm. Neither arm pays index construction inside its wall time or against its deadline.
- **Graft's cards are Grep-visible and Archex's index is not.** Both arms grant the agent `Read`, `Grep`, and `Glob` over the working tree. `graft build` gitignores `graft/` but also writes an `.ignore` file containing `!graft/`, whose own comment states the cards "should stay greppable"; ripgrep reads `.ignore` before `.gitignore`, so Graft's generated markdown cards are deliberately exposed to the agent's `Grep` and `Glob`. Archex writes no comparable agent-readable file — `.archex/` is index state. This is a real asymmetry, it is part of measuring each product as shipped, it is disclosed here rather than raised later as an unmodelled confound, and the cardinality cap above bounds how far breadth alone can carry a completeness score. Whether `graft/` is git-tracked and whether `.ignore` exists are recorded per cell.
- **`--strict-mcp-config` with an explicit `--mcp-config`.** Verified live: without `--mcp-config` the `init` event reports `mcp_servers: []`, proving ambient servers are excluded; with the arm's project `.mcp.json` passed explicitly it reports exactly that one server as `connected`. This is simultaneously the isolation mechanism and the arm-identity proof at session start.
- **The Graft MCP command is repinned.** `graft init` writes `"command": "npx", "args": ["-y", "@nanonets/graft", "mcp"]`, which resolves the *latest* published package at call time and would silently break the pin. The runner rewrites that entry to the absolute pinned binary with `CI=1` and `DO_NOT_TRACK=1`, records both the original and the rewritten command shape in the artifact, and changes nothing else. This is the only deviation from either product's literal output, and it exists to preserve R19's pinned identity.
- **Explicit base tool allowlist and denylist.** `--allowedTools` and `--disallowedTools` are frozen above. The resulting `init` tool list is asserted per cell against the arm's expected set: `Read`, `Grep`, `Glob`, plus `mcp__archex__context` and `mcp__archex__query_repo` for the control, or the six `mcp__graft__*` tools for the treatment. Archex advertises 2 of its 20 tools until retrieval opens its disclosure gate; that shipped behavior is measured, not bypassed.
- **No write tools on either arm.** The task family asks questions, so `Write`, `Edit`, `MultiEdit`, and `Bash` are denied. This makes the run cheaper and safer, and it means the frozen family performs no edits.
- **Stale-index events are recorded but expected to be zero.** With no write tools, neither index can go stale from agent activity, so this metric is structurally near-zero here. It is still recorded because the freshness probe is real and the field must exist for the later paired re-measurement, which will need an edit-bearing family to make it informative. It is reported as recorded-and-vacuous rather than as evidence of freshness parity.
- **Hook invocation is measured at an instrumented boundary.** Verified live: Claude Code surfaces `system`/`hook_started` and `hook_response` events for `SessionStart` only. Neither Archex's `PreToolUse` hook nor Graft's `UserPromptSubmit` and `PostToolUse` hooks appear in the transcript, so transcript counting would under-report both arms. The runner therefore rewrites every hook `command` the product installed to a recorder that appends one JSONL record — timestamp, hook event, matcher, tool name, stdin and stdout byte lengths, exit code, latency — and then executes the original command with the same stdin, passing stdout and the exit code through unchanged. The original command shape is recorded in the artifact. The recorder is applied identically to both arms and is covered by a fixture test asserting byte-identical stdout and exit code with and without it.
- **External deadline.** The pinned agent exposes no turn or time cap, so the runner enforces a 300-second per-cell deadline by terminating the process group. A terminated cell is a recorded failure with reason `deadline_exceeded`; its partial receipt is retained and its cost counts against the ceiling.
- **Ceiling and abort.** Cumulative modelled cost across completed cells is checked before each new cell against a **`$25`** ceiling. Reaching it aborts the run.
- **Telemetry off.** `DO_NOT_TRACK=1` and `CI=1` are set for the Graft install and every Graft invocation; `graft telemetry` must report telemetry off in the run log. Archex usage metrics are left at their shipped default and the resolved setting is recorded.
- **Repetitions are not seeded.** The agent is nondeterministic and the client exposes no temperature or seed control. The 3 repetitions exist to expose that spread, and the per-cell spread is published.

**Privacy contract, frozen:**

Committed artifacts contain only: task id, arm, repetition index, agent version, model id, the resolved tool list, MCP server name and status, tool-call names and counts, input/output/cache token counts, modelled cost, wall and setup timings, freshness state, hook records as described above, the parsed repository-relative answer paths, scoring flags, and sanitized failure reasons. They contain no prompt text — the prompt is fully determined by the task `question` already in the repository plus Appendix A's hash — no repository source content, no tool inputs or outputs, and no absolute machine paths. The repository and workspace roots are replaced with `<repo>` and `<workspace>`, and any residual `/Users/`, `/home/`, `/private/`, or `/tmp/` string aborts the run rather than being published. Raw transcripts stay in the local run directory and are not committed.

**Analysis, frozen:**

- Primary: mean required-file completeness, treatment minus control, with a 10 000-resample cluster bootstrap over the 15 repositories, seed `20260909`, reported against MWG, NIM, and EQM above and read under the power disposition. Reported twice: over all cells, and over product-using cells only.
- Pre-declared secondary, labelled exploratory: the efficiency family — tool calls, total tokens, modelled cost, and wall time — reported over all cells and separately over both-complete comparison units, where a unit is one `(task, repetition index)` whose control and treatment cells both scored completeness `1.0`. Because repetitions are unseeded, that is an index alignment, not a matched pair. The restriction exists because an arm that answers less completely can look cheaper.
- Pre-declared secondary, labelled exploratory: per-task repetition spread, so readers can see how much of any difference is agent nondeterminism.
- Pre-declared secondary, labelled exploratory: answer precision, `answer_unparsed` rate, `answer_over_broad` rate, and `no_product_use` rate per arm.
- **No exclusions.** Every recorded-failure cell — `deadline_exceeded`, rate-limited, tool-parity violation, or disconnected MCP — enters the primary completeness mean as `0.0`, identically to `answer_unparsed` and `answer_over_broad`, and is never dropped from the denominator.
- The R19 artifacts in `benchmarks/headtohead/results/` are inputs for scale comparison only; they are not regenerated and no R20 change may alter them.

## Appendix A — frozen instruction block

Appended to every cell's prompt after the task `question` and one blank line, byte for byte. SHA-256 of the block, including its trailing newline, is `39acce3ce7bff054ee09b52414c9fede8807da77ccc683b211bd16129ee1007d`; the runner asserts that hash before the first cell.

```text
Identify the files in this repository that are required to answer the question above. Use the tools available to you.

End your reply with a line containing exactly `FILES:` and nothing else, followed by one repository-relative path per line, most relevant first. Name at most 6 paths, and name only files that are required to answer the question.
```

## Post-hoc changes

None. Record only changes made after data exists, with timestamp, reason, affected field, and why the result is exploratory.
