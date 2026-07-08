# Local Metrics

This document explains what archex records in the local metrics ledger, how token savings are calculated, and which parts of the system are default-off versus explicitly opt-in.

## The boundary

No telemetry is sent by core CLI, Python API, MCP, or Docker slim workflows. archex ships with no telemetry by default.

Local metrics are optional and default-off.

What requires explicit enablement:
- CLI `query` and `scout`
- MCP `query_repo` and `scout_repo`
- Structural CLI and MCP tools that already have a reliable raw-equivalent token baseline without adding expensive new work

What is explicitly opt-in on top of metrics enablement:
- Detailed traces via `archex metrics trace enable` or `ARCHEX_USAGE_TRACE=on`
- Python API writes via `record_usage_event(...)`

What does not exist in v1:
- Hosted upload
- Team or SaaS metrics service
- Any metrics code path that makes LLM calls

To turn local metrics on, use `archex metrics enable`, `ARCHEX_USAGE_METRICS=on`, or the persisted metrics setting. To turn them off everywhere for the current process, use `ARCHEX_USAGE_METRICS=off`.

## Where the ledger lives

The machine-local SQLite ledger lives at:

```text
~/.archex/usage.sqlite
```

This is separate from repo-local `.archex/` state.

## What gets stored when metrics are enabled

When metrics are enabled, default event rows store anonymous counters only:
- surface
- tool name
- category
- returned tokens
- full-file (raw-equivalent) tokens
- saved tokens and savings percent vs full-file
- optional targeted-read tokens and savings vs targeted read
- optional whole-repo avoided tokens
- file count
- freshness
- index revision
- machine-local repo UUID

Default event rows do not store:
- query text
- file paths
- symbol names
- scout handles
- source snippets
- rendered outputs
- prompt bodies
- Git remote URLs
- org names
- repo names in event rows

The local `repos` table does map the machine-local repo UUID back to a local path so repo summaries and exports can work on the same machine. Default exports include the repo `display_name` (the local directory basename). The full `repo_root` path stays out of export output unless you opt in with `--include-local-paths`.

## Savings calculation

The metrics ledger tracks these token views per event:

1. Returned tokens
   - The size of the actual archex response delivered to the caller.

2. Full-file tokens (raw-equivalent)
   - The true token cost of reading the returned files in full (`count_tokens` of each
     file's source, summed), derived from the index — not a chunk sum, which would
     over-count synthetic per-chunk import breadcrumbs.
   - This is the naive "paste the whole file" baseline. `baseline_type` is
     `returned_full_files`.

3. Targeted-read tokens
   - The realistic counterfactual: reading only the matched line ranges plus a small
     context window (K = 5 lines each side), costed from the index.
   - Recorded when line spans are available (`query`); omitted (null) otherwise
     (for example, scout file-only results).

4. Whole-repo tokens
   - The token count for the indexed repository when archex already has that number
     cheaply. Context only, never a savings number.

The two savings numbers each name their baseline:

- Savings vs full-file paste — compression versus dumping every returned file in full.
- Savings vs realistic targeted read — the conservative number, versus how code is
  actually pulled (matched ranges plus a little context).

The formulas are:

```text
tokens_saved              = max(full_file - returned, 0)
savings_pct               = 0 if full_file <= 0 else (tokens_saved / full_file) * 100

tokens_saved_vs_targeted  = max(targeted_read - returned, 0)   # null if targeted unavailable
savings_pct_vs_targeted   = 0 if targeted_read <= 0 else (tokens_saved_vs_targeted / targeted_read) * 100

whole_repo_tokens_avoided = max(whole_repo - returned, 0)      # null if whole_repo unavailable
```

Invariant: `returned <= targeted_read <= full_file` for any input where `returned <= full_file` (the realistic case). On a tiny file whose rendered chunks exceed its source (`returned > full_file`), `targeted_read` is capped at `full_file`, so savings versus targeted reports 0.

Example:

```text
returned tokens      = 6,132
full-file tokens     = 13,120
targeted-read tokens = 8,400
whole-repo tokens    = 1,302,860

savings_pct             = (13,120 - 6,132) / 13,120 = 53.3%   (vs full-file paste)
savings_pct_vs_targeted = (8,400 - 6,132)  / 8,400  = 27.0%   (vs realistic targeted read)
whole_repo_tokens_avoided = 1,302,860 - 6,132 = 1,296,728     (context only, not savings)
```

Interpretation:
- `savings_pct` (vs full-file paste) is compression versus a naive full-file paste.
- `savings_pct_vs_targeted_read` is the realistic, conservative number.
- `whole_repo_tokens_avoided` is an upper-bound/context metric, never the headline.
- A defensible cross-tool number (vs grep / read) is not produced in-process; it is
  available only via the offline benchmark harness — see "Cross-tool efficiency" below.

Both baselines are derived from the index (per-file token totals and chunk line spans).
Neither re-reads files on the query path, and no metrics path calls a model.

## Cross-tool efficiency (offline benchmark)

The two in-process savings numbers above compare archex's returned context against a
counterfactual reconstruction of the *same* returned set. They do not answer "how many
tokens would a non-archex agent spend to localize the same code?" That cross-tool number
cannot be computed on the query path — it requires running both retrieval paths at a fixed
recall on labeled tasks — so it lives only in the benchmark harness and never enters the
`archex metrics summary` ledger or any product path.

`archex benchmark cross-tool` measures **tokens-at-fixed-recall**: the tokens archex spends
to localize a task's required files (its targeted returned regions, in rank order) versus a
naive grep/read agent (whole grep-hit files, or `+/-K` context windows around grep hits, in
grep-relevance order), tokenized with the same `cl100k_base` encoder. Recall is held equal:
a token delta is reported only for tasks where both paths reach the target required-file
recall (default 100%), so the number never compares unequal recall. The naive model is a
pure, deterministic function of the gitignore-aware corpus, the task keywords, and `K`.

The checked-in reference artifact
[`benchmarks/cross-tool-efficiency/cross-tool-comparison.json`](../benchmarks/cross-tool-efficiency/cross-tool-comparison.json)
grades the benchmark task set per corpus (localization graded separately, never merged with
comprehension). At 100% required-file recall, on the tasks where archex itself fully
localizes the required files, the token reduction versus the naive agent is:

| Corpus | Naive model | Comparable tasks | archex tokens | naive tokens | Token reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| self | full_file | 16 / 24 | 9,484 | 4,416,681 | 99.8% |
| self | grep_window | 16 / 24 | 9,484 | 2,626,845 | 99.6% |
| external-comprehension | full_file | 16 / 19 | 22,681 | 783,725 | 97.1% |
| external-comprehension | grep_window | 16 / 19 | 22,681 | 492,119 | 95.4% |
| external-localization | full_file | 20 / 21 | 13,247 | 469,836 | 97.2% |
| external-localization | grep_window | 20 / 21 | 13,247 | 408,410 | 96.8% |

"Comparable tasks" counts only tasks where both paths reach 100% required-file recall; every
token figure sums over exactly that set, so no figure compares unequal recall. The reduction
is therefore conditioned on archex fully localizing the task — it measures how much cheaper
archex localizes when it succeeds, not that archex always succeeds. In this artifact the
naive grep/read path reaches full recall on every comparable task, and all 12 excluded tasks
(of 64: 8 self, 3 external-comprehension, 1 external-localization) are archex recall misses —
cases where archex did not return all required files within its token budget (archex recall
0.0–0.8 there), excluded rather than scored at unequal recall. Regenerate the artifact from a
clean run (do not hand-edit metric values):

```bash
uv run archex benchmark cross-tool --tasks-dir benchmarks/tasks \
  --output benchmarks/cross-tool-efficiency
```

This is an offline benchmark number, not a per-event ledger metric: it is never recorded in
`~/.archex/usage.sqlite` and never shown by `archex metrics summary`.

## Surface defaults

| Surface | Default | Notes |
| --- | --- | --- |
| CLI `query` / `scout` | Not recorded | Recorded only after explicit metrics enablement. |
| MCP `query_repo` / `scout_repo` | Not recorded | Response shape stays unchanged; recording failures are non-fatal once enabled. |
| Structural CLI/MCP tools | Not recorded | After metrics enablement, archex records only tools with a cheap reliable baseline. |
| Python API `query()` / `analyze()` / `compare()` | Not recorded | No ledger write unless the caller explicitly calls `record_usage_event(...)`. |
| Detailed traces | Off by default | Local-only, opt-in on top of metrics enablement. |
| Hosted upload | Not available | Reserved for possible future work, not implemented in v1. |

## Detailed traces

Detailed traces are local-only and opt-in.

When enabled, a trace may store:
- query text
- returned file paths
- symbol names
- scout handles
- skipped-count metadata
- token math
- repo UUID
- index revision

Detailed traces never store:
- source code
- rendered output bodies
- prompt bodies

## Commands and controls

```bash
archex metrics enable
archex metrics disable
archex metrics
archex metrics summary --format json
archex metrics inspect --format json
archex metrics export --output usage.json
archex metrics delete --all
archex metrics trace enable
archex metrics trace disable
```

Environment overrides:

```bash
ARCHEX_USAGE_METRICS=on
ARCHEX_USAGE_METRICS=off
ARCHEX_USAGE_TRACE=on
ARCHEX_USAGE_TRACE=off
```

Notes:
- `archex metrics enable` or `ARCHEX_USAGE_METRICS=on` turns on local metrics recording.
- `ARCHEX_USAGE_METRICS=off` prevents writes.
- `ARCHEX_USAGE_TRACE=on|off` overrides trace recording.
- `archex metrics export` redacts local repo paths by default.
- `archex metrics export --include-local-paths` includes them explicitly.
- `archex metrics delete --all` removes the local metrics ledger.

## Python API opt-in

Python API calls do not write the metrics ledger by default.

If a Python caller wants to record usage, it must opt in explicitly:

```python
from pathlib import Path

from archex import record_usage_event
from archex.metrics.recorder import UsageEvent

record_usage_event(
    UsageEvent(
        repo_root=Path("."),
        surface="python_api",
        tool_name="query",
        category="context_retrieval",
        tokens_returned=250,
        tokens_raw_equivalent=1000,
    )
)
```

## Retention

Current local retention policy:
- raw anonymous events: 90 days
- detailed traces: 14 days
- daily aggregates: retained indefinitely

## Failure behavior

Metrics recording must never break query, scout, or MCP operations.

If recording fails:
- the user-facing operation still succeeds
- a genuine metrics-subsystem failure (storage, registry, or writer error) latches a
  warning that `archex metrics` surfaces
- the warning is **self-healing**: because health reflects the current state of the
  writer rather than its history, the next successful record on any repo clears it
- `archex metrics repair` clears a stale warning manually once recording works again,
  without deleting any accumulated savings data

The health flag lives in the single machine-local ledger (`~/.archex/usage.sqlite`), so
it is shared across every repo on the machine.

Expected, non-actionable conditions are **not** treated as failures. The full-file and
targeted-read baselines are derived from the index (per-file token totals and chunk line
spans); when those are unavailable — a legacy index built before per-file totals existed,
or a source that is not a usable local repo — the baseline is simply omitted (the event is
not recorded, or `targeted_read`/`whole_repo_tokens` become null) without latching a
warning. A reindex repopulates the per-file totals.

## Reading the output

`archex metrics summary` reports:
- returned token total
- full-file (raw-equivalent) token total
- targeted-read token total
- savings vs full-file paste (compression) and savings vs realistic targeted read
- whole-repo avoided tokens, demoted below the savings lines and labeled an
  upper-bound/context number, not savings
- a per-surface event split (`cli` / `mcp` / `python_api`)

The surface split shows how many events each surface contributed. A near-zero `mcp`
count next to a healthy `cli` count means archex is registered but agents are not
invoking the MCP tools — see the MCP surfacing notes in the
[compatibility matrix](CLIENT_COMPATIBILITY_MATRIX.md). The JSON summary exposes the
same split as `totals.by_surface`.

`archex metrics inspect` reports recent event rows.

`archex metrics repos` ranks known local repos by saved tokens over the selected time window.
