# Local Metrics

This document explains what archex records in the local metrics ledger, how token savings are calculated, and which parts of the system are default-off versus explicitly opt-in.

## The boundary

archex ships with no telemetry by default.

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
- raw-equivalent tokens
- saved tokens
- savings percent
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

The metrics ledger tracks three token views:

1. Returned tokens
   - The size of the actual archex response delivered to the caller.

2. Raw-equivalent tokens
   - The cost of returning the same result as full-file access for the files archex actually returned.
   - This is the baseline used for the headline savings number.
   - `baseline_type` is currently `returned_full_files`.

3. Whole-repo tokens
   - The token count for the indexed repository when archex already has that number cheaply.
   - This is not the headline baseline. It is context only.

The formulas are:

```text
tokens_saved = max(tokens_raw_equivalent - tokens_returned, 0)
savings_pct = 0 if tokens_raw_equivalent <= 0 else (tokens_saved / tokens_raw_equivalent) * 100
whole_repo_tokens_avoided = null if whole_repo_tokens is unavailable
whole_repo_tokens_avoided = max(whole_repo_tokens - tokens_returned, 0) otherwise
```

Example:

```text
returned tokens = 6,132
raw-equivalent tokens = 13,120
whole-repo tokens = 1,302,860

tokens_saved = 13,120 - 6,132 = 6,988
savings_pct = 6,988 / 13,120 = 53.3%
whole_repo_tokens_avoided = 1,302,860 - 6,132 = 1,296,728
```

Interpretation:
- `tokens_saved` is the headline savings number.
- `whole_repo_tokens_avoided` is an upper-bound/context metric, not the headline number.

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

Expected, non-actionable conditions are **not** treated as failures. The optional
whole-repo and raw-equivalent token baselines are computed by walking the source tree;
when a source is not a usable local repo (path gone, not a directory, no `.git`), that
baseline is simply omitted (`whole_repo_tokens` becomes null) without latching a warning.

## Reading the output

`archex metrics summary` reports:
- headline saved tokens versus returned full files
- returned token total
- raw-equivalent token total
- percentage savings
- whole-repo avoided tokens as context only
- a per-surface event split (`cli` / `mcp` / `python_api`)

The surface split shows how many events each surface contributed. A near-zero `mcp`
count next to a healthy `cli` count means archex is registered but agents are not
invoking the MCP tools — see the MCP surfacing notes in the
[compatibility matrix](CLIENT_COMPATIBILITY_MATRIX.md). The JSON summary exposes the
same split as `totals.by_surface`.

`archex metrics inspect` reports recent event rows.

`archex metrics repos` ranks known local repos by saved tokens over the selected time window.
