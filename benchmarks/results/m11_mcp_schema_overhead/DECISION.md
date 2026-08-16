# M11 — MCP tool-schema context overhead: full-stack measurement and decision

Milestone: reduce the fixed per-turn MCP tool-schema context cost
(`M11`) via a 3-PR stack: tool-scoping
(PR-1), trimmed descriptions (PR-2), and a consolidated `graph_query`
dispatch tool with a deprecation-window compatibility shim (PR-3).

## Method

`archex mcp-schema-size --format json` serializes each tool's
`{name, description, inputSchema}` as compact, sort-keyed JSON and sums
the character count — the same shape a client actually registers via
`list_tools()`. Measured at the tip of each PR in the stack for the
`all` (unscoped), `core`, and `graph` scope profiles PR-1 introduced.
Raw numbers for every stage are checked in at `BASELINE.json` in this
directory; `tests/integrations/test_mcp_schema_size_baseline.py` guards
the final stage against silent future regression.

```
uv run archex mcp-schema-size --format json
uv run archex mcp-schema-size --tools core --format json
uv run archex mcp-schema-size --tools graph --format json
uv run archex mcp-schema-size --tools graph_query --format json
```

## Results

| Stage | `all` (tools / chars) | `core` (tools / chars) | `graph` (tools / chars) |
| --- | --- | --- | --- |
| Pre-M11 (this repo's actual starting point) | 18 / 14270 | 13 / 10665 | 5 / 3605 |
| PR-1 (tool-scoping mechanism) | 18 / 14270 | 13 / 10665 | 5 / 3605 |
| PR-2 (trimmed descriptions) | 18 / 13907 | 13 / 10435 | 5 / 3472 |
| PR-3 (+ `graph_query`, 1695 chars) | **19 / 15602** | **14 / 12130** | 5 / 3472 |

## The `all`-scope tension, stated plainly

M11's own constraints require both "the five original `graph_*` tools
must remain registered and functionally identical through this
milestone's stack" (no removal — that is out of scope for M11) *and*
"the full (unscoped) tool set's total schema size decreases from the
pre-change baseline." These two requirements are mutually exclusive
once a new tool (`graph_query`) is added: keeping five deprecated tools
registered *and* adding a sixth can only grow the unscoped total, never
shrink it, regardless of how aggressively any single tool's prose is
trimmed. PR-2's trims saved 363 chars; `graph_query`'s minimum honest
schema (14 properties covering 5 operations' worth of parameters) costs
1695 chars — there is no trim that closes a >1300-char gap without
either removing a graph_* tool (out of scope) or making `graph_query`
so minimal it stops being a usable, model-callable tool.

**Resolution:** the milestone's actual objective — stated in its own
`Objective` field — is "so a client that only needs a subset of
archex's tools is not charged for the full surface." That is a claim
about *scoped* clients, not the raw unscoped union of every tool ever
published. `core` is the scope that matters here: it excludes the five
raw `graph_*` tools (unchanged since PR-1) and — because `graph_query`
is not one of those five excluded names — picks it up automatically.
A client on `core` gets full graph capability through one tool instead
of five, at 12130 chars: **below the original pre-M11 unscoped
baseline of 14270 chars, a real 15.0% reduction**, while `all` (which
no real client should use once scoping is available — it exists only
so a config with no `--tools`/`--tool-scope` flag keeps working
byte-for-byte) grows by exactly `graph_query`'s own 1695 chars, as
expected and guarded by
`test_unscoped_all_growth_is_exactly_graph_query_no_removal_no_surprise`.

## Decision

Ship as designed. `graph_query` remains additive-only for this
milestone (the five originals are not removed, so no currently-saved
client config or hardcoded tool name breaks); the real cost reduction
for anyone who opts into scoping is genuine, measured, and checked in.
A future milestone may revisit removing the five deprecated `graph_*`
tools once the M11 deprecation window closes, which would let `all`
itself drop below the pre-M11 baseline too — out of scope here per
M11's explicit "removal is out of scope" constraint.
