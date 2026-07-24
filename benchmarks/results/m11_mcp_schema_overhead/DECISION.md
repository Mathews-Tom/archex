# M11 PR-2 — Trimmed MCP tool descriptions: schema-size measurement

Milestone: reduce the fixed per-turn MCP tool-schema context cost
(`.docs/DEVELOPMENT_PLAN.md` §4 M11). This PR trims the heaviest tool
descriptions (`get_impact`, `explain_target`, and the five `graph_*`
tools) while preserving model-callable intent — every mutual-exclusion,
precedence, and behavior-selecting fact a caller needs to construct a
correct call stays in the trimmed text; only redundant framing prose
("Return ...", "from an exported artifact without indexing source
files" repeated identically across all five `graph_*` tools) was cut.

## Method

`archex mcp-schema-size --format json` serializes each tool's
`{name, description, inputSchema}` as compact, sort-keyed JSON and sums
the character count — the same shape a client actually registers via
`list_tools()`. Measured once on `m11-1-tool-scoping` (tool-scoping
mechanism only, no description changes — the "before" baseline) and
once on this PR's tip (the "after" figure), for the `all` (unscoped),
`core`, and `graph` scope profiles introduced by PR-1.

```
uv run archex mcp-schema-size --format json
uv run archex mcp-schema-size --tools core --format json
uv run archex mcp-schema-size --tools graph --format json
```

## Results

| Scope | Tools | Before (chars) | After (chars) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| `all` (unscoped) | 18 | 14270 | 13907 | 363 (2.5%) |
| `core` | 13 | 10665 | 10435 | 230 (2.2%) |
| `graph` | 5 | 3605 | 3472 | 133 (3.7%) |

Per-tool: `get_impact` 1330 -> 1118 (-212, -15.9%), `explain_target` 910
-> 892 (-18, -2.0%), `graph_lookup` 715 -> 676 (-39), `graph_neighbors`
839 -> 812 (-27), `graph_path` 849 -> 826 (-23), `graph_stats` 597 ->
574 (-23), `graph_hubs` 605 -> 584 (-21). Every other tool's schema is
byte-identical (untouched by this PR).

Raw before/after numbers and the full post-trim per-tool breakdown are
checked in at `BASELINE.json` in this directory;
`tests/integrations/test_mcp_schema_size_baseline.py` (added in this
PR) asserts the current unscoped total never exceeds the recorded
`after` value for any of the three profiles, guarding against a future
PR silently re-bloating a description back past this baseline.

## Decision

Ship as-is. The reduction is modest (2-4% depending on scope) because
the schemas are already fairly terse `inputSchema` property
descriptions dominate their size, not prose — but it is real, verified,
and stacks with PR-1's tool-scoping (which cuts far more: `core` alone
drops the unscoped 18-tool, 14270-char surface to 13-tool, 10435 chars
post-trim, a 26.9% reduction) and PR-3's `graph_query` consolidation.
No retrieval, ranking, receipt, CLI, or Python API behavior changed —
this PR only rewrites `description` string literals in
`src/archex/integrations/mcp.py`.
