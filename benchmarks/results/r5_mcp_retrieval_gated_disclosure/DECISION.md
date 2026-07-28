# R5 — MCP retrieval-gated tool disclosure: measurement and decision

Milestone: cut the fixed per-turn MCP tool-schema context cost
(`.docs/DEVELOPMENT_PLAN.md` §6 R5) via a 3-PR stack — retrieval-gated
disclosure plus token reporting (PR-1), a client compatibility path and
matrix update (PR-2), and this decision record (PR-3).

Builds directly on M11 (`../m11_mcp_schema_overhead/DECISION.md`), whose
tool-scoping mechanism and consolidated `graph_query` tool are used as-is
and not redone, per R5's design-reevaluation instruction.

## Method

`archex mcp-schema-size --tools <scope> --format json` serializes each
tool's `{name, description, inputSchema}` as compact, sort-keyed JSON —
the same shape a client registers via `list_tools()` — and reports both
the character count and the `cl100k_base` token count.

R5 added the token count. Every schema-size target this project sets is
stated in tokens, and the command measured characters only, so the
acceptance bar was uncheckable by the command named to check it.

```
uv run archex mcp-schema-size --tools all --format json
uv run archex mcp-schema-size --tools core --format json
uv run archex mcp-schema-size --tools disclosure --format json
```

Raw numbers are checked in at `BASELINE.json`;
`tests/integrations/test_mcp_disclosure.py` guards the budget and the
reduction ratio against silent future regression.

## The baseline was wrong, and correcting it made the milestone smaller

R5's objective cited **~6 000 tokens**. It is not this repo's baseline.

| Surface | Tools | Chars | Tokens |
| --- | --- | --- | --- |
| Pre-M11 unscoped (from M11's `BASELINE.json`) | 18 | 14 270 | ≈3 530 |
| `all` at R5's design gate | 19 | 15 602 | **3 859** |

No point in this repo's history approached 6 000 tokens. The `~6 000`
appears to be an unsourced estimate carried from the strategy documents.
Corrected in the plan by ledger entry `R5-DG-001` before any code landed,
because a milestone that reports "6 000 → 765" would be overstating its
own result by 55%.

## Results

| Scope | Tools | Chars | Tokens |
| --- | --- | --- | --- |
| `all` (previous default) | 19 | 15 602 | 3 859 |
| `core` (M11) | 14 | 12 130 | 2 908 |
| `graph` (M11) | 5 | 3 472 | 951 |
| **`disclosure` (new default)** | **2** | **3 286** | **765** |

**3 859 → 765 tokens: an 80.2% reduction**, against R5's 1 000-token bar.

The two disclosed tools are `context` (534 tokens) and `query_repo`
(231). Both are retrieval entry points, which is what makes the gate
coherent rather than arbitrary: a session is charged for the two tools it
needs to *start*, and for the rest only once it has demonstrated it is
actually retrieving.

`core` was the obvious alternative default and does not clear the bar at
2 908 tokens — the bar cannot be met by any scope that advertises more
than about three tools, which is why R5 needed a gate and not just a
smaller profile.

## Why changing the default is safe

The first draft of this document got the safety argument wrong, and the
correction is worth stating plainly because it changed what the code had
to do.

**The draft argument.** Narrowing what is advertised has never narrowed
what is *callable*: `build_server`'s `call_tool` dispatches by name
whatever `list_tools()` returned, a property M11 introduced. So the gate
costs discoverability, never capability.

That property is real — it is now pinned by an end-to-end session test
that calls an unadvertised `graph_hubs` through a closed gate and gets a
non-error result. But it is not sufficient, because **MCP tools are
model-controlled**: a model only calls what it was shown. Dispatch-by-name
rescues *hardcoded* callers — a script, or an agent file that names tools
directly, as archex's own `install-client --agent-file` block does. It
does nothing for a model that was never shown the tool.

**The actual argument.** What restores the wider surface for the ordinary
path is the `notifications/tools/list_changed` sent on opening, together
with the `listChanged` capability declared at initialization that entitles
a client to act on it.

That capability is the defect this review caught. The MCP SDK defaults
`NotificationOptions.tools_changed` to `False`, so the first
implementation declared `listChanged=false` and then sent the
notification anyway. A spec-compliant client is entitled to treat the tool
list as static and never re-fetch, which would have made the gate a
permanent tool-hiding mechanism for 17 of 19 tools — the exact regression
the whole design is meant to avoid. `run_stdio_server` now declares
`tools_changed=disclosure`, and a test asserts **both** polarities: an
ungated server must not promise changes it will never send.

The send itself stays best-effort — a missing session or dead transport
must not turn a successful retrieval into a failed tool call — but it is
no longer fire-and-forget. A failed send leaves the announcement pending
and is retried on the next call, and it logs at warning rather than debug,
because a silently lost notification costs a model-controlled client 17
tools for the rest of the session.

## Design decisions

**Only a successful call to a retrieval tool opens the gate.** A failed
retrieval has not demonstrated the session needs the wider surface, and
gating on any tool call at all would defeat the gate on the first turn.

**No tool was added.** R5's out-of-scope forbids adding or removing tool
capabilities. Expansion is triggered by an ordinary retrieval call rather
than by a discovery meta-tool, which would have been a new capability.

**A scope disjoint from the retrieval core is served as asked.**
`--tools graph`'s five tools share nothing with `{context, query_repo}`,
so intersecting would advertise **zero** tools while every tool stayed
callable — strictly worse than serving the scope. A client that narrowed
its own scope has already made the cost decision the gate exists to make
for it. This was a real bug, found by writing the test for it.

(An earlier draft justified this by claiming such a gate "could never
open, since no call could reach a gate-opening name". That reason is
wrong — an unadvertised `context` call dispatches fine and does open the
gate — and believing it hid the next defect: opening a disjoint scope
changed nothing about the advertised list, yet still fired a
notification, costing the client a pointless `tools/list`. The
announcement is now suppressed when the advertised set does not change.)

**Opening is one-way and per-server.** A session that has retrieved does
not fall back to the minimal surface.

## The counter-intuitive part

`--tools` does **not** disable the gate. It bounds what is advertised
*once the gate opens*, so `archex mcp --tools all` still starts at the
minimal set. `--no-disclosure` is the only opt-out.

That is the opposite of the natural guess, and the first draft of both
the plan reconciliation and PR-1's description had it wrong — they
claimed `--tools all` restored the previous surface. An operator who
believed that would think they had opted out when they had not. It is now
stated in the compatibility matrix, in `archex mcp --help`, in
`build_server`'s docstring, and pinned by
`test_an_unscoped_server_is_still_gated`.

## Compatibility

| Client behaviour | What to do |
| --- | --- |
| Honours `notifications/tools/list_changed` | Nothing. Default is correct and cheapest. |
| Ignores the notification, calls tools by hardcoded name | Nothing. Those calls still dispatch. |
| Ignores the notification and builds its list only from `list_tools()` | `--no-disclosure`. Its model would otherwise never see the other 17 tools. |
| Needs every tool visible in `list_tools()` before calling anything | `--no-disclosure`. Full per-turn cost, everything immediately. |

`install-client` and `setup` accept `--no-disclosure`, which writes
`archex mcp --no-disclosure` into the client config. The default stays
implicit — no `--disclosure` is written — so generated configs are
byte-identical to the ones archex wrote before R5 and no existing install
churns.

## Decision

Ship as designed, with the gate on by default.

The reduction is real, measured by the command that grades it, and
checked in. The default change is safe because the expansion is announced
under a declared capability, with dispatch-by-name as the fallback for
hardcoded callers — both now covered by tests through a real session. The client
classes that cannot cope have a documented, tested opt-out on both
`install-client` and `setup`, costing them nothing but the tokens they
were already paying.

What R5 does **not** claim: any improvement in retrieval quality, any
change in tool behavior, or any reduction in the cost of a session that
does retrieve — such a session pays 765 tokens up front and the full
3 859 afterwards, which is slightly *more* than 3 859 across the whole
session. The saving is for the sessions that never retrieve at all, and
for the turns before the first retrieval in the ones that do. That is the
honest shape of the win, and it is worth having because most turns in a
long session are not the first one.

## Limits

- The per-turn saving applies until the first retrieval, not for the
  whole session. A session that retrieves on turn one saves nothing and
  pays 765 extra tokens once.
- Token counts are `cl100k_base`. A client whose model tokenizes
  differently will see a different absolute number; the ratio is stable.
- No *real* client smoke test was run. The notification path is now
  covered end to end against the SDK's in-memory client — list 2 tools,
  retrieve, list 19, observe the notification — but no third-party client
  was driven. The matrix records per-client verification status, and this
  milestone did not change it.
- Through a closed gate, a call to a not-yet-advertised tool is **not
  validated client-side**. The SDK validates arguments against the schema
  it cached from `list_tools()`, so a malformed hardcoded call that used
  to fail fast in the client now reaches the server. One round trip for a
  client that honours the notification; the whole session for one that
  does not.
