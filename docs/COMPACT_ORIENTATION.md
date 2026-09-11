# Compact orientation profile

`archex onboard` renders one deterministic projection of the architecture graph in two profiles. `full` is the historical guide and is unchanged. `compact` is opt-in: a strict token budget, adaptive directory clusters, the existing graph-degree hub ranking as a reading order, and an explicit receipt of everything the budget dropped.

Nothing in this profile parses a repository, computes a second architecture map, changes retrieval or ranking, adds an MCP tool, or summarises anything with a model. It reads the same `ArchGraph` the existing onboarding guide, `archex graph *`, and the Explorer already read.

## Surfaces

Every surface defaults to the existing behaviour. The compact profile appears only when it is asked for.

| Surface | Opt-in | Default |
| --- | --- | --- |
| `archex onboard` | `--profile compact [--token-budget N]` | `--profile full`, byte-identical to previous releases |
| `archex session prime` | `--orientation-budget N` (N > 0) | `0` — records only, byte-identical |
| MCP `generate_onboarding` | `profile: "compact"`, `token_budget: N` | `profile: "full"`, no `orientation` key in the envelope |
| Claude Code `SessionStart` hook | not available | records only (see [Why the hook does not carry it](#why-the-sessionstart-hook-does-not-carry-it)) |
| Python API | `archex.onboarding.render_compact_orientation(graph, token_budget=N)` | `render_onboarding_markdown(graph)` |

The MCP surface adds **no tool**. `profile` and `token_budget` are two new properties on the existing `generate_onboarding` input schema, which grew from 715 to 1056 characters. The tool count stays 20, and the retrieval-gated default surface a client is charged for before it has retrieved anything — `context` and `query_repo`, 3286 characters / 765 tokens — is byte-identical. Verify with `archex mcp-schema-size --format json`.

## What the compact profile contains

Sections are assembled in a fixed order. Every listed item carries an exact fetch handle: a file row carries its repository-relative path, and a cluster row carries its exact directory prefix, so any row can be passed straight back to `archex query`, `archex graph neighbors`, or an editor.

1. **Overview** — file/line/node/edge counts, the top five languages with a count of the rest, the `archex` version, and the indexed commit.
2. **Entry points** — exact paths of `entry_point` nodes.
3. **Directory clusters** — adaptive prefixes over source and configuration files (see below).
4. **Recommended reading order** — entry points first, then source files ranked by the degree `GraphQuery` already computes (inbound plus outbound edges). This is the existing hub ranking, not a new one.
5. **Test surface** — adaptive prefixes over `test` nodes, with the total file count.
6. **Configuration surface** — exact paths of `config` nodes.
7. **Omissions** — one line per section that dropped anything, with counts and a reason.

The `full` profile's `Public Interfaces` and `Complexity Hotspots` sections are deliberately absent. On this repository they are the two lowest-signal sections per token: interface nodes are per-symbol, so 40 rows render a handful of distinct paths, and hotspots rank by token count, which returns lock files and benchmark result JSON.

## Directory clustering

Clustering starts from one root cluster and repeatedly refines the **largest** cluster whose children still fit the row budget, breaking ties by prefix so the output is deterministic. Single-child directory chains are compressed, so a package root is reported at the depth where it actually branches (`src/archex/`, not `src/`). Children holding fewer files than a corpus-derived threshold are folded away, and their parent stays listed as a residual row — `` - `benchmarks/` 33 file(s) outside the listed subdirectories `` — so no file loses a locator. The number of folded directories is reported in the receipt.

## Budget semantics

`token_budget` is a hard ceiling on the rendered view, measured with the same `count_tokens` (tiktoken `cl100k_base`) the session primer and context receipts use.

- The omission receipt is reserved **before** any content row is admitted, using the largest receipt the sections could produce, so the receipt is never the thing that gets truncated.
- Rows are admitted section by section in the order above. Each section reports what it dropped.
- A budget too small to hold the overview plus that reserved receipt is refused with the minimum required, rather than returning an over-budget view and calling the ceiling advisory.
- The rendered content is re-measured after assembly; `receipt.consumed_budget` is that measurement.

Omission reasons:

| Reason | Meaning |
| --- | --- |
| `token_budget` | Rows the budget could not fit. |
| `section_cap` | Items beyond a section's own item cap (entry points, reading order, configuration surface). |
| `folded_directories` | Subdirectories too small to list; their files are counted in a residual row under the parent prefix. |
| `unrenderable_path` | A path containing a control character, which cannot be rendered as one row without letting repository content forge headings. Dropped and counted. |

## Repository-controlled paths

Paths are repository content, and this view is injected into an agent's context whenever `archex session prime --orientation-budget N` is used. A backtick is a legal POSIX filename character that `git ls-files` emits unquoted, so a single-backtick row would let a repository author close the code span and write prose into the primer. Every handle — file row, cluster prefix, language, module, and the full profile's rows too — is therefore rendered with a delimiter one longer than the longest backtick run in the text, padded per CommonMark when the text starts or ends with a backtick. For any path without a backtick the output is byte-identical to a plain `` `path` ``.

Control characters cannot be contained by inline escaping, since a newline ends the row whatever the delimiter. Such paths are dropped and counted in the receipt under `unrenderable_path`, rather than emitted, so an attacker-supplied graph artifact cannot fabricate a heading or a receipt line.

`receipt.consumed_budget` on a session primer measures the records only, so the existing `consumed <= requested` invariant still holds; the orientation's own cost is in `receipt.orientation.consumed_budget`.

## Why the `SessionStart` hook does not carry it

The Claude Code `SessionStart` hook renders the primer inside a single future bounded by `DEFAULT_HOOK_TIMEOUT_SECONDS` (0.5 s, override with `ARCHEX_HOOK_TIMEOUT`), and a timeout produces **no output at all**. On this repository `render_session_primer` already costs 0.13–0.31 s and building the graph costs a further ~0.46 s, so requesting orientation on that path would reliably blow the deadline and silently suppress the records primer the hook exists to deliver.

The hook therefore stays records-only, and orientation is an explicit per-call argument on `archex session prime` and the library. There is no repository-settings or environment switch that turns it on for the hook: a repository can ship its own `.archex/settings.toml`, and a switch deciding whether archex injects repository-derived content into an agent session must not be flippable by that content.

Making the hook path carry orientation needs a bounded cached snapshot written at index time, in the style of `.archex/status-snapshot.json`. That is not part of this profile.

## Verifying it on your own repository

```bash
archex init && archex index .
archex onboard . --profile compact --token-budget 900
archex onboard .                                    # unchanged full guide
archex session prime . --format markdown --orientation-budget 300
archex mcp-schema-size --format json                # tool_count must stay 20
```
