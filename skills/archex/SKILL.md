---
name: archex
description: Use when an agent needs local-first codebase context, architecture maps, scout/fetch bundles, grep-result annotation hooks, or archex MCP setup for an existing repository.
version: 0.2.0
---

# archex

## Overview

archex is a local code-context layer. It indexes the repository, annotates search results with the code units they hit, returns structural maps and token-budgeted context bundles, and never requires hosted inference or API keys.

## Surfaces, in order

Pick the first surface the host supports. The order is an investment order, not a substitution chain: the hook cannot answer "where is X implemented" before any search has run, and the CLI is still how you fetch bodies, scout, and check blast radius.

| Order | Surface | What it does | Needs a decision from the agent? |
| --- | --- | --- | --- |
| 1 | **Hook** (`archex install-client omp --hooks`, `pi --hooks`, `opencode --hooks`, `claude-code --hooks`, `codex --hooks`) | Annotates the agent's own `grep`/`glob`/`find` results and shell `rg`, `grep`, `ugrep`, `git grep`, and path-lister (`find`, `fd`, `git ls-files`) output | No — it is always on |
| 2 | **CLI** (`archex scout`, `symbol`, `impact`, `query`) | Location, structure, bodies, blast radius | Yes — the agent must choose to run it |
| 3 | **MCP** (`archex mcp`) | The same retrieval as tools | Yes, plus a per-request tool-schema cost |

### 1. Hook: annotated search results

On oh-my-pi and Pi, `archex install-client omp --hooks` (or `pi --hooks`) installs an extension that runs after every search-tool call. On OpenCode, `archex install-client opencode --hooks` installs a `tool.execute.after` plugin that does the same for the native `grep`, `glob`, and `bash` tools, appending the lines after the tool's own text (MCP-routed calls are never touched). On Claude Code, `archex install-client claude-code --hooks` installs a `PostToolUse` hook on `Bash|Grep|Glob` that does the same, adding the lines as `additionalContext` (a system reminder next to the tool result); running it again on a settings file from an older archex replaces the retired `PreToolUse` pattern search. On Codex CLI, `archex install-client codex --hooks` installs a `PostToolUse` hook on the shell tool that annotates shell search output (`rg`, `grep`, `git grep`, and the other searches listed above) the same way (Codex records the lines as a developer message next to the command output), and replaces the retired diagnostics-only `PreToolUse` block; Codex runs a config-file hook only after you trust it (`/hooks`). In every case the search result reaches you byte-for-byte as it was, and one fact line per indexed code unit the hits fall in is appended:

```text
[archex receipt] index_revision=e44d3a393e1a units=3
[archex] src/archex/benchmark/runner.py::clone_at_commit function L196-229 · importers 22
[archex] src/archex/doctor.py module-level · units 32 · importers 3
```

Read each line as: qualified name, kind, full line span, and how many files import that file. `module-level` means the hit is outside every function and class. A unit already fully visible in the result gets no line; `+N more units` means the list was capped. Use the lines to decide which hit to open and how much of it to read (`read` with the unit's span, or `archex symbol`), instead of opening every matching file.

No lines means the index is not fresh (stale or edited since indexing), the output format was not recognised, or no hit fell in indexed code. Search results are still complete and exact; nothing was removed.

### 2. CLI: route by question type

| Question | Use |
| --- | --- |
| Exact identifier, literal string, regex, or "every occurrence" | `grep` / `rg`. Completeness matters more than ranking. |
| "Where is X implemented", "how does Y flow", unfamiliar subsystem | `archex scout . "<question>" --budget 1000 --format json`, then `archex symbol` on the returned handles |
| Body of a known symbol | `archex symbol . 'symbol:path.py::Name#kind'` rather than reading the whole file |
| Changing an exported symbol or a widely imported file | `archex impact . --changed-file <path>` first |
| A broader token-budgeted bundle | `archex query . "<question>" --format xml` |

### 3. MCP: clients without a shell

Use MCP only where the client cannot run the CLI. Install the extra and register the stdio server:

```bash
uv tool install "archex[mcp]"
```

```json
{
  "mcpServers": {
    "archex": { "command": "archex", "args": ["mcp"] }
  }
}
```

Fresh sessions advertise two retrieval tools; the rest appear after the first retrieval. For long-running sessions, keep the server warm with `archex mcp --watch --watch-path .`. The full installation and trust contract is in `docs/INSTALLATION_TRUST_CONTRACT.md`.

## First Use

Run diagnostics before retrieval:

```bash
archex doctor . --format json
```

If `index_health` is `error` because the project is uninitialized or missing an index, run:

```bash
archex init .
archex index .
```

If `index_staleness` is `warning`, run `archex index .` before relying on results. If diagnostics still fail, stop and report the exact `archex doctor` check and message.

## Scout → Fetch Protocol

Use this for location and structure questions.

1. Scout without code bodies:

```bash
archex scout . "How does authentication flow through this repo?" --budget 1000 --format json
```

2. Read `fetch_plan.recommended_strategy`:

| Strategy | Action |
| --- | --- |
| `chunk_first` | Fetch the listed `symbol:` or `chunk:` handles before running a broader query. |
| `hybrid_fetch` | Fetch the highest-scoring handles, then run `archex query` only if the fetched context is insufficient. |
| `direct_query` | Skip handle fetch and run a normal context bundle query. |

3. Fetch exact handles:

```bash
archex symbol . 'symbol:src/pkg/service.py::Service#class'
archex symbol . 'chunk:src/pkg/service.py::Service#class'
```

For file handles, run a focused query naming the file path from the handle:

```bash
archex query . "Explain src/pkg/service.py in the auth flow" --format xml
```

For Python callers, pass all scout handles directly:

```python
from archex import query
from archex.models import RepoSource

bundle = query(
    RepoSource(local_path="."),
    "How does authentication flow through this repo?",
    handles=["symbol:src/pkg/service.py::Service#class"],
)
print(bundle.to_prompt(format="xml"))
```

## Reading Receipts

Every scout and query result carries a receipt. Check `context_complete_reason` before trusting the result:

| Reason | Meaning | Do |
| --- | --- | --- |
| `no_candidates` | Retrieval found nothing for these words | Rephrase with the codebase's own identifiers, paths, or error text, or fall back to grep |
| `low_query_match` | Fewer than half the query's terms occur in what came back; `query_terms_unmatched` lists them | Same: rephrase or grep. Raising the budget returns more of the same mismatch |
| `stale_index` | The index is behind the checkout | `archex index .` |

`recommended_next_action: rephrase_query` accompanies both of the first two. archex output is context selection, not proof: verify with read or grep before editing.

## Quick Reference

| Need | Command |
| --- | --- |
| Trust check | `archex doctor .` |
| Initialize | `archex init . && archex index .` |
| Annotate search results (omp/Pi/OpenCode/Claude Code/Codex) | `archex install-client <omp\|pi\|opencode\|claude-code\|codex> --hooks` |
| Scout map | `archex scout . "question" --budget 1000 --format json` |
| Exact symbol body | `archex symbol . 'symbol:path.py::name#kind'` |
| Blast radius | `archex impact . --changed-file path.py` |
| Context bundle | `archex query . "question" --format xml` |
| Architecture guide | `archex onboard .` |

## Rules

- Use local models only. Do not add hosted embedding providers or API-key dependencies.
- Exact strings and "every occurrence" go to grep; location and structure go to scout first.
- Use exact handles from scout output; do not re-search when a handle already identifies the target.
- Keep `.archex/` generated and uncommitted.
- Run `archex doctor .` for troubleshooting before changing configuration.
