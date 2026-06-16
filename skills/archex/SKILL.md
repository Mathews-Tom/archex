---
name: archex
description: Use when an agent needs local-first codebase context, architecture maps, scout/fetch bundles, or archex MCP setup for an existing repository.
version: 0.1.0
---

# archex

## Overview

archex is a local code-context layer. It indexes the repository, returns structural maps and token-budgeted context bundles, and never requires hosted inference or API keys.

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

For exact install, MCP JSON, Docker, cache, network, freshness, and uninstall semantics, read `docs/INSTALLATION_TRUST_CONTRACT.md`.

## Scout → Fetch Protocol

Use this for broad or architectural questions.

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

## MCP Wiring

Install the MCP extra and register the stdio server with the client:

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

The full installation and trust contract is in `docs/INSTALLATION_TRUST_CONTRACT.md`.

Optional warm-container or long-running local sessions can keep the MCP server alive with watch mode:

```bash
archex mcp --watch --watch-path .
```

## Quick Reference

| Need | Command |
| --- | --- |
| Trust check | `archex doctor .` |
| Initialize | `archex init . && archex index .` |
| Scout map | `archex scout . "question" --budget 1000 --format json` |
| Context bundle | `archex query . "question" --format xml` |
| Exact symbol body | `archex symbol . 'symbol:path.py::name#kind'` |
| Architecture guide | `archex onboard .` |

## Rules

- Use local models only. Do not add hosted embedding providers or API-key dependencies.
- Prefer scout before reading files for broad questions.
- Use exact handles from scout output; do not re-search when a handle already identifies the target.
- Keep `.archex/` generated and uncommitted.
- Run `archex doctor .` for troubleshooting before changing configuration.
