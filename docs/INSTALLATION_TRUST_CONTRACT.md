# Installation and Trust Contract

This document states what archex needs to run, what it reads and writes, when it can use the network, and how to remove it. It covers the supported CLI, MCP, Docker, and in-repository Claude Code skill paths.

## Supported and expected-compatible clients

Tested in this repository:

- CLI: `archex`
- Python API: `from archex import query, analyze, compare`
- MCP stdio server: `archex mcp`
- Docker slim/full images published as `ghcr.io/mathews-tom/archex:slim` and `ghcr.io/mathews-tom/archex:full`
- In-repository Claude Code skill under `skills/archex/`

Expected-compatible but not claimed as fully verified here:

- MCP clients that support stdio servers with JSON config shaped like `mcpServers.<name>.command` plus `args`
- Headless MCP runners. Treat these as unverified until `archex doctor`, `archex mcp`, and one `archex scout` call pass in that runner.

## Guided setup

Install the CLI globally with uv and run the primary onboarding command:

```bash
uv tool install archex
archex setup
archex query "Where is cache invalidation handled?" --format xml
```

`archex setup` is the primary interactive onboarding flow. It initializes the repository, builds the first index, checks the MCP runtime (which is a standard dependency), and offers to install MCP client registrations.

## CLI-only init path

For explicit repo initialization without the full guided setup, such as in scripts:

```bash
uv tool install archex
archex init .
archex query . "Where is cache invalidation handled?" --format xml
```

`archex init` configures repo-local state and builds the first index by default. `archex index` remains available for explicit refresh and advanced indexing options.

Project dependency setup:

```bash
uv add archex
```

Core CLI indexing and BM25 retrieval do not require hosted inference, API keys, vector model downloads, or Hugging Face remote code.

## Context receipt output contract

`archex query --format json` includes a top-level `receipt` object in the serialized `ContextBundle`. `archex query --format xml` includes a compact `<receipt>` block with freshness, index revision, completeness, recommended action, budget, shown/total returned and skipped counts, returned handles, skipped candidates, and omitted dependency-edge counts.

`archex query --format markdown` and `archex scout` markdown include a `## Receipt` block with the same actionable summary: freshness, index revision, budget consumed/requested, completeness, reason, recommended action, shown/total counts, top skipped candidates, and omitted dependency-frontier edges. `archex scout --format json` includes the same receipt fields on the scout result.

MCP `query_repo` and `scout_repo` envelopes now include a top-level `receipt` field next to `content` and `_meta`. This is an intentional additive JSON-envelope change so MCP clients can inspect provenance and completeness without parsing prompt text.

The client-by-client tested/unverified matrix and bootstrap paths live in [CLIENT_COMPATIBILITY_MATRIX](CLIENT_COMPATIBILITY_MATRIX.md).

## MCP setup

The MCP runtime is a standard dependency of archex.

If your installation is damaged or missing the `mcp` package, `archex doctor` and `archex mcp` will explicitly report the missing runtime. Remediation:

```bash
# For uv tool installations:
uv tool install --force archex
# For project dependencies:
uv add archex
```

Register the stdio server exactly as:

```json
{
  "mcpServers": {
    "archex": {
      "command": "archex",
      "args": ["mcp"]
    }
  }
}
```

For warm sessions that keep the index current after file events:

```bash
archex mcp --watch --watch-path .
```

Container-backed MCP config:

```bash
docker run -d --name archex-mcp -v "$PWD:/workspace" -w /workspace ghcr.io/mathews-tom/archex:slim sleep infinity
```

```json
{
  "mcpServers": {
    "archex": {
      "command": "docker",
      "args": ["exec", "-i", "archex-mcp", "archex", "mcp"]
    }
  }
}
```

## Docker setup

Slim image, BM25-only diagnostics:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ghcr.io/mathews-tom/archex:slim archex doctor
```

Full image, local embedding workflow:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace ghcr.io/mathews-tom/archex:full archex query "Where is cache invalidation handled?" --strategy hybrid
```

The mounted repository owns `.archex/`. Index state survives when the host directory persists and is removed when you delete the host `.archex/` directory.

## Claude Code skill setup

The repository ships a skill at `skills/archex/` and a slash command at `skills/archex/commands/archex.md`. To load it from a local checkout in a Claude Code setup that reads user skills:

```bash
mkdir -p ~/.claude/skills
ln -s "$PWD/skills/archex" ~/.claude/skills/archex
```


Install (or preview) a client config with:

```bash
# Global (user-scope) install by default — writes immediately, non-destructively.
archex install-client claude-code
# Preview the exact target + config without writing anything.
archex install-client claude-code --dry-run
# Repo-local install: pass a SOURCE path or --scope project.
archex install-client claude-code . --scope project
```
Then use:

```text
/archex How does authentication flow through this repository?
```

The command procedure runs `archex doctor`, initializes/indexes when needed, runs `archex scout`, and fetches exact `symbol:` or `chunk:` handles before a broader query.

## MCP surfacing and agent guidance

Registering the MCP server is necessary but not sufficient. Three steps must all happen for an agent to use archex over MCP:

- **Registration** — `archex install-client <client>` writes the server entry. The default scope is global (user); a `[SOURCE]` path or `--scope project` installs repo-local (except Pi and oh-my-pi, which are user-scope only). Writes merge into existing config non-destructively and are idempotent; `--dry-run` previews the exact target and config with zero filesystem changes.
- **Surfacing** — the harness must expose the registered tools. Harnesses with on-demand tool discovery (e.g. oh-my-pi / Pi) keep a registered server's tools discoverable but out of the default tool set until the agent activates them.
- **Invocation** — the agent must call `query_repo` / `scout_repo` / `analyze_repo` rather than reading files by hand.

archex ships a ready-to-paste guidance prompt naming the MCP tools and the activation step. `install-client --agent-file <path>` appends it to a global or repo-specific agent file (`CLAUDE.md`, `AGENTS.md`, ...) inside a delimited, idempotent `archex:mcp-guidance` block; `--dry-run` previews it without writing:

```bash
archex install-client omp --agent-file ~/.omp/agent/AGENTS.md
archex install-client claude-code . --scope project --agent-file ./CLAUDE.md --dry-run
```

The guidance prompt and agent-file append touch only the chosen agent file and make no network or LLM calls. Use `archex metrics` (the CLI-vs-MCP surface split) to confirm agents are actually invoking the MCP tools.

## What archex reads

archex reads only paths you point it at or paths implied by the current repository:

- repository files selected for indexing
- `.git/` metadata for commit identity and working-tree freshness
- repo-local `.archex/settings.toml` when present
- global `~/.archex/config.toml` when present
- configured MCP client files checked by `archex doctor`
- local Hugging Face and FastEmbed cache directories when model-backed features are selected

archex does not need hosted API keys for core CLI, Python API, MCP, Docker slim, or BM25 retrieval.

Local usage metrics are part of the core local-first contract:

- CLI `query`/`scout` do not record metrics unless the user explicitly enables local metrics.
- MCP `query_repo`/`scout_repo` do not record metrics unless the user explicitly enables local metrics.
- Structural CLI/MCP tools do not record metrics unless the user explicitly enables local metrics; when enabled, archex only records tools that already have a cheap returned-token and raw-equivalent baseline.
- Python API calls do not write the metrics ledger unless the caller explicitly opts in with `record_usage_event(...)`.
- The local ledger path is `~/.archex/usage.sqlite`.
- No metrics code path makes LLM calls.
- No hosted upload exists in v1.

The exact token-savings math and enablement boundary are documented in [LOCAL_METRICS.md](LOCAL_METRICS.md).
## What archex writes

Generated local state:

- `.archex/settings.toml`
- `.archex/index.db`
- `.archex/metadata.json`
- `.archex/vectors/*.npz` for vector indexes in repo-local layout
- `.archex/archgraph.json` when graph export is requested
- `.archex/dogfood/` when dogfood runs are requested
- cache entries under the configured cache directory; before `archex init`, the default is `~/.archex/cache`; after repo-local initialization, the default project setting is `.archex`

Do not commit `.archex/`. The project initializer adds `.archex/` to the repository gitignore path it manages.

Explicit, user-initiated config writes happen only when you run `archex install-client`:

- the selected client's MCP config (for example `~/.claude.json`, `.mcp.json`, `~/.codex/config.toml`, `~/.cursor/mcp.json`, `~/.config/opencode/opencode.json`, `~/.pi/agent/mcp.json`, or `~/.omp/agent/mcp.json`) — merged non-destructively, never clobbering unrelated entries
- the agent file passed to `--agent-file` (for example `CLAUDE.md` or `AGENTS.md`) — a delimited `archex:mcp-guidance` block appended idempotently
- with `--hooks` (opt-in, never written by plain `install-client`): a client-specific hook file separate from the MCP config above — `~/.claude/settings.json` or `.claude/settings.json` (Claude Code), `~/.omp/agent/extensions/archex-hook.ts` or `.omp/extensions/archex-hook.ts` (oh-my-pi), `~/.pi/agent/extensions/archex-hook.ts` or `.pi/extensions/archex-hook.ts` (Pi), `~/.config/opencode/plugins/archex-hook.ts` or `.opencode/plugins/archex-hook.ts` (OpenCode), a marker-delimited block appended to the same `config.toml` as the MCP config (Codex), or `~/.cursor/hooks.json`/`.cursor/hooks.json` (Cursor). `--remove-hooks` reverses each of these. Full per-client contracts live in the [compatibility matrix](CLIENT_COMPATIBILITY_MATRIX.md).

`--dry-run` previews all of the above without writing.

A hook, once installed, additionally appends line-delimited JSON to a local diagnostics log (`~/.archex/hook-diagnostics.log` by default, overridable via `ARCHEX_HOOK_DIAGNOSTICS_LOG`) whenever a lookup degrades — a missing/stale index, a timeout, a malformed payload, or an internal error. Most entries carry only a degradation reason and timestamp. The two diagnostics-only clients are the exception, by design: since Codex and Cursor never inject the archex results they find, the log is the only place those results go, so `codex_augmentation_withheld` includes the raw Bash command that looked like a search and `cursor_context_injection_unsupported` includes the withheld archex context text itself. On every client the log is local-only, appended to a file under your control, and never read back or transmitted anywhere by archex.

## Network behavior by feature

| Feature | Network behavior |
| --- | --- |
| BM25-only CLI/query/MCP | No hosted inference and no model download required. |
| Local repository indexing | No network required. |
| Remote Git source acquisition | Uses the network to clone/fetch the requested repository URL. |
| `vector-fast` / FastEmbed | Can download the selected ONNX embedding model into the FastEmbed cache on first use. |
| `vector-torch` / sentence-transformers | Can download Hugging Face model files on first use. Built-in remote-code models require `allow_remote_code = true` or `--allow-remote-code`. |
| SPLADE | Can download the SPLADE transformers model on first use. |
| Reranker | Built-in Jina reranker can download model files and requires explicit remote-code opt-in. Custom CrossEncoder paths load with remote code disabled. |
| `archex mcp --watch` | Watches local filesystem events; no network by itself. |
| Telemetry | No telemetry is sent by core CLI, Python API, MCP, or Docker slim workflows. |

## Local metrics privacy and controls

For exact token-savings formulas, summary-field interpretation, and enablement rules, see [LOCAL_METRICS.md](LOCAL_METRICS.md).

When local metrics are enabled, default metrics rows contain:

- tool name
- category (`context_retrieval` or `structural_tools`)
- returned tokens
- raw-equivalent tokens
- saved tokens and savings percent
- optional whole-repo avoided tokens as an upper-bound/context metric
- file count
- freshness
- index revision
- repo-local random UUID reference

When local metrics are enabled, default metrics rows do not contain:

- query text
- file paths
- symbols
- scout handles
- source snippets
- rendered outputs
- prompt bodies
- Git remote URLs
- org names
- repo names in event rows

Controls:

```bash
archex metrics enable
archex metrics disable
archex metrics export --output usage.json
archex metrics delete --all
archex metrics trace enable
archex metrics trace disable
ARCHEX_USAGE_METRICS=on
ARCHEX_USAGE_METRICS=off
ARCHEX_USAGE_TRACE=on
ARCHEX_USAGE_TRACE=off
```

`archex metrics enable` or `ARCHEX_USAGE_METRICS=on` turns on local metrics recording. `ARCHEX_USAGE_METRICS=off` prevents writes. `ARCHEX_USAGE_TRACE=on|off` overrides detailed trace recording. `archex metrics export` redacts local repo paths by default unless `--include-local-paths` is passed. `archex metrics delete --all` removes the local metrics ledger.

Detailed traces are local-only and opt-in. They may store query text, returned file paths, symbols, handles, skipped counts, token math, repo ID, and index revision. They never store source code, rendered output bodies, or prompt bodies.

archex records two labeled savings numbers per event: savings versus a full-file paste (compression vs naive full-file access) and savings versus a realistic targeted read (matched line ranges plus a small context window — the conservative number). The full-file baseline equals the true per-file token cost, so it is not inflated by synthetic per-chunk import breadcrumbs. Whole-repo avoided tokens are stored separately and must be treated as an upper-bound/context metric, demoted below the savings lines and never reported as the headline savings number. A defensible cross-tool number (versus grep / read / LSP) is not produced in-process; it is available only via the offline benchmark harness.
Run `archex doctor --security --format json` to inspect selected providers, remote-code policy, revision pins, cache state, offline environment flags, and model-download implications.

## Remote-code policy

Default user-facing model loading keeps Hugging Face remote code disabled. Built-in model paths that require `trust_remote_code=True` fail fast unless opted in through one of:

```toml
[index]
allow_remote_code = true
```

```bash
ARCHEX_ALLOW_REMOTE_CODE=true archex index .
archex index . --allow-remote-code
archex query . "question" --strategy hybrid --allow-remote-code
archex benchmark run --query-fusion --allow-remote-code
```

Built-in remote-code paths use pinned model/code revisions where archex enables remote code by opt-in. Review `archex doctor --security --format json` before enabling this in a cleared environment.

## Cache locations

Model cache locations checked by doctor:

- Hugging Face: `$HF_HOME/hub` when `HF_HOME` is set, otherwise `~/.cache/huggingface/hub`
- FastEmbed: `~/.cache/fastembed`

Index and artifact locations:

- repo-local project layout: `.archex/index.db`, `.archex/vectors/`, `.archex/settings.toml`
- global default cache layout before project initialization: `~/.archex/cache`
- configured cache layout: value of `cache_dir` in config or `ARCHEX_CACHE_DIR`

## Watch and freshness semantics

- `archex index .` performs an explicit full or delta refresh.
- `archex query .` and MCP query paths check freshness and can apply small working-tree deltas before retrieval.
- `archex query --no-refresh` skips inline freshness checks and queries the existing index as-is.
- `archex mcp --watch --watch-path .` keeps a warm process subscribed to local file events.
- Unknown freshness is reported by `archex status` or `archex doctor`; run `archex index .` before relying on results.

## Minimal benchmark verification

Validate the benchmark task file shape:

```bash
archex benchmark validate --tasks-dir benchmarks/tasks
```

Run the public head-to-head report when `.archex/headtohead` results already exist:

```bash
archex benchmark headtohead report --input .archex/headtohead --format markdown
```

For a minimal local smoke benchmark, use a self-only task set that points at the current checkout, then run:

```bash
archex benchmark run --self-only --task <task-id> --output .archex/benchmark-smoke
```

## Uninstall and rollback

CLI installed through uv tool:

```bash
uv tool uninstall archex
```

Project dependency:

```bash
uv remove archex
```

Docker cleanup:

```bash
docker rm -f archex-mcp
docker image rm ghcr.io/mathews-tom/archex:slim ghcr.io/mathews-tom/archex:full
```

Remove generated local state:

```bash
rm -rf .archex
rm -rf ~/.archex/cache
rm -f ~/.archex/config.toml
rm -f .mcp.json
rm -f ~/.claude/skills/archex
```

Remove MCP client registration by deleting the `mcpServers.archex` object from the client config where you added it. Keep a copy of `.archex/settings.toml` before deleting `.archex/` if you need to restore the same indexing configuration later.
