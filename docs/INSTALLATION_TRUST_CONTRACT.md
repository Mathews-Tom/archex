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

## CLI-only setup

Install the CLI globally with uv:

```bash
uv tool install archex
archex doctor .
archex init .
archex index .
archex query . "Where is cache invalidation handled?" --format xml
```

Project dependency setup:

```bash
uv add archex
```

Core CLI indexing and BM25 retrieval do not require hosted inference, API keys, vector model downloads, or Hugging Face remote code.

## MCP setup

Install the MCP extra:

```bash
uv tool install "archex[mcp]"
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

Then use:

```text
/archex How does authentication flow through this repository?
```

The command procedure runs `archex doctor`, initializes/indexes when needed, runs `archex scout`, and fetches exact `symbol:` or `chunk:` handles before a broader query.

## What archex reads

archex reads only paths you point it at or paths implied by the current repository:

- repository files selected for indexing
- `.git/` metadata for commit identity and working-tree freshness
- repo-local `.archex/settings.toml` when present
- global `~/.archex/config.toml` when present
- configured MCP client files checked by `archex doctor`
- local Hugging Face and FastEmbed cache directories when model-backed features are selected

archex does not need hosted API keys for core CLI, Python API, MCP, Docker slim, or BM25 retrieval.

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
