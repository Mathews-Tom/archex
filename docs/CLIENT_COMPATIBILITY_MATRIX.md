# Client Compatibility Matrix

Last updated: 2026-06-20

This matrix separates config-shape verification from actual client smoke tests. `archex install-client <client>` writes the config by default (global/user scope; a SOURCE path or `--scope project` installs repo-local). Add `--dry-run` to preview the exact target and config without writing.

## Matrix

| Client / path | Tested status | Setup command / config | Watch support | Freshness semantics | Known limitations | Last verified |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Code MCP stdio | Config-path tested; client smoke unverified | `archex install-client claude-code` writes `~/.claude.json` (global); `archex install-client claude-code . --scope project` writes `.mcp.json` with `mcpServers.archex.command = "archex"` and `args = ["mcp"]`. `--dry-run` previews either. | Yes — `archex mcp --watch --watch-path .` | Inline query refresh by default; `--no-refresh` leaves freshness `unknown`; watch keeps a warm process subscribed to file events. | This stack did not run a live Claude Code UI smoke. Skill and MCP are separate rows. | 2026-06-16 |
| Claude Code skill command | Existing skill path tested in-repo; client smoke unverified | Use `skills/archex/` and the `/archex` command flow. No config file is written by `install-client`; this is command-only onboarding. | Indirect — skill can target a warm MCP server. | Same as MCP/query/scout paths underneath. | Skill setup remains repo-local documentation, not a writable client config target. | 2026-06-16 |
| CLI-only query/scout | Tested | No client config required. Run `archex doctor`, `archex scout`, `archex query`. | N/A | Query checks freshness inline unless `--no-refresh`; scout inherits query freshness in its receipt. | Not an MCP client. | 2026-06-16 |
| Generic MCP stdio client | Unverified | Use a JSON config shaped like `{ "mcpServers": { "archex": { "command": "archex", "args": ["mcp"] }}}`. `archex install-client claude-code --dry-run` prints a compatible snippet. | Client-dependent | Same server-side freshness semantics as Claude Code / Cursor. | No live generic-client smoke in this stack. | 2026-06-16 |
| Codex headless | Unverified | `archex install-client codex` writes `~/.codex/config.toml` (global); `archex install-client codex . --scope project` writes `.codex/config.toml`, appending `[mcp_servers.archex]`, `command = "archex"`, `args = ["mcp"]` without overwriting existing sections. `--dry-run` previews. | Yes — via `archex mcp --watch --watch-path .` after Codex launches the server. | Inline query refresh by default; warm watch is server-side, not Codex-specific. | Config shape verified against OpenAI Codex MCP docs; no Codex client smoke in this stack. | 2026-06-16 |
| Pi | Config shape verified; client smoke unverified | `archex install-client pi` writes `~/.pi/agent/mcp.json` with a stdio `mcpServers.archex` entry (`--dry-run` previews). User scope only. | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No Pi client smoke in this stack. | 2026-06-16 |
| OpenCode | Config shape verified; client smoke unverified | `archex install-client opencode` writes `~/.config/opencode/opencode.json` (global); `archex install-client opencode . --scope project` writes `opencode.json`, setting `mcp.archex = { type = "local", command = ["archex", "mcp"], enabled = true }`. `--dry-run` previews. | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No OpenCode client smoke in this stack. | 2026-06-16 |
| Cursor | Config shape verified; client smoke unverified | `archex install-client cursor` writes `~/.cursor/mcp.json` (global); `archex install-client cursor . --scope project` writes `.cursor/mcp.json` with `mcpServers.archex.command = "archex"`, `args = ["mcp"]`. `--dry-run` previews. | Yes — with a warm `archex mcp --watch --watch-path .` process. | Inline query refresh by default; watch keeps a warm process subscribed to file events. | No Cursor UI smoke in this stack. | 2026-06-16 |
| Dockerized MCP server | Server path tested; client smoke unverified | Run `docker run -d --name archex-mcp -v "$PWD:/workspace" -w /workspace ghcr.io/mathews-tom/archex:slim sleep infinity` then point the client to `docker exec -i archex-mcp archex mcp`. | Yes — run the MCP process with `--watch`. | Same server-side freshness semantics as stdio. | Client-specific Docker registration varies; use the same client config shapes above, but replace the command with `docker` / `exec`. | 2026-06-16 |

## First-party bootstrap command

Installs write by default; add `--dry-run` to preview the exact target and config first. The default scope is global (user); pass a SOURCE path or `--scope project` for a repo-local install.

```bash
archex install-client claude-code
archex install-client cursor
archex install-client opencode
archex install-client codex
archex install-client pi
```

Preview any of them without writing:

```bash
archex install-client claude-code --dry-run
```

Install repo-local instead of global:

```bash
archex install-client claude-code . --scope project
```

## Safe-write behavior

- Writes happen by default; `--dry-run` previews the target and config and changes nothing on disk.
- The default scope is global (user); a SOURCE path or `--scope project` selects a repo-local target.
- JSON clients merge an `archex` entry into the existing top-level server map without clobbering unrelated sections.
- Codex appends one `[mcp_servers.archex]` section to `config.toml`.
- Re-running an install with an identical `archex` entry already present is an idempotent no-op; a different existing `archex` entry is left untouched and the command fails instead of overwriting it.
- Pi only supports `--scope user`.

## Verification commands

Use these after writing config:

```bash
archex doctor .
archex scout . "How does authentication flow through this repo?" --budget 1000 --format json
archex query . "Where is cache invalidation handled?" --format json
```

For Codex, open the TUI and run `/mcp` after writing `.codex/config.toml`.
For Cursor, inspect `.cursor/mcp.json` or `~/.cursor/mcp.json` and restart the client.
For OpenCode, inspect `opencode.json` and run `opencode mcp list` if available in your installed version.
For Pi, inspect `~/.pi/agent/mcp.json` and open the MCP panel documented by your Pi build.
