# Client Compatibility Matrix

Last updated: 2026-07-06

This matrix separates config-shape verification from actual client smoke tests. `archex install-client <client>` writes the config by default (global/user scope; a SOURCE path or `--scope project` installs repo-local). Add `--dry-run` to preview the exact target and config without writing.

## Matrix

| Client / path | Tested status | Setup command / config | Watch support | Freshness semantics | Known limitations | Last verified |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Code MCP stdio | Config-path tested; client smoke unverified | `archex install-client claude-code` writes `~/.claude.json` (global); `archex install-client claude-code . --scope project` writes `.mcp.json` with `mcpServers.archex.command = "archex"` and `args = ["mcp"]`. `--dry-run` previews either. | Yes — `archex mcp --watch --watch-path .` | Inline query refresh by default; `--no-refresh` leaves freshness `unknown`; watch keeps a warm process subscribed to file events. | This stack did not run a live Claude Code UI smoke. Skill and MCP are separate rows. | 2026-06-16 |
| Claude Code PreToolUse hook (opt-in) | Config-shape tested end-to-end (install, remove, non-destructive merge, matcher-only-Grep/Glob assertion); no live Claude Code UI smoke | `archex install-client claude-code --hooks` writes `~/.claude/settings.json` (global) or `.claude/settings.json` (project) — a different file from the MCP config above. `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#claude-code-pretooluse-hook-opt-in) for the full contract. | N/A — one subprocess per matched tool call, not a warm process | Every injected block carries `index_revision=`/`generated_at=` receipt fields; a missing/stale index degrades to no injected context plus a diagnostics log line. | Opt-in, never installed by default; augments only (`additionalContext`, never `permissionDecision`); matcher is `Grep`/`Glob` only, `Read` is never intercepted; hard ~500ms lookup timeout; diagnostics at `~/.archex/hook-diagnostics.log`. | 2026-07-06 |
| Claude Code skill command | Existing skill path tested in-repo; client smoke unverified | Use `skills/archex/` and the `/archex` command flow. No config file is written by `install-client`; this is command-only onboarding. | Indirect — skill can target a warm MCP server. | Same as MCP/query/scout paths underneath. | Skill setup remains repo-local documentation, not a writable client config target. | 2026-06-16 |
| CLI-only query/scout | Tested | No client config required. Run `archex doctor`, `archex scout`, `archex query`. | N/A | Query checks freshness inline unless `--no-refresh`; scout inherits query freshness in its receipt. | Not an MCP client. | 2026-06-16 |
| Generic MCP stdio client | Unverified | Use a JSON config shaped like `{ "mcpServers": { "archex": { "command": "archex", "args": ["mcp"] }}}`. `archex install-client claude-code --dry-run` prints a compatible snippet. | Client-dependent | Same server-side freshness semantics as Claude Code / Cursor. | No live generic-client smoke in this stack. | 2026-06-16 |
| Codex headless | Unverified | `archex install-client codex` writes `~/.codex/config.toml` (global); `archex install-client codex . --scope project` writes `.codex/config.toml`, appending `[mcp_servers.archex]`, `command = "archex"`, `args = ["mcp"]` without overwriting existing sections. `--dry-run` previews. | Yes — via `archex mcp --watch --watch-path .` after Codex launches the server. | Inline query refresh by default; warm watch is server-side, not Codex-specific. | Config shape verified against OpenAI Codex MCP docs; no Codex client smoke in this stack. | 2026-06-16 |
| Pi | Config shape verified; client smoke unverified | `archex install-client pi` writes `~/.pi/agent/mcp.json` with a stdio `mcpServers.archex` entry (`--dry-run` previews). User scope only. | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No Pi client smoke in this stack. | 2026-06-16 |
| oh-my-pi (omp) MCP stdio | Config shape verified; client smoke unverified | `archex install-client omp` writes `~/.omp/agent/mcp.json` (user scope only) with `mcpServers.archex.command = "archex"`, `args = ["mcp"]`, plus the oh-my-pi `$schema` (`--dry-run` previews). | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No oh-my-pi client smoke in this stack. Discovery-gated harness — tools must be activated before use (see below). | 2026-06-20 |
| oh-my-pi (omp) `tool_result` hook (opt-in) | Config-shape tested end-to-end (install, remove, idempotent reinstall, dispatch-table excludes `read`); no live oh-my-pi UI smoke | `archex install-client omp --hooks` writes a TypeScript extension module to `.omp/extensions/archex-hook.ts` (project scope) or `~/.omp/agent/extensions/archex-hook.ts` (user scope, default) — a different file/mechanism from the MCP config above. `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#oh-my-pi-omp--pi-tool_result-hook-opt-in) for the full contract. | N/A — one subprocess per matched tool call, not a warm process | Same receipt/timeout/diagnostics semantics as the Claude Code hook below — this module shells out to the identical `python -m archex.integrations.hook` subprocess unmodified. | Opt-in, never installed by default; augments `grep`/`glob` `tool_result` content only, never `read`. Supports both project and user scope (unlike the MCP config row above, which is user-scope only). | 2026-07-06 |
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
archex install-client omp
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
- Pi and oh-my-pi (omp) only support `--scope user`.

## Registration is not surfacing is not invocation

Registering an MCP server is necessary but not sufficient for an agent to actually use archex. Three distinct steps must all happen:

- **Registration** — `install-client` writes the MCP server entry into the client config (this command).
- **Surfacing** — the client/harness must expose the registered tools to the agent. Harnesses with on-demand tool discovery (e.g. oh-my-pi / Pi) treat a registered server's tools as *discoverable* but keep them out of the default tool set; the agent must activate them before the first call.
- **Invocation** — the agent must choose to call `query_repo` / `scout_repo` / `analyze_repo` instead of reading files by hand.

archex cannot change a harness's tool-gating, but it ships a ready-to-paste agent-file guidance prompt that names the MCP tools and the activation step. Append it to a global or repo-specific agent file (`CLAUDE.md`, `AGENTS.md`, ...):

```bash
archex install-client omp --agent-file ~/.omp/agent/AGENTS.md
archex install-client claude-code . --scope project --agent-file ./CLAUDE.md --dry-run
```

The append is non-destructive and idempotent (a delimited `archex:mcp-guidance` block, never duplicated on re-run), and `--dry-run` previews the block without writing.

To check whether agents actually route through MCP, `archex metrics` reports a CLI-vs-MCP surface split; a near-zero `mcp` count means the tools are registered but not being invoked.

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
For oh-my-pi, inspect `~/.omp/agent/mcp.json` and confirm the archex tools are activated in your session (discovery-gated harnesses require explicit activation).

## Claude Code PreToolUse hook (opt-in)

`archex install-client claude-code --hooks` installs a Claude Code `PreToolUse` hook (`src/archex/integrations/hook.py`, invoked as `python -m archex.integrations.hook`) that augments `Grep`/`Glob` tool calls with archex symbol-search results, injected as `additionalContext`. It is opt-in — plain `archex install-client claude-code` never installs it — and it writes to a different file than MCP server registration:

```bash
archex install-client claude-code --hooks                     # global: ~/.claude/settings.json
archex install-client claude-code . --hooks --scope project   # repo-local: .claude/settings.json
archex install-client claude-code --hooks --dry-run           # preview only, writes nothing
archex install-client claude-code --remove-hooks              # clean uninstall
```

Installed config shape (the `command` path is the Python interpreter active when `--hooks` was run, so the hook always runs in the same environment archex was installed into):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Glob|Grep",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/venv/bin/python",
            "args": ["-m", "archex.integrations.hook"]
          }
        ]
      }
    ]
  }
}
```

Non-blocking contract:

- **Never intercepts `Read`.** The installed matcher is `Grep`/`Glob` only — read-before-edit semantics are preserved. Every install writes exactly this matcher; a repo-level assertion test (`tests/cli/test_install_client_hooks.py`) checks the written config never reaches an archex hook entry through a matcher that includes `Read`.
- **Exits 0 on every path.** A missing or stale index, a malformed payload, a timeout, or any internal error all degrade to no injected context — never a blocked or errored tool call. Failures are never surfaced to the agent; they are appended as JSON lines to a local diagnostics log instead (`~/.archex/hook-diagnostics.log` by default, override with `ARCHEX_HOOK_DIAGNOSTICS_LOG`).
- **Hard ~500ms lookup timeout** (override with `ARCHEX_HOOK_TIMEOUT_SECONDS`) — a lookup still running past the budget is abandoned in place and the process exits immediately, so a stuck lookup can never block the agent loop.
- **Every injected block carries a freshness/receipt marker** — `index_revision=<hash prefix> generated_at=<UTC timestamp>` — mirroring the receipt contract `query`/`scout` already expose, so a downstream agent can tell how current the injected context is.
- **Non-destructive install/uninstall.** Any other hooks already configured in the same settings file — other tools' `PreToolUse` matcher groups, other hook events entirely, unrelated top-level settings — are left untouched by both `--hooks` and `--remove-hooks`. Re-running `--hooks` is an idempotent no-op once installed.

Manual verification (bypassing Claude Code entirely — this is exactly what the hook receives on stdin for a `Grep` tool call):

```bash
echo '{"tool_name":"Grep","tool_input":{"pattern":"compute_delta"}}' \
  | python -m archex.integrations.hook
```

A repo with a fresh index returns JSON on stdout shaped `{"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": "..."}}`. A repo with no index, a stale index, or a `cwd` outside a Git working tree exits 0 with empty stdout and a diagnostics log line instead.

Section G of the development plan (M20–M23) extends this same `python -m archex.integrations.hook` subprocess contract to other clients with per-client installer shims. M20 (oh-my-pi implemented below; Pi confirmation and wiring pending) is in progress; Codex CLI, OpenCode are not yet implemented; Cursor's weaker prompt-level mechanism is a planned exception.

## oh-my-pi (omp) / Pi `tool_result` hook (opt-in)

`archex install-client omp --hooks` installs a shared TypeScript extension module (`archex-hook.ts`) that registers a `pi.on("tool_result", ...)` handler scoped to `grep`/`glob`-equivalent tool calls, augmenting their content by shelling out to the identical `python -m archex.integrations.hook` subprocess documented above — no lookup, ranking, timeout, or diagnostics logic is reimplemented for this client. It is opt-in — plain `archex install-client omp` never installs it:

```bash
archex install-client omp --hooks                      # user (default): ~/.omp/agent/extensions/archex-hook.ts
archex install-client omp . --hooks --scope project    # project-local: .omp/extensions/archex-hook.ts
archex install-client omp --hooks --dry-run            # preview only, writes nothing
archex install-client omp --remove-hooks               # clean uninstall
```

Unlike the Claude Code hook (a JSON command entry merged into `settings.json`), this installs a standalone `.ts` file discovered by oh-my-pi's own native extension auto-discovery (project: `<cwd>/.omp/extensions/*.ts`; user: `~/.omp/agent/extensions/*.ts`). The Python interpreter invoked (`ARCHEX_PYTHON_COMMAND` baked into the file) is the one active when `--hooks` was run, exactly like the Claude Code hook's `command` field.

Contract, mirroring the Claude Code hook:

- **Never intercepts `read`.** The module's only tool-name dispatch is a lookup table (`ARCHEX_QUERY_FIELDS`) keyed on `grep`, `glob`, and `find` — `read` is absent from that table by construction, not filtered by a conditional that could be gotten wrong. `tests/cli/test_install_client_hooks.py`'s `test_omp_ts_hook_module_query_field_table_excludes_read` asserts this structurally on every generated module.
- **Exits without throwing on every path.** A missing/stale index, a subprocess spawn failure, a timeout, or a malformed subprocess response all degrade to leaving `tool_result` content unmodified. Failures append a JSON line to the same diagnostics log the Python subprocess uses (`~/.archex/hook-diagnostics.log` by default, override with `ARCHEX_HOOK_DIAGNOSTICS_LOG`) — never surfaced to the agent.
- **Hard ~500ms lookup timeout**, matching `DEFAULT_HOOK_TIMEOUT_SECONDS`: the module's own `setTimeout` guard `SIGKILL`s a subprocess still running past the budget, independent of whatever timeout the subprocess enforces on itself.
- **Field-name translation happens at the edge.** oh-my-pi's `glob` tool carries its query in a field named `path` (it has no `pattern` field); Pi has no `glob` tool at all — its glob-equivalent is `find`, whose query is already in a field named `pattern`. Both are translated into the subprocess's own `{"tool_name": "Glob", "tool_input": {"pattern": ...}}` shape before the subprocess is ever invoked, so `archex.integrations.hook` stays unaware of any client but Claude Code. See `tests/integrations/test_hooks.py`'s `test_omp_glob_shim_translates_path_field_to_subprocess_glob_payload` and `test_pi_find_shim_translates_pattern_field_to_subprocess_glob_payload`.
- **Uniform across dispatch contexts.** The handler is registered once, unconditionally, with no subagent/session discriminator — oh-my-pi's `tool_result` event carries no such field, so a subagent-issued `grep`/`glob` call is augmented identically to a top-level one.

Manual verification (bypassing oh-my-pi entirely — same fixture payload the Claude Code section above uses, since the subprocess contract is unmodified):

```bash
echo '{"tool_name":"Grep","tool_input":{"pattern":"compute_delta"}}' \
  | python -m archex.integrations.hook
```

Pi's own installer (`archex install-client pi --hooks`) and its extension-directory/`tool_result` contract confirmation are tracked as a separate, not-yet-landed PR in the M20 stack.
