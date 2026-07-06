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
| Codex CLI PreToolUse hook (opt-in, diagnostics-only) | Config-shape tested end-to-end (install, remove, idempotent reinstall, preserves unrelated `config.toml` content, matcher-only-`Bash` assertion); no live Codex CLI smoke | `archex install-client codex --hooks` appends a marker-delimited `[[hooks.PreToolUse]]` block to the same `config.toml` the MCP registration above writes to (`~/.codex/config.toml` global, `.codex/config.toml` project). `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#codex-cli-pretooluse-hook-opt-in-diagnostics-only) for the full contract and the confirmation-spike findings. | N/A — one subprocess per matched tool call, not a warm process | Diagnostics-only — no injected context, ever (see limitations); a missing/stale index degrades to a diagnostics log line via the same `archex.integrations.hook` engine the Claude Code hook uses. | Codex's schema supports `additionalContext` augmentation, but Codex has no Grep/Glob-equivalent tool-call event — only a generic `Bash` tool covers every shell command. This hook detects search-shaped `Bash` invocations and logs what archex would have surfaced, but never injects it (would otherwise require intercepting every shell command, not just searches). Requires a one-time hash-based hook-trust review (`/hooks` in Codex) after install. | 2026-07-06 |
| Pi MCP stdio | Config shape verified; client smoke unverified | `archex install-client pi` writes `~/.pi/agent/mcp.json` with a stdio `mcpServers.archex` entry (`--dry-run` previews). User scope only. | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No Pi client smoke in this stack. | 2026-06-16 |
| Pi `tool_result` hook (opt-in) | Config-shape tested end-to-end (install, remove, byte-identical module content vs. the oh-my-pi row below); no live Pi UI smoke | `archex install-client pi --hooks` writes the identical TypeScript extension module to `.pi/extensions/archex-hook.ts` (project scope) or `~/.pi/agent/extensions/archex-hook.ts` (user scope, default) — confirmed against the installed `@mariozechner/pi-coding-agent` 0.68.1. `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#oh-my-pi-omp--pi-tool_result-hook-opt-in) for the full contract. | N/A — one subprocess per matched tool call, not a warm process | Same receipt/timeout/diagnostics semantics as the oh-my-pi hook — Pi's `tool_result` event contract is identical, so this is the same generated module, not a Pi-specific variant. | Opt-in, never installed by default; Pi has no `glob` tool — its glob-equivalent is `find`, already covered by the shared module's dispatch table. | 2026-07-06 |
| oh-my-pi (omp) MCP stdio | Config shape verified; client smoke unverified | `archex install-client omp` writes `~/.omp/agent/mcp.json` (user scope only) with `mcpServers.archex.command = "archex"`, `args = ["mcp"]`, plus the oh-my-pi `$schema` (`--dry-run` previews). | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No oh-my-pi client smoke in this stack. Discovery-gated harness — tools must be activated before use (see below). | 2026-06-20 |
| oh-my-pi (omp) `tool_result` hook (opt-in) | Config-shape tested end-to-end (install, remove, idempotent reinstall, dispatch-table excludes `read`); no live oh-my-pi UI smoke | `archex install-client omp --hooks` writes a TypeScript extension module to `.omp/extensions/archex-hook.ts` (project scope) or `~/.omp/agent/extensions/archex-hook.ts` (user scope, default) — a different file/mechanism from the MCP config above. `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#oh-my-pi-omp--pi-tool_result-hook-opt-in) for the full contract. | N/A — one subprocess per matched tool call, not a warm process | Same receipt/timeout/diagnostics semantics as the Claude Code hook below — this module shells out to the identical `python -m archex.integrations.hook` subprocess unmodified. | Opt-in, never installed by default; augments `grep`/`glob` `tool_result` content only, never `read`. Supports both project and user scope (unlike the MCP config row above, which is user-scope only). | 2026-07-06 |
| OpenCode | Config shape verified; client smoke unverified | `archex install-client opencode` writes `~/.config/opencode/opencode.json` (global); `archex install-client opencode . --scope project` writes `opencode.json`, setting `mcp.archex = { type = "local", command = ["archex", "mcp"], enabled = true }`. `--dry-run` previews. | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No OpenCode client smoke in this stack. | 2026-06-16 |
| OpenCode `tool.execute.after` plugin (opt-in) | Config-shape tested end-to-end (install, remove, idempotent reinstall, native-vs-MCP dispatch-table assertion, subagent-dispatch reachability confirmed both structurally and via a live session); confirmed against a real installed `opencode-ai` 1.14.33 client | `archex install-client opencode --hooks` writes a standalone plugin file to `.opencode/plugins/archex-hook.ts` (project scope) or `~/.config/opencode/plugins/archex-hook.ts` (user scope, default) — OpenCode auto-loads local plugin files from these directories, so no `opencode.json` entry is written or needed. `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#opencode-toolexecuteafter-plugin-opt-in) for the full contract, the confirmation-spike findings, and the live verification. | N/A — one subprocess per matched tool call, not a warm process | Same receipt/timeout/diagnostics semantics as the Claude Code hook above — this module shells out to the identical `python -m archex.integrations.hook` subprocess, unmodified. | Opt-in, never installed by default; augments only native `grep`/`glob` tool calls, never `read` or any MCP-routed tool. | 2026-07-06 |
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
- Pi and oh-my-pi (omp) MCP server config only supports `--scope user`; their `--hooks`/`--remove-hooks` installers support both `--scope user` and `--scope project`.

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

Section G of the development plan (M20–M23) extends this same `python -m archex.integrations.hook` subprocess contract to other clients with per-client installer shims. M20 (oh-my-pi and Pi), M21 (Codex CLI, diagnostics-only), and M22 (OpenCode) are implemented; Cursor's weaker prompt-level mechanism is a planned exception.

## oh-my-pi (omp) / Pi `tool_result` hook (opt-in)

`archex install-client omp --hooks` / `archex install-client pi --hooks` install the identical shared TypeScript extension module (`archex-hook.ts`) that registers a `pi.on("tool_result", ...)` handler scoped to `grep`/`glob`-equivalent tool calls, augmenting their content by shelling out to the identical `python -m archex.integrations.hook` subprocess documented above — no lookup, ranking, timeout, or diagnostics logic is reimplemented for either client. It is opt-in — plain `archex install-client omp`/`pi` never installs it:

```bash
archex install-client omp --hooks                      # user (default): ~/.omp/agent/extensions/archex-hook.ts
archex install-client omp . --hooks --scope project    # project-local: .omp/extensions/archex-hook.ts
archex install-client pi --hooks                       # user (default): ~/.pi/agent/extensions/archex-hook.ts
archex install-client pi . --hooks --scope project     # project-local: .pi/extensions/archex-hook.ts
archex install-client omp --hooks --dry-run            # preview only, writes nothing (pi identical)
archex install-client omp --remove-hooks               # clean uninstall (pi identical)
```

Unlike the Claude Code hook (a JSON command entry merged into `settings.json`), this installs a standalone `.ts` file discovered by each host's own native extension auto-discovery — oh-my-pi: project `<cwd>/.omp/extensions/*.ts`, user `~/.omp/agent/extensions/*.ts`; Pi: project `.pi/extensions/*.ts`, user `~/.pi/agent/extensions/*.ts` (confirmed by reading the installed `@mariozechner/pi-coding-agent` 0.68.1's own `docs/extensions.md`, closing the `.docs/DEVELOPMENT_PLAN.md` §2 GAP). The Python interpreter invoked (`ARCHEX_PYTHON_COMMAND` baked into the file) is the one active when `--hooks` was run, exactly like the Claude Code hook's `command` field.

**Pi confirmation findings (M20 §2 GAP, closed):** Pi's `pi.on("tool_result", ...)` event contract is structurally identical to oh-my-pi's — same `{ content, details, isError }` partial-patch return shape, same handler-chaining semantics. The one difference is tool naming: Pi has no `glob` tool at all (its built-in tool set is `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`); its glob-equivalent is `find`, which — unlike oh-my-pi's `glob` — already carries its query in a field named `pattern`, identical to Claude Code's own Glob tool. Because the contract matches, **PR-1's shared module is reused byte-for-byte for Pi** (`test_write_hook_install_plan_pi_and_omp_share_identical_module_content`) — no Pi-specific variant exists; only the installer's target-path resolution differs.

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

Confirmed manually against the installed Pi CLI (0.68.1) with both `node`/`tsx` (matching Pi's `jiti`-based TypeScript loading) and `bun`: a `grep` and a `find` `tool_result` event both received appended archex context, and a `read` event was left untouched, using the exact file written by `archex install-client pi --hooks`.

## Codex CLI PreToolUse hook (opt-in, diagnostics-only)

`archex install-client codex --hooks` installs `src/archex/integrations/codex_hook.py` (invoked as `python -m archex.integrations.codex_hook`) as a Codex `PreToolUse` hook. Unlike the Claude Code and oh-my-pi/Pi hooks above, this one never injects context — it is diagnostics-only. It is opt-in — plain `archex install-client codex` never installs it — and it writes to the *same* `config.toml` the MCP registration above writes to, as a separate marker-delimited block:

```bash
archex install-client codex --hooks                     # global: ~/.codex/config.toml
archex install-client codex . --hooks --scope project   # repo-local: .codex/config.toml
archex install-client codex --hooks --dry-run           # preview only, writes nothing
archex install-client codex --remove-hooks              # clean uninstall
```

### Confirmation-spike findings (M21 §2 GAP)

DEVELOPMENT_PLAN.md's §2 GAP asked whether Codex's hook schema supports content/context augmentation the way Claude Code's `additionalContext` does. Read directly against `openai/codex`'s Rust source (`codex-rs/hooks/`, `codex-rs/core/src/tools/`), not secondary docs:

- **Augmentation IS supported.** `PreToolUseHookSpecificOutputWire` (`codex-rs/hooks/src/schema.rs`) has an `additional_context: Option<String>` field that serializes to the wire as `additionalContext` — the same field name Claude Code uses.
- **But there is no Grep/Glob-equivalent tool-call event to scope it to.** Codex's only `PreToolUse` tool names are `Bash` (every shell invocation — `HookToolName::bash()` in `codex-rs/core/src/tools/hook_names.rs`), `apply_patch` (file edits, aliased to `Write`/`Edit`), `spawn_agent`, and MCP tools under their own names. File search and reads both happen through the generic `Bash` tool by shelling out to `grep`/`rg`/`find`/`cat`. There is no tool name that means "this is a search."

Hooking `Bash` unconditionally to inject `additionalContext` would intercept *every* shell command Codex runs, including destructive ones — a materially broader and riskier surface than the Grep/Glob-only pattern the other hooks use, and it would fail "matches the Grep/Glob-equivalent tool only" on its face since no such tool exists. This hook therefore ships the diagnostics-only fallback: it detects `Bash` invocations shaped like a search command and logs what archex would have surfaced, but never mutates or blocks the tool call.

### Installed config shape

The block is appended to `config.toml` between marker comments so a re-run or `--remove-hooks` can find and replace exactly this block without disturbing any other section (including the `[mcp_servers.archex]` registration above):

```toml
# archex:codex-hook start
[[hooks.PreToolUse]]
matcher = "^Bash$"

[[hooks.PreToolUse.hooks]]
type = "command"
command = "/path/to/venv/bin/python -m archex.integrations.codex_hook"
timeout = 1
# archex:codex-hook end
```

Contract:

- **Never returns augmented output.** No code path ever sets `additionalContext`, `permissionDecision`, `updatedInput`, or any other `hookSpecificOutput` field — every invocation writes `{}` to stdout (Codex defaults every schema field, so an empty object means "no decision").
- **Matches `Bash` only — there is no `Read` or Grep/Glob-equivalent hook in Codex to accidentally match instead.** `tests/cli/test_install_client_hooks.py`'s `test_codex_hook_toml_block_matcher_never_reaches_read` asserts the installed matcher is exactly `^Bash$` and never matches `Read`.
- **Exits 0 on every path.** A missing/stale index, a malformed payload, a non-search `Bash` command, a timeout, or any internal error all degrade to no diagnostic (or a diagnostic-only log line) — never a blocked or errored tool call.
- **Reuses `archex.integrations.hook`'s engine in-process.** `lookup_with_timeout`/`log_diagnostic` are called directly (no second subprocess spawned), so freshness/timeout/diagnostics semantics exactly match the Claude Code hook, appending to the same `~/.archex/hook-diagnostics.log` (override with `ARCHEX_HOOK_DIAGNOSTICS_LOG`).
- **Codex's own hook-level `timeout` field is whole seconds only** (`HookHandlerConfig::Command.timeout_sec: Option<u64>`, confirmed via `codex-rs/config/src/hook_config.rs`) — the installer sets it to `1` as an outer backstop; the real ~500ms budget is enforced internally by the reused `archex.integrations.hook` engine.
- **Non-destructive install/uninstall.** Any other `config.toml` content — including the `[mcp_servers.archex]` MCP registration — is left untouched by both `--hooks` and `--remove-hooks`. Re-running `--hooks` is an idempotent no-op once installed, even across a venv move (the block converges on the active `sys.executable` each time).
- **One-time hash-based trust review.** Codex requires reviewing and trusting a hook by its content hash before it runs (`/hooks` in the Codex TUI); a hook that changes after trust (e.g. a subsequent archex upgrade rewriting the `command` path) needs re-trusting.

Manual verification (bypassing Codex entirely — this is exactly what the hook receives on stdin for a `Bash` tool call):

```bash
echo '{"tool_name":"Bash","tool_input":{"command":"grep -rn compute_delta ."},"cwd":"'"$PWD"'"}' \
  | python -m archex.integrations.codex_hook
```

Always exits 0 with `{}` on stdout. A repo with a fresh index and a search-shaped command appends a `codex_augmentation_withheld` diagnostic line to the log describing the match, instead of injecting it; a repo with no index, a stale index, or a non-search command produces no diagnostic (or the same `index_not_fresh`/`status_error` diagnostic the Claude Code hook logs) and no injected context either way.

## OpenCode `tool.execute.after` plugin (opt-in)

`archex install-client opencode --hooks` installs a standalone TypeScript plugin file that registers a `tool.execute.after` handler scoped to OpenCode's native `grep`/`glob` tool calls, augmenting their output by shelling out to the identical `python -m archex.integrations.hook` subprocess documented above — no lookup, ranking, timeout, or diagnostics logic is reimplemented. It is opt-in — plain `archex install-client opencode` never installs it:

```bash
archex install-client opencode --hooks                      # user (default): ~/.config/opencode/plugins/archex-hook.ts
archex install-client opencode . --hooks --scope project    # project-local: .opencode/plugins/archex-hook.ts
archex install-client opencode --hooks --dry-run            # preview only, writes nothing
archex install-client opencode --remove-hooks                # clean uninstall
```

Unlike the MCP config row above (an `opencode.json` entry), this installs a standalone `.ts` file — OpenCode's own docs state that files in `.opencode/plugins/` (project) and `~/.config/opencode/plugins/` (global) "are automatically loaded at startup," so no `opencode.json` change is required or written.

### Why this isn't the oh-my-pi/Pi module verbatim

OpenCode's `tool.execute.after` hook contract is structurally different from oh-my-pi/Pi's `tool_result`: its handler signature is `(input, output) => Promise<void>` — it mutates the `output.output` string **in place** rather than returning a content patch. `input` carries `{tool, sessionID, callID, args}`. Both of OpenCode's native `grep` and `glob` tools already carry their query pattern in a field named `pattern` (confirmed against the installed `opencode-ai` 1.14.33's own tool definitions), so — unlike oh-my-pi's `glob`, which needs a `path`→`pattern` remap — the plugin's translation onto the subprocess's existing Grep/Glob contract is an identity mapping. Its dispatch table (`ARCHEX_AUGMENTED_TOOLS`) is keyed directly on OpenCode's own tool ids: `{"grep": "Grep", "glob": "Glob"}`.

### Confirmation spike: OpenCode-side reliability gaps (M22 §2 GAP)

DEVELOPMENT_PLAN.md flagged two OpenCode-side reliability gaps this milestone's own tests had to check rather than assume. Both were resolved by reading `opencode-ai` 1.14.33's own tool-resolution source directly (`packages/opencode/src/session/prompt.ts`, the version installed during development), not secondary documentation:

1. **`tool.execute.after`-on-MCP-tools output inconsistency — confirmed real, and moot for this plugin.** The MCP tool-execution branch does trigger `tool.execute.after`, but passes the tool's raw MCP `CallToolResult` (`{content, metadata}`) as `output`, not the `{title, output, metadata}` shape the type declares. The text actually sent to the model is rebuilt from `result.content` *after* the hook call, discarding any `output.output` mutation — so augmenting an MCP tool through this hook would silently do nothing even if a plugin tried. This is moot here: `ARCHEX_AUGMENTED_TOOLS` never contains an MCP-shaped id, and OpenCode registers every MCP tool under a mandatory `{server}_{tool}` prefix (confirmed in the same source), so an exact `"grep"`/`"glob"` collision with an MCP tool id is structurally impossible, not merely unlikely.
2. **`tool.execute.before`'s documented subagent-bypass bug does not extend to the `.after` hook this plugin uses.** `TaskTool.execute`'s `ops.prompt({sessionID: nextSession.id, ...})` is bound to the exact same `prompt()` closure that processes a top-level turn's own tool resolution, which wraps every native tool's `execute` with the identical `tool.execute.before`/`.after` triggers regardless of which session is being processed. Confirmed both from source and live (see below) — a subagent-issued `grep`/`glob` call is augmented identically to a top-level one in the version this was verified against.

### Manual verification (live, real `opencode-ai` 1.14.33, free model — not part of the pytest suite; no Node/Bun in CI)

Installed the plugin on a fixture repo and ran real `opencode run` sessions:

- A native `grep` call for a real symbol: the exported session's tool part (`opencode export <sessionID>`) shows the archex receipt block appended directly to `state.output`, after the grep results.
- A native `glob` call: same — archex context appended in place.
- A `read` call: the exported `state.output` is the exact raw file content, byte-for-byte — no archex text anywhere.
- **Subagent dispatch:** instructed the top-level agent to launch a `subagent_type: general` Task and have *that subagent* call `grep` itself. Exporting the **subagent's own child session** directly (not the parent's relayed summary) showed the archex receipt block appended to the subagent's own `grep` tool part output — direct, first-party confirmation that `tool.execute.after` reaches subagent-issued native tool calls.
- A missing/stale index (a fresh, never-`archex init`'d repo): the plugin invocation degrades to unmodified output plus an `index_not_fresh` diagnostics log line.
- A hung subprocess (replaced with a 30s sleep for the test): the ~500ms timeout guard fired, `SIGKILL`ed the process, left output unmodified, and logged a `ts_timeout` diagnostic — measured at ~526ms end-to-end.
- Malformed subprocess stdout (non-JSON): output stays unmodified, no throw.

Manual verification (bypassing OpenCode entirely — same fixture payload the sections above use, since the subprocess contract is unmodified):

```bash
echo '{"tool_name":"Grep","tool_input":{"pattern":"compute_delta"}}' \
  | python -m archex.integrations.hook
```

### Contract, mirroring the oh-my-pi/Pi hook

- **Never intercepts `read`.** The plugin's only tool-name dispatch is `ARCHEX_AUGMENTED_TOOLS`, keyed exactly on `{"grep", "glob"}` — `read` is absent by construction, and an MCP-routed tool id can never collide with it (see above). `tests/cli/test_install_client_hooks.py`'s `test_opencode_ts_hook_module_native_vs_mcp_tool_routing` and `test_cli_hooks_opencode_installed_file_never_targets_read_or_mcp_tool_ids` assert this structurally, against both the in-memory plan and the file the CLI actually writes.
- **Exits without throwing on every path.** A missing/stale index, a subprocess spawn failure, a timeout, or a malformed subprocess response all degrade to leaving `output` untouched. Failures append a JSON line to the same diagnostics log the Python subprocess and the oh-my-pi/Pi module use (`~/.archex/hook-diagnostics.log` by default, override with `ARCHEX_HOOK_DIAGNOSTICS_LOG`) — never surfaced to the agent.
- **Hard ~500ms lookup timeout**, matching `DEFAULT_HOOK_TIMEOUT_SECONDS`: the plugin's own `setTimeout` guard `SIGKILL`s a subprocess still running past the budget, independent of whatever timeout the subprocess enforces on itself.
- **No field-name translation needed.** Both `grep` and `glob` already carry their query in a field named `pattern`, so the plugin's mapping onto the subprocess's `{"tool_name": "Grep"|"Glob", "tool_input": {"pattern": ...}}` contract is an identity mapping on the field, keyed only on the tool id.
- **Subagent-reachable.** The handler registers unconditionally with no `sessionID`/`callID`-based gating, and both source inspection and a live nested-subagent run confirm `tool.execute.after` fires for subagent-issued calls in the verified version — see the confirmation spike above.
