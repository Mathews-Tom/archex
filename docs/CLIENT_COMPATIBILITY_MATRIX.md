# Client Compatibility Matrix

Last updated: 2026-09-10

This matrix separates config-shape verification from actual client smoke tests. `archex install-client <client>` writes the config by default (global/user scope; a SOURCE path or `--scope project` installs repo-local). Add `--dry-run` to preview the exact target and config without writing.

## Matrix

| Client / path | Tested status | Setup command / config | Watch support | Freshness semantics | Known limitations | Last verified |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Code MCP stdio | Config-path tested; client smoke unverified | `archex install-client claude-code` writes `~/.claude.json` (global); `archex install-client claude-code . --scope project` writes `.mcp.json` with `mcpServers.archex.command = "archex"` and `args = ["mcp"]`. `--dry-run` previews either. | Yes — `archex mcp --watch --watch-path .` | Inline query refresh by default; `--no-refresh` leaves freshness `unknown`; watch keeps a warm process subscribed to file events. | This stack did not run a live Claude Code UI smoke. Skill and MCP are separate rows. | 2026-06-16 |
| Claude Code PostToolUse search-annotation hook (opt-in) | Config-shape tested end-to-end (install, remove, idempotent reinstall, replacement of the retired `PreToolUse` entry, foreign handlers and the post-edit and `SessionStart` entries surviving in both install orders). The adapter is run as a subprocess against a real index (output shape, unknown payload fields, three calls in one session, malformed payloads, stale/dirty index, non-search `Bash`, over-budget exit) **and live-verified in a real Claude Code 2.1.285 session** against a local Anthropic-Messages stub (no hosted call): the annotation appeared in the transcript's `hook_additional_context` attachment and beside the tool result in the next captured request body | `archex install-client claude-code --hooks` writes `~/.claude/settings.json` (global) or `.claude/settings.json` (project) — a different file from the MCP config above — adding one `PostToolUse` handler on `Bash\|Grep\|Glob` and removing the retired archex `PreToolUse` handler. `--dry-run` previews, `--remove-hooks` removes both. See [below](#claude-code-posttooluse-search-annotation-hook-opt-in) for the full contract. | N/A — one in-process-guarded subprocess per matched tool call, not a warm process | `archex annotate` maps each hit to its enclosing indexed code unit and returns one fact line per unit as `additionalContext`, stamped with `index_revision=`; only a `fresh` index annotates, so stale, dirty, missing, or unrecognised output adds nothing plus a diagnostics log line. | Opt-in, never installed by default; appends only (`additionalContext`, never `updatedToolOutput`), and the original result reaches the model unchanged. Because Claude Code on macOS/Linux searches through `Bash`, the matcher spawns the hook for every shell command; a command that is not a search exits before the index and tokenizer are imported (p50 36 ms) and writes no ledger line. A search costs p50 365 ms / p95 389 ms on a 24k-chunk index, and the first call in a cold process can exceed the 500 ms hard budget, in which case nothing is added. One ledger line per search call (`host: claude-code`, `toolCallId` = `tool_use_id`) at `~/.archex/annotation-ledger.jsonl`; diagnostics at `~/.archex/hook-diagnostics.log`. | 2026-09-30 |
| Claude Code SessionStart session primer (opt-in) | Config-shape tested end-to-end (install, remove, idempotent reinstall, preserves unrelated hook groups) and live hook-process proof for both `startup` and `resume`; no live Claude Code UI smoke | `archex install-client claude-code --session-primer` writes `~/.claude/settings.json` (global) or `.claude/settings.json` (project), adding only an owned `SessionStart` handler. `--dry-run` previews; `--remove-session-primer` uninstalls. See [below](#claude-code-sessionstart-session-primer-opt-in). | N/A — one bounded subprocess per session start/resume | Injects only a fresh-index, explicit, receipt-bearing session primer; stale/missing state, malformed payloads, errors, and timeout emit no context and exit 0. | Opt-in. It does not capture records, index, inject opaque transcript state, or run for any SessionStart source other than `startup` and `resume`. | 2026-08-18 |
| Claude Code PostToolUse post-edit impact hook (opt-in) | Config-shape tested end-to-end (install, remove, idempotent reinstall, coexistence with the search-annotation and SessionStart surfaces) **and live-verified in a real Claude Code session**: a `claude -p` run that edited a file produced a `hook_additional_context` attachment with `hookName: PostToolUse:Edit` carrying the archex block, recorded in the session transcript | `archex install-client claude-code --post-edit-hooks` writes `~/.claude/settings.json` (global) or `.claude/settings.json` (project), adding only an owned `PostToolUse` handler with matcher `Edit|Write`. `--dry-run` previews; `--remove-post-edit-hooks` uninstalls. See [below](#post-edit-impact-hooks-opt-in). | N/A — one bounded subprocess per successful edit | Emits only from a generation whose recorded working-tree signature still matches the tree on disk; every block carries `generation=`/`index_revision=`/`generated_at=`/`confidence=`. Stale, unverifiable, or timed-out cycles emit nothing and leave the edit state dirty for the next event. | Opt-in. `PostToolUse` fires after the tool ran, so the hook cannot block or fail an edit. Risk is file-scoped (import/reference edges), never call-graph. Bounded by `ARCHEX_POST_EDIT_TIMEOUT_SECONDS` (default 8s). | 2026-09-10 |
| Claude Code skill command | Existing skill path tested in-repo; client smoke unverified | Use `skills/archex/` and the `/archex` command flow. No config file is written by `install-client`; this is command-only onboarding. | Indirect — skill can target a warm MCP server. | Same as MCP/query/scout paths underneath. | Skill setup remains repo-local documentation, not a writable client config target. | 2026-06-16 |
| CLI-only query/scout | Tested | No client config required. Run `archex doctor`, `archex scout`, `archex query`. | N/A | Query checks freshness inline unless `--no-refresh`; scout inherits query freshness in its receipt. | Not an MCP client. | 2026-06-16 |
| Generic MCP stdio client | Unverified | Use a JSON config shaped like `{ "mcpServers": { "archex": { "command": "archex", "args": ["mcp"] }}}`. `archex install-client claude-code --dry-run` prints a compatible snippet. | Client-dependent | Same server-side freshness semantics as Claude Code / Cursor. | No live generic-client smoke in this stack. | 2026-06-16 |
| Codex headless | Unverified | `archex install-client codex` writes `~/.codex/config.toml` (global); `archex install-client codex . --scope project` writes `.codex/config.toml`, appending `[mcp_servers.archex]`, `command = "archex"`, `args = ["mcp"]` without overwriting existing sections. `--dry-run` previews. | Yes — via `archex mcp --watch --watch-path .` after Codex launches the server. | Inline query refresh by default; warm watch is server-side, not Codex-specific. | Config shape verified against OpenAI Codex MCP docs; no Codex client smoke in this stack. | 2026-06-16 |
| Codex CLI PostToolUse search-annotation hook (opt-in) | Config-shape tested end-to-end (install, remove, idempotent reinstall, in-place replacement of the retired `PreToolUse` block, coexistence with the post-edit block and the MCP registration in every install order). The adapter is run as a subprocess against a real index (output shape, unknown payload fields, three calls in one session, `workdir` recovered from the session rollout, malformed payloads, stale/dirty index, non-search commands, over-budget exit) **and live-verified in a real Codex CLI 0.153.4 session** against a local Responses-API stub (no hosted call, isolated `CODEX_HOME`): the annotation appeared as a `hooks.additional_context` developer message in the session rollout and in the next captured request body | `archex install-client codex --hooks` appends a marker-delimited `[[hooks.PostToolUse]]` block (matcher `^Bash$`) to the same `config.toml` the MCP registration above writes to (`~/.codex/config.toml` global, `.codex/config.toml` project) and replaces the retired diagnostics-only `PreToolUse` block. `--dry-run` previews, `--remove-hooks` removes both. See [below](#codex-cli-posttooluse-search-annotation-hook-opt-in) for the full contract. | N/A — one in-process-guarded subprocess per matched tool call, not a warm process | `archex annotate` maps each hit to its enclosing indexed code unit and returns one fact line per unit as `additionalContext`, stamped with `index_revision=`; only a `fresh` index annotates, so stale, dirty, missing, or unrecognised output adds nothing plus a diagnostics log line. | Opt-in, never installed by default; appends only (`additionalContext`, recorded by Codex as a developer message beside the result), and the original output reaches the model unchanged. Codex has one shell tool (`Bash`), so the matcher spawns the hook for every shell command; a command that is not a search exits before the index and tokenizer are imported (p50 42 ms) and writes no ledger line. A search costs p50 396 ms / p95 444 ms on a 24k-chunk index (first call 478 ms), and a cold first call can exceed the 500 ms hard budget, in which case nothing is added. Codex 0.153.4 runs a config-file hook only once trusted (`trusted_hash` in `hooks.state`; the TUI's `/hooks` records it), so a freshly installed hook does nothing until reviewed. `tool_input` carries no `workdir`; it is read back from the rollout, and without a rollout a search run from a subdirectory can resolve to a same-named file under `cwd`. One ledger line per search call (`host: codex`, `toolCallId` = `tool_use_id`) at `~/.archex/annotation-ledger.jsonl`; diagnostics at `~/.archex/hook-diagnostics.log`. | 2026-09-30 |
| Codex CLI PostToolUse post-edit impact hook (opt-in) | Config-shape tested end-to-end (install, remove, idempotent reinstall, coexistence with the search-annotation block) and real installed-hook subprocess smoke against a live index; no live Codex CLI session smoke | `archex install-client codex --post-edit-hooks` appends a marker-delimited `[[hooks.PostToolUse]]` block (matcher `^apply_patch$`) to the same `config.toml` the MCP registration and search hook use, under its own `# archex:codex-post-edit-hook` markers. `--dry-run` previews; `--remove-post-edit-hooks` uninstalls. | N/A — one bounded subprocess per successful patch | Same freshness contract as the Claude Code row. | Opt-in. Returns `hookSpecificOutput.additionalContext`: Codex's `PostToolUseHookSpecificOutputWire` supports it, and `apply_patch`'s `tool_input.command` carries the V4A patch text whose `*** Add/Update/Delete File:` and `*** Move to:` markers give the edited paths. Not exercised through a live Codex session — the local OAuth refresh token was expired at verification time. | 2026-09-10 |
| Pi MCP stdio | Config shape verified; client smoke unverified | `archex install-client pi` writes `~/.pi/agent/mcp.json` with a stdio `mcpServers.archex` entry (`--dry-run` previews). User scope only. | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No Pi client smoke in this stack. | 2026-06-16 |
| Pi `tool_result` hook (opt-in) | Same generated module as the oh-my-pi row below (byte-identical; install/remove tested). Executed under Bun against a real index, not in a live Pi session | `archex install-client pi --hooks` writes the identical TypeScript extension module to `.pi/extensions/archex-hook.ts` (project scope) or `~/.pi/agent/extensions/archex-hook.ts` (user scope, default) — confirmed against the installed `@mariozechner/pi-coding-agent` 0.68.1. `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#oh-my-pi-omp--pi-tool_result-hook-opt-in) for the full contract. | N/A — one subprocess per search-tool result, not a warm process | Annotates the agent's own `grep`/`find`/bash-search results via `archex annotate`; stale or dirty index adds nothing. | Opt-in, never installed by default. Pi's glob-equivalent is `find`; its output format was not observed in a local Pi transcript, so headerless `find` output is read as one path per line. | 2026-09-29 |
| Pi post-edit `tool_result` hook (opt-in) | Config-shape tested end-to-end and live-executed: the installed module was loaded under Bun with a real `tool_result` dispatch, spawned the Python subprocess, and returned a content patch carrying `client=pi` | `archex install-client pi --post-edit-hooks` writes `.pi/extensions/archex-post-edit-hook.ts` (project) or `~/.pi/agent/extensions/archex-post-edit-hook.ts` (user) — a different file from the search hook's `archex-hook.ts`, so both install and uninstall independently. | N/A — one bounded subprocess per successful edit | Same freshness contract as the Claude Code row. | Opt-in. Fires only for `edit`/`write` with `isError !== true`. `tool_result` is post-execution and cannot block the edit. Verified against `@mariozechner/pi-coding-agent` 0.68.1, whose `edit`/`write` schemas both name the target `path`. | 2026-09-10 |
| oh-my-pi (omp) MCP stdio | Config shape verified; client smoke unverified | `archex install-client omp` writes `~/.omp/agent/mcp.json` (user scope only) with `mcpServers.archex.command = "archex"`, `args = ["mcp"]`, plus the oh-my-pi `$schema` (`--dry-run` previews). | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No oh-my-pi client smoke in this stack. Discovery-gated harness — tools must be activated before use (see below). | 2026-06-20 |
| oh-my-pi (omp) `tool_result` hook (opt-in) | Install, remove, and idempotent reinstall tested beside a foreign extension; module executed under Bun against a real index; **live omp 18.4.2 session** (local stub provider, no hosted call) showed the appended block in the session JSONL and in the next provider request | `archex install-client omp --hooks` writes a TypeScript extension module to `.omp/extensions/archex-hook.ts` (project scope) or `~/.omp/agent/extensions/archex-hook.ts` (user scope, default) — a different file/mechanism from the MCP config above. `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#oh-my-pi-omp--pi-tool_result-hook-opt-in) for the full contract. | N/A — one subprocess per search-tool result, not a warm process | Annotates the agent's own `grep`/`glob`/bash-search results via `archex annotate`; stale or dirty index adds nothing; per-call ledger at `~/.archex/annotation-ledger.jsonl`. | Opt-in, never installed by default; never touches `read`. Supports both project and user scope (unlike the MCP config row above, which is user-scope only). | 2026-09-29 |
| oh-my-pi (omp) post-edit `tool_result` hook (opt-in) | Config-shape tested end-to-end and live-executed: the installed module was loaded under Bun with a real `tool_result` dispatch, spawned the Python subprocess, and returned a content patch carrying `client=omp` | `archex install-client omp --post-edit-hooks` writes `.omp/extensions/archex-post-edit-hook.ts` (project) or `~/.omp/agent/extensions/archex-post-edit-hook.ts` (user). | N/A — one bounded subprocess per successful edit | Same freshness contract as the Claude Code row. | Opt-in. Fires only for `edit`/`write` with `isError !== true`. Verified against `@oh-my-pi/pi-coding-agent` 18.1.16, whose `BUILTIN_TOOL_NAMES` lists `edit` and `write` and whose schemas both name the target `path`. | 2026-09-10 |
| OpenCode | Config shape verified; client smoke unverified | `archex install-client opencode` writes `~/.config/opencode/opencode.json` (global); `archex install-client opencode . --scope project` writes `opencode.json`, setting `mcp.archex = { type = "local", command = ["archex", "mcp"], enabled = true }`. `--dry-run` previews. | Client-dependent; server supports `--watch`. | Same server-side freshness semantics as other stdio clients. | No OpenCode client smoke in this stack. | 2026-06-16 |
| OpenCode `tool.execute.after` plugin (opt-in) | Install, remove, and idempotent reinstall tested beside a foreign plugin; plugin executed under Bun against a real index (byte-identical prefix, three calls and three ledger lines, malformed input, dirty index, budget overrun kills the subprocess); **live OpenCode 1.14.33 session** (local chat-completions stub via `@ai-sdk/openai-compatible`, no hosted call) showed the appended block in the next provider request and in the session's SQLite `part` rows | `archex install-client opencode --hooks` writes a standalone plugin file to `.opencode/plugins/archex-hook.ts` (project scope) or `~/.config/opencode/plugins/archex-hook.ts` (user scope, default) — OpenCode auto-loads local plugin files from these directories, so no `opencode.json` entry is written or needed. `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#opencode-toolexecuteafter-plugin-opt-in) for the full contract and the verification. | N/A — one subprocess per matched tool call, not a warm process | Annotates the native `grep`, `glob`, and `bash` search results through the shared annotation core: fresh index only, about 300 tokens per call, the original text a byte-for-byte prefix. Stale, dirty, missing, or unprovenanced indexes add nothing. | Opt-in, never installed by default; never touches `read` or any MCP-routed tool (OpenCode discards `output.output` edits on MCP results). The first call after a long idle can exceed the 0.5 s budget and then adds nothing. | 2026-09-30 |
| OpenCode post-edit `tool.execute.after` plugin (opt-in) | Config-shape tested end-to-end and live-executed: the installed plugin was loaded under Bun against a real `tool.execute.after` invocation, spawned the Python subprocess, and appended a block carrying `client=opencode` to `output.output` | `archex install-client opencode --post-edit-hooks` writes `.opencode/plugins/archex-post-edit-hook.ts` (project) or `~/.config/opencode/plugins/archex-post-edit-hook.ts` (user). OpenCode auto-loads plugin files from those directories, so no `opencode.json` entry is written. | N/A — one bounded subprocess per successful edit | Same freshness contract as the Claude Code row. | Opt-in. Fires only for the `edit`/`write` tools. The event exposes no error flag, so a failed edit is not distinguishable at the boundary; the Python side revalidates every recorded path against the real working tree, so a phantom path produces no output. Verified against `opencode-ai` 1.14.33; `edit`/`write` name the target `filePath`. | 2026-09-10 |
| Cursor | Config shape verified; client smoke unverified | `archex install-client cursor` writes `~/.cursor/mcp.json` (global); `archex install-client cursor . --scope project` writes `.cursor/mcp.json` with `mcpServers.archex.command = "archex"`, `args = ["mcp"]`. `--dry-run` previews. | Yes — with a warm `archex mcp --watch --watch-path .` process. | Inline query refresh by default; watch keeps a warm process subscribed to file events. | No Cursor UI smoke in this stack. | 2026-06-16 |
| Cursor `beforeSubmitPrompt` hook (opt-in, diagnostics-only, prompt-level) | Config-shape tested end-to-end (install, remove, idempotent reinstall, preserves unrelated `hooks.json` content, `beforeReadFile`-never-wired assertion); no live Cursor UI smoke | `archex install-client cursor --hooks` writes `~/.cursor/hooks.json` (global) or `.cursor/hooks.json` (project) — a different file from the MCP config above. `--dry-run` previews, `--remove-hooks` uninstalls. See [below](#cursor-beforesubmitprompt-hook-opt-in-diagnostics-only) for the full contract and the confirmation-spike findings. | N/A — one subprocess per submitted prompt, not a warm process | Diagnostics-only — no injected context, ever (see limitations); a missing/stale index degrades to no diagnostic, or the same `index_not_fresh`/`status_error` diagnostic the other hooks log. | Prompt-level, not per-tool-call: fires on every submitted prompt regardless of whether a lookup is relevant. Cursor's `beforeSubmitPrompt` output schema has no context-injection field at all (confirmed against Cursor's own docs), so this is diagnostics-only, unlike the augmenting Claude Code/omp/Pi/OpenCode hooks above. | 2026-07-06 |
| Cursor post-edit impact hook | **Not supported** — `archex install-client cursor --post-edit-hooks` fails with the reason rather than writing anything | N/A | N/A | N/A | `afterFileEdit` is Cursor's only event carrying an edited file path (`{file_path, edits}`) and its documented schema has **no output fields at all**, so it cannot return impact to the agent. `postToolUse` does document an `additional_context` output and lists `Write` among its matchers, but documents no path field for that tool's input, so edited paths cannot be extracted from the one event that could surface text. Archex ships no adapter it cannot show to work. | 2026-09-10 |
| Claude Code status line (opt-in) | Config-shape tested end-to-end (install, remove, idempotent reinstall, foreign-statusLine refusal, coexistence with all three hook surfaces) and the installed renderer executed for real against every snapshot state under an empty `PATH` | `archex install-client claude-code --statusline` writes `~/.claude/settings.json` (global) or `.claude/settings.json` (project) plus the renderer script `archex-statusline.sh` beside it. `--dry-run` previews; `--remove-statusline` uninstalls. See [below](#persistent-status-surfaces-opt-in). | N/A — reports whether a watch refresh was observed; starts no watcher | Renders the cached snapshot only: `fresh`, `dirty`, `pending`, `stale`, `missing`, `corrupt`, and `unsupported` are distinct. Never opens the index, runs a parser, or starts an archex process on repaint. | Opt-in. `statusLine` is a scalar settings key, so a status line archex did not install is refused rather than replaced. The `stale` label needs a shell clock (`$EPOCHSECONDS`, bash 5+/zsh); under macOS `/bin/sh` (bash 3.2) the renderer reports the measured state without an age instead of forking `date`. Never reports token savings. | 2026-09-10 |
| oh-my-pi (omp) / Pi status extension (opt-in) | Config-shape tested end-to-end (per-host placement, byte-identical modules, idempotent reinstall, independence from the search and post-edit modules in the same directory) **and executed under Bun**: the rendered module was loaded in a real runtime, all three events dispatched, and every snapshot state rendered through a captured `ctx.ui.setStatus` | `archex install-client omp --statusline` writes `.omp/extensions/archex-status.ts` (project) or `~/.omp/agent/extensions/archex-status.ts` (user); `archex install-client pi --statusline` writes `.pi/extensions/archex-status.ts` or `~/.pi/agent/extensions/archex-status.ts`. `--dry-run` previews; `--remove-statusline` uninstalls. | N/A — reports whether a watch refresh was observed; starts no watcher | Same cached-snapshot contract as the Claude Code row, including `stale`, which this renderer can always compute because it has a real clock. | Opt-in. Refreshes on `turn_start`, `tool_result`, and `turn_end`; a host with `hasUI` false (print/RPC mode) receives no call. Verified against `@oh-my-pi/pi-coding-agent` 18.1.16 and `@mariozechner/pi-coding-agent` 0.68.1, whose `ExtensionUIContext` both declare `setStatus(key, text)`. No live TUI session was driven: the status refresh fires on turn events, and running one would require a hosted model call. | 2026-09-10 |
| Status surfaces for codex, cursor, and opencode | **Not supported** — `archex install-client <client> --statusline` fails with the reason rather than writing anything | N/A | N/A | N/A | The Codex CLI exposes no status-line configuration and no persistently rendered hook output field. Cursor's configuration surface is hooks only, with no status or footer API. OpenCode's plugin surface exposes tool and chat hooks plus observable TUI events whose only status-shaped member is the transient `tui.toast.show`; a toast disappears, so it cannot carry a persistent freshness indicator. Use `archex status --cached`. | 2026-09-10 |
| Dockerized MCP server | Server path tested; client smoke unverified | Run `docker run -d --name archex-mcp -v "$PWD:/workspace" -w /workspace ghcr.io/mathews-tom/archex:slim sleep infinity` then point the client to `docker exec -i archex-mcp archex mcp`. | Yes — run the MCP process with `--watch`. | Same server-side freshness semantics as stdio. | Client-specific Docker registration varies; use the same client config shapes above, but replace the command with `docker` / `exec`. | 2026-06-16 |

## First-party bootstrap command

Run bare `install-client` to automatically discover config paths for all supported clients and interactively select which to install:

```bash
archex install-client
```

Or configure a specific client directly (installs write by default; add `--dry-run` to preview the exact target and config first). The default scope is global (user); pass a SOURCE path or `--scope project` for a repo-local install:

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
## Registration vs runtime startability

A valid MCP client registration config is separate from whether `archex mcp` can actually start. `install-client` checks the MCP runtime health before writing any client config. If the runtime is damaged or missing, `install-client` fails fast with an error message detailing how to remediate the installation. You can bypass this check with `--allow-missing-mcp` if you are packaging or know what you are doing, but writing known-broken client config is blocked by default.

## Registration is not surfacing is not invocation

Registering an MCP server is necessary but not sufficient for an agent to actually use archex. Three distinct steps must all happen:

- **Registration** — `install-client` writes the MCP server entry into the client config (this command).
- **Surfacing** — the client/harness must expose the registered tools to the agent. Harnesses with on-demand tool discovery (e.g. oh-my-pi / Pi) treat a registered server's tools as *discoverable* but keep them out of the default tool set; the agent must activate them before the first call.
- **Invocation** — the agent must choose to call `query_repo` / `scout_repo` / `analyze_repo` instead of reading files by hand.

### Retrieval-gated disclosure (R5)

`archex mcp` advertises only the two retrieval entry points — `context` and `query_repo` — until the client calls one of them, then advertises everything and sends `notifications/tools/list_changed`. That cuts the fixed per-turn schema cost from **4 192 tokens to 765**, an 81.8% reduction, measurable with a bare `archex mcp-schema-size`, which reports the gated cost a fresh session is actually charged alongside the expanded cost the same session pays after it retrieves. `--no-disclosure` reports the ungated surface.

**What makes this safe is the notification, not the dispatch.** MCP tools are model-controlled: a model only calls what it was shown. So for the ordinary path, the thing that puts the other 18 tools back in front of the model is `tools/list_changed` — which archex is entitled to have honoured because the gated server declares the `listChanged` capability at initialization.

Dispatch-by-name surviving a closed gate is a *fallback*, not the mechanism: `call_tool` dispatches by name whatever `list_tools()` returned, which rescues **hardcoded** callers — a script, or an agent file that names tools directly, as archex's own guidance block does. It does not help a model that was never shown the tool.

| Client behaviour | What to do |
| --- | --- |
| Honours `notifications/tools/list_changed` | Nothing. The default is correct and cheapest. |
| Ignores the notification, calls tools by hardcoded name | Nothing. Those calls still dispatch. |
| Ignores the notification and builds its tool list only from `list_tools()` | `--no-disclosure`. Its model would otherwise never see the other 18 tools. |
| Needs every tool visible in `list_tools()` before it will call anything | `archex install-client <client> --no-disclosure`, `archex setup --no-disclosure`, or `archex mcp --no-disclosure` by hand. Pays the full per-turn cost, sees everything immediately. |

One behavioural consequence worth knowing: the MCP client SDK validates a call's arguments against the schema it has cached from `list_tools()`, so through a closed gate a call to a not-yet-advertised tool is **not** validated client-side and reaches the server instead of failing fast. The window is one round trip for a client that honours the notification, and the whole session for one that does not — another reason for the third row above.

`--tools` does **not** disable the gate. It bounds what is advertised *once the gate opens*, so `archex mcp --tools all` still starts at the minimal set. `--no-disclosure` is the only opt-out. That is the opposite of the natural guess, which is why it is stated here and in `archex mcp --help`.

Configs written without `--no-disclosure` are byte-identical to the ones archex wrote before R5, so no existing install churns.

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

## Claude Code PostToolUse search-annotation hook (opt-in)

`archex install-client claude-code --hooks` installs a Claude Code `PostToolUse` hook (`src/archex/integrations/claude_code_annotate_hook.py`, invoked as `python -m archex.integrations.claude_code_annotate_hook`) on the tools `Bash`, `Grep` and `Glob`. When one of them ran a search, the hook maps each hit to the indexed code unit that contains it (`archex annotate`) and returns one fact line per unit as `hookSpecificOutput.additionalContext`; Claude Code puts that text in a system reminder next to the tool result, so the model sees the original result unchanged plus the annotation. It never uses `updatedToolOutput`, so it cannot replace or drop the result. It is opt-in — plain `archex install-client claude-code` never installs it — and it writes to a different file than MCP server registration:

```bash
archex install-client claude-code --hooks                     # global: ~/.claude/settings.json
archex install-client claude-code . --hooks --scope project   # repo-local: .claude/settings.json
archex install-client claude-code --hooks --dry-run           # preview only, writes nothing
archex install-client claude-code --remove-hooks              # clean uninstall
```

`--hooks` also removes the retired archex `PreToolUse` entry (`python -m archex.integrations.hook`, matcher `Glob|Grep`) that archex 0.33 and earlier wrote, and `--remove-hooks` removes both, so an upgrade needs one `--hooks` run and leaves one archex search hook behind.

Installed config shape (the `command` path is the Python interpreter active when `--hooks` was run, so the hook always runs in the same environment archex was installed into):

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash|Grep|Glob",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/venv/bin/python",
            "args": ["-m", "archex.integrations.claude_code_annotate_hook"]
          }
        ]
      }
    ]
  }
}
```

Result text the hook reads from the payload (shapes recorded at a live Claude Code 2.1.285 hook): `Bash` — `tool_response.stdout`; `Grep` — `tool_response.content` when `tool_input.output_mode` is `content`, otherwise `tool_response.filenames` one per line (`count` mode adds nothing); `Glob` — `tool_response.filenames` one per line. On macOS and Linux Claude Code searches through `Bash` (an embedded `ugrep`/`bfs`) unless `Grep`/`Glob` are named in `--tools`, so the `Bash` matcher is the one that fires in practice; a `Bash` command that is not a search (`ls`, `git status`, a test run) is recognised from its command line alone and exits before the index, the tokenizer, or pydantic is imported.

Non-blocking contract:

- **Never intercepts `Read`, and never blocks.** `PostToolUse` fires after the tool ran, and the matcher is `Bash|Grep|Glob` only. A repo-level test (`tests/cli/test_install_client_hooks.py`) checks the written config reaches the archex entry only through that `PostToolUse` group.
- **Exits 0 on every path, prints stdout only on success.** A missing or stale index, a dirty tree, an unrecognised output format, a malformed payload, a timeout, or any internal error all add nothing; the reason goes to a local diagnostics log (`~/.archex/hook-diagnostics.log` by default, override with `ARCHEX_HOOK_DIAGNOSTICS_LOG`), never to the agent.
- **Hard 0.5 s wall-clock guard** (override with `ARCHEX_HOOK_TIMEOUT_SECONDS`), counted from the start of the hook's `main`: once the budget is spent the process exits 0 with no stdout, however far the annotation got, and a `timeout` ledger line is written.
- **Bounded and deterministic.** About 30 tokens per line and about 300 per call, `+N more units` past the cap, units already fully visible in the result get no line, and only a `fresh` index annotates. No network, no model call. Every block starts with an `[archex receipt] index_revision=<hash prefix> units=<n>` line.
- **One ledger line per search call**, none for a shell command that is not a search: `{timestamp, host: "claude-code", toolCallId, tool, eligible, annotated, units, tokens, freshness, reason, latency_ms}` appended to `ARCHEX_ANNOTATION_LEDGER` (default `~/.archex/annotation-ledger.jsonl`), with `tool_use_id` as `toolCallId`. The schema is the one the omp/Pi module writes; the Python writer lives in `archex.integrations.annotate_hook` for the other adapters to reuse.
- **Non-destructive install/uninstall.** Any other hooks in the same settings file — other tools' matcher groups under any event, the archex post-edit `PostToolUse` entry, the `SessionStart` primer, unrelated top-level settings — are left untouched by `--hooks`, a reinstall, and `--remove-hooks`, whichever archex surface was installed first. Re-running `--hooks` converges on one archex entry.

Measured on a fresh 24,219-chunk index (60 runs each, hook run as a subprocess with a Claude-shaped payload): a non-search `Bash` payload p50 36 ms / p95 40 ms; a search that annotates three units p50 365 ms / p95 389 ms. Python start-up and imports are most of a search's cost, and the first call in a process tree with cold caches can exceed the 500 ms budget, in which case that one call is not annotated.

Manual verification (bypassing Claude Code — this is what the hook receives on stdin for a `Bash` search):

```bash
echo '{"tool_name":"Bash","tool_use_id":"toolu_demo","cwd":"'"$PWD"'","tool_input":{"command":"rg -n compute_delta src"},"tool_response":{"stdout":"src/archex/index/delta.py:285:def compute_delta(","stderr":"","interrupted":false,"isImage":false}}' \
  | python -m archex.integrations.claude_code_annotate_hook
```

A repo with a fresh index prints `{"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "[archex receipt] ..."}}`. A repo with no index, a stale or dirty index, or output the parser does not recognise exits 0 with empty stdout and a diagnostics or ledger line instead.

Live verification (Claude Code 2.1.285, no hosted call): `scripts/swe_ab_stub_provider.py` answers Claude Code's Anthropic Messages requests (`ANTHROPIC_BASE_URL` pointing at it) with a scripted sequence of `Bash`, `Grep`, and `Glob` search calls. The session transcript's `hook_additional_context` attachment (`hookName: PostToolUse:Bash`) and the tool result in the next captured request body both carry the annotation, with the original result text first and the annotation in a `<system-reminder>` after it.

## Claude Code SessionStart session primer (opt-in)

`archex install-client claude-code --session-primer` installs a separate Claude Code `SessionStart` hook (`src/archex/integrations/session_hook.py`, invoked as `python -m archex.integrations.session_hook`). It delivers the bounded primer rendered from the explicit project-session ledger only when an existing index is fresh. It is opt-in: plain MCP installation and `--hooks` search-hook installation never add it.

```bash
archex install-client claude-code --session-primer                   # global: ~/.claude/settings.json
archex install-client claude-code . --session-primer --scope project # repo-local: .claude/settings.json
archex install-client claude-code --session-primer --dry-run          # preview only, writes nothing
archex install-client claude-code --remove-session-primer             # clean uninstall
```

The installer owns only its `SessionStart` handler, with matcher `resume|startup` and `args = ["-m", "archex.integrations.session_hook"]`. It preserves every other `SessionStart` handler, all search-hook groups, other hook events, and unrelated settings. Re-installation converges on one canonical handler; removal removes only that owned handler.

The hook accepts `cwd` and only `source: "startup"` or `"resume"` from Claude Code's SessionStart payload. It renders through `render_session_primer`, whose receipt checks index freshness and record revisions before the hook emits `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}`. A missing/stale index, empty eligible primer, malformed payload, render failure, or elapsed `ARCHEX_HOOK_TIMEOUT_SECONDS` budget emits no stdout context, writes a local diagnostic where relevant, and exits 0. It never captures a record, reindexes, reads conversation transcripts, or blocks session start.

Manual process-level check after recording a session item and indexing the repo:

```bash
printf '%s' '{"source":"startup","cwd":"'"$PWD"'"}' \
  | python -m archex.integrations.session_hook
```

`tests/integrations/test_hooks.py::test_session_start_hook_injects_only_fresh_explicit_context` exercises both supported sources through the actual module process and proves a modified repository emits no primer. This is hook-process evidence, not a claim of a live Claude Code UI smoke.

## oh-my-pi (omp) / Pi `tool_result` hook (opt-in)

`archex install-client omp --hooks` / `archex install-client pi --hooks` install one shared TypeScript extension module (`archex-hook.ts`) that registers a `pi.on("tool_result", ...)` handler. For every search-tool result — `grep`, `glob` (oh-my-pi), `find` (Pi), and `bash` search commands — it sends the host, the tool name, the tool's input, and the tool's own result text to the host-neutral annotation entry point (`python -m archex.integrations.annotate_hook`, the same core as `archex annotate`) and appends the returned lines. It is opt-in — plain `archex install-client omp`/`pi` never installs it:

```bash
archex install-client omp --hooks                      # user (default): ~/.omp/agent/extensions/archex-hook.ts
archex install-client omp . --hooks --scope project    # project-local: .omp/extensions/archex-hook.ts
archex install-client pi --hooks                       # user (default): ~/.pi/agent/extensions/archex-hook.ts
archex install-client pi . --hooks --scope project     # project-local: .pi/extensions/archex-hook.ts
archex install-client omp --hooks --dry-run            # preview only, writes nothing (pi identical)
archex install-client omp --remove-hooks               # clean uninstall (pi identical)
```

Unlike the Claude Code hook (a JSON command entry merged into `settings.json`), this installs a standalone `.ts` file discovered by each host's own native extension auto-discovery — oh-my-pi: project `<cwd>/.omp/extensions/*.ts`, user `~/.omp/agent/extensions/*.ts`; Pi: project `.pi/extensions/*.ts`, user `~/.pi/agent/extensions/*.ts` (confirmed by reading the installed `@mariozechner/pi-coding-agent` 0.68.1's own `docs/extensions.md`). Install, reinstall, and remove touch only `archex-hook.ts`; other modules in the same directory are left alone. The Python interpreter invoked (`ARCHEX_PYTHON_COMMAND` baked into the file) is the one active when `--hooks` was run. Pi's `tool_result` contract matches oh-my-pi's (`{ content, details, isError }` partial patch), so both hosts get the byte-identical module.

What gets appended — one neutral fact line per distinct indexed code unit the search hit, under a receipt header:

```text
[archex receipt] index_revision=e44d3a393e1a units=3
[archex] src/archex/benchmark/delta_strategies.py::prepare_repo function L42-63 · importers 2
[archex] src/archex/benchmark/determinism_economics.py::_prepare_task_repo function L318-350 · importers 2
[archex] src/archex/benchmark/runner.py::clone_at_commit function L196-229 · importers 22
```

Contract:

- **Augment, never replace.** The patch is a deep copy of the host's content array with exactly one text block appended; every original block, including fields the module does not know, survives unchanged. `details` and `isError` are never returned.
- **Parsing lives in Python.** `archex annotate` recognises the omp `grep` single-file, header-tree, and bare formats, the omp `glob` tree, and shell search output: `path:line:` output of `rg -n`/`grep -rn`/`ugrep -n`/`git grep -n`, bare `line:text` output of a one-file search (the file is the command's single path operand), and path lists from `find`, `bfs`, `fd`, `rg --files`, `git ls-files`, and `-l`. Counting (`-c`) and quiet (`-q`) searches are not searches. The entry point decides whether a call is a search before it imports the index, the tokenizer, or pydantic: a non-search call costs about 40 ms end to end. A bash command naming none of the search programs is not sent at all; anything unrecognised adds nothing.
- **Bounded.** One line per unit, about 30 tokens per line and 300 per call (cl100k), excess units collapsed into `+N more units`. A unit whose whole span is already visible in the result gets no line.
- **Fresh index only.** A stale, dirty, missing, or unprovenanced index adds nothing.
- **Fail open.** A spawn failure, a timeout past the budget (`ARCHEX_HOOK_TIMEOUT_SECONDS`, default 0.5s; the subprocess is `SIGKILL`ed), a malformed response, or any error leaves the result untouched; faults go to `~/.archex/hook-diagnostics.log` (`ARCHEX_HOOK_DIAGNOSTICS_LOG`).
- **Per-call ledger.** One JSON line per search-tool result at `~/.archex/annotation-ledger.jsonl` (`ARCHEX_ANNOTATION_LEDGER`): `host`, `toolCallId`, tool, eligible, annotated, units, tokens, freshness, the reason when nothing was added, and latency. The transcript alone cannot show whether an annotation was applied.
- **Degree is importers only.** The index stores file-level import edges and no call edges, so no caller count is rendered.

Evidence: `tests/integrations/test_annotate_hook_module.py` executes the generated module under Bun against a real index (byte-identical original, unknown-field survival, three calls in one process all annotated and ledgered, malformed events, dirty index, budget overrun). A live omp 18.4.2 session driven by a local OpenAI-compatible stub provider (`--no-extensions -e archex-hook.ts`, no hosted call) recorded the appended block in the session JSONL `toolResult` and in the next provider request.

The OpenCode plugin and the Claude Code hook annotate search results the same way: OpenCode from a `tool.execute.after` plugin ([below](#opencode-toolexecuteafter-plugin-opt-in)), Claude Code from a `PostToolUse` handler ([above](#claude-code-posttooluse-search-annotation-hook-opt-in)).

Manual verification (bypassing the host — pipe any real search result through the same command the module runs):

```bash
rg -n "def main" src | archex annotate --tool bash --input-json '{"command": "rg -n \"def main\" src"}'
```

## Codex CLI PostToolUse search-annotation hook (opt-in)

`archex install-client codex --hooks` installs a Codex `PostToolUse` hook (`src/archex/integrations/codex_annotate_hook.py`, invoked as `python -m archex.integrations.codex_annotate_hook`) on Codex's shell tool. When a shell command ran a search (`rg`, `grep`, `egrep`, `fgrep`, `git grep`, optionally after one leading `cd <dir> &&`), the hook maps each hit to the indexed code unit that contains it (`archex annotate`) and returns one fact line per unit as `hookSpecificOutput.additionalContext`. Codex records that text as a separate developer message next to the tool result, so the model sees the original output unchanged plus the annotation; the hook never asks Codex to block or rewrite anything. It is opt-in — plain `archex install-client codex` never installs it — and it writes to the *same* `config.toml` the MCP registration above writes to, as a separate marker-delimited block:

```bash
archex install-client codex --hooks                     # global: ~/.codex/config.toml
archex install-client codex . --hooks --scope project   # repo-local: .codex/config.toml
archex install-client codex --hooks --dry-run           # preview only, writes nothing
archex install-client codex --remove-hooks              # clean uninstall
```

`--hooks` also replaces the retired diagnostics-only `[[hooks.PreToolUse]]` block (`# archex:codex-hook`, `python -m archex.integrations.codex_hook`) that archex 0.33 and earlier wrote, in place, and `--remove-hooks` removes both, so an upgrade needs one `--hooks` run and leaves one archex search hook behind. Codex 0.153.4 runs a config-file hook only once it is trusted: it is inert until its content hash is recorded as `trusted_hash` in the hook's `hooks.state` entry (`hooks/src/engine/discovery.rs`; reviewing it in the TUI's `/hooks` is the interactive way), and a change to the block (for example an archex upgrade moving the interpreter path) needs a new review. `codex exec --dangerously-bypass-hook-trust` skips the check for one invocation and is only meant for automation that already vets its hooks.

### Installed config shape

The block sits between marker comments so a re-run or `--remove-hooks` finds and replaces exactly this block without disturbing any other section (the `[mcp_servers.archex]` registration above and the post-edit block below included); an existing block is replaced where it sits, so a reinstall never reorders the archex blocks. The `command` path is the Python interpreter active when `--hooks` was run:

```toml
# archex:codex-annotate-hook start
[[hooks.PostToolUse]]
matcher = "^Bash$"

[[hooks.PostToolUse.hooks]]
type = "command"
command = "/path/to/venv/bin/python -m archex.integrations.codex_annotate_hook"
timeout = 2
# archex:codex-annotate-hook end
```

### What Codex 0.153.4 sends, and where the annotation goes

Checked against the `rust-v0.153.4` source and recorded from a live hook (the `tool_input` and `tool_response` shapes below are what the hook received in a real session):

- **Tool name.** Shell commands run through `exec_command`, and its `PostToolUse` tool name is the fixed `Bash` (`post_unified_exec_tool_use_payload` in `codex-rs/core/src/tools/handlers/unified_exec.rs`). There is no separate search tool: `rg`/`grep` go through the same `Bash`, so the hook decides "is this a search?" from the command line alone and a command that is not a search exits before the index, the tokenizer, or pydantic is imported.
- **`tool_input`** is `{"command": "<the cmd string>"}`. The model-facing `workdir` argument is not forwarded. Codex writes the `exec_command` call, arguments included, to the session rollout (`transcript_path`) before running it, so the hook reads `workdir` back from the last megabyte of that file by `tool_use_id`. With no rollout, no such call, or no `workdir`, the base of relative paths is the payload's `cwd`, which is where Codex runs the command when the model sets no `workdir`. Limitation: if the rollout cannot be read (`transcript_path` is null when Codex keeps no local rollout) and the model ran the search from a subdirectory, a relative hit can resolve to a same-named file under `cwd`; a path that does not exist there is dropped.
- **`tool_response`** is the command's output as a bare JSON string, without the `Chunk ID`/`Wall time` header the model sees (`post_tool_use_response` in `codex-rs/core/src/tools/context.rs`). Codex sends no `PostToolUse` for a command still running past its yield time and none for a failed tool call.
- **Common fields** on stdin: `session_id`, `turn_id`, `transcript_path` (nullable), `cwd`, `hook_event_name`, `model`, `permission_mode`, `tool_name`, `tool_input`, `tool_response`, `tool_use_id` (`hooks/schema/generated/post-tool-use.command.input.schema.json`). Unknown fields are ignored.
- **Output.** `hookSpecificOutput.additionalContext` (`hooks/src/schema.rs`, `PostToolUseHookSpecificOutputWire`, camelCase, unknown fields denied) is collected per handler and recorded with `record_additional_contexts` (`codex-rs/core/src/tools/registry.rs`, `codex-rs/core/src/hook_runtime.rs`) as a developer message in the conversation, so the next model request carries it. In a live session the message appears in the rollout with `content_item_kinds: ["hooks.additional_context"]` and in the next request's `input` between the `function_call` and its `function_call_output`.

Non-blocking contract:

- **Matches the shell tool only.** `tests/cli/test_install_client_hooks.py` asserts the installed matcher is exactly `^Bash$`: not `apply_patch` (the post-edit hook's tool), not MCP tools, not `Read`.
- **Exits 0 on every path, prints stdout only on success.** A missing or stale index, a dirty tree, an unrecognised output format, a malformed payload, a timeout, or any internal error all add nothing; the reason goes to a local diagnostics log (`~/.archex/hook-diagnostics.log` by default, override with `ARCHEX_HOOK_DIAGNOSTICS_LOG`), never to the agent.
- **Hard 0.5 s wall-clock guard** (override with `ARCHEX_HOOK_TIMEOUT_SECONDS`), counted from the start of the hook's `main`: once the budget is spent the process exits 0 with no stdout and a `timeout` ledger line is written. Codex's own hook `timeout` is whole seconds; the installer sets `2` as the outer bound.
- **Bounded and deterministic.** About 30 tokens per line and about 300 per call, `+N more units` past the cap, units already fully visible in the result get no line, and only a `fresh` index annotates. No network, no model call. Every block starts with an `[archex receipt] index_revision=<hash prefix> units=<n>` line.
- **One ledger line per search call**, none for a shell command that is not a search: `{timestamp, host: "codex", toolCallId, tool: "Bash", eligible, annotated, units, tokens, freshness, reason, latency_ms}` appended to `ARCHEX_ANNOTATION_LEDGER` (default `~/.archex/annotation-ledger.jsonl`), with `tool_use_id` as `toolCallId`. The Claude Code hook writes the same schema through the same runner (`archex.integrations.post_tool_use_annotate`).
- **Non-destructive install/uninstall.** Any other `config.toml` content — the `[mcp_servers.archex]` registration, the post-edit block, unrelated sections — is left untouched by `--hooks`, a reinstall, and `--remove-hooks`, in every install order.

Measured on a fresh 24,219-chunk index (60 runs each, hook run as a subprocess with a Codex-shaped payload and a 690 KB rollout): a non-search command p50 42 ms / p95 52 ms (first call 60 ms); a search that annotates three units p50 396 ms / p95 444 ms / max 478 ms (first call 478 ms). Python start-up and imports are most of a search's cost; recovering `workdir` from the rollout adds no measurable time. A cold first call can exceed the 500 ms budget, in which case that one call is not annotated.

Manual verification (bypassing Codex — this is what the hook receives on stdin for a shell search):

```bash
echo '{"tool_name":"Bash","tool_use_id":"call_demo","cwd":"'"$PWD"'","transcript_path":null,"tool_input":{"command":"rg -n compute_delta src"},"tool_response":"src/archex/index/delta.py:285:def compute_delta(\n"}' \
  | python -m archex.integrations.codex_annotate_hook
```

A repo with a fresh index prints `{"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "[archex receipt] ..."}}`. A repo with no index, a stale or dirty index, or output the parser does not recognise exits 0 with empty stdout and a diagnostics or ledger line instead.

Live verification (Codex CLI 0.153.4, no hosted call, isolated `CODEX_HOME`): a `[model_providers.stub]` table with `base_url = "http://127.0.0.1:<port>/v1"` and `wire_api = "responses"` points Codex at `scripts/swe_ab_stub_provider.py`, which answers Codex's Responses requests with a scripted `exec_command` running `rg -n …`, then `ls -la`, then `grep -rn …`. With the hook installed by `install-client codex --hooks` and run under `--dangerously-bypass-hook-trust`, the two searches were annotated (two ledger lines, keyed by the calls' `tool_use_id`; none for `ls`), the annotation is in the session rollout as a `hooks.additional_context` developer message, and the next captured request body carries it beside the untouched command output. Without the trust bypass and without a reviewed hash the same session ran no hook and wrote no ledger line.

## OpenCode `tool.execute.after` plugin (opt-in)

`archex install-client opencode --hooks` installs a standalone TypeScript plugin file that registers a `tool.execute.after` handler for OpenCode's native `grep`, `glob`, and `bash` tools. For every call it sends the host, the tool name, the tool's `args`, and the tool's own output text to the host-neutral annotation entry point (`python -m archex.integrations.annotate_hook`, the same core as `archex annotate`) and appends the returned lines to `output.output`. It is opt-in — plain `archex install-client opencode` never installs it:

```bash
archex install-client opencode --hooks                      # user (default): ~/.config/opencode/plugins/archex-hook.ts
archex install-client opencode . --hooks --scope project    # project-local: .opencode/plugins/archex-hook.ts
archex install-client opencode --hooks --dry-run            # preview only, writes nothing
archex install-client opencode --remove-hooks                # clean uninstall
```

Unlike the MCP config row above (an `opencode.json` entry), this installs a standalone `.ts` file that OpenCode loads from its plugin directories (`.opencode/plugins/` for a project, `~/.config/opencode/plugins/` globally), so no `opencode.json` change is written. Install, reinstall, and remove touch only `archex-hook.ts`; a foreign plugin, and archex's own `archex-post-edit-hook.ts`, in the same directory are left alone. The Python interpreter invoked (`ARCHEX_PYTHON_COMMAND` baked into the file) is the one active when `--hooks` was run. This replaces the earlier plugin that searched the index for the `grep`/`glob` pattern (`python -m archex.integrations.hook`) and never saw the results; reinstalling with `--hooks` overwrites it.

What gets appended after the tool's own text, in this shape (recorded live, below):

```text
[archex receipt] index_revision=a42a5205d59d units=3
[archex] services/auth.py module-level · units 1 · importers 1
[archex] services/auth.py::AuthService.login method L15-18 · importers 1
[archex] utils.py::hash_password function L9-10 · importers 2
```

### How the hook reaches the model

OpenCode v1.14.33 (`packages/opencode/src/session/prompt.ts:432-466`) runs a native tool, builds `output = {...result, attachments}`, calls `plugin.trigger("tool.execute.after", {tool, sessionID, callID, args}, output)`, and returns that same `output` object to the AI SDK executor. `Plugin.trigger` passes every hook the same object (`plugin/index.ts:260-275`). A hook that reassigns `output.output` therefore changes the text the model receives and the text stored in the session. The public type is `input: {tool, sessionID, callID, args}` and `output: {title, output, metadata}` (`packages/plugin/src/index.ts:295-303`); `callID` becomes the ledger's `toolCallId`. Native tool shapes the annotation core parses: `grep` (`{pattern, path?, include?}`) prints `Found N matches`, then per file an absolute `path:` line and `  Line N: text` entries; `glob` (`{pattern, path?}`) prints absolute paths or `No files found`; `bash` (`{command, workdir?, timeout?, description}`) prints the command output, followed by a `<bash_metadata>` block when it was cut short or aborted.

### Contract, mirroring the oh-my-pi/Pi hook

- **Augment, never replace.** `output.output` is only ever reassigned to the original text plus `"\n\n"` plus the annotation, so the host's own text stays a byte-for-byte prefix. `title`, `metadata`, `attachments`, and every field the plugin does not know are never touched.
- **Search tools only.** The plugin's only tool-name dispatch is `ANNOTATED_TOOLS`, keyed exactly on `{"grep", "glob", "bash"}` — the set `archex.annotate` accepts for host `opencode`. `read` and every other tool are absent by construction. A `bash` command that names none of `rg`, `grep`, `find`, `bfs`, `fd`, or `ls-files` is never sent to Python (0 ms); the Python entry decides whether the rest are searches (`rg`, `grep`, `ugrep`, `git grep`, or path listers such as `find`, `fd`, `git ls-files`) before it imports the index or the tokenizer.
- **MCP calls stay excluded.** OpenCode registers every MCP tool under a mandatory `{server}_{tool}` id, so no MCP tool can collide with the table. The exclusion also has a functional reason: for an MCP call `tool.execute.after` receives the raw MCP result (`{content}`, `prompt.ts:480-510`), which OpenCode rebuilds into the model-visible text afterwards from `result.content` (`prompt.ts:512-557`), discarding any `output.output` mutation.
- **Parsing lives in Python.** The plugin gathers inputs and appends output; formats, freshness, caps (about 30 tokens per line, 300 per call, `+N more units`), and the fresh-index requirement are all in `archex.annotate`.
- **Fail open.** A stale or dirty index, an unrecognised format, a spawn failure, a timeout, a malformed response, or any thrown error leaves `output` untouched; faults go to `~/.archex/hook-diagnostics.log` (`ARCHEX_HOOK_DIAGNOSTICS_LOG`), never to the agent.
- **Bounded.** The subprocess runs under a 0.5 s budget (`ARCHEX_HOOK_TIMEOUT_SECONDS`) and is `SIGKILL`ed past it. The first call after a long idle can miss the budget while Python's files are cold; it then adds nothing.
- **Per-call ledger.** One JSON line per `grep`/`glob`/`bash` call at `~/.archex/annotation-ledger.jsonl` (`ARCHEX_ANNOTATION_LEDGER`): `host: "opencode"`, `toolCallId` = OpenCode's `callID`, tool, eligible, annotated, units, tokens, freshness, the reason when nothing was added, and latency. A `bash` command that is not a search writes an ineligible `not_search_command` line.
- **Subagent-reachable.** The handler registers unconditionally with no `sessionID`-based gating. A subagent's turn runs through the same tool-resolution code as a top-level turn, so its native `grep`/`glob`/`bash` calls reach `tool.execute.after` too (confirmed for the earlier plugin on the same hook by exporting a subagent's own child session).

### Verification performed

- `tests/integrations/test_opencode_annotate_plugin.py` executes the generated plugin under Bun against a real index with recorded-shape `(input, output)` pairs: a byte-identical prefix with `title`/`metadata`/unknown fields untouched, three calls in one process with three ledger lines and distinct `callID`s, `glob` and `bash` search annotated, non-search `bash`, `read`, and an MCP-shaped call untouched, malformed `args`/`output`, a dirty index, a subprocess that outlives the budget (recorded pid confirmed killed), and a missing interpreter. `tests/cli/test_install_client_hooks.py` covers install, remove, idempotent reinstall, the dispatch table, and foreign-plugin survival.
- **Live, OpenCode 1.14.33, no hosted call.** A custom provider (`@ai-sdk/openai-compatible`, `baseURL` on a local stub, `scripts/swe_ab_stub_provider.py`) scripted `grep`, `bash rg -n …`, and `bash echo hello` in a committed, freshly indexed repository with the plugin installed at project scope by `install-client opencode --hooks --scope project`, under isolated `XDG_*` directories. The captured chat-completions request that followed each call carried the annotation in its `role: "tool"` message (`tool_call_id` `call_0`/`call_1`), and the session's SQLite `part` rows (`state.output`) held byte-identical text; the `echo` call carried none. A run of the same script with `--pure` (no external plugins) produced the unannotated outputs; each annotated output began with exactly that text and added 252 characters. The ledger held three lines with `toolCallId` `call_0`–`call_2`, `host: "opencode"`, 3 units and 78 tokens on the two searches, and `not_search_command` on the echo. An earlier run of the same script hit the 0.5 s budget on the first call (a cold Python after idle): that call's output was left untouched, the ledger recorded `timeout` and the diagnostics log a `ts_timeout` line, and the two later calls were annotated.
- **Latency through the plugin** (Bun, 60 sequential calls per scenario, a 24k-chunk index, machine load average ≈ 17 from unrelated processes, so absolute values run above an idle machine): `bash` naming no search program p50 0 ms (no process started); `bash` naming one but not a search p50 43 ms / p95 55 ms; `bash rg` search p50 442 ms / p95 483 ms (59 of 60 annotated); `grep` search p50 456 ms / p95 502 ms (51 of 60 annotated, the other 9 hit the budget and added nothing); first call in a fresh Bun process 441–464 ms. The same request sent straight to `python -m archex.integrations.annotate_hook` at the same load measured p50 469 ms / p95 502 ms, so the plugin adds no measurable overhead over the Python entry.

Manual verification (bypassing the host — the same request the plugin sends):

```bash
echo '{"host":"opencode","tool":"bash","input":{"command":"rg -n compute_delta src"},"text":"src/delta.py:12:def compute_delta(a, b):\n","cwd":"."}' \
  | python -m archex.integrations.annotate_hook
```

## Post-edit impact hooks (opt-in)

Installed separately from every other archex surface with `--post-edit-hooks`, removed with `--remove-post-edit-hooks`. Never installed by default. Each supported client gets a different mechanism but the same core contract, because all of them shell into one Python entry point (`archex.integrations.post_edit_hook`); no client-specific impact logic exists.

### What it does

After a *successful* edit the adapter records the edited paths, synchronizes the index through the existing delta/full-fallback path, and emits a bounded block naming the edited files, the files that depend on them, and the tests in the affected set.

```text
[archex post-edit receipt] generation=5e04ec901fb1 index_revision=ab8d87d134b4 generated_at=2026-09-10T01:13:20Z client=claude-code confidence=complete
archex post-edit impact — file-scoped: derived from indexed import and reference edges between files, not from call-graph analysis.
Edited files (1):
- pkg/core.py
Files that depend on them (2):
- pkg/consumer.py
- tests/test_core.py
Tests in the affected set (1):
- tests/test_core.py
```

### The two halves of the contract

**Fail open at the client boundary.** The hook never blocks or fails an edit. Every adapter is wired to a post-execution event that upstream documents as unable to reject the tool call, and every code path exits 0 with no output rather than raising. A malformed payload, an unmanaged repository, a path outside the repository, a stale index, a timeout, or an internal error all degrade to silence plus a line in `~/.archex/hook-diagnostics.log` (override with `ARCHEX_HOOK_DIAGNOSTICS_LOG`).

**Fail closed on freshness.** Content reaches the agent only when three conditions hold on the refreshed store: a persisted generation id exists, the store is not flagged for reindex, and the working-tree signature recorded at publication still equals the signature recomputed from disk. The third catches an edit landing between the refresh and the report. Any failure emits nothing and leaves the recorded edit pending, so the next edit event retries.

### What it deliberately does not claim

- **Risk is file-scoped.** Dependents come from indexed import and reference edges between files. archex has no `CALLS` evidence, so no symbol-level blast radius is claimed and the rendered line says so.
- **Incompleteness is stated, not implied.** Truncated lists carry their full count, changed paths archex does not index are named as unindexed with their dependents declared unknown, and edits dropped by the recorded-edit cap are counted. Any of the three downgrades the receipt from `confidence=complete` to `confidence=partial`.

### Bounds

| Bound | Default | Override |
| --- | --- | --- |
| Whole synchronize-and-report cycle | 8 s | `ARCHEX_POST_EDIT_TIMEOUT_SECONDS` |
| TypeScript shim kill timer | 15 s | not configurable (deliberately above the Python deadline so that one fires first and logs) |
| Paths accepted from one event | 512 | not configurable |
| Pending paths retained | 200 | not configurable |
| Characters per path | 1024 | not configurable |

### Shared state

`.archex/post-edit-state.json` — versioned, bounded, written atomically under an exclusive lock because several hook subprocesses run concurrently when an agent edits several files in one turn. Lock contention past a short deadline abandons the write rather than delaying the agent; the working-tree delta re-derives changed files independently, so a skipped record costs scoping precision, not correctness.

### Per-client mechanism

| Client | Event | Success signal | How impact reaches the agent |
| --- | --- | --- | --- |
| claude-code | `PostToolUse`, matcher `Edit|Write` | event fires only after the tool ran; an explicit `tool_response.success: false` or `error` is honoured | `hookSpecificOutput.additionalContext` |
| codex | `PostToolUse`, matcher `^apply_patch$` | as above | `hookSpecificOutput.additionalContext` |
| omp / pi | `tool_result` on `edit`/`write` | `isError !== true` | appended to the result's `content` |
| opencode | `tool.execute.after` on `edit`/`write` | event fires after execution; no error flag exists, so recorded paths are revalidated against the real working tree | appended to `output.output` |
| cursor | none | — | not supported; see the matrix row |

### Verification performed

- A real `claude -p` session edited a file with the hook installed; Claude Code's own transcript recorded a `hook_additional_context` attachment (`hookName: PostToolUse:Edit`) carrying the archex block.
- The Codex adapter, and the omp, pi, and OpenCode modules, were each executed against a live index: the Python hooks as real subprocesses driven by the installed configuration, the TypeScript modules loaded under Bun and dispatched with a real event. Each returned a fresh receipt naming the correct client.
- A forced 1 ms budget produced exit 0, no output, a `post_edit_timeout` diagnostic, and a still-dirty state ready for the next event.
- The Codex CLI itself was not driven end to end: the local OAuth refresh token was expired at verification time.

## Persistent status surfaces (opt-in)

Installed separately from every other archex surface with `--statusline`, removed with `--remove-statusline`. Never installed by default. Supported on `claude-code` (a `statusLine` command), `omp`, and `pi` (a status extension module); `codex`, `cursor`, and `opencode` have no persistent status surface and are refused explicitly.

Archex publishes a bounded, versioned status snapshot to `.archex/status-snapshot.json` whenever indexing establishes that the store describes the current tree (a full, delta, or unchanged-tree publication, and the validated cache hit every warm query takes), whenever a post-edit hook records or synchronizes an edit, and whenever `archex status` runs. Renderers only read that file. **No renderer opens the index, runs a parser, or starts an archex process on repaint.**

### States

| State | Meaning | Who determines it |
| --- | --- | --- |
| `fresh` | The index describes the working tree and nothing is awaiting synchronization. | Writer |
| `dirty` | The index no longer describes the tree, or the store is flagged for a reindex. | Writer |
| `pending` | An edit was recorded at or after the last index measurement and has not been synchronized. | Writer |
| `stale` | The snapshot is valid but older than the freshness budget (`900s`, override with `ARCHEX_STATUS_STALE_AFTER_SECONDS`), so it is no longer evidence of freshness. | Reader, needs a clock |
| `missing` | No snapshot has been published, or `archex reset` cleared it. | Reader |
| `corrupt` | A snapshot exists but is unreadable, unparsable, or invalid. Remedy: re-publish with `archex status`. | Reader |
| `unsupported` | A snapshot exists and parses but declares a schema version this build does not read. Remedy: upgrade archex. | Reader |

`working_tree_dirty` is reported as its own field and never drives the state: a synchronized index describes a tree with uncommitted edits exactly as well as a clean one. Status never reports estimated token savings.

The snapshot's serialized layout is part of its versioned contract: sorted keys and two-space indentation, so every scalar sits on its own `  "key": value` line. That is what lets the POSIX `sh` renderer read it with shell builtins and no JSON parser, and a test pins it. A document that archex did not write — hand-edited, or reformatted — may therefore read as `corrupt` in that renderer even where a JSON parser would accept it; `archex status` re-publishes it in the canonical form.

### CLI

```bash
archex status .                      # authoritative: opens the index, and refreshes the snapshot
archex status . --cached             # reads only the cached snapshot; never opens the index
archex status . --cached --format json
archex status . --cached --strict    # exit 1 unless the cached state is fresh
```

`--cached` exits 1 for `corrupt` and `unsupported`, and (with `--strict`) for anything but `fresh`. It is the mode for scripts and repeated calls; the default mode is the one to run when a client's status surface looks wrong, because it re-measures and republishes.

Both modes work from a subdirectory. `--cached` walks up to the nearest published snapshot, exactly as the shell and TypeScript renderers do, so all three surfaces answer for the same document.

### Claude Code status line

```bash
archex install-client claude-code --statusline                     # global: ~/.claude/settings.json
archex install-client claude-code . --statusline --scope project    # repo-local: .claude/settings.json
archex install-client claude-code --statusline --dry-run            # preview only, writes nothing
archex install-client claude-code --remove-statusline               # clean uninstall
```

Two artifacts: the renderer script at `~/.claude/archex-statusline.sh` (or `.claude/archex-statusline.sh` for project scope), and a `statusLine` entry in the same `settings.json` the hook installers merge into:

```json
{
  "statusLine": {
    "type": "command",
    "command": "/bin/zsh \"/abs/path/.claude/archex-statusline.sh\"",
    "padding": 0,
    "refreshInterval": 10
  }
}
```

The interpreter is chosen at install time from the shells present on the host, preferring one that can read a clock without forking (`zsh` after the builtin `zmodload zsh/datetime`, or bash 5+); `sh` is the fallback where none can. That choice is what makes the `stale` label reachable on macOS, whose `/bin/sh` and `/bin/bash` are both bash 3.2.

`statusLine` is a scalar settings key, not a matcher group, so there is no way for two status lines to coexist. Consequences, both tested:

- A `statusLine` that archex did not install is **refused, not replaced** — the command fails and changes nothing. Remove it first, or install into the other scope.
- `--remove-statusline` deletes only an archex-owned entry (its command names `archex-statusline.sh`) and only an archex-owned script (its body carries the `archex:statusline` marker). A foreign status line and a foreign script of the same name are both left untouched.

Unrelated settings keys, the `PostToolUse` search-annotation hook, the `SessionStart` primer, and the `PostToolUse` post-edit hook all survive install and removal; each surface owns its own key or marker.

Sample output:

```text
archex fresh - 1234 files, 5678 chunks - rev 01234567 - 12s ago
archex pending - 3 awaiting sync - rev 01234567 - 4s ago
archex dirty - reindex required - rev 01234567
archex stale - unverified since measurement - rev 01234567 - 31m ago
archex missing - no status snapshot - run: archex index
archex corrupt - unreadable snapshot - run: archex status
archex unsupported - snapshot v2 - upgrade archex
```

A trailing ` - watch` appears when a watch-driven refresh was observed within the last `300s`. Its absence means "no recent watch refresh", never "no watcher is running" — an idle watcher publishes nothing because nothing changed.

### Why the renderer is a shell script

Claude Code re-runs the status-line command on every repaint, debounced at 300 ms, and cancels an in-flight script when a new update arrives. A cold Python start per repaint is exactly what R23 excludes, so the renderer is POSIX `sh` using **shell builtins only**: no `jq`, no `python`, no `date`, no `archex`, and no command substitution anywhere. It reads the session directory out of the JSON payload Claude Code writes to stdin (falling back to `$PWD`), walks up to the nearest `.archex/status-snapshot.json`, and parses that document line by line with parameter expansion.

One consequence is handled rather than hidden: computing the `stale` label needs a clock, and POSIX `sh` has no builtin one. The renderer reads `$EPOCHSECONDS`, which bash 5+ provides natively and zsh provides after the builtin `zmodload zsh/datetime`, and the installer therefore picks a clock-bearing interpreter when the host has one. On a host where none does, the renderer reports the measured state without an age or `stale` label instead of spending a `date` fork on every repaint, and `archex status --cached` — which always has a clock — remains the surface that always reports `stale`.

### Verification performed for the Claude Code status line

- The installed script was executed through `/bin/sh` **with an empty `PATH`** against `fresh`, `dirty` (both variants), `pending` (complete and truncated views), `missing`, `corrupt` (unparsable bytes and an unknown state value), and `unsupported` snapshots. Every run printed the expected line, exited 0, and wrote nothing to stderr — which no renderer that shelled out to `jq`, `python`, `date`, or `archex` could do.
- The `stale`, age, and watch segments were exercised under `/bin/zsh`, the local shell that can read a clock without forking.
- Session-directory resolution was exercised three ways: the stdin `cwd` payload from an unrelated working directory, a subdirectory of the session repository, and an unrelated directory (which reports `missing` rather than another repository's status).
- `archex status --cached` was verified to render every state, to resolve the snapshot from a subdirectory, and to work with `IndexStore.__init__` patched to raise.
- A refused install (a `statusLine` archex did not write) was verified to leave no renderer script behind.

### oh-my-pi (omp) and Pi status extension

```bash
archex install-client omp --statusline                    # user: ~/.omp/agent/extensions/archex-status.ts
archex install-client omp . --statusline --scope project  # repo-local: .omp/extensions/archex-status.ts
archex install-client pi --statusline                     # user: ~/.pi/agent/extensions/archex-status.ts
archex install-client omp --remove-statusline             # clean uninstall
```

Both hosts receive a byte-identical TypeScript module, a different file from the search hook's `archex-hook.ts` and the post-edit hook's `archex-post-edit-hook.ts`, so all three surfaces install and remove independently.

Unlike the Claude Code status line — a command the client re-runs per repaint — this is a module the host already has loaded. It registers `turn_start`, `tool_result`, and `turn_end` handlers, and each one reads the snapshot with `readFileSync` and pushes the rendered line through `ctx.ui.setStatus("archex", …)`, which both hosts render in the footer and in the `status` status-line segment. A repaint therefore launches nothing at all: no subprocess, no index, no parser. A host reporting `hasUI: false` (print and RPC modes, where `setStatus` is a documented no-op) receives no call.

Because the module runs in a JavaScript runtime it always has a clock, so it reports `stale` on every platform, and it reads `ARCHEX_STATUS_STALE_AFTER_SECONDS` like the other two renderers, so a tuned freshness budget cannot make the three surfaces disagree about one snapshot.

The snapshot version, freshness budget, watch TTL, and artifact path are substituted into the module from the Python constants at install time, so this renderer cannot drift from the document it reads.

### Clients with no persistent status surface

`archex install-client <client> --statusline` fails with the upstream reason and writes nothing for `codex`, `cursor`, and `opencode`:

| Client | Upstream reason |
| --- | --- |
| codex | The Codex CLI exposes no status-line configuration, and no hook output field that renders persistently — its hooks surface text only as `additional_context` on a tool event. |
| cursor | Cursor's configuration surface is hooks only; there is no status or footer API for an extension to write into. |
| opencode | The plugin surface exposes tool and chat hooks plus observable TUI events, whose only status-shaped member is the transient `tui.toast.show`. A toast disappears, so it cannot carry a persistent freshness indicator. |

`archex status --cached` is the supported surface for those clients. Archex ships no adapter it cannot show to work.

### Verification performed for the omp/Pi module

- The rendered module was loaded under Bun in a real runtime, all three events were dispatched, and every published status was captured from `ctx.ui.setStatus`: `fresh`, `pending` (complete and truncated views), `dirty` (both variants), `stale`, `missing`, `corrupt` (unparsable bytes and an unknown state value), `unsupported`, and the watch segment all rendered distinctly.
- Snapshot discovery was exercised through the real upward walk from the process working directory, not only through the test override. A present-but-unreadable snapshot classifies as `corrupt` rather than being walked past to a parent repository's document.
- The `ARCHEX_STATUS_STALE_AFTER_SECONDS` override was exercised: the same snapshot renders `fresh` under the default budget and `stale` under a tightened one.
- A host reporting `hasUI: false` received no `setStatus` call.
- Placement, byte identity between the two hosts, idempotent reinstall, and independence from the post-edit module in the same directory are covered by `tests/cli/test_install_client_status_adapters.py`; the Bun execution lives in `tests/integrations/test_status_extension_module.py` and skips where `bun` is absent.
- No live omp or Pi TUI session was driven: the refresh fires on turn events, and producing one would require a hosted model call. Upstream support rests on each host's own type declarations (`ExtensionUIContext.setStatus`, plus the `turn_start`/`tool_result`/`turn_end` event declarations) at the installed versions, and on the executed module.

## Cursor `beforeSubmitPrompt` hook (opt-in, diagnostics-only)

`archex install-client cursor --hooks` installs `src/archex/integrations/cursor_hook.py` (invoked as `python -m archex.integrations.cursor_hook`) as a Cursor `beforeSubmitPrompt` hook. **This is prompt-level context, not per-tool-call augmentation** — it fires once per submitted prompt, not once per Grep/Glob-equivalent tool call the way the Claude Code/omp/Pi/OpenCode hooks above do, and (per the confirmation spike below) it never injects anything into the conversation at all. It is opt-in — plain `archex install-client cursor` never installs it — and it writes to a *different* file than MCP server registration:

```bash
archex install-client cursor --hooks                     # global: ~/.cursor/hooks.json
archex install-client cursor . --hooks --scope project   # repo-local: .cursor/hooks.json
archex install-client cursor --hooks --dry-run           # preview only, writes nothing
archex install-client cursor --remove-hooks              # clean uninstall
```

### Confirmation-spike findings (M23 §2 assumption, corrected)

The historical M23 assumption framed `beforeSubmitPrompt` as Cursor's "content-bearing" prompt-level hook and planned to inject an archex `scout`-style summary through its output. Read directly against Cursor's own official docs (`cursor.com/docs/hooks`, `cursor.com/docs/reference/third-party-hooks`, fetched 2026-07-06), not secondary sources:

- Cursor has no Grep/Glob-equivalent tool-call hook at all — `preToolUse`/`postToolUse` fire generically for every tool with no per-tool augmentation scoping, unlike Claude Code's `Grep`/`Glob` matcher, oh-my-pi/Pi's `grep`/`glob`/`find` dispatch, or OpenCode's native-tool `tool.execute.after`.
- **`beforeSubmitPrompt`'s own output schema is `{"continue": bool, "user_message": str | None}` only.** There is no context-injection output field, nested or flat. This differs from `sessionStart` and `postToolUse`, which both support an `additional_context`/`additionalContext` output field. The "Response Format Compatibility" section of the third-party-hooks doc documents a Claude-Code-style nested `hookSpecificOutput` translation only for `PreToolUse` and `Stop`/`SubagentStop` — none is documented for `UserPromptSubmit` (which Cursor maps to `beforeSubmitPrompt`), and no `additionalContext` passthrough exists for it despite Claude Code's own `UserPromptSubmit` hook supporting that field.
- `user_message` is documented as shown only when a submission is blocked (`continue: false`) — deny/blocking behavior is explicitly out of scope for this milestone, so it cannot carry injected context on the normal path either.

**Consequence:** as specified, Cursor cannot deliver prompt-level context injection today — there is no output field to carry it. Per the same discipline M21 applied when Codex's hook schema turned out to have no Grep/Glob-equivalent event to scope augmentation to, this hook ships the plan's own accepted fallback instead: **diagnostics-only**. It performs the same lookup and logs what it would have injected, but never returns it to Cursor and never blocks prompt submission.

### Installed config shape

Unlike Claude Code (a JSON entry merged into `settings.json`) or Codex (a marker-delimited TOML block), this merges a single-entry array under a `hooks.beforeSubmitPrompt` key into `hooks.json` — a key structurally separate from `hooks.beforeReadFile`, so "never touches `beforeReadFile`" is a property of only ever writing under this one key, not something a shared matcher regex has to get right:

```json
{
  "version": 1,
  "hooks": {
    "beforeSubmitPrompt": [
      {
        "command": "/path/to/venv/bin/python -m archex.integrations.cursor_hook",
        "timeout": 1
      }
    ]
  }
}
```

Contract:

- **Never wires anything to `beforeReadFile`, or any hook besides `beforeSubmitPrompt`.** The installer only ever reads and writes `hooks.beforeSubmitPrompt`; any other `hooks.*` key already present (including `beforeReadFile`) is left byte-for-byte untouched by both install and remove. `tests/cli/test_install_client_hooks.py`'s `test_write_hook_install_plan_cursor_config_assertion_never_wires_before_read_file` and `test_cli_hooks_cursor_installed_file_never_targets_before_read_file` assert this against both the in-memory plan and the file the CLI actually writes, seeded with a pre-existing `beforeReadFile` entry.
- **Never returns context injection, and never blocks.** No code path in `archex.integrations.cursor_hook` ever sets anything but `{"continue": true}` — there is no field to carry injected context (see the confirmation spike above), and deny/blocking behavior is out of scope for this milestone regardless.
- **Exits 0 on every path.** A missing/stale index, a malformed payload, a prompt with no identifier-like tokens, a timeout, or any internal error all degrade to no diagnostic (or a diagnostic-only log line) — never a blocked or errored prompt submission.
- **Reuses `archex.integrations.hook`'s engine in-process.** `lookup_with_timeout`/`log_diagnostic` are called directly (no second subprocess spawned), so freshness/timeout/diagnostics semantics exactly match the pattern-search subprocess the OpenCode plugin runs (the engine the Claude Code hook ran before it moved to result annotation), appending to the same `~/.archex/hook-diagnostics.log` (override with `ARCHEX_HOOK_DIAGNOSTICS_LOG`).
- **Query extraction picks a single strongest token, not the whole prompt.** `IndexStore.search_symbols` phrase-matches its entire query as one quoted FTS5 phrase, so passing a full natural-language sentence would require that exact word sequence to appear verbatim in the indexed content — true for a single Grep/Glob pattern fragment, never true for prose. `cursor_hook._extract_query` instead picks the single longest identifier-like token in the prompt as a heuristic stand-in for "the symbol the user is probably asking about."
- **Prompt-level, not per-tool-call.** Every submitted prompt triggers a lookup attempt regardless of whether the agent is about to search for anything — there is no matcher-based scoping the way Grep/Glob-only hooks have, because `beforeSubmitPrompt` is not a per-tool event.
- **Non-destructive install/uninstall.** Any other `hooks.json` content — other hook types, unrelated top-level keys — is left untouched by both `--hooks` and `--remove-hooks`. Re-running `--hooks` is an idempotent no-op once installed, even across a venv move (the command converges on the active `sys.executable` each time).

Manual verification (bypassing Cursor entirely — this is exactly what the hook receives on stdin for a submitted prompt):

```bash
echo '{"prompt":"How does compute_delta handle renames?","attachments":[]}' \
  | python -m archex.integrations.cursor_hook
```

Always exits 0 with `{"continue": true}` on stdout. A repo with a fresh index and a prompt containing a real identifier appends a `cursor_context_injection_unsupported` diagnostic line to the log describing the withheld match, instead of injecting it; a repo with no index, a stale index, or a prompt with no identifier-like tokens produces no diagnostic (or the same `index_not_fresh`/`status_error` diagnostic the pattern-search engine logs) and no injected context either way.
