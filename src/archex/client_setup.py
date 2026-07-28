"""Client install/bootstrap helpers for MCP-compatible archex setups."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from archex.integrations.codex_hook import HOOK_MATCHER as CODEX_HOOK_MATCHER
from archex.integrations.hook import HOOK_MATCHER
from archex.integrations.mcp import resolve_tool_scope

ClientName = Literal["claude-code", "codex", "cursor", "opencode", "pi", "omp"]
ClientScope = Literal["project", "user"]

_USER_ONLY_CLIENTS: frozenset[ClientName] = frozenset({"pi", "omp"})

HookAction = Literal["install", "remove"]

#: Substring in a hook handler's ``args`` that identifies it as archex-owned,
#: so install/remove can find and replace our own entry without disturbing any
#: other hook the user has configured for the same matcher group.
_HOOK_ARGS_MARKER = "archex.integrations.hook"
_OMP_SCHEMA = (
    "https://raw.githubusercontent.com/can1357/oh-my-pi/main/"
    "packages/coding-agent/src/config/mcp-schema.json"
)
_OPENCODE_SCHEMA = "https://opencode.ai/config.json"
_CLIENT_SCHEMA: dict[ClientName, str] = {
    "opencode": _OPENCODE_SCHEMA,
    "omp": _OMP_SCHEMA,
}

AGENT_GUIDANCE_START = "<!-- archex:mcp-guidance start -->"
AGENT_GUIDANCE_END = "<!-- archex:mcp-guidance end -->"
AGENT_GUIDANCE_PROMPT = "\n".join(
    [
        "## Repository context via archex (MCP)",
        (
            'For architecture, ownership, dependency, or "where is X" questions, use '
            "the archex MCP tools before reading files by hand:"
        ),
        "- `scout_repo` — compact structural map",
        "- `query_repo` — ranked code context for a question",
        "- `analyze_repo` — module/package architecture",
        "- `search_symbols` / `get_symbol` — exact symbol lookup",
        (
            "Pass the repository path as `repo_url`. In harnesses with on-demand tool "
            "discovery, activate the archex tools first. Treat archex output as context "
            "selection, not proof — verify with reads/tests before editing."
        ),
    ]
)


@dataclass(frozen=True)
class ClientInstallPlan:
    client: ClientName
    scope: ClientScope
    target_path: Path
    content: str
    description: str
    tested_status: str
    last_verified: str


@dataclass(frozen=True)
class DiscoveredClient:
    client: ClientName
    scope: ClientScope
    config_path: Path
    is_installed: bool
    evidence: str


def get_client_config_candidates(
    client: ClientName, repo_root: Path, scope: ClientScope
) -> list[Path]:
    home = Path.home()
    if scope == "project":
        if client == "claude-code":
            return [repo_root / ".mcp.json"]
        if client == "cursor":
            return [repo_root / ".cursor" / "mcp.json"]
        if client == "codex":
            return [repo_root / ".codex" / "config.toml"]
        if client == "opencode":
            return [repo_root / "opencode.json"]
        return []
    else:
        if client == "claude-code":
            return [
                home / ".claude.json",
                home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
                home / ".config" / "claude" / "claude_desktop_config.json",
            ]
        if client == "cursor":
            return [home / ".cursor" / "mcp.json"]
        if client == "codex":
            return [home / ".codex" / "config.toml"]
        if client == "opencode":
            return [home / ".config" / "opencode" / "opencode.json"]
        if client == "pi":
            return [home / ".pi" / "agent" / "mcp.json"]
        if client == "omp":
            return [home / ".omp" / "agent" / "mcp.json"]
        return []


def discover_agent_files(repo_root: Path) -> list[Path]:
    home = Path.home()
    candidates = [
        home / ".omp" / "agent" / "AGENTS.md",
        home / ".pi" / "agent" / "AGENTS.md",
        repo_root / "AGENTS.md",
        repo_root / "CLAUDE.md",
        repo_root / ".cursorrules",
    ]
    return [p for p in candidates if p.exists() and p.is_file()]


def discover_clients(source: str | Path | None = None) -> list[DiscoveredClient]:
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    discovered: list[DiscoveredClient] = []

    # Order based on the spec
    all_clients: list[ClientName] = ["omp", "codex", "claude-code", "cursor", "opencode", "pi"]

    for client in all_clients:
        scopes: list[ClientScope] = (
            ["user"] if client in _USER_ONLY_CLIENTS else ["user", "project"]
        )
        for scope in scopes:
            candidates = get_client_config_candidates(client, repo_root, scope)
            found = False
            for candidate in candidates:
                if candidate.exists():
                    discovered.append(
                        DiscoveredClient(
                            client=client,
                            scope=scope,
                            config_path=candidate,
                            is_installed=True,
                            evidence=f"{candidate} exists",
                        )
                    )
                    found = True
                    break
            if not found and candidates:
                # Pick the first one as the default to display if none found
                default_path = candidates[0]
                discovered.append(
                    DiscoveredClient(
                        client=client,
                        scope=scope,
                        config_path=default_path,
                        is_installed=False,
                        evidence=f"{default_path} not found",
                    )
                )

    return discovered


def build_discovered_install_plans(
    discovered: list[DiscoveredClient],
    source: str | Path | None = None,
    tool_scope: str | None = None,
    disclosure: bool = True,
) -> list[ClientInstallPlan]:
    plans: list[ClientInstallPlan] = []
    for d in discovered:
        if d.is_installed:
            s = source if d.scope == "project" else None
            plans.append(
                build_client_install_plan(
                    d.client, s, scope=d.scope, tool_scope=tool_scope, disclosure=disclosure
                )
            )
    return plans


def render_multiple_install_preview(plans: list[ClientInstallPlan]) -> str:
    if not plans:
        return "No client configurations detected to update.\n"

    lines = ["Will write:"]
    for plan in plans:
        if _is_toml_plan(plan):
            lines.append(f"- {plan.target_path}: add [mcp_servers.archex]")
        elif plan.client == "opencode":
            lines.append(f"- {plan.target_path}: add mcp.archex")
        else:
            lines.append(f"- {plan.target_path}: add mcpServers.archex")

    lines.append("\nNo existing non-archex entries will be removed.")
    return "\n".join(lines) + "\n"


def build_client_install_plan(
    client: ClientName,
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
    tool_scope: str | None = None,
    disclosure: bool = True,
) -> ClientInstallPlan:
    selected_scope = _resolve_scope(client, source, scope)
    if client in _USER_ONLY_CLIENTS and selected_scope != "user":
        raise ValueError(f"{client} client config supports only --scope user")
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    target_path = _target_path(client, repo_root, selected_scope)
    content = _render_content(client, tool_scope, disclosure=disclosure)
    return ClientInstallPlan(
        client=client,
        scope=selected_scope,
        target_path=target_path,
        content=content,
        description=_description(client, selected_scope),
        tested_status=_tested_status(client),
        last_verified="2026-06-16",
    )


def write_client_install_plan(plan: ClientInstallPlan) -> Path:
    target = plan.target_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_toml_plan(plan):
        block = plan.content.strip()
        if target.exists():
            existing = target.read_text(encoding="utf-8")
            if "[mcp_servers.archex]" in existing:
                if block in existing:
                    return target
                raise ValueError(f"archex already configured in {target}")
            if existing.strip():
                new_content = existing.rstrip() + "\n\n" + plan.content
            else:
                new_content = plan.content
        else:
            new_content = plan.content
        target.write_text(new_content, encoding="utf-8")
        return target

    key = _json_server_key(plan.client)
    payload_obj: object = json.loads(plan.content)
    if not isinstance(payload_obj, dict):
        raise ValueError(f"expected JSON object in generated content for {plan.client}")
    payload = cast("dict[str, object]", payload_obj)
    if target.exists():
        existing_payload_obj: object = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(existing_payload_obj, dict):
            raise ValueError(f"expected JSON object in {target}")
        existing_payload = cast("dict[str, object]", existing_payload_obj)
    else:
        existing_payload: dict[str, object] = {}
    raw_container = existing_payload.get(key)
    if raw_container is None:
        container: dict[str, object] = {}
        existing_payload[key] = container
    elif isinstance(raw_container, dict):
        container = cast("dict[str, object]", raw_container)
    else:
        raise ValueError(f"expected object at {key} in {target}")
    payload_container_obj = payload.get(key)
    if not isinstance(payload_container_obj, dict):
        raise ValueError(f"expected object at {key} in generated content for {plan.client}")
    payload_container = cast("dict[str, object]", payload_container_obj)
    archex_entry_obj = payload_container.get("archex")
    if not isinstance(archex_entry_obj, dict):
        raise ValueError(f"expected archex entry in generated content for {plan.client}")
    archex_entry = cast("dict[str, object]", archex_entry_obj)
    existing_archex = container.get("archex")
    if existing_archex is not None:
        if existing_archex == archex_entry:
            return target
        raise ValueError(f"archex already configured in {target}")
    container["archex"] = archex_entry
    schema = _CLIENT_SCHEMA.get(plan.client)
    if schema is not None and "$schema" not in existing_payload:
        existing_payload["$schema"] = schema
    target.write_text(json.dumps(existing_payload, indent=2) + "\n", encoding="utf-8")
    return target


def render_client_install_preview(plan: ClientInstallPlan) -> str:
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {plan.target_path}",
        f"Status: {plan.tested_status}",
        f"Last verified: {plan.last_verified}",
        f"Description: {plan.description}",
        "",
        "Dry run. Re-run without --dry-run to write this config.",
        "",
        plan.content.rstrip(),
    ]
    return "\n".join(lines) + "\n"


def render_agent_guidance_block() -> str:
    return f"{AGENT_GUIDANCE_START}\n{AGENT_GUIDANCE_PROMPT}\n{AGENT_GUIDANCE_END}\n"


def append_agent_guidance(agent_file: Path) -> bool:
    """Append the archex MCP guidance block to ``agent_file`` exactly once.

    Returns True if the block was written, False if it was already present.
    The append is non-destructive: existing content is preserved and the block
    is never duplicated on re-run.
    """
    block = render_agent_guidance_block()
    if agent_file.exists():
        existing = agent_file.read_text(encoding="utf-8")
        if AGENT_GUIDANCE_START in existing:
            return False
        new_content = existing.rstrip("\n") + "\n\n" + block if existing.strip() else block
    else:
        agent_file.parent.mkdir(parents=True, exist_ok=True)
        new_content = block
    agent_file.write_text(new_content, encoding="utf-8")
    return True


def render_agent_guidance_preview(agent_file: Path) -> str:
    already_present = agent_file.exists() and AGENT_GUIDANCE_START in agent_file.read_text(
        encoding="utf-8"
    )
    status = (
        "archex MCP guidance already present; no change."
        if already_present
        else "Append archex MCP guidance block (idempotent):"
    )
    return f"Agent file: {agent_file}\n{status}\n\n{render_agent_guidance_block()}"


@dataclass(frozen=True)
class ClaudeCodeHookInstallPlan:
    """Install or remove the Claude Code PreToolUse hook (M19; claude-code only)."""

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    hook_entry: dict[str, object]


@dataclass(frozen=True)
class TsHookInstallPlan:
    """Install or remove the shared TS hook module (M20 omp/pi; M22 opencode).

    Unlike the Claude Code hook (a JSON command entry merged into an existing
    settings file), this installs a standalone TypeScript module — see
    ``_render_ts_hook_module`` for the per-client template dispatch. omp and
    pi share one module (``_TS_HOOK_MODULE_TEMPLATE``, a
    ``pi.on("tool_result", ...)`` handler returning a content patch);
    opencode uses a structurally different one
    (``_OPENCODE_HOOK_MODULE_TEMPLATE``, a ``tool.execute.after`` plugin
    that mutates its output argument in place instead).
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    module_content: str


@dataclass(frozen=True)
class CodexHookInstallPlan:
    """Install or remove the Codex CLI diagnostics-only PreToolUse hook (M21).

    Unlike the Claude Code hook (a JSON command entry merged into
    ``settings.json``) or the omp/pi hook (a standalone ``.ts`` module), this
    appends a marker-delimited TOML block to the *same* ``config.toml`` the
    MCP server registration already writes to (``_target_path`` for
    ``client == "codex"``), mirroring that file's non-destructive append
    behavior for a ``[[hooks.PreToolUse]]`` table instead of
    ``[mcp_servers.archex]``. See ``archex.integrations.codex_hook`` for why
    this ships a diagnostics-only hook rather than Grep/Glob-scoped
    augmentation (Codex has no such tool-call event).
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    block_content: str


@dataclass(frozen=True)
class CursorHookInstallPlan:
    """Install or remove the Cursor ``beforeSubmitPrompt`` hook (M23; prompt-level).

    Unlike the Claude Code hook (matcher-grouped entries under
    ``hooks.PreToolUse``) or the Codex hook (a marker-delimited TOML block
    appended to ``config.toml``), this merges a single-entry array under
    ``hooks.beforeSubmitPrompt`` into a *separate* file, ``hooks.json`` —
    Cursor's own hook config lives apart from ``mcp.json``. Because
    ``beforeSubmitPrompt`` is its own top-level key, distinct from
    ``hooks.beforeReadFile``, "never touches ``beforeReadFile``" is a
    structural property of only ever writing under this one key, rather than
    something a shared matcher regex has to get right. See
    ``archex.integrations.cursor_hook`` for why this ships diagnostics-only
    rather than context injection (Cursor's ``beforeSubmitPrompt`` output
    schema has no context-injection field at all).
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    hook_entry: dict[str, object]


#: Either hook install plan shape. ``install_client_cmd.py`` treats both
#: uniformly; ``write_hook_install_plan``/``render_hook_install_preview``
#: dispatch on the concrete type.
HookInstallPlan = (
    ClaudeCodeHookInstallPlan | TsHookInstallPlan | CodexHookInstallPlan | CursorHookInstallPlan
)


def build_hook_install_plan(
    client: ClientName,
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
    action: HookAction,
) -> HookInstallPlan:
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    if client == "claude-code":
        selected_scope = _resolve_hook_scope(source, scope)
        return ClaudeCodeHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=_hook_settings_path(repo_root, selected_scope),
            action=action,
            hook_entry=_render_hook_entry(),
        )
    if client in {"omp", "pi", "opencode"}:
        selected_scope = _resolve_hook_scope(source, scope)
        return TsHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=_ts_hook_module_path(client, repo_root, selected_scope),
            action=action,
            module_content=_render_ts_hook_module(client),
        )
    if client == "codex":
        selected_scope = _resolve_hook_scope(source, scope)
        return CodexHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=_target_path(client, repo_root, selected_scope),
            action=action,
            block_content=_render_codex_hook_block(),
        )
    if client == "cursor":
        selected_scope = _resolve_hook_scope(source, scope)
        return CursorHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=_cursor_hook_settings_path(repo_root, selected_scope),
            action=action,
            hook_entry=_render_cursor_hook_entry(),
        )
    raise ValueError(
        "--hooks/--remove-hooks is only supported for claude-code (M19), omp, pi (M20), "
        f"codex (M21), opencode (M22), cursor (M23); got {client!r}"
    )


def write_hook_install_plan(plan: HookInstallPlan) -> Path:
    if isinstance(plan, TsHookInstallPlan):
        return _write_ts_hook_plan(plan)
    if isinstance(plan, CodexHookInstallPlan):
        return _write_codex_hook_plan(plan)
    if isinstance(plan, CursorHookInstallPlan):
        return _write_cursor_hook_plan(plan)
    target = plan.target_path
    existing = _read_json_object(target) if target.exists() else {}
    updated, changed = _apply_hook_action(existing, plan)
    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return target


def _write_ts_hook_plan(plan: TsHookInstallPlan) -> Path:
    target = plan.target_path
    if plan.action == "remove":
        if target.exists():
            target.unlink()
        return target
    existing = target.read_text(encoding="utf-8") if target.exists() else None
    if existing != plan.module_content:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(plan.module_content, encoding="utf-8")
    return target


def _write_codex_hook_plan(plan: CodexHookInstallPlan) -> Path:
    target = plan.target_path
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    updated = _apply_codex_hook_block(existing, plan)
    if updated == existing:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(updated, encoding="utf-8")
    return target


def render_hook_install_preview(plan: HookInstallPlan) -> str:
    if isinstance(plan, TsHookInstallPlan):
        return _render_ts_hook_preview(plan)
    if isinstance(plan, CodexHookInstallPlan):
        return _render_codex_hook_preview(plan)
    if isinstance(plan, CursorHookInstallPlan):
        return _render_cursor_hook_preview(plan)
    existing = _read_json_object(plan.target_path) if plan.target_path.exists() else {}
    updated, changed = _apply_hook_action(existing, plan)
    action_label = "Install" if plan.action == "install" else "Remove"
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {plan.target_path}",
        f"Action: {action_label} PreToolUse hook (matcher: {HOOK_MATCHER!r})",
    ]
    if not changed:
        lines.append(
            "No change: hook already in the requested state (idempotent no-op)."
            if plan.action == "install"
            else "No change: no archex hook is installed."
        )
    else:
        lines.append("Dry run. Re-run without --dry-run to write this config.")
    lines.append("")
    lines.append(json.dumps(updated, indent=2))
    return "\n".join(lines) + "\n"


def _render_ts_hook_preview(plan: TsHookInstallPlan) -> str:
    target = plan.target_path
    existing = target.read_text(encoding="utf-8") if target.exists() else None
    action_label = "Install" if plan.action == "install" else "Remove"
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {target}",
        (
            f"Action: {action_label} archex {_ts_hook_event_label(plan.client)} "
            "(grep/glob-equivalent tools only)"
        ),
    ]
    if plan.action == "install":
        lines.append(
            "No change: hook module already installed and up to date (idempotent no-op)."
            if existing == plan.module_content
            else "Dry run. Re-run without --dry-run to write this file."
        )
        lines.append("")
        lines.append(plan.module_content)
    else:
        lines.append(
            "No change: no archex hook module is installed."
            if existing is None
            else "Dry run. Re-run without --dry-run to remove this file."
        )
    return "\n".join(lines) + "\n"


def _render_codex_hook_preview(plan: CodexHookInstallPlan) -> str:
    target = plan.target_path
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    updated = _apply_codex_hook_block(existing, plan)
    action_label = "Install" if plan.action == "install" else "Remove"
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {target}",
        (
            f"Action: {action_label} PreToolUse hook "
            f"(matcher: {CODEX_HOOK_MATCHER!r}, diagnostics-only)"
        ),
    ]
    if updated == existing:
        lines.append(
            "No change: hook already in the requested state (idempotent no-op)."
            if plan.action == "install"
            else "No change: no archex hook is installed."
        )
    else:
        lines.append("Dry run. Re-run without --dry-run to write this config.")
    lines.append("")
    lines.append(updated)
    return "\n".join(lines) + "\n"


def _hook_settings_path(repo_root: Path, scope: ClientScope) -> Path:
    return (
        repo_root / ".claude" / "settings.json"
        if scope == "project"
        else Path.home() / ".claude" / "settings.json"
    )


def _render_hook_entry() -> dict[str, object]:
    return {
        "type": "command",
        "command": sys.executable,
        "args": ["-m", _HOOK_ARGS_MARKER],
    }


_TS_HOOK_MODULE_FILENAME = "archex-hook.ts"


def _ts_hook_module_path(client: ClientName, repo_root: Path, scope: ClientScope) -> Path:
    if client == "omp":
        return (
            repo_root / ".omp" / "extensions" / _TS_HOOK_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".omp" / "agent" / "extensions" / _TS_HOOK_MODULE_FILENAME
        )
    if client == "pi":
        return (
            repo_root / ".pi" / "extensions" / _TS_HOOK_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".pi" / "agent" / "extensions" / _TS_HOOK_MODULE_FILENAME
        )
    if client == "opencode":
        return (
            repo_root / ".opencode" / "plugins" / _TS_HOOK_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".config" / "opencode" / "plugins" / _TS_HOOK_MODULE_FILENAME
        )
    raise ValueError(f"unsupported TS hook client: {client}")


def _ts_hook_event_label(client: ClientName) -> str:
    """Preview wording for the two structurally different TS hook shapes.

    omp/pi install a `tool_result` handler returning a content patch;
    opencode installs a `tool.execute.after` plugin that mutates its output
    argument in place (see `_OPENCODE_HOOK_MODULE_TEMPLATE`).
    """
    return "tool.execute.after plugin" if client == "opencode" else "tool_result hook module"


def _render_ts_hook_module(client: ClientName) -> str:
    template = _OPENCODE_HOOK_MODULE_TEMPLATE if client == "opencode" else _TS_HOOK_MODULE_TEMPLATE
    return template.replace("__ARCHEX_PYTHON_COMMAND__", json.dumps(sys.executable))


_TS_HOOK_MODULE_TEMPLATE = r"""/**
 * archex shared `tool_result` hook module (M20 — oh-my-pi / Pi).
 *
 * Installed by `archex install-client omp --hooks` / `archex install-client
 * pi --hooks` (opt-in; never installed by default). Mirrors the Claude Code
 * PreToolUse hook (M19, `src/archex/integrations/hook.py`) which this module
 * shells out to unmodified — no lookup/ranking/freshness logic lives here.
 *
 * Contract:
 * - Only grep/glob-equivalent tool calls are inspected via
 *   `ARCHEX_QUERY_FIELDS` below. `read` is never touched, and no branch here
 *   ever matches `toolName === "read"` — this must never interfere with
 *   read-before-edit semantics.
 * - Every path resolves without throwing. A missing/stale index, a spawn
 *   failure, a timeout, or a malformed subprocess response all degrade to
 *   returning `undefined` (no content override); failures are appended to
 *   the same diagnostics log the Python subprocess uses
 *   (`ARCHEX_HOOK_DIAGNOSTICS_LOG` or `~/.archex/hook-diagnostics.log`),
 *   never surfaced to the agent flow.
 * - The lookup runs under the same ~500ms wall-clock budget as the Python
 *   hook's own internal timeout; this module additionally guards the
 *   subprocess call itself so a hung `python` process can never block the
 *   host agent past the budget.
 *
 * Shared across oh-my-pi and Pi: both expose an identical
 * `pi.on("tool_result", handler)` extension event with the same
 * `{ content, details, isError }` partial-patch return contract. The one
 * difference between hosts is which native tool name plays the "glob" role
 * (oh-my-pi: `glob`, pattern in `input.path`; Pi: `find`, pattern already in
 * `input.pattern`) — both entries are listed below so this exact file works
 * unmodified on either host: only the tool names that actually exist on the
 * running host will ever fire.
 */

import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// --- Baked in at install time (`archex install-client <client> --hooks`) ---

/** Python interpreter active when `--hooks` ran — mirrors the Claude Code
 * JSON hook's `command`, so this always runs in the same environment archex
 * was installed into. */
const ARCHEX_PYTHON_COMMAND = __ARCHEX_PYTHON_COMMAND__;
const ARCHEX_PYTHON_ARGS = ["-m", "archex.integrations.hook"];

/** Matches `DEFAULT_HOOK_TIMEOUT_SECONDS` in `archex.integrations.hook`. */
const ARCHEX_HOOK_TIMEOUT_MS = 500;

const ARCHEX_DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG";

// --- Native tool name -> archex query-field mapping ---
//
// Claude Code's Grep/Glob tools both carry their query in an input field
// named `pattern` (the subprocess's own contract). Each client's native tool
// names and field names are translated to that shape here, at the edge, so
// `archex.integrations.hook` never needs to know about any client but
// Claude Code.

interface ToolQueryMapping {
  /** `tool_name` value the Python subprocess expects (`Grep` or `Glob`). */
  claudeToolName: "Grep" | "Glob";
  /** Field on the native tool's `input` object holding the query string. */
  field: string;
}

const ARCHEX_QUERY_FIELDS: Readonly<Record<string, ToolQueryMapping>> = {
  grep: { claudeToolName: "Grep", field: "pattern" },
  // oh-my-pi's glob tool carries its glob pattern in `path`.
  glob: { claudeToolName: "Glob", field: "path" },
  // Pi has no `glob` tool; its glob-equivalent is `find`, whose pattern is
  // already in a field named `pattern`.
  find: { claudeToolName: "Glob", field: "pattern" },
};

// --- Minimal structural types for the `tool_result` contract ---
//
// Declared locally (never imported from either host package) so this module
// has zero import-resolution dependency on which host loaded it.

interface ToolResultEventLike {
  toolName: string;
  input?: Record<string, unknown>;
  content?: unknown[];
  details?: unknown;
  isError?: boolean;
}

interface ToolResultPatch {
  content?: unknown[];
  details?: unknown;
  isError?: boolean;
}

type ToolResultHandler = (
  event: ToolResultEventLike,
  ctx: unknown,
) => Promise<ToolResultPatch | undefined>;

interface HookHost {
  on(event: "tool_result", handler: ToolResultHandler): unknown;
}

// --- Diagnostics (parity with hook.py's `log_diagnostic`) ---

function diagnosticsLogPath(): string {
  const override = process.env[ARCHEX_DIAGNOSTICS_LOG_ENV_VAR];
  if (override && override.trim().length > 0) return override;
  return join(homedir(), ".archex", "hook-diagnostics.log");
}

function logDiagnostic(kind: string, detail: string, cwd?: string): void {
  try {
    const path = diagnosticsLogPath();
    mkdirSync(dirname(path), { recursive: true });
    const entry: Record<string, string> = {
      timestamp: new Date().toISOString(),
      kind,
      detail,
    };
    if (cwd) entry.cwd = cwd;
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf-8");
  } catch {
    // Diagnostics logging must never raise into the hook's return path.
  }
}

// --- Subprocess call: `python -m archex.integrations.hook` ---

function runArchexHookSubprocess(
  payload: Record<string, unknown>,
  cwd: string,
): Promise<string | null> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value: string | null): void => {
      if (settled) return;
      settled = true;
      resolve(value);
    };

    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(ARCHEX_PYTHON_COMMAND, ARCHEX_PYTHON_ARGS, {
        cwd,
        stdio: ["pipe", "pipe", "ignore"],
      });
    } catch (err) {
      logDiagnostic("ts_spawn_error", String(err), cwd);
      finish(null);
      return;
    }

    const timer = setTimeout(() => {
      logDiagnostic("ts_timeout", `lookup exceeded ${ARCHEX_HOOK_TIMEOUT_MS}ms`, cwd);
      try {
        child.kill("SIGKILL");
      } catch {
        // Already exited.
      }
      finish(null);
    }, ARCHEX_HOOK_TIMEOUT_MS);

    let stdout = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      logDiagnostic("ts_spawn_error", String(err), cwd);
      finish(null);
    });
    child.on("close", () => {
      clearTimeout(timer);
      finish(stdout.length > 0 ? stdout : null);
    });

    try {
      child.stdin?.write(JSON.stringify(payload));
      child.stdin?.end();
    } catch (err) {
      clearTimeout(timer);
      logDiagnostic("ts_stdin_error", String(err), cwd);
      finish(null);
    }
  });
}

function extractAdditionalContext(rawStdout: string): string | null {
  try {
    const parsed: unknown = JSON.parse(rawStdout);
    if (typeof parsed !== "object" || parsed === null) return null;
    const hookSpecificOutput = (parsed as Record<string, unknown>).hookSpecificOutput;
    if (typeof hookSpecificOutput !== "object" || hookSpecificOutput === null) return null;
    const context = (hookSpecificOutput as Record<string, unknown>).additionalContext;
    return typeof context === "string" && context.length > 0 ? context : null;
  } catch {
    return null;
  }
}

// --- Extension entry point ---

export default function archexHook(pi: HookHost): void {
  pi.on("tool_result", async (event) => {
    try {
      const mapping = ARCHEX_QUERY_FIELDS[event.toolName];
      if (!mapping) return undefined; // never touches "read" or any other tool

      const pattern = event.input?.[mapping.field];
      if (typeof pattern !== "string" || pattern.trim().length === 0) return undefined;

      const cwd = process.cwd();
      const rawStdout = await runArchexHookSubprocess(
        { tool_name: mapping.claudeToolName, tool_input: { pattern }, cwd },
        cwd,
      );
      if (rawStdout === null) return undefined;

      const context = extractAdditionalContext(rawStdout);
      if (context === null) return undefined;

      const existingContent = Array.isArray(event.content) ? event.content : [];
      return {
        content: [...existingContent, { type: "text", text: `\n\n${context}` }],
      };
    } catch (err) {
      logDiagnostic("ts_internal_error", String(err));
      return undefined;
    }
  });
}
"""


_OPENCODE_HOOK_MODULE_TEMPLATE = r"""/**
 * archex OpenCode `tool.execute.after` plugin (M22 — OpenCode hook integration).
 *
 * Installed by `archex install-client opencode --hooks` (opt-in; never
 * installed by default) as a standalone plugin file OpenCode auto-loads from
 * its native plugin directory -- `.opencode/plugins/archex-hook.ts`
 * (project-local) or `~/.config/opencode/plugins/archex-hook.ts` (global).
 * No `opencode.json` entry is required: per OpenCode's own docs, files in
 * these directories "are automatically loaded at startup."
 *
 * Mirrors the oh-my-pi/Pi `tool_result` hook (M20, `_TS_HOOK_MODULE_TEMPLATE`)
 * which this module shells out to unmodified -- no lookup/ranking/freshness
 * logic lives here, only the `python -m archex.integrations.hook` subprocess
 * contract from M19.
 *
 * Contract:
 * - Only OpenCode's native `grep` and `glob` tools are inspected
 *   (`ARCHEX_AUGMENTED_TOOLS`, this module's only tool-name dispatch).
 *   `read` is never touched, and an MCP-routed tool call can never match
 *   this table: OpenCode registers every MCP tool under a mandatory
 *   `{server}_{tool}` id (confirmed against the installed `opencode-ai`
 *   1.14.33's own MCP tool-registration code), so an exact `"grep"`/`"glob"`
 *   collision with an MCP tool id is structurally impossible, not merely
 *   unlikely.
 * - `tool.execute.after`'s contract differs structurally from oh-my-pi/Pi's
 *   `tool_result`: its handler signature is `(input, output) => Promise<void>`
 *   -- it mutates the `output.output` string IN PLACE rather than returning
 *   a patch object. A degraded path (spawn failure, timeout, malformed
 *   subprocess response, or any thrown error) simply returns without
 *   touching `output`, leaving the native tool's own result untouched.
 * - Every path resolves without throwing past this handler. A missing/stale
 *   index, a spawn failure, a timeout, or a malformed subprocess response
 *   all degrade to leaving `output` untouched; failures are appended to the
 *   same diagnostics log the Python subprocess and the M20 TS module use
 *   (`ARCHEX_HOOK_DIAGNOSTICS_LOG` or `~/.archex/hook-diagnostics.log`),
 *   never surfaced to the agent flow.
 * - The lookup runs under the same ~500ms wall-clock budget as the Python
 *   hook's own internal timeout; this module additionally guards the
 *   subprocess call itself so a hung `python` process can never block the
 *   host agent past the budget.
 *
 * Two OpenCode-side reliability gaps this milestone's own tests assert
 * against rather than assume away (`.docs/DEVELOPMENT_PLAN.md` §2), both
 * confirmed by reading `opencode-ai` 1.14.33's own tool-resolution source
 * (the version installed during development), not secondary documentation:
 * - MCP tool calls DO trigger `tool.execute.after`, but the hook receives
 *   the tool's raw MCP `CallToolResult` as `output` (a `{content, metadata}`
 *   shape), not the `{title, output, metadata}` shape this type declares --
 *   the text actually sent to the model is rebuilt from `result.content`
 *   AFTER the hook runs, discarding any `output.output` mutation. Moot for
 *   this plugin: its dispatch table never contains an MCP-shaped tool id.
 * - A Task-tool-spawned subagent's own turn is processed by the exact same
 *   tool-resolution code path as a top-level turn (the subagent's prompt
 *   loop is a recursive call into the identical function that built the
 *   top-level session's own tool table), so a subagent-issued `grep`/`glob`
 *   call triggers `tool.execute.after` identically to a top-level one in
 *   the version this was verified against. This module itself makes no
 *   session/agent distinction either way -- see the installer test suite
 *   for the specific structural check and its citation.
 */

import type { Plugin } from "@opencode-ai/plugin";
import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// --- Baked in at install time (`archex install-client opencode --hooks`) ---

/** Python interpreter active when `--hooks` ran -- mirrors the Claude Code
 * JSON hook's `command`, so this always runs in the same environment archex
 * was installed into. */
const ARCHEX_PYTHON_COMMAND = __ARCHEX_PYTHON_COMMAND__;
const ARCHEX_PYTHON_ARGS = ["-m", "archex.integrations.hook"];

/** Matches `DEFAULT_HOOK_TIMEOUT_SECONDS` in `archex.integrations.hook`. */
const ARCHEX_HOOK_TIMEOUT_MS = 500;

const ARCHEX_DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG";

// --- OpenCode native tool id -> archex subprocess Claude-shape tool_name ---
//
// OpenCode's `grep` and `glob` tools both carry their query pattern in an
// `args.pattern` field (confirmed against the bundled tool definitions) --
// both translate directly onto the subprocess's existing
// `{"tool_name": "Grep"|"Glob", "tool_input": {"pattern": ...}}` contract.
// This table is this module's *only* tool-name dispatch: `read`, every
// other native tool, and every MCP-routed tool id fall through unmatched.
const ARCHEX_AUGMENTED_TOOLS: Readonly<Record<string, "Grep" | "Glob">> = {
  grep: "Grep",
  glob: "Glob",
};

// --- Diagnostics (parity with hook.py's `log_diagnostic`) ---

function logDiagnostic(kind: string, detail: string, cwd?: string): void {
  try {
    const override = process.env[ARCHEX_DIAGNOSTICS_LOG_ENV_VAR];
    const path = override && override.trim().length > 0
      ? override
      : join(homedir(), ".archex", "hook-diagnostics.log");
    mkdirSync(dirname(path), { recursive: true });
    const entry: Record<string, string> = {
      timestamp: new Date().toISOString(),
      kind,
      detail,
    };
    if (cwd) entry.cwd = cwd;
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf-8");
  } catch {
    // Diagnostics logging must never raise into the hook's return path.
  }
}

// --- Subprocess call: `python -m archex.integrations.hook` ---

function runArchexHookSubprocess(
  payload: Record<string, unknown>,
  cwd: string,
): Promise<string | null> {
  const { promise, resolve } = Promise.withResolvers<string | null>();
  let settled = false;
  const finish = (value: string | null): void => {
    if (settled) return;
    settled = true;
    resolve(value);
  };

  let child: ChildProcess;
  try {
    child = spawn(ARCHEX_PYTHON_COMMAND, ARCHEX_PYTHON_ARGS, {
      cwd,
      stdio: ["pipe", "pipe", "ignore"],
    });
  } catch (err) {
    logDiagnostic("ts_spawn_error", String(err), cwd);
    finish(null);
    return promise;
  }

  const timer = setTimeout(() => {
    logDiagnostic("ts_timeout", `lookup exceeded ${ARCHEX_HOOK_TIMEOUT_MS}ms`, cwd);
    try {
      child.kill("SIGKILL");
    } catch {
      // Already exited.
    }
    finish(null);
  }, ARCHEX_HOOK_TIMEOUT_MS);

  let stdout = "";
  child.stdout?.on("data", (chunk: Buffer) => {
    stdout += chunk.toString("utf-8");
  });
  child.on("error", (err) => {
    clearTimeout(timer);
    logDiagnostic("ts_spawn_error", String(err), cwd);
    finish(null);
  });
  child.on("close", () => {
    clearTimeout(timer);
    finish(stdout.length > 0 ? stdout : null);
  });

  try {
    child.stdin?.write(JSON.stringify(payload));
    child.stdin?.end();
  } catch (err) {
    clearTimeout(timer);
    logDiagnostic("ts_stdin_error", String(err), cwd);
    finish(null);
  }

  return promise;
}

function extractAdditionalContext(rawStdout: string): string | null {
  try {
    const parsed: unknown = JSON.parse(rawStdout);
    if (typeof parsed !== "object" || parsed === null) return null;
    const hookSpecificOutput = (parsed as Record<string, unknown>).hookSpecificOutput;
    if (typeof hookSpecificOutput !== "object" || hookSpecificOutput === null) return null;
    const context = (hookSpecificOutput as Record<string, unknown>).additionalContext;
    return typeof context === "string" && context.length > 0 ? context : null;
  } catch {
    return null;
  }
}

// --- Plugin entry point ---

export const ArchexHookPlugin: Plugin = async ({ directory }) => {
  return {
    "tool.execute.after": async (input, output) => {
      try {
        const claudeToolName = ARCHEX_AUGMENTED_TOOLS[input.tool];
        if (!claudeToolName) return; // never "read", never an MCP-routed tool

        const args = (input.args ?? {}) as Record<string, unknown>;
        const pattern = args.pattern;
        if (typeof pattern !== "string" || pattern.trim().length === 0) return;

        const rawStdout = await runArchexHookSubprocess(
          { tool_name: claudeToolName, tool_input: { pattern }, cwd: directory },
          directory,
        );
        if (rawStdout === null) return;

        const context = extractAdditionalContext(rawStdout);
        if (context === null) return;

        output.output = `${output.output}\n\n${context}`;
      } catch (err) {
        logDiagnostic("ts_internal_error", String(err));
      }
    },
  };
};
"""


#: Marker comments delimiting the archex-owned block appended to Codex's
#: ``config.toml`` -- lets install/remove find and replace exactly the block
#: this installer wrote (and nothing else a user configured in the same
#: file) without parsing/re-serializing TOML.
_CODEX_HOOK_BLOCK_START = "# archex:codex-hook start"
_CODEX_HOOK_BLOCK_END = "# archex:codex-hook end"


def _render_codex_hook_block() -> str:
    command = f"{sys.executable} -m archex.integrations.codex_hook"
    return (
        "\n".join(
            [
                _CODEX_HOOK_BLOCK_START,
                "[[hooks.PreToolUse]]",
                f'matcher = "{CODEX_HOOK_MATCHER}"',
                "",
                "[[hooks.PreToolUse.hooks]]",
                'type = "command"',
                f'command = "{command}"',
                "timeout = 1",
                _CODEX_HOOK_BLOCK_END,
            ]
        )
        + "\n"
    )


def _strip_codex_hook_block(existing: str) -> str:
    start = existing.find(_CODEX_HOOK_BLOCK_START)
    if start == -1:
        return existing
    end = existing.find(_CODEX_HOOK_BLOCK_END, start)
    if end == -1:
        return existing  # malformed marker pair -- leave untouched rather than guess
    end += len(_CODEX_HOOK_BLOCK_END)
    if end < len(existing) and existing[end] == "\n":
        end += 1
    before, after = existing[:start], existing[end:]
    # `_render_codex_hook_block`/`_apply_codex_hook_block` always separate a
    # freshly appended block from prior content with exactly one blank line
    # -- strip that same separator back out so a strip+re-add round-trips
    # byte-for-byte (idempotent reinstall, and a clean `remove`).
    if before.endswith("\n\n"):
        before = before[:-1]
    return before + after


def _apply_codex_hook_block(existing: str, plan: CodexHookInstallPlan) -> str:
    without_block = _strip_codex_hook_block(existing)
    if plan.action == "remove":
        return without_block
    if not without_block.strip():
        return plan.block_content
    return without_block.rstrip("\n") + "\n\n" + plan.block_content


#: Substring in a Cursor hook entry's ``command`` string that identifies it
#: as archex-owned, so install/remove can find and replace our own entry
#: without disturbing any other ``beforeSubmitPrompt`` hook the user has
#: configured in the same file.
_CURSOR_HOOK_COMMAND_MARKER = "archex.integrations.cursor_hook"


def _cursor_hook_settings_path(repo_root: Path, scope: ClientScope) -> Path:
    return (
        repo_root / ".cursor" / "hooks.json"
        if scope == "project"
        else Path.home() / ".cursor" / "hooks.json"
    )


def _render_cursor_hook_entry() -> dict[str, object]:
    return {
        "command": f"{sys.executable} -m archex.integrations.cursor_hook",
        "timeout": 1,
    }


def _is_archex_cursor_hook_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    command = cast("dict[str, object]", entry).get("command")
    return isinstance(command, str) and _CURSOR_HOOK_COMMAND_MARKER in command


def _apply_cursor_hook_action(
    payload: dict[str, object], plan: CursorHookInstallPlan
) -> tuple[dict[str, object], bool]:
    """Return ``(updated_payload, changed)`` without mutating ``payload``.

    Both install and remove start by stripping every archex-owned entry out
    of ``hooks.beforeSubmitPrompt``, then install re-adds exactly one
    canonical entry -- a second install (even after ``sys.executable``
    changed, e.g. a venv move) converges on the same representation instead
    of accumulating duplicates. This never reads or writes any other key
    under ``hooks`` (in particular, never ``hooks.beforeReadFile``).
    """
    updated = copy.deepcopy(payload)
    if plan.action == "install":
        updated.setdefault("version", 1)
    hooks_root_obj = updated.get("hooks")
    if hooks_root_obj is None:
        hooks_root: dict[str, object] = {}
        updated["hooks"] = hooks_root
    elif isinstance(hooks_root_obj, dict):
        hooks_root = cast("dict[str, object]", hooks_root_obj)
    else:
        raise ValueError("expected object at 'hooks' in existing hooks.json")

    before_submit_obj = hooks_root.get("beforeSubmitPrompt")
    if before_submit_obj is None:
        before_submit: list[object] = []
    elif isinstance(before_submit_obj, list):
        before_submit = cast("list[object]", before_submit_obj)
    else:
        raise ValueError("expected array at 'hooks.beforeSubmitPrompt' in existing hooks.json")

    kept = [entry for entry in before_submit if not _is_archex_cursor_hook_entry(entry)]
    merged = [*kept, plan.hook_entry] if plan.action == "install" else kept

    if merged:
        hooks_root["beforeSubmitPrompt"] = merged
    else:
        hooks_root.pop("beforeSubmitPrompt", None)
    if not hooks_root:
        updated.pop("hooks", None)

    return updated, updated != payload


def _write_cursor_hook_plan(plan: CursorHookInstallPlan) -> Path:
    target = plan.target_path
    existing = _read_json_object(target) if target.exists() else {}
    updated, changed = _apply_cursor_hook_action(existing, plan)
    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return target


def _render_cursor_hook_preview(plan: CursorHookInstallPlan) -> str:
    existing = _read_json_object(plan.target_path) if plan.target_path.exists() else {}
    updated, changed = _apply_cursor_hook_action(existing, plan)
    action_label = "Install" if plan.action == "install" else "Remove"
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {plan.target_path}",
        f"Action: {action_label} beforeSubmitPrompt hook (prompt-level, diagnostics-only)",
    ]
    if not changed:
        lines.append(
            "No change: hook already in the requested state (idempotent no-op)."
            if plan.action == "install"
            else "No change: no archex hook is installed."
        )
    else:
        lines.append("Dry run. Re-run without --dry-run to write this config.")
    lines.append("")
    lines.append(json.dumps(updated, indent=2))
    return "\n".join(lines) + "\n"


def _is_archex_hook_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    args = cast("dict[str, object]", entry).get("args")
    if not isinstance(args, list):
        return False
    items = cast("list[object]", args)
    return any(isinstance(item, str) and _HOOK_ARGS_MARKER in item for item in items)


def _read_json_object(path: Path) -> dict[str, object]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast("dict[str, object]", payload_obj)


def _apply_hook_action(
    payload: dict[str, object], plan: ClaudeCodeHookInstallPlan
) -> tuple[dict[str, object], bool]:
    """Return ``(updated_payload, changed)`` without mutating ``payload``.

    Both install and remove start by stripping every archex-owned hook entry
    (identified by ``_HOOK_ARGS_MARKER``) out of ``hooks.PreToolUse``, then
    install re-adds exactly one canonical entry. That makes a second install
    (even after ``sys.executable`` changed, e.g. a venv move) converge on the
    same representation instead of accumulating duplicates, and never touches
    hook entries archex does not own.
    """
    updated = copy.deepcopy(payload)
    hooks_root_obj = updated.get("hooks")
    if hooks_root_obj is None:
        hooks_root: dict[str, object] = {}
        updated["hooks"] = hooks_root
    elif isinstance(hooks_root_obj, dict):
        hooks_root = cast("dict[str, object]", hooks_root_obj)
    else:
        raise ValueError("expected object at 'hooks' in existing settings")

    pre_tool_use_obj = hooks_root.get("PreToolUse")
    if pre_tool_use_obj is None:
        pre_tool_use: list[object] = []
    elif isinstance(pre_tool_use_obj, list):
        pre_tool_use = cast("list[object]", pre_tool_use_obj)
    else:
        raise ValueError("expected array at 'hooks.PreToolUse' in existing settings")

    stripped_groups = _strip_archex_hook_entries(pre_tool_use)
    merged_groups = (
        _merge_hook_entry(stripped_groups, plan.hook_entry)
        if plan.action == "install"
        else stripped_groups
    )

    if merged_groups:
        hooks_root["PreToolUse"] = merged_groups
    else:
        hooks_root.pop("PreToolUse", None)
    if not hooks_root:
        updated.pop("hooks", None)

    return updated, updated != payload


def _strip_archex_hook_entries(groups: list[object]) -> list[object]:
    stripped: list[object] = []
    for group in groups:
        if not isinstance(group, dict):
            stripped.append(group)
            continue
        group_dict = cast("dict[str, object]", group)
        handlers_obj = group_dict.get("hooks")
        if not isinstance(handlers_obj, list):
            stripped.append(group_dict)
            continue
        handlers = cast("list[object]", handlers_obj)
        kept = [h for h in handlers if not _is_archex_hook_entry(h)]
        if not kept:
            continue  # the group only ever held our own entry
        stripped.append({**group_dict, "hooks": kept} if len(kept) != len(handlers) else group_dict)
    return stripped


def _merge_hook_entry(groups: list[object], hook_entry: dict[str, object]) -> list[object]:
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_dict = cast("dict[str, object]", group)
        if group_dict.get("matcher") == HOOK_MATCHER:
            handlers_obj = group_dict.get("hooks")
            handlers = cast("list[object]", handlers_obj) if isinstance(handlers_obj, list) else []
            group_dict["hooks"] = [*handlers, hook_entry]
            return groups
    return [*groups, {"matcher": HOOK_MATCHER, "hooks": [hook_entry]}]


def _resolve_scope(
    client: ClientName, source: str | Path | None, scope: ClientScope | None
) -> ClientScope:
    if scope is not None:
        return scope
    if client in _USER_ONLY_CLIENTS:
        return "user"
    return "project" if source is not None else "user"


def _resolve_hook_scope(source: str | Path | None, scope: ClientScope | None) -> ClientScope:
    """Scope resolution for hook installs (claude-code, omp, pi).

    Unlike ``_resolve_scope`` (MCP server config, where omp/pi are user-scope
    only per the ``CLIENT_COMPATIBILITY_MATRIX.md`` convention), the hook
    installer supports both scopes for every client it handles: project scope
    when a repo ``source`` is given, user scope otherwise.
    """
    if scope is not None:
        return scope
    return "project" if source is not None else "user"


def _target_path(client: ClientName, repo_root: Path, scope: ClientScope) -> Path:
    home = Path.home()
    if client == "claude-code":
        return repo_root / ".mcp.json" if scope == "project" else home / ".claude.json"
    if client == "cursor":
        return (
            repo_root / ".cursor" / "mcp.json"
            if scope == "project"
            else home / ".cursor" / "mcp.json"
        )
    if client == "opencode":
        return (
            repo_root / "opencode.json"
            if scope == "project"
            else home / ".config" / "opencode" / "opencode.json"
        )
    if client == "codex":
        return (
            repo_root / ".codex" / "config.toml"
            if scope == "project"
            else home / ".codex" / "config.toml"
        )
    if client == "pi":
        return home / ".pi" / "agent" / "mcp.json"
    if client == "omp":
        return home / ".omp" / "agent" / "mcp.json"
    raise ValueError(f"unsupported client: {client}")


def _mcp_args(tool_scope: str | None, *, disclosure: bool = True) -> list[str]:
    """CLI args for the `archex mcp` server command.

    `None` preserves the existing unscoped `["mcp"]` args exactly (backward
    compatible with every config archex has ever written). A non-`None`
    scope is validated via `resolve_tool_scope` before being embedded --
    an unknown tool name in `tool_scope` fails at install time, not
    silently inside a client's own MCP server subprocess.

    `disclosure=False` writes `--no-disclosure`, which is the compatibility
    path for a client that cannot re-fetch its tool list: it pays the full
    schema cost every turn but sees every tool from the first `list_tools()`.
    The default is left implicit rather than written as `--disclosure`, so
    configs stay byte-identical to the ones archex already wrote.
    """
    args = ["mcp"]
    if tool_scope is not None and resolve_tool_scope(tool_scope) is not None:
        args += ["--tools", tool_scope]
    if not disclosure:
        args.append("--no-disclosure")
    return args


def _render_content(
    client: ClientName, tool_scope: str | None = None, *, disclosure: bool = True
) -> str:
    args = _mcp_args(tool_scope, disclosure=disclosure)
    if client == "codex":
        return f'[mcp_servers.archex]\ncommand = "archex"\nargs = {json.dumps(args)}\n'
    if client == "opencode":
        payload = {
            "$schema": _OPENCODE_SCHEMA,
            "mcp": {
                "archex": {
                    "type": "local",
                    "command": ["archex", *args],
                    "enabled": True,
                }
            },
        }
        return json.dumps(payload, indent=2) + "\n"
    if client == "pi":
        payload = {
            "mcpServers": {
                "archex": {
                    "command": "archex",
                    "args": args,
                }
            }
        }
        return json.dumps(payload, indent=2) + "\n"
    if client == "omp":
        payload = {
            "$schema": _OMP_SCHEMA,
            "mcpServers": {
                "archex": {
                    "command": "archex",
                    "args": args,
                }
            },
        }
        return json.dumps(payload, indent=2) + "\n"
    payload = {
        "mcpServers": {
            "archex": {
                "command": "archex",
                "args": args,
            }
        }
    }
    return json.dumps(payload, indent=2) + "\n"


def _json_server_key(client: ClientName) -> str:
    return "mcp" if client == "opencode" else "mcpServers"


def _is_toml_plan(plan: ClientInstallPlan) -> bool:
    return plan.client == "codex"


def _description(client: ClientName, scope: ClientScope) -> str:
    if client == "codex":
        return "Codex CLI config.toml MCP server registration"
    if client == "opencode":
        return f"OpenCode {'project' if scope == 'project' else 'user'} config"
    if client == "pi":
        return "Pi agent MCP config"
    if client == "omp":
        return "oh-my-pi agent MCP config"
    if client == "cursor":
        return f"Cursor {'project' if scope == 'project' else 'user'} MCP config"
    return f"Claude Code {'project' if scope == 'project' else 'user'} MCP config"


def _tested_status(client: ClientName) -> str:
    if client == "claude-code":
        return "config-path tested; client smoke unverified"
    if client == "cursor":
        return "config-shape verified; client smoke unverified"
    if client == "opencode":
        return "config-shape verified; client smoke unverified"
    if client == "codex":
        return "unverified client smoke"
    if client == "omp":
        return "config-shape verified; client smoke unverified"
    return "config-shape verified; client smoke unverified"
