"""Client install/bootstrap helpers for MCP-compatible archex setups."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from archex.integrations.codex_hook import HOOK_MATCHER as CODEX_HOOK_MATCHER
from archex.integrations.codex_post_edit_hook import POST_EDIT_MATCHER as CODEX_POST_EDIT_MATCHER
from archex.integrations.hook import HOOK_MATCHER
from archex.integrations.mcp import resolve_tool_scope
from archex.integrations.post_edit_hook import POST_EDIT_MATCHER
from archex.integrations.session_hook import SESSION_START_MATCHER
from archex.project import PROJECT_DIR_NAME
from archex.status_snapshot import (
    DEFAULT_STALE_AFTER_SECONDS,
    SNAPSHOT_FILENAME,
    STATUS_SNAPSHOT_VERSION,
    WATCH_OBSERVATION_TTL_SECONDS,
)

ClientName = Literal["claude-code", "codex", "cursor", "opencode", "pi", "omp"]
ClientScope = Literal["project", "user"]

_USER_ONLY_CLIENTS: frozenset[ClientName] = frozenset({"pi", "omp"})

HookAction = Literal["install", "remove"]

#: Substring in a hook handler's ``args`` that identifies it as archex-owned,
#: so install/remove can find and replace our own entry without disturbing any
#: other hook the user has configured for the same matcher group.
_HOOK_ARGS_MARKER = "archex.integrations.hook"

#: Substring in a SessionStart handler's ``args`` that identifies it as
#: archex-owned, so the session-primer installer can preserve existing search
#: hooks and unrelated SessionStart handlers.
_SESSION_PRIMER_ARGS_MARKER = "archex.integrations.session_hook"
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


@dataclass(frozen=True)
class ClaudeCodeSessionPrimerInstallPlan:
    """Install or remove the opt-in Claude Code SessionStart primer hook."""

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


def build_session_primer_install_plan(
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
    action: HookAction,
) -> ClaudeCodeSessionPrimerInstallPlan:
    """Build a separate opt-in Claude Code SessionStart primer install plan."""
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    selected_scope = _resolve_hook_scope(source, scope)
    return ClaudeCodeSessionPrimerInstallPlan(
        client="claude-code",
        scope=selected_scope,
        target_path=_hook_settings_path(repo_root, selected_scope),
        action=action,
        hook_entry=_render_session_primer_hook_entry(),
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


def write_session_primer_install_plan(plan: ClaudeCodeSessionPrimerInstallPlan) -> Path:
    """Write a SessionStart primer plan without disturbing other settings."""
    target = plan.target_path
    existing = _read_json_object(target) if target.exists() else {}
    updated, changed = _apply_session_primer_action(existing, plan)
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


def render_session_primer_install_preview(plan: ClaudeCodeSessionPrimerInstallPlan) -> str:
    """Render a no-write preview for the SessionStart primer installer."""
    existing = _read_json_object(plan.target_path) if plan.target_path.exists() else {}
    updated, changed = _apply_session_primer_action(existing, plan)
    action_label = "Install" if plan.action == "install" else "Remove"
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {plan.target_path}",
        (
            "Action: "
            f"{action_label} SessionStart session-primer hook "
            f"(matcher: {SESSION_START_MATCHER!r})"
        ),
    ]
    if not changed:
        lines.append(
            "No change: session-primer hook already in the requested state (idempotent no-op)."
            if plan.action == "install"
            else "No change: no archex session-primer hook is installed."
        )
    else:
        lines.append("Dry run. Re-run without --dry-run to write this config.")
    lines.extend(["", json.dumps(updated, indent=2)])
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


def _render_session_primer_hook_entry() -> dict[str, object]:
    return {
        "type": "command",
        "command": sys.executable,
        "args": ["-m", _SESSION_PRIMER_ARGS_MARKER],
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
 * against rather than assume away, both confirmed by reading `opencode-ai`
 * 1.14.33's own tool-resolution source (the version installed during
 * development), not secondary documentation:
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


def _is_archex_hook_entry(entry: object, marker: str = _HOOK_ARGS_MARKER) -> bool:
    if not isinstance(entry, dict):
        return False
    args = cast("dict[str, object]", entry).get("args")
    if not isinstance(args, list):
        return False
    items = cast("list[object]", args)
    return any(isinstance(item, str) and marker in item for item in items)


def _read_json_object(path: Path) -> dict[str, object]:
    payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast("dict[str, object]", payload_obj)


def _apply_claude_event_hook_action(
    payload: dict[str, object],
    *,
    event: str,
    marker: str,
    matcher: str,
    hook_entry: dict[str, object],
    action: HookAction,
) -> tuple[dict[str, object], bool]:
    """Return ``(updated_payload, changed)`` without mutating ``payload``.

    Both install and remove start by stripping every archex-owned handler
    (identified by ``marker``) out of ``hooks.<event>``, then install re-adds
    exactly one canonical entry. That makes a second install (even after
    ``sys.executable`` changed, e.g. a venv move) converge on the same
    representation instead of accumulating duplicates, and never touches
    hook entries archex does not own. Each archex hook owns a distinct
    ``event``/``marker`` pair, so installing one never disturbs another.
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

    group_obj = hooks_root.get(event)
    if group_obj is None:
        groups: list[object] = []
    elif isinstance(group_obj, list):
        groups = cast("list[object]", group_obj)
    else:
        raise ValueError(f"expected array at 'hooks.{event}' in existing settings")

    stripped_groups = _strip_archex_hook_entries(groups, marker=marker)
    merged_groups = (
        _merge_hook_entry(stripped_groups, hook_entry, matcher=matcher)
        if action == "install"
        else stripped_groups
    )

    if merged_groups:
        hooks_root[event] = merged_groups
    else:
        hooks_root.pop(event, None)
    if not hooks_root:
        updated.pop("hooks", None)

    return updated, updated != payload


def _apply_hook_action(
    payload: dict[str, object], plan: ClaudeCodeHookInstallPlan
) -> tuple[dict[str, object], bool]:
    """Merge one owned PreToolUse search handler into Claude Code settings."""
    return _apply_claude_event_hook_action(
        payload,
        event="PreToolUse",
        marker=_HOOK_ARGS_MARKER,
        matcher=HOOK_MATCHER,
        hook_entry=plan.hook_entry,
        action=plan.action,
    )


def _apply_session_primer_action(
    payload: dict[str, object], plan: ClaudeCodeSessionPrimerInstallPlan
) -> tuple[dict[str, object], bool]:
    """Merge one owned SessionStart handler without touching other settings."""
    return _apply_claude_event_hook_action(
        payload,
        event="SessionStart",
        marker=_SESSION_PRIMER_ARGS_MARKER,
        matcher=SESSION_START_MATCHER,
        hook_entry=plan.hook_entry,
        action=plan.action,
    )


def _strip_archex_hook_entries(
    groups: list[object], *, marker: str = _HOOK_ARGS_MARKER
) -> list[object]:
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
        kept = [h for h in handlers if not _is_archex_hook_entry(h, marker)]
        if not kept:
            continue  # the group only ever held our own entry
        stripped.append({**group_dict, "hooks": kept} if len(kept) != len(handlers) else group_dict)
    return stripped


def _merge_hook_entry(
    groups: list[object], hook_entry: dict[str, object], *, matcher: str = HOOK_MATCHER
) -> list[object]:
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_dict = cast("dict[str, object]", group)
        if group_dict.get("matcher") == matcher:
            handlers_obj = group_dict.get("hooks")
            handlers = cast("list[object]", handlers_obj) if isinstance(handlers_obj, list) else []
            group_dict["hooks"] = [*handlers, hook_entry]
            return groups
    return [*groups, {"matcher": matcher, "hooks": [hook_entry]}]


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


# ---------------------------------------------------------------------------
# R21 — post-edit impact hooks
#
# A separate, independently installable and removable surface from the
# PreToolUse search hook above. It owns its own event, its own ownership
# marker, and its own module filename on every client, so installing or
# removing one never disturbs the other.
#
# Client dispositions are evidence-based, not aspirational:
#   claude-code  PostToolUse, matcher `Edit|Write`, additionalContext output
#   codex        PostToolUse, matcher `^apply_patch$`, additionalContext output
#   omp / pi     tool_result on `edit`/`write`, content patch
#   opencode     tool.execute.after on `edit`/`write`, output.output mutation
#   cursor       UNSUPPORTED — see `POST_EDIT_UNSUPPORTED_CLIENTS`
# ---------------------------------------------------------------------------

#: Substring in a PostToolUse handler's ``args`` identifying it as the
#: archex post-edit hook. Distinct from ``_HOOK_ARGS_MARKER`` (and not a
#: superstring of it), so the two installers cannot strip each other.
_POST_EDIT_ARGS_MARKER = "archex.integrations.post_edit_hook"
_CODEX_POST_EDIT_ARGS_MARKER = "archex.integrations.codex_post_edit_hook"

_CODEX_POST_EDIT_BLOCK_START = "# archex:codex-post-edit-hook start"
_CODEX_POST_EDIT_BLOCK_END = "# archex:codex-post-edit-hook end"

_TS_POST_EDIT_MODULE_FILENAME = "archex-post-edit-hook.ts"

#: Clients with no post-edit adapter, and the upstream reason. Surfaced by
#: the CLI so an unsupported client gets an explicit refusal rather than a
#: silent no-op or a fabricated success message.
POST_EDIT_UNSUPPORTED_CLIENTS: dict[ClientName, str] = {
    "cursor": (
        "Cursor's afterFileEdit is the only event carrying an edited file path "
        "and its documented schema has no output fields at all, so it cannot "
        "return impact to the agent; postToolUse does document an "
        "additional_context output but documents no path field for its Write "
        "tool input, so edited paths cannot be extracted from it. Archex will "
        "not ship a post-edit adapter that cannot be shown to work."
    ),
}


@dataclass(frozen=True)
class ClaudeCodePostEditHookInstallPlan:
    """Install or remove the Claude Code PostToolUse post-edit hook (R21)."""

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    hook_entry: dict[str, object]


@dataclass(frozen=True)
class CodexPostEditHookInstallPlan:
    """Install or remove the Codex CLI PostToolUse post-edit hook (R21).

    Appends a marker-delimited ``[[hooks.PostToolUse]]`` block to the same
    ``config.toml`` the MCP registration and the M21 ``PreToolUse`` block
    already write to, using its own marker pair so the three coexist.
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    block_content: str


@dataclass(frozen=True)
class TsPostEditHookInstallPlan:
    """Install or remove the standalone TS post-edit module (R21).

    omp and pi share a ``tool_result`` module; opencode uses a structurally
    different ``tool.execute.after`` plugin that mutates its output argument
    in place. Both are written under a filename distinct from the search
    hook's, so the two surfaces install and remove independently.
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    module_content: str


PostEditHookInstallPlan = (
    ClaudeCodePostEditHookInstallPlan | CodexPostEditHookInstallPlan | TsPostEditHookInstallPlan
)


def build_post_edit_hook_install_plan(
    client: ClientName,
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
    action: HookAction,
) -> PostEditHookInstallPlan:
    """Build a post-edit hook plan, refusing clients with no proven event."""
    if client in POST_EDIT_UNSUPPORTED_CLIENTS:
        raise ValueError(
            f"{client} has no supported post-edit event. {POST_EDIT_UNSUPPORTED_CLIENTS[client]}"
        )
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    selected_scope = _resolve_hook_scope(source, scope)
    if client == "claude-code":
        return ClaudeCodePostEditHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=_hook_settings_path(repo_root, selected_scope),
            action=action,
            hook_entry=_render_post_edit_hook_entry(),
        )
    if client == "codex":
        return CodexPostEditHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=_target_path(client, repo_root, selected_scope),
            action=action,
            block_content=_render_codex_post_edit_block(),
        )
    if client in {"omp", "pi", "opencode"}:
        return TsPostEditHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=_ts_post_edit_module_path(client, repo_root, selected_scope),
            action=action,
            module_content=_render_ts_post_edit_module(client),
        )
    raise ValueError(
        f"post-edit hooks are supported for claude-code, codex, omp, pi, and opencode; got {client}"
    )


def write_post_edit_hook_install_plan(plan: PostEditHookInstallPlan) -> Path:
    """Apply a post-edit hook plan, leaving unrelated configuration intact."""
    if isinstance(plan, TsPostEditHookInstallPlan):
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
    if isinstance(plan, CodexPostEditHookInstallPlan):
        target = plan.target_path
        existing_toml = target.read_text(encoding="utf-8") if target.exists() else ""
        updated_toml = _apply_codex_post_edit_block(existing_toml, plan)
        if updated_toml == existing_toml:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated_toml, encoding="utf-8")
        return target
    target = plan.target_path
    existing_json = _read_json_object(target) if target.exists() else {}
    updated_json, changed = _apply_post_edit_hook_action(existing_json, plan)
    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(updated_json, indent=2) + "\n", encoding="utf-8")
    return target


def render_post_edit_hook_install_preview(plan: PostEditHookInstallPlan) -> str:
    """Render a no-write preview of exactly what the plan would change."""
    action_label = "Install" if plan.action == "install" else "Remove"
    header = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {plan.target_path}",
    ]
    if isinstance(plan, TsPostEditHookInstallPlan):
        event = (
            "tool.execute.after plugin" if plan.client == "opencode" else "tool_result hook module"
        )
        header.append(f"Action: {action_label} post-edit {event}")
        existing = (
            plan.target_path.read_text(encoding="utf-8") if plan.target_path.exists() else None
        )
        if plan.action == "install" and existing == plan.module_content:
            header.append("No change: module already installed (idempotent no-op).")
        elif plan.action == "remove" and existing is None:
            header.append("No change: no archex post-edit module is installed.")
        else:
            header.append("Dry run. Re-run without --dry-run to write this module.")
        body = "" if plan.action == "remove" else plan.module_content
        return "\n".join([*header, "", body]).rstrip("\n") + "\n"
    if isinstance(plan, CodexPostEditHookInstallPlan):
        header.append(
            f"Action: {action_label} PostToolUse hook block (matcher: {CODEX_POST_EDIT_MATCHER!r})"
        )
        existing_toml = (
            plan.target_path.read_text(encoding="utf-8") if plan.target_path.exists() else ""
        )
        updated_toml = _apply_codex_post_edit_block(existing_toml, plan)
        if updated_toml == existing_toml:
            header.append(
                "No change: hook already in the requested state (idempotent no-op)."
                if plan.action == "install"
                else "No change: no archex post-edit hook block is installed."
            )
        else:
            header.append("Dry run. Re-run without --dry-run to write this config.")
        return "\n".join([*header, "", updated_toml]).rstrip("\n") + "\n"
    header.append(f"Action: {action_label} PostToolUse hook (matcher: {POST_EDIT_MATCHER!r})")
    existing_json = _read_json_object(plan.target_path) if plan.target_path.exists() else {}
    updated_json, changed = _apply_post_edit_hook_action(existing_json, plan)
    if not changed:
        header.append(
            "No change: hook already in the requested state (idempotent no-op)."
            if plan.action == "install"
            else "No change: no archex post-edit hook is installed."
        )
    else:
        header.append("Dry run. Re-run without --dry-run to write this config.")
    return "\n".join([*header, "", json.dumps(updated_json, indent=2)]) + "\n"


def _apply_post_edit_hook_action(
    payload: dict[str, object], plan: ClaudeCodePostEditHookInstallPlan
) -> tuple[dict[str, object], bool]:
    return _apply_claude_event_hook_action(
        payload,
        event="PostToolUse",
        marker=_POST_EDIT_ARGS_MARKER,
        matcher=POST_EDIT_MATCHER,
        hook_entry=plan.hook_entry,
        action=plan.action,
    )


def _render_post_edit_hook_entry() -> dict[str, object]:
    return {
        "type": "command",
        "command": sys.executable,
        "args": ["-m", _POST_EDIT_ARGS_MARKER],
    }


def _render_codex_post_edit_block() -> str:
    command = f"{sys.executable} -m {_CODEX_POST_EDIT_ARGS_MARKER}"
    return (
        "\n".join(
            [
                _CODEX_POST_EDIT_BLOCK_START,
                "[[hooks.PostToolUse]]",
                f'matcher = "{CODEX_POST_EDIT_MATCHER}"',
                "",
                "[[hooks.PostToolUse.hooks]]",
                'type = "command"',
                f'command = "{command}"',
                "timeout = 15",
                _CODEX_POST_EDIT_BLOCK_END,
            ]
        )
        + "\n"
    )


def _strip_codex_post_edit_block(existing: str) -> str:
    start = existing.find(_CODEX_POST_EDIT_BLOCK_START)
    if start == -1:
        return existing
    end = existing.find(_CODEX_POST_EDIT_BLOCK_END, start)
    if end == -1:
        return existing  # malformed marker pair -- leave untouched rather than guess
    end += len(_CODEX_POST_EDIT_BLOCK_END)
    if end < len(existing) and existing[end] == "\n":
        end += 1
    before, after = existing[:start], existing[end:]
    if before.endswith("\n\n"):
        before = before[:-1]
    return before + after


def _apply_codex_post_edit_block(existing: str, plan: CodexPostEditHookInstallPlan) -> str:
    without_block = _strip_codex_post_edit_block(existing)
    if plan.action == "remove":
        return without_block
    if not without_block.strip():
        return plan.block_content
    return without_block.rstrip("\n") + "\n\n" + plan.block_content


def _ts_post_edit_module_path(client: ClientName, repo_root: Path, scope: ClientScope) -> Path:
    if client == "omp":
        return (
            repo_root / ".omp" / "extensions" / _TS_POST_EDIT_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".omp" / "agent" / "extensions" / _TS_POST_EDIT_MODULE_FILENAME
        )
    if client == "pi":
        return (
            repo_root / ".pi" / "extensions" / _TS_POST_EDIT_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".pi" / "agent" / "extensions" / _TS_POST_EDIT_MODULE_FILENAME
        )
    if client == "opencode":
        return (
            repo_root / ".opencode" / "plugins" / _TS_POST_EDIT_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".config" / "opencode" / "plugins" / _TS_POST_EDIT_MODULE_FILENAME
        )
    raise ValueError(f"unsupported TS post-edit client: {client}")


def _render_ts_post_edit_module(client: ClientName) -> str:
    template = (
        _OPENCODE_POST_EDIT_MODULE_TEMPLATE
        if client == "opencode"
        else _TS_POST_EDIT_MODULE_TEMPLATE
    )
    return template.replace("__ARCHEX_PYTHON_COMMAND__", json.dumps(sys.executable)).replace(
        "__ARCHEX_CLIENT__", client
    )


_TS_POST_EDIT_MODULE_TEMPLATE = r"""/**
 * archex shared post-edit `tool_result` module (R21 — oh-my-pi / Pi).
 *
 * Installed by `archex install-client omp --post-edit-hooks` /
 * `archex install-client pi --post-edit-hooks` (opt-in; never installed by
 * default). A separate file from the search hook's `archex-hook.ts`, so the
 * two surfaces install and remove independently.
 *
 * Upstream contract (verified against each host's own source, not docs):
 * - Both hosts expose `pi.on("tool_result", handler)`, fired after a tool
 *   executes, with `{ toolName, input, content, details, isError }` and a
 *   partial-patch return. `isError` is the explicit success signal — this
 *   module records an edit only when it is not `true`.
 * - `tool_result` fires after execution and cannot block or fail the edit;
 *   the blocking event is the separate pre-execution `tool_call`.
 * - Both hosts' `edit` and `write` schemas carry the target in `path`
 *   (oh-my-pi `dist/types/edit/schemas.d.ts` and `dist/types/tools/write.d.ts`;
 *   Pi `dist/core/tools/edit.d.ts` and `.../write.d.ts`).
 *
 * Every path resolves without throwing. A spawn failure, a timeout, a stale
 * index, or a malformed response all degrade to returning `undefined` (no
 * content patch), logged to the same diagnostics file the Python hooks use.
 */
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// --- Baked in at install time (`archex install-client <client> --post-edit-hooks`) ---

/** Python interpreter active when the installer ran, so this always runs in
 * the environment archex was installed into. */
const ARCHEX_PYTHON_COMMAND = __ARCHEX_PYTHON_COMMAND__;
const ARCHEX_PYTHON_ARGS = ["-m", "archex.integrations.post_edit_hook"];

/** Wall-clock kill timer for the subprocess. Deliberately larger than the
 * search hook's 500ms — a post-edit cycle reparses changed files — and
 * larger than the Python side's own DEFAULT_POST_EDIT_TIMEOUT_SECONDS so
 * the Python deadline, which logs a diagnostic, normally fires first. */
const ARCHEX_POST_EDIT_TIMEOUT_MS = 15000;

const ARCHEX_DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG";

/** Native edit tool name -> the Claude Code tool name the Python subprocess
 * expects. Verified against each host's own tool definitions, not docs. */
const ARCHEX_EDIT_TOOLS: Readonly<Record<string, "Edit" | "Write">> = {
  edit: "Edit",
  write: "Write",
};

/** Input fields that can hold the edited path, most-specific first. Both
 * oh-my-pi's and Pi's `edit`/`write` schemas name it `path`; OpenCode's name
 * it `filePath`. The others are accepted defensively — the Python side
 * rejects anything that is not a real path inside the repository, so a wrong
 * guess degrades to no output rather than a wrong claim. */
const ARCHEX_PATH_FIELDS = ["filePath", "path", "file_path", "notebook_path"] as const;

/** Declared to the Python subprocess as `archex_client` so the emitted
 * receipt names the host that actually ran the edit. The subprocess
 * allowlists this value; an unknown one falls back to `claude-code`. */
const ARCHEX_CLIENT = "__ARCHEX_CLIENT__";

function firstPath(args: Record<string, unknown> | undefined): string | null {
  if (!args) return null;
  for (const field of ARCHEX_PATH_FIELDS) {
    const value = args[field];
    if (typeof value === "string" && value.trim().length > 0) return value;
  }
  return null;
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
      timestamp: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
      kind,
      detail,
    };
    if (cwd) entry.cwd = cwd;
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf-8");
  } catch {
    // Diagnostics logging must never raise into the host's edit path.
  }
}

// --- Subprocess call: `python -m archex.integrations.post_edit_hook` ---

function runArchexPostEditSubprocess(
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
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
      finish(null);
      return;
    }

    const timer = setTimeout(() => {
      logDiagnostic(
        "ts_post_edit_timeout",
        `post-edit cycle exceeded ${ARCHEX_POST_EDIT_TIMEOUT_MS}ms`,
        cwd,
      );
      try {
        child.kill("SIGKILL");
      } catch {
        // Already exited.
      }
      finish(null);
    }, ARCHEX_POST_EDIT_TIMEOUT_MS);

    let stdout = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
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
      logDiagnostic("ts_post_edit_stdin_error", String(err), cwd);
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

// --- Extension entry point ---

export default function archexPostEditHook(pi: HookHost): void {
  pi.on("tool_result", async (event) => {
    try {
      const claudeToolName = ARCHEX_EDIT_TOOLS[event.toolName];
      if (!claudeToolName) return undefined; // never touches read/grep/glob/bash
      if (event.isError === true) return undefined; // only successful edits count

      const filePath = firstPath(event.input);
      if (filePath === null) return undefined;

      const cwd = process.cwd();
      const rawStdout = await runArchexPostEditSubprocess(
        {
          tool_name: claudeToolName,
          tool_input: { file_path: filePath },
          cwd,
          archex_client: ARCHEX_CLIENT,
        },
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
      logDiagnostic("ts_post_edit_internal_error", String(err));
      return undefined;
    }
  });
}
"""


_OPENCODE_POST_EDIT_MODULE_TEMPLATE = r"""/**
 * archex OpenCode post-edit `tool.execute.after` plugin (R21).
 *
 * Installed by `archex install-client opencode --post-edit-hooks` (opt-in).
 * A separate plugin file from the search hook's `archex-hook.ts`, so the two
 * surfaces install and remove independently. OpenCode auto-loads plugin
 * files from `.opencode/plugins/` and `~/.config/opencode/plugins/`, so no
 * `opencode.json` entry is written or needed.
 *
 * Upstream contract (verified against `packages/plugin/src/index.ts`):
 * - `"tool.execute.after"` receives `input { tool, sessionID, callID, args }`
 *   and a mutable `output { title, output, metadata }`. It runs after the
 *   tool executed and has no documented way to block or fail it.
 * - The hook exposes no error flag, so a failed edit is not distinguishable
 *   here; the Python side revalidates every recorded path against the real
 *   working tree before anything is emitted, so a phantom path simply
 *   produces no output.
 * - Text reaches the agent only by appending to `output.output`; there is no
 *   additional-context field on this event.
 * - `packages/opencode/src/tool/edit.ts` and `.../write.ts` define the
 *   built-in `edit` and `write` tools, both parameterised by `filePath`.
 *
 * Every path resolves without throwing and leaves `output` untouched when
 * archex has nothing fresh to add.
 */
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// --- Baked in at install time (`archex install-client <client> --post-edit-hooks`) ---

/** Python interpreter active when the installer ran, so this always runs in
 * the environment archex was installed into. */
const ARCHEX_PYTHON_COMMAND = __ARCHEX_PYTHON_COMMAND__;
const ARCHEX_PYTHON_ARGS = ["-m", "archex.integrations.post_edit_hook"];

/** Wall-clock kill timer for the subprocess. Deliberately larger than the
 * search hook's 500ms — a post-edit cycle reparses changed files — and
 * larger than the Python side's own DEFAULT_POST_EDIT_TIMEOUT_SECONDS so
 * the Python deadline, which logs a diagnostic, normally fires first. */
const ARCHEX_POST_EDIT_TIMEOUT_MS = 15000;

const ARCHEX_DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG";

/** Native edit tool name -> the Claude Code tool name the Python subprocess
 * expects. Verified against each host's own tool definitions, not docs. */
const ARCHEX_EDIT_TOOLS: Readonly<Record<string, "Edit" | "Write">> = {
  edit: "Edit",
  write: "Write",
};

/** Input fields that can hold the edited path, most-specific first. Both
 * oh-my-pi's and Pi's `edit`/`write` schemas name it `path`; OpenCode's name
 * it `filePath`. The others are accepted defensively — the Python side
 * rejects anything that is not a real path inside the repository, so a wrong
 * guess degrades to no output rather than a wrong claim. */
const ARCHEX_PATH_FIELDS = ["filePath", "path", "file_path", "notebook_path"] as const;

/** Declared to the Python subprocess as `archex_client` so the emitted
 * receipt names the host that actually ran the edit. The subprocess
 * allowlists this value; an unknown one falls back to `claude-code`. */
const ARCHEX_CLIENT = "__ARCHEX_CLIENT__";

function firstPath(args: Record<string, unknown> | undefined): string | null {
  if (!args) return null;
  for (const field of ARCHEX_PATH_FIELDS) {
    const value = args[field];
    if (typeof value === "string" && value.trim().length > 0) return value;
  }
  return null;
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
      timestamp: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
      kind,
      detail,
    };
    if (cwd) entry.cwd = cwd;
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf-8");
  } catch {
    // Diagnostics logging must never raise into the host's edit path.
  }
}

// --- Subprocess call: `python -m archex.integrations.post_edit_hook` ---

function runArchexPostEditSubprocess(
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
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
      finish(null);
      return;
    }

    const timer = setTimeout(() => {
      logDiagnostic(
        "ts_post_edit_timeout",
        `post-edit cycle exceeded ${ARCHEX_POST_EDIT_TIMEOUT_MS}ms`,
        cwd,
      );
      try {
        child.kill("SIGKILL");
      } catch {
        // Already exited.
      }
      finish(null);
    }, ARCHEX_POST_EDIT_TIMEOUT_MS);

    let stdout = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
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
      logDiagnostic("ts_post_edit_stdin_error", String(err), cwd);
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

// --- Minimal structural types for the plugin contract ---

interface ToolExecuteAfterInput {
  tool: string;
  sessionID?: string;
  callID?: string;
  args?: Record<string, unknown>;
}

interface ToolExecuteAfterOutput {
  title?: string;
  output?: string;
  metadata?: unknown;
}

type Plugin = (context: { directory?: string }) => Promise<{
  "tool.execute.after"?: (
    input: ToolExecuteAfterInput,
    output: ToolExecuteAfterOutput,
  ) => Promise<void>;
}>;

// --- Plugin entry point ---

export const ArchexPostEditPlugin: Plugin = async ({ directory }) => {
  return {
    "tool.execute.after": async (input, output) => {
      try {
        const claudeToolName = ARCHEX_EDIT_TOOLS[input.tool];
        if (!claudeToolName) return; // never touches read/grep/glob/bash

        const filePath = firstPath(input.args);
        if (filePath === null) return;

        const cwd = directory ?? process.cwd();
        const rawStdout = await runArchexPostEditSubprocess(
          {
            tool_name: claudeToolName,
            tool_input: { file_path: filePath },
            cwd,
            archex_client: ARCHEX_CLIENT,
          },
          cwd,
        );
        if (rawStdout === null) return;

        const context = extractAdditionalContext(rawStdout);
        if (context === null) return;

        output.output = `${output.output ?? ""}\n\n${context}`;
      } catch (err) {
        logDiagnostic("ts_post_edit_internal_error", String(err));
      }
    },
  };
};

export default ArchexPostEditPlugin;
"""


# ---------------------------------------------------------------------------
# Persistent status surfaces (R23)
# ---------------------------------------------------------------------------

#: Filename of the installed Claude Code status-line renderer. Doubles as the
#: ownership marker: ``statusLine`` is a single settings key rather than a
#: list, so install and remove only ever touch an entry whose command
#: references this filename. Any other configured status line is a user's own
#: and is refused rather than replaced.
STATUSLINE_SCRIPT_FILENAME = "archex-statusline.sh"

#: Marker comment inside the script, so an installed file is identifiable as
#: archex-owned even after it is moved or copied.
STATUSLINE_SCRIPT_MARKER = "archex:statusline"

#: How often Claude Code re-runs the command in addition to its event-driven
#: repaints. The renderer's freshness and age segments are time-based, so
#: without a timer a snapshot would appear fresh through an idle session.
STATUSLINE_REFRESH_INTERVAL_SECONDS = 10


@dataclass(frozen=True)
class ClaudeCodeStatuslineInstallPlan:
    """Install or remove the Claude Code status-line renderer (R23).

    Two artifacts, one plan: the renderer script beside the client's other
    archex-owned files, and the ``statusLine`` entry in the same
    ``settings.json`` the hook installers merge into.
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    script_path: Path
    action: HookAction
    script_content: str
    statusline_entry: dict[str, object]


#: Filename of the installed omp/Pi status extension module. Its own name is
#: the ownership marker, exactly as the hook modules' filenames are.
_TS_STATUS_MODULE_FILENAME = "archex-status.ts"

#: Status key the extension registers under. Both hosts sort footer statuses
#: by key and render them inline, so the key is user-visible ordering.
TS_STATUS_KEY = "archex"

#: Clients with no persistent status surface, and the upstream reason.
#: Surfaced by the CLI so an unsupported client is refused explicitly rather
#: than silently no-op'ing or being given a transient stand-in.
STATUSLINE_UNSUPPORTED_CLIENTS: dict[ClientName, str] = {
    "opencode": (
        "OpenCode's plugin surface exposes tool and chat hooks plus observable "
        "TUI events, whose only status-shaped member is the transient "
        "tui.toast.show. A toast disappears, so it cannot carry a persistent "
        "freshness indicator; archex ships no adapter it cannot show to work."
    ),
    "codex": (
        "The Codex CLI exposes no status-line configuration and no hook output "
        "field that renders persistently; its hooks surface text only as "
        "additional_context on a tool event."
    ),
    "cursor": (
        "Cursor's configuration surface is hooks only, with no status or "
        "footer API for an extension to write into."
    ),
}


@dataclass(frozen=True)
class TsStatusInstallPlan:
    """Install or remove the omp/Pi status extension module (R23).

    Unlike the Claude Code status line -- a command the client re-runs per
    repaint -- this is a module the host already has loaded. It reads the
    snapshot in-process with `readFileSync` and pushes the rendered line
    through `ctx.ui.setStatus`, so a repaint launches nothing at all.
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    module_content: str


StatusInstallPlan = ClaudeCodeStatuslineInstallPlan | TsStatusInstallPlan


def _ts_status_module_path(client: ClientName, repo_root: Path, scope: ClientScope) -> Path:
    if client == "omp":
        return (
            repo_root / ".omp" / "extensions" / _TS_STATUS_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".omp" / "agent" / "extensions" / _TS_STATUS_MODULE_FILENAME
        )
    return (
        repo_root / ".pi" / "extensions" / _TS_STATUS_MODULE_FILENAME
        if scope == "project"
        else Path.home() / ".pi" / "agent" / "extensions" / _TS_STATUS_MODULE_FILENAME
    )


def render_ts_status_module() -> str:
    """Render the omp/Pi status module with the snapshot contract baked in."""
    return (
        _TS_STATUS_MODULE_TEMPLATE.replace(
            "__ARCHEX_SNAPSHOT_VERSION__", str(STATUS_SNAPSHOT_VERSION)
        )
        .replace("__ARCHEX_STALE_DEFAULT__", str(DEFAULT_STALE_AFTER_SECONDS))
        .replace("__ARCHEX_WATCH_TTL__", str(WATCH_OBSERVATION_TTL_SECONDS))
        .replace("__ARCHEX_PROJECT_DIR__", PROJECT_DIR_NAME)
        .replace("__ARCHEX_SNAPSHOT_FILENAME__", SNAPSHOT_FILENAME)
        .replace("__ARCHEX_STATUS_KEY__", TS_STATUS_KEY)
    )


_TS_STATUS_MODULE_TEMPLATE = r"""/**
 * archex status extension module (R23 - oh-my-pi / Pi).
 *
 * Installed by `archex install-client omp --statusline` /
 * `archex install-client pi --statusline` (opt-in; never installed by
 * default). A separate file from the search and post-edit hook modules, so
 * every surface installs and removes independently.
 *
 * Upstream contract (verified against each host's own type declarations, not
 * docs):
 * - `ctx.ui.setStatus(key, text)` sets keyed footer/status-bar text, rendered
 *   by the built-in footer and by the `status` status-line segment, sorted by
 *   key (oh-my-pi `src/modes/controllers/extension-ui-controller.ts` wires it
 *   to `setHookStatus`; Pi `dist/core/extensions/types.d.ts` declares it on
 *   `ExtensionUIContext`).
 * - Both hosts declare `turn_start`, `turn_end`, and `tool_result` events with
 *   the same `(event, ctx)` handler shape, and `ctx.hasUI` is false in
 *   print/RPC modes where `setStatus` is a documented no-op.
 *
 * Unlike the hook modules, this one spawns no subprocess and opens no index:
 * a repaint is a single `readFileSync` of the bounded snapshot archex
 * publishes plus string formatting, inside a process the host already runs.
 * Every path resolves without throwing; an unreadable, malformed, or
 * unknown-version snapshot renders as an explicit state instead.
 */
import { existsSync, readFileSync } from "node:fs";
import { join, parse } from "node:path";

// --- Baked in at install time from the Python contract ---

const ARCHEX_SNAPSHOT_VERSION = __ARCHEX_SNAPSHOT_VERSION__;
const ARCHEX_STALE_AFTER_SECONDS = __ARCHEX_STALE_DEFAULT__;
const ARCHEX_WATCH_TTL_SECONDS = __ARCHEX_WATCH_TTL__;
const ARCHEX_PROJECT_DIR = "__ARCHEX_PROJECT_DIR__";
const ARCHEX_SNAPSHOT_FILENAME = "__ARCHEX_SNAPSHOT_FILENAME__";
const ARCHEX_STATUS_KEY = "__ARCHEX_STATUS_KEY__";

/** Override for tests and for a host started outside the repository. */
const ARCHEX_SNAPSHOT_ENV_VAR = "ARCHEX_STATUS_SNAPSHOT";

/** Freshness-budget override, honoured by every archex status renderer. */
const ARCHEX_STALE_ENV_VAR = "ARCHEX_STATUS_STALE_AFTER_SECONDS";

/** Reader-state lines, worded exactly as the shell renderer words them. */
const ARCHEX_MISSING_LINE = "archex missing - no status snapshot - run: archex index";
const ARCHEX_CORRUPT_LINE = "archex corrupt - unreadable snapshot - run: archex status";

interface ArchexSnapshot {
  version?: unknown;
  state?: unknown;
  index_revision?: unknown;
  files_indexed?: unknown;
  chunks_indexed?: unknown;
  pending_delta_files?: unknown;
  pending_view_complete?: unknown;
  reindex_required?: unknown;
  written_at_epoch?: unknown;
  watch_observed_epoch?: unknown;
}

function snapshotPath(start: string): string | null {
  const override = process.env[ARCHEX_SNAPSHOT_ENV_VAR];
  if (override && override.trim().length > 0) return override;
  let current = start;
  for (;;) {
    const candidate = join(current, ARCHEX_PROJECT_DIR, ARCHEX_SNAPSHOT_FILENAME);
    // Existence, not readability: a present-but-unreadable snapshot must
    // classify as `corrupt` here, exactly as it does in the shell renderer
    // and the Python reader. Probing with a read would skip it and report
    // either `missing` or -- worse -- a parent repository's status.
    if (existsSync(candidate)) return candidate;
    const parent = parse(current).dir;
    if (!parent || parent === current) return null;
    current = parent;
  }
}

/** Freshness budget in seconds, overridable exactly as the CLI allows. */
function staleAfterSeconds(): number {
  const raw = process.env[ARCHEX_STALE_ENV_VAR];
  if (raw === undefined) return ARCHEX_STALE_AFTER_SECONDS;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : ARCHEX_STALE_AFTER_SECONDS;
}

function readNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function readText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function ageSegment(writtenAtEpoch: number, now: number): string {
  if (writtenAtEpoch <= 0) return "";
  const age = Math.max(0, now - writtenAtEpoch);
  if (age < 60) return ` - ${age}s ago`;
  if (age < 3600) return ` - ${Math.floor(age / 60)}m ago`;
  return ` - ${Math.floor(age / 3600)}h ago`;
}

/** Render the status text for `cwd`, or a state naming why it cannot. */
export function renderArchexStatus(cwd: string, nowSeconds?: number): string {
  const path = snapshotPath(cwd);
  if (path === null) return ARCHEX_MISSING_LINE;

  // An absent file is `missing`; unreadable or malformed bytes are
  // `corrupt`. The two have different remedies -- publish one versus
  // re-publish this one -- so they must not collapse into each other.
  let raw: string;
  try {
    raw = readFileSync(path, "utf-8");
  } catch (err) {
    const code = (err as { code?: string }).code;
    return code === "ENOENT" ? ARCHEX_MISSING_LINE : ARCHEX_CORRUPT_LINE;
  }

  let parsed: ArchexSnapshot;
  try {
    const document: unknown = JSON.parse(raw);
    if (typeof document !== "object" || document === null || Array.isArray(document)) {
      return ARCHEX_CORRUPT_LINE;
    }
    parsed = document as ArchexSnapshot;
  } catch {
    return ARCHEX_CORRUPT_LINE;
  }

  if (typeof parsed.version !== "number") return ARCHEX_CORRUPT_LINE;
  if (parsed.version !== ARCHEX_SNAPSHOT_VERSION) {
    return `archex unsupported - snapshot v${parsed.version} - upgrade archex`;
  }

  const persisted = readText(parsed.state);
  if (persisted !== "fresh" && persisted !== "dirty" && persisted !== "pending") {
    return ARCHEX_CORRUPT_LINE;
  }

  const now = nowSeconds ?? Math.floor(Date.now() / 1000);
  const writtenAtEpoch = readNumber(parsed.written_at_epoch);
  const age = writtenAtEpoch > 0 ? Math.max(0, now - writtenAtEpoch) : null;
  const stale = age !== null && age > staleAfterSeconds();
  const state = stale ? "stale" : persisted;

  let detail = "";
  if (state === "fresh") {
    const files = readNumber(parsed.files_indexed);
    const chunks = readNumber(parsed.chunks_indexed);
    detail = ` - ${files} files, ${chunks} chunks`;
  } else if (state === "pending") {
    const pending = readNumber(parsed.pending_delta_files);
    const complete = parsed.pending_view_complete !== false;
    detail = ` - ${pending}${complete ? "" : "+"} awaiting sync`;
  } else if (state === "dirty") {
    detail = parsed.reindex_required === true ? " - reindex required" : " - index behind tree";
  } else {
    detail = " - unverified since measurement";
  }

  const revision = readText(parsed.index_revision);
  const revisionSegment = revision.length > 0 ? ` - rev ${revision.slice(0, 8)}` : "";

  const watchEpoch = readNumber(parsed.watch_observed_epoch);
  const watching = watchEpoch > 0 && now - watchEpoch <= ARCHEX_WATCH_TTL_SECONDS;

  return `archex ${state}${detail}${revisionSegment}${ageSegment(writtenAtEpoch, now)}${
    watching ? " - watch" : ""
  }`;
}

// --- Minimal structural types for the events used ---
//
// Declared locally (never imported from either host package) so this module
// has zero import-resolution dependency on which host loaded it.

interface StatusUiLike {
  setStatus(key: string, text: string | undefined): void;
}

interface StatusContextLike {
  ui?: StatusUiLike;
  hasUI?: boolean;
}

type StatusHandler = (event: unknown, ctx: StatusContextLike) => void;

interface StatusHost {
  on(event: "turn_start" | "turn_end" | "tool_result", handler: StatusHandler): unknown;
}

/** Refresh the footer status, swallowing every failure. */
export function publishArchexStatus(ctx: StatusContextLike): void {
  try {
    if (ctx.hasUI === false || !ctx.ui) return;
    ctx.ui.setStatus(ARCHEX_STATUS_KEY, renderArchexStatus(process.cwd()));
  } catch {
    // A status refresh must never disturb the host's turn.
  }
}

export default function archexStatusExtension(pi: StatusHost): void {
  const refresh: StatusHandler = (_event, ctx) => {
    publishArchexStatus(ctx);
  };
  // Three refresh points: entering a turn, after each tool result (so an edit
  // shows as pending mid-turn), and at rest. All three read one small file.
  pi.on("turn_start", refresh);
  pi.on("tool_result", refresh);
  pi.on("turn_end", refresh);
}
"""


def build_statusline_install_plan(
    client: ClientName,
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
    action: HookAction,
) -> StatusInstallPlan:
    """Build a status plan, refusing clients with no persistent surface."""
    if client in STATUSLINE_UNSUPPORTED_CLIENTS:
        message = (
            f"{client} has no persistent status surface. "
            f"{STATUSLINE_UNSUPPORTED_CLIENTS[client]} "
            "Use `archex status --cached` instead."
        )
        raise ValueError(message)
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    selected_scope = _resolve_hook_scope(source, scope)
    if client == "claude-code":
        script_path = _statusline_script_path(repo_root, selected_scope)
        return ClaudeCodeStatuslineInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=_hook_settings_path(repo_root, selected_scope),
            script_path=script_path,
            action=action,
            script_content=render_statusline_script(),
            statusline_entry=_render_statusline_entry(script_path),
        )
    if client in {"omp", "pi"}:
        return TsStatusInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=_ts_status_module_path(client, repo_root, selected_scope),
            action=action,
            module_content=render_ts_status_module(),
        )
    message = f"status surfaces are supported for claude-code, omp, and pi; got {client}"
    raise ValueError(message)


def write_statusline_install_plan(plan: StatusInstallPlan) -> Path:
    """Apply a status plan, leaving unrelated configuration intact."""
    if isinstance(plan, TsStatusInstallPlan):
        target = plan.target_path
        if plan.action == "remove":
            if target.exists():
                target.unlink()
            return target
        existing_module = target.read_text(encoding="utf-8") if target.exists() else None
        if existing_module != plan.module_content:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(plan.module_content, encoding="utf-8")
        return target

    target = plan.target_path
    existing = _read_json_object(target) if target.exists() else {}
    # Decided before anything is written: a foreign status line raises here,
    # and a refused install must leave no renderer script behind.
    updated, changed = _apply_statusline_action(existing, plan)

    if plan.action == "install":
        plan.script_path.parent.mkdir(parents=True, exist_ok=True)
        existing_script = (
            plan.script_path.read_text(encoding="utf-8") if plan.script_path.exists() else None
        )
        if existing_script != plan.script_content:
            plan.script_path.write_text(plan.script_content, encoding="utf-8")
        plan.script_path.chmod(0o755)
    elif plan.script_path.exists() and _is_archex_statusline_script(plan.script_path):
        plan.script_path.unlink()

    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return target


def render_statusline_install_preview(plan: StatusInstallPlan) -> str:
    """Render a no-write preview of exactly what the plan would change."""
    action_label = "Install" if plan.action == "install" else "Remove"
    if isinstance(plan, TsStatusInstallPlan):
        header = [
            f"Client: {plan.client}",
            f"Scope: {plan.scope}",
            f"Target: {plan.target_path}",
            f"Action: {action_label} status extension module",
        ]
        existing_module = (
            plan.target_path.read_text(encoding="utf-8") if plan.target_path.exists() else None
        )
        if plan.action == "install" and existing_module == plan.module_content:
            header.append("No change: module already installed (idempotent no-op).")
        elif plan.action == "remove" and existing_module is None:
            header.append("No change: no archex status module is installed.")
        else:
            header.append("Dry run. Re-run without --dry-run to write this module.")
        body = "" if plan.action == "remove" else plan.module_content
        return "\n".join([*header, "", body]).rstrip("\n") + "\n"

    existing = _read_json_object(plan.target_path) if plan.target_path.exists() else {}
    updated, changed = _apply_statusline_action(existing, plan)
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {plan.target_path}",
        f"Renderer: {plan.script_path}",
        f"Action: {action_label} statusLine command",
    ]
    if not changed:
        lines.append(
            "No change: status line already in the requested state (idempotent no-op)."
            if plan.action == "install"
            else "No change: no archex status line is installed."
        )
    else:
        lines.append("Dry run. Re-run without --dry-run to write this config.")
    body = json.dumps(updated, indent=2)
    return "\n".join([*lines, "", body]).rstrip("\n") + "\n"


def _statusline_script_path(repo_root: Path, scope: ClientScope) -> Path:
    return (
        repo_root / ".claude" / STATUSLINE_SCRIPT_FILENAME
        if scope == "project"
        else Path.home() / ".claude" / STATUSLINE_SCRIPT_FILENAME
    )


def statusline_interpreter() -> str:
    """Shell to run the renderer with, preferring one that can read a clock.

    The renderer reports `stale` and an age only when the shell exposes a
    builtin clock, because forking `date` on every repaint is what R23
    excludes. `sh` is bash 3.2 on macOS and dash on many Linux images, and
    neither has one -- so the interpreter is chosen at install time from the
    shells present on the host, and `sh` remains the fallback. Probing costs
    one subprocess per install, never per repaint.
    """
    for candidate, probe in (
        ("/bin/zsh", "zmodload zsh/datetime 2>/dev/null; echo ${EPOCHSECONDS:-}"),
        ("/opt/homebrew/bin/bash", "echo ${EPOCHSECONDS:-}"),
        ("/usr/local/bin/bash", "echo ${EPOCHSECONDS:-}"),
        ("/bin/bash", "echo ${EPOCHSECONDS:-}"),
    ):
        if not Path(candidate).exists():
            continue
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, absolute path
                [candidate, "-c", probe],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.stdout.strip().isdigit():
            return candidate
    return "sh"


def _render_statusline_entry(script_path: Path) -> dict[str, object]:
    # The path is quoted rather than interpolated bare: a home or repository
    # directory containing a space would otherwise split into two arguments.
    # Invoking through an interpreter also keeps the entry working if the
    # executable bit is lost (a copied dotfiles tree, a restored backup).
    return {
        "type": "command",
        "command": f'{statusline_interpreter()} "{script_path}"',
        "padding": 0,
        "refreshInterval": STATUSLINE_REFRESH_INTERVAL_SECONDS,
    }


def _is_archex_statusline(entry: object) -> bool:
    """Whether a configured ``statusLine`` value is the archex-owned one."""
    if not isinstance(entry, dict):
        return False
    command = cast("dict[str, object]", entry).get("command")
    return isinstance(command, str) and STATUSLINE_SCRIPT_FILENAME in command


def _is_archex_statusline_script(path: Path) -> bool:
    """Whether the file at ``path`` is an archex-owned renderer script."""
    try:
        return STATUSLINE_SCRIPT_MARKER in path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False


def _apply_statusline_action(
    payload: dict[str, object], plan: ClaudeCodeStatuslineInstallPlan
) -> tuple[dict[str, object], bool]:
    """Return ``(updated_payload, changed)`` without mutating ``payload``.

    ``statusLine`` is a scalar key, so there is no matcher group to merge
    into and no way to coexist with another status line. A foreign entry is
    therefore refused rather than replaced -- the same rule the MCP installer
    applies to a foreign ``archex`` server entry -- and remove leaves a
    foreign entry alone.
    """
    existing = payload.get("statusLine")
    if plan.action == "remove":
        if existing is None or not _is_archex_statusline(existing):
            return payload, False
        updated = copy.deepcopy(payload)
        del updated["statusLine"]
        return updated, True
    if existing is not None and not _is_archex_statusline(existing):
        message = (
            f"{plan.target_path} already configures a different statusLine command. "
            "Remove it first, or install into the other scope; archex will not overwrite it."
        )
        raise ValueError(message)
    if existing == plan.statusline_entry:
        return payload, False
    updated = copy.deepcopy(payload)
    updated["statusLine"] = plan.statusline_entry
    return updated, True


def render_statusline_script() -> str:
    """Render the status-line script with the snapshot contract baked in."""
    return (
        _STATUSLINE_SCRIPT_TEMPLATE.replace(
            "__ARCHEX_SNAPSHOT_VERSION__", str(STATUS_SNAPSHOT_VERSION)
        )
        .replace("__ARCHEX_STALE_DEFAULT__", str(DEFAULT_STALE_AFTER_SECONDS))
        .replace("__ARCHEX_WATCH_TTL__", str(WATCH_OBSERVATION_TTL_SECONDS))
        .replace("__ARCHEX_SNAPSHOT_RELATIVE_PATH__", f"{PROJECT_DIR_NAME}/{SNAPSHOT_FILENAME}")
    )


_STATUSLINE_SCRIPT_TEMPLATE = r"""#!/bin/sh
# archex:statusline - Claude Code status line renderer (R23).
#
# Installed by `archex install-client claude-code --statusline`, removed by
# `--remove-statusline`. Renders the bounded status snapshot archex publishes
# at <repo>/__ARCHEX_SNAPSHOT_RELATIVE_PATH__.
#
# Claude Code re-runs this command on every repaint (debounced at 300ms, and
# it cancels an in-flight script when a new update arrives), so the renderer
# uses shell builtins exclusively: no `jq`, no `python`, no `date`, no
# `archex`, and no command substitution anywhere. It therefore forks no
# process, opens no index, and parses no source. The test suite proves that
# by running it with an empty PATH.
#
# Snapshot version this renderer understands: __ARCHEX_SNAPSHOT_VERSION__.
# A different version prints `unsupported` rather than guessing at fields.

set -u

archex_version_supported=__ARCHEX_SNAPSHOT_VERSION__
archex_stale_default=__ARCHEX_STALE_DEFAULT__
archex_watch_ttl=__ARCHEX_WATCH_TTL__

# --- Locate the snapshot ----------------------------------------------------
# Claude Code writes one line of session JSON to stdin, carrying the session
# directory as "cwd". Extracting it by parameter expansion avoids a JSON
# parser; an unreadable payload falls back to the process working directory,
# and the upward walk then handles being started in a subdirectory.
payload=""
IFS= read -r payload 2>/dev/null || :

archex_dir=""
case $payload in
*'"cwd":'*)
	archex_rest=${payload#*'"cwd":'}
	while :; do
		case $archex_rest in
		' '*) archex_rest=${archex_rest# } ;;
		*) break ;;
		esac
	done
	case $archex_rest in
	'"'*)
		archex_rest=${archex_rest#\"}
		archex_dir=${archex_rest%%\"*}
		;;
	esac
	;;
esac
# Only an absolute, existing directory is trusted. A payload whose earlier
# values happen to contain the literal `"cwd":` text, or a relative or
# nonexistent path, falls back to the process working directory rather than
# steering the walk below somewhere else.
case $archex_dir in
/*) ;;
*) archex_dir="" ;;
esac
if [ -z "$archex_dir" ] || [ ! -d "$archex_dir" ]; then
	archex_dir=$PWD
fi

archex_snapshot=${ARCHEX_STATUS_SNAPSHOT:-}
if [ -z "$archex_snapshot" ]; then
	archex_probe=$archex_dir
	while [ -n "$archex_probe" ]; do
		if [ -f "$archex_probe/__ARCHEX_SNAPSHOT_RELATIVE_PATH__" ]; then
			archex_snapshot="$archex_probe/__ARCHEX_SNAPSHOT_RELATIVE_PATH__"
			break
		fi
		case $archex_probe in
		*/?*) archex_probe=${archex_probe%/*} ;;
		*) archex_probe="" ;;
		esac
	done
fi

if [ -z "$archex_snapshot" ] || [ ! -f "$archex_snapshot" ]; then
	printf 'archex missing - no status snapshot - run: archex index\n'
	exit 0
fi

# --- Read it ----------------------------------------------------------------
# The document is written with sorted keys and two-space indentation, so every
# scalar occupies one line as `  "key": value,`. Splitting that with
# parameter expansion is exact for this writer and degrades to an empty field
# (hence `corrupt`) for anything else.
archex_field_version=""
archex_field_state=""
archex_field_revision=""
archex_field_files=0
archex_field_chunks=0
archex_field_pending=0
archex_field_pending_complete=""
archex_field_reindex=""
archex_field_epoch=0
archex_field_watch_epoch=0

archex_line=""
while IFS= read -r archex_line || [ -n "$archex_line" ]; do
	case $archex_line in
	*'": '*) ;;
	*) continue ;;
	esac
	archex_key=${archex_line#*\"}
	archex_key=${archex_key%%\"*}
	archex_value=${archex_line#*\": }
	archex_value=${archex_value%,}
	case $archex_value in
	\"*\")
		archex_value=${archex_value#\"}
		archex_value=${archex_value%\"}
		;;
	esac
	case $archex_key in
	version) archex_field_version=$archex_value ;;
	state) archex_field_state=$archex_value ;;
	index_revision) archex_field_revision=$archex_value ;;
	files_indexed) archex_field_files=$archex_value ;;
	chunks_indexed) archex_field_chunks=$archex_value ;;
	pending_delta_files) archex_field_pending=$archex_value ;;
	pending_view_complete) archex_field_pending_complete=$archex_value ;;
	reindex_required) archex_field_reindex=$archex_value ;;
	written_at_epoch) archex_field_epoch=$archex_value ;;
	watch_observed_epoch) archex_field_watch_epoch=$archex_value ;;
	esac
done < "$archex_snapshot"

# Numeric fields are normalized before any arithmetic. Non-digits become 0,
# and a leading zero is stripped: POSIX arithmetic reads `08` as octal and
# aborts the whole script on the invalid digit, which would replace a
# `corrupt` line with an error on stderr and no status at all.
for archex_numeric in files chunks pending epoch watch_epoch; do
	case $archex_numeric in
	files) archex_probe_value=$archex_field_files ;;
	chunks) archex_probe_value=$archex_field_chunks ;;
	pending) archex_probe_value=$archex_field_pending ;;
	epoch) archex_probe_value=$archex_field_epoch ;;
	*) archex_probe_value=$archex_field_watch_epoch ;;
	esac
	case $archex_probe_value in
	'' | *[!0-9]*) archex_probe_value=0 ;;
	*)
		while :; do
			case $archex_probe_value in
			0?*) archex_probe_value=${archex_probe_value#0} ;;
			*) break ;;
			esac
		done
		;;
	esac
	case $archex_numeric in
	files) archex_field_files=$archex_probe_value ;;
	chunks) archex_field_chunks=$archex_probe_value ;;
	pending) archex_field_pending=$archex_probe_value ;;
	epoch) archex_field_epoch=$archex_probe_value ;;
	*) archex_field_watch_epoch=$archex_probe_value ;;
	esac
done

# --- Classify ---------------------------------------------------------------
# `corrupt` and `unsupported` are deliberately separate: a document that is
# unparsable or carries no usable version needs re-publishing, while a
# document from a schema this build does not read needs an archex upgrade.
# The Python reader and the omp/Pi module classify identically.
case $archex_field_version in
'' | *[!0-9]*)
	printf 'archex corrupt - unreadable snapshot - run: archex status\n'
	exit 0
	;;
esac
if [ "$archex_field_version" != "$archex_version_supported" ]; then
	printf 'archex unsupported - snapshot v%s - upgrade archex\n' "$archex_field_version"
	exit 0
fi
case $archex_field_state in
fresh | dirty | pending) ;;
*)
	printf 'archex corrupt - unreadable snapshot - run: archex status\n'
	exit 0
	;;
esac

# `EPOCHSECONDS` is a shell builtin variable in bash 5+ and zsh, and absent in
# dash and bash 3.2 (macOS `/bin/sh`). Where it is absent this renderer has no
# clock it can read without forking `date`, so it reports the measured state
# and its measurement time and leaves the `stale` judgement to
# `archex status --cached`, which has a clock. Paying a fork on every repaint
# to gain one label is the wrong trade.
if [ -n "${ZSH_VERSION:-}" ]; then
	# `zmodload` is a zsh builtin, so this costs no process. Guarded on
	# ZSH_VERSION because under sh/dash the word would be an external
	# command, which is exactly what this renderer refuses to launch.
	zmodload zsh/datetime 2>/dev/null || :
fi
archex_now=${EPOCHSECONDS:-}
case $archex_now in
'' | *[!0-9]*) archex_now="" ;;
esac

archex_age=""
if [ -n "$archex_now" ] && [ "$archex_field_epoch" -gt 0 ]; then
	archex_age=$((archex_now - archex_field_epoch))
	if [ "$archex_age" -lt 0 ]; then
		archex_age=0
	fi
fi

archex_budget=${ARCHEX_STATUS_STALE_AFTER_SECONDS:-$archex_stale_default}
case $archex_budget in
'' | *[!0-9]*) archex_budget=$archex_stale_default ;;
esac
if [ -n "$archex_age" ] && [ "$archex_age" -gt "$archex_budget" ]; then
	archex_field_state=stale
fi

archex_watch=""
if [ -n "$archex_now" ] && [ "$archex_field_watch_epoch" -gt 0 ]; then
	if [ $((archex_now - archex_field_watch_epoch)) -le "$archex_watch_ttl" ]; then
		archex_watch=" - watch"
	fi
fi

# --- Render -----------------------------------------------------------------
archex_detail=""
case $archex_field_state in
fresh) archex_detail=" - $archex_field_files files, $archex_field_chunks chunks" ;;
pending)
	# The model's field defaults to true, so an absent field means a complete
	# view here too; only an explicit `false` marks the count as a lower bound.
	if [ "$archex_field_pending_complete" = "false" ]; then
		archex_detail=" - $archex_field_pending+ awaiting sync"
	else
		archex_detail=" - $archex_field_pending awaiting sync"
	fi
	;;
dirty)
	if [ "$archex_field_reindex" = "true" ]; then
		archex_detail=" - reindex required"
	else
		archex_detail=" - index behind tree"
	fi
	;;
stale) archex_detail=" - unverified since measurement" ;;
esac

archex_revision_segment=""
if [ -n "$archex_field_revision" ]; then
	archex_revision_short=$archex_field_revision
	# Trim to the first eight characters, but only when there are more than
	# eight: for a shorter value the `????????` pattern does not match and the
	# suffix trim would collapse the whole string to empty.
	case $archex_field_revision in
	?????????*)
		archex_revision_tail=${archex_field_revision#????????}
		archex_revision_short=${archex_field_revision%"$archex_revision_tail"}
		;;
	esac
	archex_revision_segment=" - rev $archex_revision_short"
fi

archex_age_segment=""
if [ -n "$archex_age" ]; then
	if [ "$archex_age" -lt 60 ]; then
		archex_age_segment=" - ${archex_age}s ago"
	elif [ "$archex_age" -lt 3600 ]; then
		archex_age_segment=" - $((archex_age / 60))m ago"
	else
		archex_age_segment=" - $((archex_age / 3600))h ago"
	fi
fi

printf 'archex %s%s%s%s%s\n' \
	"$archex_field_state" \
	"$archex_detail" \
	"$archex_revision_segment" \
	"$archex_age_segment" \
	"$archex_watch"
"""
