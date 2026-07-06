"""Client install/bootstrap helpers for MCP-compatible archex setups."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from archex.integrations.hook import HOOK_MATCHER

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


def build_client_install_plan(
    client: ClientName,
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
) -> ClientInstallPlan:
    selected_scope = _resolve_scope(client, source, scope)
    if client in _USER_ONLY_CLIENTS and selected_scope != "user":
        raise ValueError(f"{client} client config supports only --scope user")
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    target_path = _target_path(client, repo_root, selected_scope)
    content = _render_content(client)
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
class HookInstallPlan:
    """Install or remove the Claude Code PreToolUse hook (M19; claude-code only)."""

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    hook_entry: dict[str, object]


def build_hook_install_plan(
    client: ClientName,
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
    action: HookAction,
) -> HookInstallPlan:
    if client != "claude-code":
        raise ValueError(
            f"--hooks/--remove-hooks is only supported for claude-code (M19 scope); got {client!r}"
        )
    selected_scope = _resolve_scope(client, source, scope)
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    return HookInstallPlan(
        client=client,
        scope=selected_scope,
        target_path=_hook_settings_path(repo_root, selected_scope),
        action=action,
        hook_entry=_render_hook_entry(),
    )


def write_hook_install_plan(plan: HookInstallPlan) -> Path:
    target = plan.target_path
    existing = _read_json_object(target) if target.exists() else {}
    updated, changed = _apply_hook_action(existing, plan)
    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return target


def render_hook_install_preview(plan: HookInstallPlan) -> str:
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
    payload: dict[str, object], plan: HookInstallPlan
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


def _render_content(client: ClientName) -> str:
    if client == "codex":
        return '[mcp_servers.archex]\ncommand = "archex"\nargs = ["mcp"]\n'
    if client == "opencode":
        payload = {
            "$schema": _OPENCODE_SCHEMA,
            "mcp": {
                "archex": {
                    "type": "local",
                    "command": ["archex", "mcp"],
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
                    "args": ["mcp"],
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
                    "args": ["mcp"],
                }
            },
        }
        return json.dumps(payload, indent=2) + "\n"
    payload = {
        "mcpServers": {
            "archex": {
                "command": "archex",
                "args": ["mcp"],
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
