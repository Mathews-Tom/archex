"""Client install/bootstrap helpers for MCP-compatible archex setups."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

ClientName = Literal["claude-code", "codex", "cursor", "opencode", "pi"]
ClientScope = Literal["project", "user"]

_USER_ONLY_CLIENTS: frozenset[ClientName] = frozenset({"pi"})


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
    if plan.client == "opencode" and "$schema" not in existing_payload:
        existing_payload["$schema"] = "https://opencode.ai/config.json"
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
    raise ValueError(f"unsupported client: {client}")


def _render_content(client: ClientName) -> str:
    if client == "codex":
        return '[mcp_servers.archex]\ncommand = "archex"\nargs = ["mcp"]\n'
    if client == "opencode":
        payload = {
            "$schema": "https://opencode.ai/config.json",
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
    return "config-shape verified; client smoke unverified"
