"""MCP registration targets, rendering and install writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from archex.client_setup.io import write_config_atomic
from archex.client_setup.models import ClientInstallPlan, ClientName, ClientScope
from archex.integrations.mcp import resolve_tool_scope

USER_ONLY_CLIENTS: frozenset[ClientName] = frozenset({"pi", "omp"})
_OMP_SCHEMA = (
    "https://raw.githubusercontent.com/can1357/oh-my-pi/main/"
    "packages/coding-agent/src/config/mcp-schema.json"
)
_OPENCODE_SCHEMA = "https://opencode.ai/config.json"
_CLIENT_SCHEMA: dict[ClientName, str] = {
    "opencode": _OPENCODE_SCHEMA,
    "omp": _OMP_SCHEMA,
}


def build_client_install_plan(
    client: ClientName,
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
    tool_scope: str | None = None,
    disclosure: bool = True,
) -> ClientInstallPlan:
    selected_scope = _resolve_scope(client, source, scope)
    if client in USER_ONLY_CLIENTS and selected_scope != "user":
        raise ValueError(f"{client} client config supports only --scope user")
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    target_path = client_config_path(client, repo_root, selected_scope)
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
    if is_toml_plan(plan):
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
        write_config_atomic(target, new_content)
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
    write_config_atomic(target, json.dumps(existing_payload, indent=2) + "\n")
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
    if client in USER_ONLY_CLIENTS:
        return "user"
    return "project" if source is not None else "user"


def resolve_hook_scope(source: str | Path | None, scope: ClientScope | None) -> ClientScope:
    """Scope resolution for hook installs (claude-code, omp, pi).

    Unlike ``_resolve_scope`` (MCP server config, where omp/pi are user-scope
    only per the ``CLIENT_COMPATIBILITY_MATRIX.md`` convention), the hook
    installer supports both scopes for every client it handles: project scope
    when a repo ``source`` is given, user scope otherwise.
    """
    if scope is not None:
        return scope
    return "project" if source is not None else "user"


def client_config_path(client: ClientName, repo_root: Path, scope: ClientScope) -> Path:
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


def is_toml_plan(plan: ClientInstallPlan) -> bool:
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
