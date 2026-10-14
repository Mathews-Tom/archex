"""Client and agent-file discovery with multi-client previews."""

from __future__ import annotations

from pathlib import Path

from archex.client_setup.models import ClientInstallPlan, ClientName, ClientScope, DiscoveredClient
from archex.client_setup.registration import (
    USER_ONLY_CLIENTS,
    build_client_install_plan,
    is_toml_plan,
)


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
        scopes: list[ClientScope] = ["user"] if client in USER_ONLY_CLIENTS else ["user", "project"]
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
        if is_toml_plan(plan):
            lines.append(f"- {plan.target_path}: add [mcp_servers.archex]")
        elif plan.client == "opencode":
            lines.append(f"- {plan.target_path}: add mcp.archex")
        else:
            lines.append(f"- {plan.target_path}: add mcpServers.archex")

    lines.append("\nNo existing non-archex entries will be removed.")
    return "\n".join(lines) + "\n"
