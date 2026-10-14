"""Client install/bootstrap helpers with an explicit public surface."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import cast

from archex.client_setup.agent_guidance import (
    AGENT_GUIDANCE_END,
    AGENT_GUIDANCE_PROMPT,
    AGENT_GUIDANCE_START,
    append_agent_guidance,
    render_agent_guidance_block,
    render_agent_guidance_preview,
)
from archex.client_setup.claude_code import (
    build_session_primer_install_plan,
    hook_settings_path,
    render_session_primer_install_preview,
    write_session_primer_install_plan,
)
from archex.client_setup.discovery import (
    build_discovered_install_plans,
    discover_agent_files,
    discover_clients,
    get_client_config_candidates,
    render_multiple_install_preview,
)
from archex.client_setup.hooks import (
    build_hook_install_plan,
    render_hook_install_preview,
    write_hook_install_plan,
)
from archex.client_setup.io import read_json_object, write_config_atomic
from archex.client_setup.models import (
    POST_EDIT_UNSUPPORTED_CLIENTS,
    STATUSLINE_REFRESH_INTERVAL_SECONDS,
    STATUSLINE_SCRIPT_FILENAME,
    STATUSLINE_SCRIPT_MARKER,
    STATUSLINE_UNSUPPORTED_CLIENTS,
    TS_STATUS_KEY,
    ClaudeCodeHookInstallPlan,
    ClaudeCodePostEditHookInstallPlan,
    ClaudeCodeSessionPrimerInstallPlan,
    ClaudeCodeStatuslineInstallPlan,
    ClientInstallPlan,
    ClientName,
    ClientScope,
    CodexHookInstallPlan,
    CodexPostEditHookInstallPlan,
    CursorHookInstallPlan,
    DiscoveredClient,
    HookAction,
    HookInstallPlan,
    PostEditHookInstallPlan,
    StatusInstallPlan,
    TsHookInstallPlan,
    TsPostEditHookInstallPlan,
    TsStatusInstallPlan,
)
from archex.client_setup.post_edit import (
    build_post_edit_hook_install_plan,
    render_post_edit_hook_install_preview,
    write_post_edit_hook_install_plan,
)
from archex.client_setup.registration import (
    build_client_install_plan,
    render_client_install_preview,
    resolve_hook_scope,
    write_client_install_plan,
)
from archex.client_setup.statusline_script import render_statusline_script
from archex.client_setup.statusline_ts import render_ts_status_module, ts_status_module_path
from archex.client_setup.ts_annotation import render_annotation_hook_module

__all__ = [
    "AGENT_GUIDANCE_END",
    "AGENT_GUIDANCE_PROMPT",
    "AGENT_GUIDANCE_START",
    "ClaudeCodeHookInstallPlan",
    "ClaudeCodePostEditHookInstallPlan",
    "ClaudeCodeSessionPrimerInstallPlan",
    "ClaudeCodeStatuslineInstallPlan",
    "ClientInstallPlan",
    "ClientName",
    "ClientScope",
    "CodexHookInstallPlan",
    "CodexPostEditHookInstallPlan",
    "CursorHookInstallPlan",
    "DiscoveredClient",
    "HookAction",
    "HookInstallPlan",
    "POST_EDIT_UNSUPPORTED_CLIENTS",
    "PostEditHookInstallPlan",
    "STATUSLINE_REFRESH_INTERVAL_SECONDS",
    "STATUSLINE_SCRIPT_FILENAME",
    "STATUSLINE_SCRIPT_MARKER",
    "STATUSLINE_UNSUPPORTED_CLIENTS",
    "StatusInstallPlan",
    "TS_STATUS_KEY",
    "TsHookInstallPlan",
    "TsPostEditHookInstallPlan",
    "TsStatusInstallPlan",
    "append_agent_guidance",
    "build_client_install_plan",
    "build_discovered_install_plans",
    "build_hook_install_plan",
    "build_post_edit_hook_install_plan",
    "build_session_primer_install_plan",
    "build_statusline_install_plan",
    "discover_agent_files",
    "discover_clients",
    "get_client_config_candidates",
    "render_agent_guidance_block",
    "render_agent_guidance_preview",
    "render_annotation_hook_module",
    "render_client_install_preview",
    "render_hook_install_preview",
    "render_multiple_install_preview",
    "render_post_edit_hook_install_preview",
    "render_session_primer_install_preview",
    "render_statusline_install_preview",
    "render_statusline_script",
    "render_ts_status_module",
    "statusline_interpreter",
    "write_client_install_plan",
    "write_config_atomic",
    "write_hook_install_plan",
    "write_post_edit_hook_install_plan",
    "write_session_primer_install_plan",
    "write_statusline_install_plan",
]


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
    selected_scope = resolve_hook_scope(source, scope)
    if client == "claude-code":
        script_path = _statusline_script_path(repo_root, selected_scope)
        return ClaudeCodeStatuslineInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=hook_settings_path(repo_root, selected_scope),
            script_path=script_path,
            action=action,
            script_content=render_statusline_script(),
            statusline_entry=_render_statusline_entry(script_path),
        )
    if client in {"omp", "pi"}:
        return TsStatusInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=ts_status_module_path(client, repo_root, selected_scope),
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
            write_config_atomic(target, plan.module_content)
        return target

    target = plan.target_path
    existing = read_json_object(target) if target.exists() else {}
    # Decided before anything is written: a foreign status line raises here,
    # and a refused install must leave no renderer script behind.
    updated, changed = _apply_statusline_action(existing, plan)

    if plan.action == "install":
        plan.script_path.parent.mkdir(parents=True, exist_ok=True)
        existing_script = (
            plan.script_path.read_text(encoding="utf-8") if plan.script_path.exists() else None
        )
        if existing_script != plan.script_content:
            write_config_atomic(plan.script_path, plan.script_content)
        plan.script_path.chmod(0o755)
    elif plan.script_path.exists() and _is_archex_statusline_script(plan.script_path):
        plan.script_path.unlink()

    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    write_config_atomic(target, json.dumps(updated, indent=2) + "\n")
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

    existing = read_json_object(plan.target_path) if plan.target_path.exists() else {}
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
