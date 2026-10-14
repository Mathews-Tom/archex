"""Annotation hook plan builders and per-host dispatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from archex.client_setup.claude_code import apply_hook_action, hook_settings_path, render_hook_entry
from archex.client_setup.codex import (
    render_codex_hook_block,
    render_codex_hook_preview,
    write_codex_hook_plan,
)
from archex.client_setup.cursor import (
    cursor_hook_settings_path,
    render_cursor_hook_preview,
    write_cursor_hook_plan,
)
from archex.client_setup.io import read_json_object, write_config_atomic
from archex.client_setup.models import (
    ClaudeCodeHookInstallPlan,
    ClientName,
    ClientScope,
    CodexHookInstallPlan,
    CursorHookInstallPlan,
    HookAction,
    HookInstallPlan,
    TsHookInstallPlan,
)
from archex.client_setup.registration import client_config_path, resolve_hook_scope
from archex.client_setup.ts_annotation import (
    render_ts_hook_module,
    render_ts_hook_preview,
    ts_hook_module_path,
    write_ts_hook_plan,
)
from archex.integrations.claude_code_annotate_hook import HOOK_MATCHER


def build_hook_install_plan(
    client: ClientName,
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
    action: HookAction,
) -> HookInstallPlan:
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    if client == "claude-code":
        selected_scope = resolve_hook_scope(source, scope)
        return ClaudeCodeHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=hook_settings_path(repo_root, selected_scope),
            action=action,
            hook_entry=render_hook_entry(),
        )
    if client in {"omp", "pi", "opencode"}:
        selected_scope = resolve_hook_scope(source, scope)
        return TsHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=ts_hook_module_path(client, repo_root, selected_scope),
            action=action,
            module_content=render_ts_hook_module(client),
        )
    if client == "codex":
        selected_scope = resolve_hook_scope(source, scope)
        return CodexHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=client_config_path(client, repo_root, selected_scope),
            action=action,
            block_content=render_codex_hook_block(),
        )
    if client == "cursor":
        selected_scope = resolve_hook_scope(source, scope)
        return CursorHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=cursor_hook_settings_path(repo_root, selected_scope),
            action=action,
            hook_entry=_render_cursor_hook_entry(),
        )
    raise ValueError(
        "--hooks/--remove-hooks is only supported for claude-code (M19), omp, pi (M20), "
        f"codex (M21), opencode (M22), cursor (M23); got {client!r}"
    )


def write_hook_install_plan(plan: HookInstallPlan) -> Path:
    if isinstance(plan, TsHookInstallPlan):
        return write_ts_hook_plan(plan)
    if isinstance(plan, CodexHookInstallPlan):
        return write_codex_hook_plan(plan)
    if isinstance(plan, CursorHookInstallPlan):
        return write_cursor_hook_plan(plan)
    target = plan.target_path
    existing = read_json_object(target) if target.exists() else {}
    updated, changed = apply_hook_action(existing, plan)
    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    write_config_atomic(target, json.dumps(updated, indent=2) + "\n")
    return target


def render_hook_install_preview(plan: HookInstallPlan) -> str:
    if isinstance(plan, TsHookInstallPlan):
        return render_ts_hook_preview(plan)
    if isinstance(plan, CodexHookInstallPlan):
        return render_codex_hook_preview(plan)
    if isinstance(plan, CursorHookInstallPlan):
        return render_cursor_hook_preview(plan)
    existing = read_json_object(plan.target_path) if plan.target_path.exists() else {}
    updated, changed = apply_hook_action(existing, plan)
    action_label = "Install" if plan.action == "install" else "Remove"
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {plan.target_path}",
        (
            f"Action: {action_label} PostToolUse search-annotation hook "
            f"(matcher: {HOOK_MATCHER!r}; appends the indexed code units each hit falls in "
            "to Bash/Grep/Glob search results) and remove the retired archex PreToolUse "
            "pattern-search entry"
            if plan.action == "install"
            else f"Action: {action_label} the PostToolUse search-annotation hook "
            f"(matcher: {HOOK_MATCHER!r}) and any retired archex PreToolUse "
            "pattern-search entry"
        ),
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


def _render_cursor_hook_entry() -> dict[str, object]:
    return {
        "command": f"{sys.executable} -m archex.integrations.cursor_hook",
        "timeout": 1,
    }
