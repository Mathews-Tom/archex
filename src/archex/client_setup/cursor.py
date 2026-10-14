"""Cursor annotation hook JSON rendering and installation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from archex.client_setup.io import read_json_object, write_config_atomic

if TYPE_CHECKING:
    from archex.client_setup.models import ClientScope, CursorHookInstallPlan

#: Substring in a Cursor hook entry's ``command`` string that identifies it
#: as archex-owned, so install/remove can find and replace our own entry
#: without disturbing any other ``beforeSubmitPrompt`` hook the user has
#: configured in the same file.
_CURSOR_HOOK_COMMAND_MARKER = "archex.integrations.cursor_hook"


def cursor_hook_settings_path(repo_root: Path, scope: ClientScope) -> Path:
    return (
        repo_root / ".cursor" / "hooks.json"
        if scope == "project"
        else Path.home() / ".cursor" / "hooks.json"
    )


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


def write_cursor_hook_plan(plan: CursorHookInstallPlan) -> Path:
    target = plan.target_path
    existing = read_json_object(target) if target.exists() else {}
    updated, changed = _apply_cursor_hook_action(existing, plan)
    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    write_config_atomic(target, json.dumps(updated, indent=2) + "\n")
    return target


def render_cursor_hook_preview(plan: CursorHookInstallPlan) -> str:
    existing = read_json_object(plan.target_path) if plan.target_path.exists() else {}
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
