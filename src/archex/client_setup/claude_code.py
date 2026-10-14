"""Claude Code annotation and session-primer hook settings."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import cast

from archex.client_setup.io import read_json_object, write_config_atomic
from archex.client_setup.models import (
    ClaudeCodeHookInstallPlan,
    ClaudeCodeSessionPrimerInstallPlan,
    ClientScope,
    HookAction,
)
from archex.client_setup.registration import resolve_hook_scope
from archex.integrations.claude_code_annotate_hook import HOOK_MATCHER
from archex.integrations.session_hook import SESSION_START_MATCHER

#: Substring in a Claude Code ``PostToolUse`` handler's ``args`` that
#: identifies it as the archex search-annotation hook, so install/remove can
#: find and replace our own entry without disturbing any other hook the user
#: has configured for the same matcher group.
_HOOK_ARGS_MARKER = "archex.integrations.claude_code_annotate_hook"

#: The retired Claude Code ``PreToolUse`` pattern-search hook. Install and
#: remove both strip any handler carrying this marker from ``PreToolUse``, so a
#: settings file written by an older archex converges on the annotation hook.
_LEGACY_SEARCH_HOOK_ARGS_MARKER = "archex.integrations.hook"
_LEGACY_SEARCH_HOOK_EVENT = "PreToolUse"
_LEGACY_SEARCH_HOOK_MATCHER = "Glob|Grep"

#: Substring in a SessionStart handler's ``args`` that identifies it as
#: archex-owned, so the session-primer installer can preserve existing search
#: hooks and unrelated SessionStart handlers.
_SESSION_PRIMER_ARGS_MARKER = "archex.integrations.session_hook"


def build_session_primer_install_plan(
    source: str | Path | None = None,
    *,
    scope: ClientScope | None = None,
    action: HookAction,
) -> ClaudeCodeSessionPrimerInstallPlan:
    """Build a separate opt-in Claude Code SessionStart primer install plan."""
    repo_root = Path(source if source is not None else ".").expanduser().resolve()
    selected_scope = resolve_hook_scope(source, scope)
    return ClaudeCodeSessionPrimerInstallPlan(
        client="claude-code",
        scope=selected_scope,
        target_path=hook_settings_path(repo_root, selected_scope),
        action=action,
        hook_entry=_render_session_primer_hook_entry(),
    )


def write_session_primer_install_plan(plan: ClaudeCodeSessionPrimerInstallPlan) -> Path:
    """Write a SessionStart primer plan without disturbing other settings."""
    target = plan.target_path
    existing = read_json_object(target) if target.exists() else {}
    updated, changed = _apply_session_primer_action(existing, plan)
    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    write_config_atomic(target, json.dumps(updated, indent=2) + "\n")
    return target


def render_session_primer_install_preview(plan: ClaudeCodeSessionPrimerInstallPlan) -> str:
    """Render a no-write preview for the SessionStart primer installer."""
    existing = read_json_object(plan.target_path) if plan.target_path.exists() else {}
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


def hook_settings_path(repo_root: Path, scope: ClientScope) -> Path:
    return (
        repo_root / ".claude" / "settings.json"
        if scope == "project"
        else Path.home() / ".claude" / "settings.json"
    )


def render_hook_entry() -> dict[str, object]:
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


def _is_archex_hook_entry(entry: object, marker: str = _HOOK_ARGS_MARKER) -> bool:
    if not isinstance(entry, dict):
        return False
    args = cast("dict[str, object]", entry).get("args")
    if not isinstance(args, list):
        return False
    items = cast("list[object]", args)
    return any(isinstance(item, str) and marker in item for item in items)


def apply_claude_event_hook_action(
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


def apply_hook_action(
    payload: dict[str, object], plan: ClaudeCodeHookInstallPlan
) -> tuple[dict[str, object], bool]:
    """Merge the owned PostToolUse annotation handler into Claude Code settings.

    The retired archex ``PreToolUse`` pattern-search handler is stripped first
    on every action, so install replaces it and remove clears both.
    """
    without_legacy, _ = apply_claude_event_hook_action(
        payload,
        event=_LEGACY_SEARCH_HOOK_EVENT,
        marker=_LEGACY_SEARCH_HOOK_ARGS_MARKER,
        matcher=_LEGACY_SEARCH_HOOK_MATCHER,
        hook_entry=plan.hook_entry,
        action="remove",
    )
    updated, _ = apply_claude_event_hook_action(
        without_legacy,
        event="PostToolUse",
        marker=_HOOK_ARGS_MARKER,
        matcher=HOOK_MATCHER,
        hook_entry=plan.hook_entry,
        action=plan.action,
    )
    return updated, updated != payload


def _apply_session_primer_action(
    payload: dict[str, object], plan: ClaudeCodeSessionPrimerInstallPlan
) -> tuple[dict[str, object], bool]:
    """Merge one owned SessionStart handler without touching other settings."""
    return apply_claude_event_hook_action(
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
