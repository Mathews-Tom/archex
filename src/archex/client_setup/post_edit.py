"""Post-edit hook plan builders, rendering and per-host installation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from archex.client_setup.claude_code import apply_claude_event_hook_action, hook_settings_path
from archex.client_setup.io import (
    read_json_object,
    strip_marked_toml_block,
    upsert_marked_toml_block,
    write_config_atomic,
)
from archex.client_setup.models import (
    POST_EDIT_UNSUPPORTED_CLIENTS,
    ClaudeCodePostEditHookInstallPlan,
    ClientName,
    ClientScope,
    CodexPostEditHookInstallPlan,
    HookAction,
    PostEditHookInstallPlan,
    TsPostEditHookInstallPlan,
)
from archex.client_setup.omp_post_edit import TS_POST_EDIT_MODULE_TEMPLATE
from archex.client_setup.opencode_post_edit import OPENCODE_POST_EDIT_MODULE_TEMPLATE
from archex.client_setup.registration import client_config_path, resolve_hook_scope
from archex.integrations.codex_post_edit_hook import POST_EDIT_MATCHER as CODEX_POST_EDIT_MATCHER
from archex.integrations.post_edit_hook import POST_EDIT_MATCHER

# ---------------------------------------------------------------------------
# R21 — post-edit impact hooks
#
# A separate, independently installable and removable surface from the
# PostToolUse search-annotation hook above. It owns its own ownership
# marker and its own module filename on every client, so installing or
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
#: archex post-edit hook. Distinct from ``_HOOK_ARGS_MARKER`` and from
#: ``_LEGACY_SEARCH_HOOK_ARGS_MARKER`` (and not a superstring of either), so
#: the installers cannot strip each other.
_POST_EDIT_ARGS_MARKER = "archex.integrations.post_edit_hook"
_CODEX_POST_EDIT_ARGS_MARKER = "archex.integrations.codex_post_edit_hook"

_CODEX_POST_EDIT_BLOCK_START = "# archex:codex-post-edit-hook start"
_CODEX_POST_EDIT_BLOCK_END = "# archex:codex-post-edit-hook end"

_TS_POST_EDIT_MODULE_FILENAME = "archex-post-edit-hook.ts"


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
    selected_scope = resolve_hook_scope(source, scope)
    if client == "claude-code":
        return ClaudeCodePostEditHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=hook_settings_path(repo_root, selected_scope),
            action=action,
            hook_entry=_render_post_edit_hook_entry(),
        )
    if client == "codex":
        return CodexPostEditHookInstallPlan(
            client=client,
            scope=selected_scope,
            target_path=client_config_path(client, repo_root, selected_scope),
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
            write_config_atomic(target, plan.module_content)
        return target
    if isinstance(plan, CodexPostEditHookInstallPlan):
        target = plan.target_path
        existing_toml = target.read_text(encoding="utf-8") if target.exists() else ""
        updated_toml = _apply_codex_post_edit_block(existing_toml, plan)
        if updated_toml == existing_toml:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        write_config_atomic(target, updated_toml)
        return target
    target = plan.target_path
    existing_json = read_json_object(target) if target.exists() else {}
    updated_json, changed = _apply_post_edit_hook_action(existing_json, plan)
    if not changed:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    write_config_atomic(target, json.dumps(updated_json, indent=2) + "\n")
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
    existing_json = read_json_object(plan.target_path) if plan.target_path.exists() else {}
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
    return apply_claude_event_hook_action(
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
    return strip_marked_toml_block(
        existing, _CODEX_POST_EDIT_BLOCK_START, _CODEX_POST_EDIT_BLOCK_END
    )


def _apply_codex_post_edit_block(existing: str, plan: CodexPostEditHookInstallPlan) -> str:
    if plan.action == "remove":
        return _strip_codex_post_edit_block(existing)
    return upsert_marked_toml_block(
        existing, plan.block_content, _CODEX_POST_EDIT_BLOCK_START, _CODEX_POST_EDIT_BLOCK_END
    )


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
        OPENCODE_POST_EDIT_MODULE_TEMPLATE if client == "opencode" else TS_POST_EDIT_MODULE_TEMPLATE
    )
    return template.replace("__ARCHEX_PYTHON_COMMAND__", json.dumps(sys.executable)).replace(
        "__ARCHEX_CLIENT__", client
    )
