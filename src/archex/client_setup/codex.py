"""Codex annotation hook TOML rendering and installation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from archex.client_setup.io import (
    marked_block_span,
    strip_marked_toml_block,
    upsert_marked_toml_block,
    write_config_atomic,
)
from archex.integrations.codex_annotate_hook import HOOK_MATCHER as CODEX_HOOK_MATCHER

if TYPE_CHECKING:
    from archex.client_setup.models import CodexHookInstallPlan

#: Substring in the Codex ``PostToolUse`` hook command that identifies it as the
#: archex search-annotation hook.
_CODEX_HOOK_ARGS_MARKER = "archex.integrations.codex_annotate_hook"


def write_codex_hook_plan(plan: CodexHookInstallPlan) -> Path:
    target = plan.target_path
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    updated = _apply_codex_hook_block(existing, plan)
    if updated == existing:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    write_config_atomic(target, updated)
    return target


def render_codex_hook_preview(plan: CodexHookInstallPlan) -> str:
    target = plan.target_path
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    updated = _apply_codex_hook_block(existing, plan)
    action_label = "Install" if plan.action == "install" else "Remove"
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {target}",
        (
            f"Action: {action_label} PostToolUse search-annotation hook "
            f"(matcher: {CODEX_HOOK_MATCHER!r}; appends the indexed code units each hit "
            "falls in to shell search results) and remove the retired archex PreToolUse "
            "diagnostics block"
            if plan.action == "install"
            else f"Action: {action_label} the PostToolUse search-annotation hook "
            f"(matcher: {CODEX_HOOK_MATCHER!r}) and any retired archex PreToolUse "
            "diagnostics block"
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


#: Marker comments delimiting the archex-owned block appended to Codex's
#: ``config.toml`` -- lets install/remove find and replace exactly the block
#: this installer wrote (and nothing else a user configured in the same
#: file) without parsing/re-serializing TOML.
_CODEX_HOOK_BLOCK_START = "# archex:codex-annotate-hook start"
_CODEX_HOOK_BLOCK_END = "# archex:codex-annotate-hook end"

#: Markers of the retired diagnostics-only ``PreToolUse`` block. It is stripped
#: on install and remove so an upgrade leaves exactly one archex search hook.
_CODEX_LEGACY_HOOK_BLOCK_START = "# archex:codex-hook start"
_CODEX_LEGACY_HOOK_BLOCK_END = "# archex:codex-hook end"

#: Codex kills a hook command after this many seconds. The hook enforces its own
#: 0.5 s budget (``ARCHEX_HOOK_TIMEOUT_SECONDS``); this is only the outer bound,
#: and Codex clamps it to at least one second.
_CODEX_HOOK_TIMEOUT_SECONDS = 2


def render_codex_hook_block() -> str:
    command = f"{sys.executable} -m {_CODEX_HOOK_ARGS_MARKER}"
    return (
        "\n".join(
            [
                _CODEX_HOOK_BLOCK_START,
                "[[hooks.PostToolUse]]",
                f'matcher = "{CODEX_HOOK_MATCHER}"',
                "",
                "[[hooks.PostToolUse.hooks]]",
                'type = "command"',
                f'command = "{command}"',
                f"timeout = {_CODEX_HOOK_TIMEOUT_SECONDS}",
                _CODEX_HOOK_BLOCK_END,
            ]
        )
        + "\n"
    )


def _strip_codex_hook_block(existing: str) -> str:
    """Remove the annotation hook block and the retired diagnostics-only block."""
    without_legacy = strip_marked_toml_block(
        existing, _CODEX_LEGACY_HOOK_BLOCK_START, _CODEX_LEGACY_HOOK_BLOCK_END
    )
    return strip_marked_toml_block(without_legacy, _CODEX_HOOK_BLOCK_START, _CODEX_HOOK_BLOCK_END)


def _apply_codex_hook_block(existing: str, plan: CodexHookInstallPlan) -> str:
    if plan.action == "remove":
        return _strip_codex_hook_block(existing)
    # A block is replaced where it sits (the annotation block on a reinstall,
    # else the retired one on an upgrade), so neighbouring archex blocks never
    # reorder; only a first install appends.
    own = marked_block_span(existing, _CODEX_HOOK_BLOCK_START, _CODEX_HOOK_BLOCK_END)
    legacy = marked_block_span(
        existing, _CODEX_LEGACY_HOOK_BLOCK_START, _CODEX_LEGACY_HOOK_BLOCK_END
    )
    if own is None and legacy is not None:
        return existing[: legacy[0]] + plan.block_content + existing[legacy[1] :]
    upserted = upsert_marked_toml_block(
        existing, plan.block_content, _CODEX_HOOK_BLOCK_START, _CODEX_HOOK_BLOCK_END
    )
    return strip_marked_toml_block(
        upserted, _CODEX_LEGACY_HOOK_BLOCK_START, _CODEX_LEGACY_HOOK_BLOCK_END
    )
