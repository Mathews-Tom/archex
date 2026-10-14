"""TypeScript annotation module targets, rendering and installation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from archex.client_setup.io import write_config_atomic
from archex.client_setup.omp_annotation import TS_HOOK_MODULE_TEMPLATE
from archex.client_setup.opencode_annotation import OPENCODE_HOOK_MODULE_TEMPLATE

if TYPE_CHECKING:
    from archex.client_setup.models import ClientName, ClientScope, TsHookInstallPlan


def write_ts_hook_plan(plan: TsHookInstallPlan) -> Path:
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


def render_ts_hook_preview(plan: TsHookInstallPlan) -> str:
    target = plan.target_path
    existing = target.read_text(encoding="utf-8") if target.exists() else None
    action_label = "Install" if plan.action == "install" else "Remove"
    lines = [
        f"Client: {plan.client}",
        f"Scope: {plan.scope}",
        f"Target: {target}",
        f"Action: {action_label} archex {_ts_hook_event_label(plan.client)} "
        f"({_ts_hook_coverage(plan.client)})",
    ]
    if plan.action == "install":
        lines.append(
            "No change: hook module already installed and up to date (idempotent no-op)."
            if existing == plan.module_content
            else "Dry run. Re-run without --dry-run to write this file."
        )
        lines.append("")
        lines.append(plan.module_content)
    else:
        lines.append(
            "No change: no archex hook module is installed."
            if existing is None
            else "Dry run. Re-run without --dry-run to remove this file."
        )
    return "\n".join(lines) + "\n"


def _ts_hook_coverage(client: ClientName) -> str:
    """What the installed module acts on, as the preview states it."""
    if client == "opencode":
        return (
            "native grep, glob, and bash search results: appends one line per indexed code unit hit"
        )
    finder = "glob" if client == "omp" else "find"
    return f"grep, {finder}, and bash search results: appends one line per indexed code unit hit"


_TS_HOOK_MODULE_FILENAME = "archex-hook.ts"


def ts_hook_module_path(client: ClientName, repo_root: Path, scope: ClientScope) -> Path:
    if client == "omp":
        return (
            repo_root / ".omp" / "extensions" / _TS_HOOK_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".omp" / "agent" / "extensions" / _TS_HOOK_MODULE_FILENAME
        )
    if client == "pi":
        return (
            repo_root / ".pi" / "extensions" / _TS_HOOK_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".pi" / "agent" / "extensions" / _TS_HOOK_MODULE_FILENAME
        )
    if client == "opencode":
        return (
            repo_root / ".opencode" / "plugins" / _TS_HOOK_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".config" / "opencode" / "plugins" / _TS_HOOK_MODULE_FILENAME
        )
    raise ValueError(f"unsupported TS hook client: {client}")


def _ts_hook_event_label(client: ClientName) -> str:
    """Preview wording for the two structurally different TS hook shapes.

    omp/pi install a `tool_result` handler returning a content patch;
    opencode installs a `tool.execute.after` plugin that appends the
    annotation to its output argument's text in place (see
    `_OPENCODE_HOOK_MODULE_TEMPLATE`).
    """
    return "tool.execute.after plugin" if client == "opencode" else "tool_result hook module"


def render_ts_hook_module(client: ClientName) -> str:
    if client == "opencode":
        return OPENCODE_HOOK_MODULE_TEMPLATE.replace(
            "__ARCHEX_PYTHON_COMMAND__", json.dumps(sys.executable)
        )
    return render_annotation_hook_module(sys.executable)


def render_annotation_hook_module(python_command: str) -> str:
    """The omp/Pi annotation module, running archex from ``python_command``.

    `install-client --hooks` bakes in the installer's own interpreter; the SWE
    A/B harness renders the same module against the interpreter of an archex
    installed inside a task container.
    """
    return TS_HOOK_MODULE_TEMPLATE.replace("__ARCHEX_PYTHON_COMMAND__", json.dumps(python_command))
