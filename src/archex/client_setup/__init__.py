"""Client install/bootstrap helpers with an explicit public surface."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
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
    apply_claude_event_hook_action,
    apply_hook_action,
    build_session_primer_install_plan,
    hook_settings_path,
    render_hook_entry,
    render_session_primer_install_preview,
    write_session_primer_install_plan,
)
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
from archex.client_setup.discovery import (
    build_discovered_install_plans,
    discover_agent_files,
    discover_clients,
    get_client_config_candidates,
    render_multiple_install_preview,
)
from archex.client_setup.io import (
    read_json_object,
    strip_marked_toml_block,
    upsert_marked_toml_block,
    write_config_atomic,
)
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
from archex.client_setup.registration import (
    build_client_install_plan,
    client_config_path,
    render_client_install_preview,
    resolve_hook_scope,
    write_client_install_plan,
)
from archex.client_setup.ts_annotation import (
    render_annotation_hook_module,
    render_ts_hook_module,
    render_ts_hook_preview,
    ts_hook_module_path,
    write_ts_hook_plan,
)
from archex.integrations.claude_code_annotate_hook import HOOK_MATCHER
from archex.integrations.codex_post_edit_hook import POST_EDIT_MATCHER as CODEX_POST_EDIT_MATCHER
from archex.integrations.post_edit_hook import POST_EDIT_MATCHER
from archex.project import PROJECT_DIR_NAME
from archex.status_snapshot import (
    DEFAULT_STALE_AFTER_SECONDS,
    SNAPSHOT_FILENAME,
    STATUS_SNAPSHOT_VERSION,
    WATCH_OBSERVATION_TTL_SECONDS,
)

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


TS_POST_EDIT_MODULE_TEMPLATE = r"""/**
 * archex shared post-edit `tool_result` module (R21 — oh-my-pi / Pi).
 *
 * Installed by `archex install-client omp --post-edit-hooks` /
 * `archex install-client pi --post-edit-hooks` (opt-in; never installed by
 * default). A separate file from the search hook's `archex-hook.ts`, so the
 * two surfaces install and remove independently.
 *
 * Upstream contract (verified against each host's own source, not docs):
 * - Both hosts expose `pi.on("tool_result", handler)`, fired after a tool
 *   executes, with `{ toolName, input, content, details, isError }` and a
 *   partial-patch return. `isError` is the explicit success signal — this
 *   module records an edit only when it is not `true`.
 * - `tool_result` fires after execution and cannot block or fail the edit;
 *   the blocking event is the separate pre-execution `tool_call`.
 * - Both hosts' `edit` and `write` schemas carry the target in `path`
 *   (oh-my-pi `dist/types/edit/schemas.d.ts` and `dist/types/tools/write.d.ts`;
 *   Pi `dist/core/tools/edit.d.ts` and `.../write.d.ts`).
 *
 * Every path resolves without throwing. A spawn failure, a timeout, a stale
 * index, or a malformed response all degrade to returning `undefined` (no
 * content patch), logged to the same diagnostics file the Python hooks use.
 */
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// --- Baked in at install time (`archex install-client <client> --post-edit-hooks`) ---

/** Python interpreter active when the installer ran, so this always runs in
 * the environment archex was installed into. */
const ARCHEX_PYTHON_COMMAND = __ARCHEX_PYTHON_COMMAND__;
const ARCHEX_PYTHON_ARGS = ["-m", "archex.integrations.post_edit_hook"];

/** Wall-clock kill timer for the subprocess. Deliberately larger than the
 * search hook's 500ms — a post-edit cycle reparses changed files — and
 * larger than the Python side's own DEFAULT_POST_EDIT_TIMEOUT_SECONDS so
 * the Python deadline, which logs a diagnostic, normally fires first. */
const ARCHEX_POST_EDIT_TIMEOUT_MS = 15000;

const ARCHEX_DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG";

/** Native edit tool name -> the Claude Code tool name the Python subprocess
 * expects. Verified against each host's own tool definitions, not docs. */
const ARCHEX_EDIT_TOOLS: Readonly<Record<string, "Edit" | "Write">> = {
  edit: "Edit",
  write: "Write",
};

/** Input fields that can hold the edited path, most-specific first. Both
 * oh-my-pi's and Pi's `edit`/`write` schemas name it `path`; OpenCode's name
 * it `filePath`. The others are accepted defensively — the Python side
 * rejects anything that is not a real path inside the repository, so a wrong
 * guess degrades to no output rather than a wrong claim. */
const ARCHEX_PATH_FIELDS = ["filePath", "path", "file_path", "notebook_path"] as const;

/** Declared to the Python subprocess as `archex_client` so the emitted
 * receipt names the host that actually ran the edit. The subprocess
 * allowlists this value; an unknown one falls back to `claude-code`. */
const ARCHEX_CLIENT = "__ARCHEX_CLIENT__";

function firstPath(args: Record<string, unknown> | undefined): string | null {
  if (!args) return null;
  for (const field of ARCHEX_PATH_FIELDS) {
    const value = args[field];
    if (typeof value === "string" && value.trim().length > 0) return value;
  }
  return null;
}

// --- Diagnostics (parity with hook.py's `log_diagnostic`) ---

function diagnosticsLogPath(): string {
  const override = process.env[ARCHEX_DIAGNOSTICS_LOG_ENV_VAR];
  if (override && override.trim().length > 0) return override;
  return join(homedir(), ".archex", "hook-diagnostics.log");
}

function logDiagnostic(kind: string, detail: string, cwd?: string): void {
  try {
    const path = diagnosticsLogPath();
    mkdirSync(dirname(path), { recursive: true });
    const entry: Record<string, string> = {
      timestamp: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
      kind,
      detail,
    };
    if (cwd) entry.cwd = cwd;
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf-8");
  } catch {
    // Diagnostics logging must never raise into the host's edit path.
  }
}

// --- Subprocess call: `python -m archex.integrations.post_edit_hook` ---

function runArchexPostEditSubprocess(
  payload: Record<string, unknown>,
  cwd: string,
): Promise<string | null> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value: string | null): void => {
      if (settled) return;
      settled = true;
      resolve(value);
    };

    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(ARCHEX_PYTHON_COMMAND, ARCHEX_PYTHON_ARGS, {
        cwd,
        stdio: ["pipe", "pipe", "ignore"],
      });
    } catch (err) {
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
      finish(null);
      return;
    }

    const timer = setTimeout(() => {
      logDiagnostic(
        "ts_post_edit_timeout",
        `post-edit cycle exceeded ${ARCHEX_POST_EDIT_TIMEOUT_MS}ms`,
        cwd,
      );
      try {
        child.kill("SIGKILL");
      } catch {
        // Already exited.
      }
      finish(null);
    }, ARCHEX_POST_EDIT_TIMEOUT_MS);

    let stdout = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
      finish(null);
    });
    child.on("close", () => {
      clearTimeout(timer);
      finish(stdout.length > 0 ? stdout : null);
    });

    try {
      child.stdin?.write(JSON.stringify(payload));
      child.stdin?.end();
    } catch (err) {
      clearTimeout(timer);
      logDiagnostic("ts_post_edit_stdin_error", String(err), cwd);
      finish(null);
    }
  });
}

function extractAdditionalContext(rawStdout: string): string | null {
  try {
    const parsed: unknown = JSON.parse(rawStdout);
    if (typeof parsed !== "object" || parsed === null) return null;
    const hookSpecificOutput = (parsed as Record<string, unknown>).hookSpecificOutput;
    if (typeof hookSpecificOutput !== "object" || hookSpecificOutput === null) return null;
    const context = (hookSpecificOutput as Record<string, unknown>).additionalContext;
    return typeof context === "string" && context.length > 0 ? context : null;
  } catch {
    return null;
  }
}

// --- Minimal structural types for the `tool_result` contract ---
//
// Declared locally (never imported from either host package) so this module
// has zero import-resolution dependency on which host loaded it.

interface ToolResultEventLike {
  toolName: string;
  input?: Record<string, unknown>;
  content?: unknown[];
  details?: unknown;
  isError?: boolean;
}

interface ToolResultPatch {
  content?: unknown[];
  details?: unknown;
  isError?: boolean;
}

type ToolResultHandler = (
  event: ToolResultEventLike,
  ctx: unknown,
) => Promise<ToolResultPatch | undefined>;

interface HookHost {
  on(event: "tool_result", handler: ToolResultHandler): unknown;
}

// --- Extension entry point ---

export default function archexPostEditHook(pi: HookHost): void {
  pi.on("tool_result", async (event) => {
    try {
      const claudeToolName = ARCHEX_EDIT_TOOLS[event.toolName];
      if (!claudeToolName) return undefined; // never touches read/grep/glob/bash
      if (event.isError === true) return undefined; // only successful edits count

      const filePath = firstPath(event.input);
      if (filePath === null) return undefined;

      const cwd = process.cwd();
      const rawStdout = await runArchexPostEditSubprocess(
        {
          tool_name: claudeToolName,
          tool_input: { file_path: filePath },
          cwd,
          archex_client: ARCHEX_CLIENT,
        },
        cwd,
      );
      if (rawStdout === null) return undefined;

      const context = extractAdditionalContext(rawStdout);
      if (context === null) return undefined;

      const existingContent = Array.isArray(event.content) ? event.content : [];
      return {
        content: [...existingContent, { type: "text", text: `\n\n${context}` }],
      };
    } catch (err) {
      logDiagnostic("ts_post_edit_internal_error", String(err));
      return undefined;
    }
  });
}
"""


OPENCODE_POST_EDIT_MODULE_TEMPLATE = r"""/**
 * archex OpenCode post-edit `tool.execute.after` plugin (R21).
 *
 * Installed by `archex install-client opencode --post-edit-hooks` (opt-in).
 * A separate plugin file from the search hook's `archex-hook.ts`, so the two
 * surfaces install and remove independently. OpenCode auto-loads plugin
 * files from `.opencode/plugins/` and `~/.config/opencode/plugins/`, so no
 * `opencode.json` entry is written or needed.
 *
 * Upstream contract (verified against `packages/plugin/src/index.ts`):
 * - `"tool.execute.after"` receives `input { tool, sessionID, callID, args }`
 *   and a mutable `output { title, output, metadata }`. It runs after the
 *   tool executed and has no documented way to block or fail it.
 * - The hook exposes no error flag, so a failed edit is not distinguishable
 *   here; the Python side revalidates every recorded path against the real
 *   working tree before anything is emitted, so a phantom path simply
 *   produces no output.
 * - Text reaches the agent only by appending to `output.output`; there is no
 *   additional-context field on this event.
 * - `packages/opencode/src/tool/edit.ts` and `.../write.ts` define the
 *   built-in `edit` and `write` tools, both parameterised by `filePath`.
 *
 * Every path resolves without throwing and leaves `output` untouched when
 * archex has nothing fresh to add.
 */
import { spawn } from "node:child_process";
import { appendFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

// --- Baked in at install time (`archex install-client <client> --post-edit-hooks`) ---

/** Python interpreter active when the installer ran, so this always runs in
 * the environment archex was installed into. */
const ARCHEX_PYTHON_COMMAND = __ARCHEX_PYTHON_COMMAND__;
const ARCHEX_PYTHON_ARGS = ["-m", "archex.integrations.post_edit_hook"];

/** Wall-clock kill timer for the subprocess. Deliberately larger than the
 * search hook's 500ms — a post-edit cycle reparses changed files — and
 * larger than the Python side's own DEFAULT_POST_EDIT_TIMEOUT_SECONDS so
 * the Python deadline, which logs a diagnostic, normally fires first. */
const ARCHEX_POST_EDIT_TIMEOUT_MS = 15000;

const ARCHEX_DIAGNOSTICS_LOG_ENV_VAR = "ARCHEX_HOOK_DIAGNOSTICS_LOG";

/** Native edit tool name -> the Claude Code tool name the Python subprocess
 * expects. Verified against each host's own tool definitions, not docs. */
const ARCHEX_EDIT_TOOLS: Readonly<Record<string, "Edit" | "Write">> = {
  edit: "Edit",
  write: "Write",
};

/** Input fields that can hold the edited path, most-specific first. Both
 * oh-my-pi's and Pi's `edit`/`write` schemas name it `path`; OpenCode's name
 * it `filePath`. The others are accepted defensively — the Python side
 * rejects anything that is not a real path inside the repository, so a wrong
 * guess degrades to no output rather than a wrong claim. */
const ARCHEX_PATH_FIELDS = ["filePath", "path", "file_path", "notebook_path"] as const;

/** Declared to the Python subprocess as `archex_client` so the emitted
 * receipt names the host that actually ran the edit. The subprocess
 * allowlists this value; an unknown one falls back to `claude-code`. */
const ARCHEX_CLIENT = "__ARCHEX_CLIENT__";

function firstPath(args: Record<string, unknown> | undefined): string | null {
  if (!args) return null;
  for (const field of ARCHEX_PATH_FIELDS) {
    const value = args[field];
    if (typeof value === "string" && value.trim().length > 0) return value;
  }
  return null;
}

// --- Diagnostics (parity with hook.py's `log_diagnostic`) ---

function diagnosticsLogPath(): string {
  const override = process.env[ARCHEX_DIAGNOSTICS_LOG_ENV_VAR];
  if (override && override.trim().length > 0) return override;
  return join(homedir(), ".archex", "hook-diagnostics.log");
}

function logDiagnostic(kind: string, detail: string, cwd?: string): void {
  try {
    const path = diagnosticsLogPath();
    mkdirSync(dirname(path), { recursive: true });
    const entry: Record<string, string> = {
      timestamp: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
      kind,
      detail,
    };
    if (cwd) entry.cwd = cwd;
    appendFileSync(path, `${JSON.stringify(entry)}\n`, "utf-8");
  } catch {
    // Diagnostics logging must never raise into the host's edit path.
  }
}

// --- Subprocess call: `python -m archex.integrations.post_edit_hook` ---

function runArchexPostEditSubprocess(
  payload: Record<string, unknown>,
  cwd: string,
): Promise<string | null> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value: string | null): void => {
      if (settled) return;
      settled = true;
      resolve(value);
    };

    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(ARCHEX_PYTHON_COMMAND, ARCHEX_PYTHON_ARGS, {
        cwd,
        stdio: ["pipe", "pipe", "ignore"],
      });
    } catch (err) {
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
      finish(null);
      return;
    }

    const timer = setTimeout(() => {
      logDiagnostic(
        "ts_post_edit_timeout",
        `post-edit cycle exceeded ${ARCHEX_POST_EDIT_TIMEOUT_MS}ms`,
        cwd,
      );
      try {
        child.kill("SIGKILL");
      } catch {
        // Already exited.
      }
      finish(null);
    }, ARCHEX_POST_EDIT_TIMEOUT_MS);

    let stdout = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf-8");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      logDiagnostic("ts_post_edit_spawn_error", String(err), cwd);
      finish(null);
    });
    child.on("close", () => {
      clearTimeout(timer);
      finish(stdout.length > 0 ? stdout : null);
    });

    try {
      child.stdin?.write(JSON.stringify(payload));
      child.stdin?.end();
    } catch (err) {
      clearTimeout(timer);
      logDiagnostic("ts_post_edit_stdin_error", String(err), cwd);
      finish(null);
    }
  });
}

function extractAdditionalContext(rawStdout: string): string | null {
  try {
    const parsed: unknown = JSON.parse(rawStdout);
    if (typeof parsed !== "object" || parsed === null) return null;
    const hookSpecificOutput = (parsed as Record<string, unknown>).hookSpecificOutput;
    if (typeof hookSpecificOutput !== "object" || hookSpecificOutput === null) return null;
    const context = (hookSpecificOutput as Record<string, unknown>).additionalContext;
    return typeof context === "string" && context.length > 0 ? context : null;
  } catch {
    return null;
  }
}

// --- Minimal structural types for the plugin contract ---

interface ToolExecuteAfterInput {
  tool: string;
  sessionID?: string;
  callID?: string;
  args?: Record<string, unknown>;
}

interface ToolExecuteAfterOutput {
  title?: string;
  output?: string;
  metadata?: unknown;
}

type Plugin = (context: { directory?: string }) => Promise<{
  "tool.execute.after"?: (
    input: ToolExecuteAfterInput,
    output: ToolExecuteAfterOutput,
  ) => Promise<void>;
}>;

// --- Plugin entry point ---

export const ArchexPostEditPlugin: Plugin = async ({ directory }) => {
  return {
    "tool.execute.after": async (input, output) => {
      try {
        const claudeToolName = ARCHEX_EDIT_TOOLS[input.tool];
        if (!claudeToolName) return; // never touches read/grep/glob/bash

        const filePath = firstPath(input.args);
        if (filePath === null) return;

        const cwd = directory ?? process.cwd();
        const rawStdout = await runArchexPostEditSubprocess(
          {
            tool_name: claudeToolName,
            tool_input: { file_path: filePath },
            cwd,
            archex_client: ARCHEX_CLIENT,
          },
          cwd,
        );
        if (rawStdout === null) return;

        const context = extractAdditionalContext(rawStdout);
        if (context === null) return;

        output.output = `${output.output ?? ""}\n\n${context}`;
      } catch (err) {
        logDiagnostic("ts_post_edit_internal_error", String(err));
      }
    },
  };
};

export default ArchexPostEditPlugin;
"""


#: Filename of the installed omp/Pi status extension module. Its own name is
#: the ownership marker, exactly as the hook modules' filenames are.
_TS_STATUS_MODULE_FILENAME = "archex-status.ts"


def ts_status_module_path(client: ClientName, repo_root: Path, scope: ClientScope) -> Path:
    if client == "omp":
        return (
            repo_root / ".omp" / "extensions" / _TS_STATUS_MODULE_FILENAME
            if scope == "project"
            else Path.home() / ".omp" / "agent" / "extensions" / _TS_STATUS_MODULE_FILENAME
        )
    return (
        repo_root / ".pi" / "extensions" / _TS_STATUS_MODULE_FILENAME
        if scope == "project"
        else Path.home() / ".pi" / "agent" / "extensions" / _TS_STATUS_MODULE_FILENAME
    )


def render_ts_status_module() -> str:
    """Render the omp/Pi status module with the snapshot contract baked in."""
    return (
        _TS_STATUS_MODULE_TEMPLATE.replace(
            "__ARCHEX_SNAPSHOT_VERSION__", str(STATUS_SNAPSHOT_VERSION)
        )
        .replace("__ARCHEX_STALE_DEFAULT__", str(DEFAULT_STALE_AFTER_SECONDS))
        .replace("__ARCHEX_WATCH_TTL__", str(WATCH_OBSERVATION_TTL_SECONDS))
        .replace("__ARCHEX_PROJECT_DIR__", PROJECT_DIR_NAME)
        .replace("__ARCHEX_SNAPSHOT_FILENAME__", SNAPSHOT_FILENAME)
        .replace("__ARCHEX_STATUS_KEY__", TS_STATUS_KEY)
    )


_TS_STATUS_MODULE_TEMPLATE = r"""/**
 * archex status extension module (R23 - oh-my-pi / Pi).
 *
 * Installed by `archex install-client omp --statusline` /
 * `archex install-client pi --statusline` (opt-in; never installed by
 * default). A separate file from the search and post-edit hook modules, so
 * every surface installs and removes independently.
 *
 * Upstream contract (verified against each host's own type declarations, not
 * docs):
 * - `ctx.ui.setStatus(key, text)` sets keyed footer/status-bar text, rendered
 *   by the built-in footer and by the `status` status-line segment, sorted by
 *   key (oh-my-pi `src/modes/controllers/extension-ui-controller.ts` wires it
 *   to `setHookStatus`; Pi `dist/core/extensions/types.d.ts` declares it on
 *   `ExtensionUIContext`).
 * - Both hosts declare `turn_start`, `turn_end`, and `tool_result` events with
 *   the same `(event, ctx)` handler shape, and `ctx.hasUI` is false in
 *   print/RPC modes where `setStatus` is a documented no-op.
 *
 * Unlike the hook modules, this one spawns no subprocess and opens no index:
 * a repaint is a single `readFileSync` of the bounded snapshot archex
 * publishes plus string formatting, inside a process the host already runs.
 * Every path resolves without throwing; an unreadable, malformed, or
 * unknown-version snapshot renders as an explicit state instead.
 */
import { existsSync, readFileSync } from "node:fs";
import { join, parse } from "node:path";

// --- Baked in at install time from the Python contract ---

const ARCHEX_SNAPSHOT_VERSION = __ARCHEX_SNAPSHOT_VERSION__;
const ARCHEX_STALE_AFTER_SECONDS = __ARCHEX_STALE_DEFAULT__;
const ARCHEX_WATCH_TTL_SECONDS = __ARCHEX_WATCH_TTL__;
const ARCHEX_PROJECT_DIR = "__ARCHEX_PROJECT_DIR__";
const ARCHEX_SNAPSHOT_FILENAME = "__ARCHEX_SNAPSHOT_FILENAME__";
const ARCHEX_STATUS_KEY = "__ARCHEX_STATUS_KEY__";

/** Override for tests and for a host started outside the repository. */
const ARCHEX_SNAPSHOT_ENV_VAR = "ARCHEX_STATUS_SNAPSHOT";

/** Freshness-budget override, honoured by every archex status renderer. */
const ARCHEX_STALE_ENV_VAR = "ARCHEX_STATUS_STALE_AFTER_SECONDS";

/** Reader-state lines, worded exactly as the shell renderer words them. */
const ARCHEX_MISSING_LINE = "archex missing - no status snapshot - run: archex index";
const ARCHEX_CORRUPT_LINE = "archex corrupt - unreadable snapshot - run: archex status";

interface ArchexSnapshot {
  version?: unknown;
  state?: unknown;
  index_revision?: unknown;
  files_indexed?: unknown;
  chunks_indexed?: unknown;
  pending_delta_files?: unknown;
  pending_view_complete?: unknown;
  reindex_required?: unknown;
  written_at_epoch?: unknown;
  watch_observed_epoch?: unknown;
}

function snapshotPath(start: string): string | null {
  const override = process.env[ARCHEX_SNAPSHOT_ENV_VAR];
  if (override && override.trim().length > 0) return override;
  let current = start;
  for (;;) {
    const candidate = join(current, ARCHEX_PROJECT_DIR, ARCHEX_SNAPSHOT_FILENAME);
    // Existence, not readability: a present-but-unreadable snapshot must
    // classify as `corrupt` here, exactly as it does in the shell renderer
    // and the Python reader. Probing with a read would skip it and report
    // either `missing` or -- worse -- a parent repository's status.
    if (existsSync(candidate)) return candidate;
    const parent = parse(current).dir;
    if (!parent || parent === current) return null;
    current = parent;
  }
}

/** Freshness budget in seconds, overridable exactly as the CLI allows. */
function staleAfterSeconds(): number {
  const raw = process.env[ARCHEX_STALE_ENV_VAR];
  if (raw === undefined) return ARCHEX_STALE_AFTER_SECONDS;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : ARCHEX_STALE_AFTER_SECONDS;
}

function readNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function readText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function ageSegment(writtenAtEpoch: number, now: number): string {
  if (writtenAtEpoch <= 0) return "";
  const age = Math.max(0, now - writtenAtEpoch);
  if (age < 60) return ` - ${age}s ago`;
  if (age < 3600) return ` - ${Math.floor(age / 60)}m ago`;
  return ` - ${Math.floor(age / 3600)}h ago`;
}

/** Render the status text for `cwd`, or a state naming why it cannot. */
export function renderArchexStatus(cwd: string, nowSeconds?: number): string {
  const path = snapshotPath(cwd);
  if (path === null) return ARCHEX_MISSING_LINE;

  // An absent file is `missing`; unreadable or malformed bytes are
  // `corrupt`. The two have different remedies -- publish one versus
  // re-publish this one -- so they must not collapse into each other.
  let raw: string;
  try {
    raw = readFileSync(path, "utf-8");
  } catch (err) {
    const code = (err as { code?: string }).code;
    return code === "ENOENT" ? ARCHEX_MISSING_LINE : ARCHEX_CORRUPT_LINE;
  }

  let parsed: ArchexSnapshot;
  try {
    const document: unknown = JSON.parse(raw);
    if (typeof document !== "object" || document === null || Array.isArray(document)) {
      return ARCHEX_CORRUPT_LINE;
    }
    parsed = document as ArchexSnapshot;
  } catch {
    return ARCHEX_CORRUPT_LINE;
  }

  if (typeof parsed.version !== "number") return ARCHEX_CORRUPT_LINE;
  if (parsed.version !== ARCHEX_SNAPSHOT_VERSION) {
    return `archex unsupported - snapshot v${parsed.version} - upgrade archex`;
  }

  const persisted = readText(parsed.state);
  if (persisted !== "fresh" && persisted !== "dirty" && persisted !== "pending") {
    return ARCHEX_CORRUPT_LINE;
  }

  const now = nowSeconds ?? Math.floor(Date.now() / 1000);
  const writtenAtEpoch = readNumber(parsed.written_at_epoch);
  const age = writtenAtEpoch > 0 ? Math.max(0, now - writtenAtEpoch) : null;
  const stale = age !== null && age > staleAfterSeconds();
  const state = stale ? "stale" : persisted;

  let detail = "";
  if (state === "fresh") {
    const files = readNumber(parsed.files_indexed);
    const chunks = readNumber(parsed.chunks_indexed);
    detail = ` - ${files} files, ${chunks} chunks`;
  } else if (state === "pending") {
    const pending = readNumber(parsed.pending_delta_files);
    const complete = parsed.pending_view_complete !== false;
    detail = ` - ${pending}${complete ? "" : "+"} awaiting sync`;
  } else if (state === "dirty") {
    detail = parsed.reindex_required === true ? " - reindex required" : " - index behind tree";
  } else {
    detail = " - unverified since measurement";
  }

  const revision = readText(parsed.index_revision);
  const revisionSegment = revision.length > 0 ? ` - rev ${revision.slice(0, 8)}` : "";

  const watchEpoch = readNumber(parsed.watch_observed_epoch);
  const watching = watchEpoch > 0 && now - watchEpoch <= ARCHEX_WATCH_TTL_SECONDS;

  return `archex ${state}${detail}${revisionSegment}${ageSegment(writtenAtEpoch, now)}${
    watching ? " - watch" : ""
  }`;
}

// --- Minimal structural types for the events used ---
//
// Declared locally (never imported from either host package) so this module
// has zero import-resolution dependency on which host loaded it.

interface StatusUiLike {
  setStatus(key: string, text: string | undefined): void;
}

interface StatusContextLike {
  ui?: StatusUiLike;
  hasUI?: boolean;
}

type StatusHandler = (event: unknown, ctx: StatusContextLike) => void;

interface StatusHost {
  on(event: "turn_start" | "turn_end" | "tool_result", handler: StatusHandler): unknown;
}

/** Refresh the footer status, swallowing every failure. */
export function publishArchexStatus(ctx: StatusContextLike): void {
  try {
    if (ctx.hasUI === false || !ctx.ui) return;
    ctx.ui.setStatus(ARCHEX_STATUS_KEY, renderArchexStatus(process.cwd()));
  } catch {
    // A status refresh must never disturb the host's turn.
  }
}

export default function archexStatusExtension(pi: StatusHost): void {
  const refresh: StatusHandler = (_event, ctx) => {
    publishArchexStatus(ctx);
  };
  // Three refresh points: entering a turn, after each tool result (so an edit
  // shows as pending mid-turn), and at rest. All three read one small file.
  pi.on("turn_start", refresh);
  pi.on("tool_result", refresh);
  pi.on("turn_end", refresh);
}
"""


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


def render_statusline_script() -> str:
    """Render the status-line script with the snapshot contract baked in."""
    return (
        _STATUSLINE_SCRIPT_TEMPLATE.replace(
            "__ARCHEX_SNAPSHOT_VERSION__", str(STATUS_SNAPSHOT_VERSION)
        )
        .replace("__ARCHEX_STALE_DEFAULT__", str(DEFAULT_STALE_AFTER_SECONDS))
        .replace("__ARCHEX_WATCH_TTL__", str(WATCH_OBSERVATION_TTL_SECONDS))
        .replace("__ARCHEX_SNAPSHOT_RELATIVE_PATH__", f"{PROJECT_DIR_NAME}/{SNAPSHOT_FILENAME}")
    )


_STATUSLINE_SCRIPT_TEMPLATE = r"""#!/bin/sh
# archex:statusline - Claude Code status line renderer (R23).
#
# Installed by `archex install-client claude-code --statusline`, removed by
# `--remove-statusline`. Renders the bounded status snapshot archex publishes
# at <repo>/__ARCHEX_SNAPSHOT_RELATIVE_PATH__.
#
# Claude Code re-runs this command on every repaint (debounced at 300ms, and
# it cancels an in-flight script when a new update arrives), so the renderer
# uses shell builtins exclusively: no `jq`, no `python`, no `date`, no
# `archex`, and no command substitution anywhere. It therefore forks no
# process, opens no index, and parses no source. The test suite proves that
# by running it with an empty PATH.
#
# Snapshot version this renderer understands: __ARCHEX_SNAPSHOT_VERSION__.
# A different version prints `unsupported` rather than guessing at fields.

set -u

archex_version_supported=__ARCHEX_SNAPSHOT_VERSION__
archex_stale_default=__ARCHEX_STALE_DEFAULT__
archex_watch_ttl=__ARCHEX_WATCH_TTL__

# --- Locate the snapshot ----------------------------------------------------
# Claude Code writes one line of session JSON to stdin, carrying the session
# directory as "cwd". Extracting it by parameter expansion avoids a JSON
# parser; an unreadable payload falls back to the process working directory,
# and the upward walk then handles being started in a subdirectory.
payload=""
IFS= read -r payload 2>/dev/null || :

archex_dir=""
case $payload in
*'"cwd":'*)
	archex_rest=${payload#*'"cwd":'}
	while :; do
		case $archex_rest in
		' '*) archex_rest=${archex_rest# } ;;
		*) break ;;
		esac
	done
	case $archex_rest in
	'"'*)
		archex_rest=${archex_rest#\"}
		archex_dir=${archex_rest%%\"*}
		;;
	esac
	;;
esac
# Only an absolute, existing directory is trusted. A payload whose earlier
# values happen to contain the literal `"cwd":` text, or a relative or
# nonexistent path, falls back to the process working directory rather than
# steering the walk below somewhere else.
case $archex_dir in
/*) ;;
*) archex_dir="" ;;
esac
if [ -z "$archex_dir" ] || [ ! -d "$archex_dir" ]; then
	archex_dir=$PWD
fi

archex_snapshot=${ARCHEX_STATUS_SNAPSHOT:-}
if [ -z "$archex_snapshot" ]; then
	archex_probe=$archex_dir
	while [ -n "$archex_probe" ]; do
		if [ -f "$archex_probe/__ARCHEX_SNAPSHOT_RELATIVE_PATH__" ]; then
			archex_snapshot="$archex_probe/__ARCHEX_SNAPSHOT_RELATIVE_PATH__"
			break
		fi
		case $archex_probe in
		*/?*) archex_probe=${archex_probe%/*} ;;
		*) archex_probe="" ;;
		esac
	done
fi

if [ -z "$archex_snapshot" ] || [ ! -f "$archex_snapshot" ]; then
	printf 'archex missing - no status snapshot - run: archex index\n'
	exit 0
fi

# --- Read it ----------------------------------------------------------------
# The document is written with sorted keys and two-space indentation, so every
# scalar occupies one line as `  "key": value,`. Splitting that with
# parameter expansion is exact for this writer and degrades to an empty field
# (hence `corrupt`) for anything else.
archex_field_version=""
archex_field_state=""
archex_field_revision=""
archex_field_files=0
archex_field_chunks=0
archex_field_pending=0
archex_field_pending_complete=""
archex_field_reindex=""
archex_field_epoch=0
archex_field_watch_epoch=0

archex_line=""
while IFS= read -r archex_line || [ -n "$archex_line" ]; do
	case $archex_line in
	*'": '*) ;;
	*) continue ;;
	esac
	archex_key=${archex_line#*\"}
	archex_key=${archex_key%%\"*}
	archex_value=${archex_line#*\": }
	archex_value=${archex_value%,}
	case $archex_value in
	\"*\")
		archex_value=${archex_value#\"}
		archex_value=${archex_value%\"}
		;;
	esac
	case $archex_key in
	version) archex_field_version=$archex_value ;;
	state) archex_field_state=$archex_value ;;
	index_revision) archex_field_revision=$archex_value ;;
	files_indexed) archex_field_files=$archex_value ;;
	chunks_indexed) archex_field_chunks=$archex_value ;;
	pending_delta_files) archex_field_pending=$archex_value ;;
	pending_view_complete) archex_field_pending_complete=$archex_value ;;
	reindex_required) archex_field_reindex=$archex_value ;;
	written_at_epoch) archex_field_epoch=$archex_value ;;
	watch_observed_epoch) archex_field_watch_epoch=$archex_value ;;
	esac
done < "$archex_snapshot"

# Numeric fields are normalized before any arithmetic. Non-digits become 0,
# and a leading zero is stripped: POSIX arithmetic reads `08` as octal and
# aborts the whole script on the invalid digit, which would replace a
# `corrupt` line with an error on stderr and no status at all.
for archex_numeric in files chunks pending epoch watch_epoch; do
	case $archex_numeric in
	files) archex_probe_value=$archex_field_files ;;
	chunks) archex_probe_value=$archex_field_chunks ;;
	pending) archex_probe_value=$archex_field_pending ;;
	epoch) archex_probe_value=$archex_field_epoch ;;
	*) archex_probe_value=$archex_field_watch_epoch ;;
	esac
	case $archex_probe_value in
	'' | *[!0-9]*) archex_probe_value=0 ;;
	*)
		while :; do
			case $archex_probe_value in
			0?*) archex_probe_value=${archex_probe_value#0} ;;
			*) break ;;
			esac
		done
		;;
	esac
	case $archex_numeric in
	files) archex_field_files=$archex_probe_value ;;
	chunks) archex_field_chunks=$archex_probe_value ;;
	pending) archex_field_pending=$archex_probe_value ;;
	epoch) archex_field_epoch=$archex_probe_value ;;
	*) archex_field_watch_epoch=$archex_probe_value ;;
	esac
done

# --- Classify ---------------------------------------------------------------
# `corrupt` and `unsupported` are deliberately separate: a document that is
# unparsable or carries no usable version needs re-publishing, while a
# document from a schema this build does not read needs an archex upgrade.
# The Python reader and the omp/Pi module classify identically.
case $archex_field_version in
'' | *[!0-9]*)
	printf 'archex corrupt - unreadable snapshot - run: archex status\n'
	exit 0
	;;
esac
if [ "$archex_field_version" != "$archex_version_supported" ]; then
	printf 'archex unsupported - snapshot v%s - upgrade archex\n' "$archex_field_version"
	exit 0
fi
case $archex_field_state in
fresh | dirty | pending) ;;
*)
	printf 'archex corrupt - unreadable snapshot - run: archex status\n'
	exit 0
	;;
esac

# `EPOCHSECONDS` is a shell builtin variable in bash 5+ and zsh, and absent in
# dash and bash 3.2 (macOS `/bin/sh`). Where it is absent this renderer has no
# clock it can read without forking `date`, so it reports the measured state
# and its measurement time and leaves the `stale` judgement to
# `archex status --cached`, which has a clock. Paying a fork on every repaint
# to gain one label is the wrong trade.
if [ -n "${ZSH_VERSION:-}" ]; then
	# `zmodload` is a zsh builtin, so this costs no process. Guarded on
	# ZSH_VERSION because under sh/dash the word would be an external
	# command, which is exactly what this renderer refuses to launch.
	zmodload zsh/datetime 2>/dev/null || :
fi
archex_now=${EPOCHSECONDS:-}
case $archex_now in
'' | *[!0-9]*) archex_now="" ;;
esac

archex_age=""
if [ -n "$archex_now" ] && [ "$archex_field_epoch" -gt 0 ]; then
	archex_age=$((archex_now - archex_field_epoch))
	if [ "$archex_age" -lt 0 ]; then
		archex_age=0
	fi
fi

archex_budget=${ARCHEX_STATUS_STALE_AFTER_SECONDS:-$archex_stale_default}
case $archex_budget in
'' | *[!0-9]*) archex_budget=$archex_stale_default ;;
esac
if [ -n "$archex_age" ] && [ "$archex_age" -gt "$archex_budget" ]; then
	archex_field_state=stale
fi

archex_watch=""
if [ -n "$archex_now" ] && [ "$archex_field_watch_epoch" -gt 0 ]; then
	if [ $((archex_now - archex_field_watch_epoch)) -le "$archex_watch_ttl" ]; then
		archex_watch=" - watch"
	fi
fi

# --- Render -----------------------------------------------------------------
archex_detail=""
case $archex_field_state in
fresh) archex_detail=" - $archex_field_files files, $archex_field_chunks chunks" ;;
pending)
	# The model's field defaults to true, so an absent field means a complete
	# view here too; only an explicit `false` marks the count as a lower bound.
	if [ "$archex_field_pending_complete" = "false" ]; then
		archex_detail=" - $archex_field_pending+ awaiting sync"
	else
		archex_detail=" - $archex_field_pending awaiting sync"
	fi
	;;
dirty)
	if [ "$archex_field_reindex" = "true" ]; then
		archex_detail=" - reindex required"
	else
		archex_detail=" - index behind tree"
	fi
	;;
stale) archex_detail=" - unverified since measurement" ;;
esac

archex_revision_segment=""
if [ -n "$archex_field_revision" ]; then
	archex_revision_short=$archex_field_revision
	# Trim to the first eight characters, but only when there are more than
	# eight: for a shorter value the `????????` pattern does not match and the
	# suffix trim would collapse the whole string to empty.
	case $archex_field_revision in
	?????????*)
		archex_revision_tail=${archex_field_revision#????????}
		archex_revision_short=${archex_field_revision%"$archex_revision_tail"}
		;;
	esac
	archex_revision_segment=" - rev $archex_revision_short"
fi

archex_age_segment=""
if [ -n "$archex_age" ]; then
	if [ "$archex_age" -lt 60 ]; then
		archex_age_segment=" - ${archex_age}s ago"
	elif [ "$archex_age" -lt 3600 ]; then
		archex_age_segment=" - $((archex_age / 60))m ago"
	else
		archex_age_segment=" - $((archex_age / 3600))h ago"
	fi
fi

printf 'archex %s%s%s%s%s\n' \
	"$archex_field_state" \
	"$archex_detail" \
	"$archex_revision_segment" \
	"$archex_age_segment" \
	"$archex_watch"
"""
