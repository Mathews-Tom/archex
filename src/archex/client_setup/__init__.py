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
from archex.client_setup.ts_annotation import render_annotation_hook_module
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
