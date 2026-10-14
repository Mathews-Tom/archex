"""Shared client install-plan types and capability constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ClientName = Literal["claude-code", "codex", "cursor", "opencode", "pi", "omp"]
ClientScope = Literal["project", "user"]

HookAction = Literal["install", "remove"]


@dataclass(frozen=True)
class ClientInstallPlan:
    client: ClientName
    scope: ClientScope
    target_path: Path
    content: str
    description: str
    tested_status: str
    last_verified: str


@dataclass(frozen=True)
class DiscoveredClient:
    client: ClientName
    scope: ClientScope
    config_path: Path
    is_installed: bool
    evidence: str


@dataclass(frozen=True)
class ClaudeCodeHookInstallPlan:
    """Install or remove the Claude Code ``PostToolUse`` search-annotation hook.

    Installing also removes the retired ``PreToolUse`` pattern-search entry;
    removing removes both (claude-code only).
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    hook_entry: dict[str, object]


@dataclass(frozen=True)
class TsHookInstallPlan:
    """Install or remove the shared TS hook module (M20 omp/pi; M22 opencode).

    Unlike the Claude Code hook (a JSON command entry merged into an existing
    settings file), this installs a standalone TypeScript module — see
    ``ts_annotation.render_ts_hook_module`` for the per-client template dispatch.
    omp and pi share one module (``omp_annotation.TS_HOOK_MODULE_TEMPLATE``, a
    ``pi.on("tool_result", ...)`` handler returning a content patch);
    opencode uses a structurally different one
    (``opencode_annotation.OPENCODE_HOOK_MODULE_TEMPLATE``, a ``tool.execute.after`` plugin
    that appends to its output argument's text in place instead).
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    module_content: str


@dataclass(frozen=True)
class CodexHookInstallPlan:
    """Install or remove the Codex CLI PostToolUse search-annotation hook.

    Unlike the Claude Code hook (a JSON command entry merged into
    ``settings.json``) or the omp/pi hook (a standalone ``.ts`` module), this
    appends a marker-delimited TOML block to the *same* ``config.toml`` the
    MCP server registration already writes to (``registration.client_config_path`` for
    ``client == "codex"``), mirroring that file's non-destructive append
    behavior for a ``[[hooks.PostToolUse]]`` table instead of
    ``[mcp_servers.archex]``. Install also removes the retired diagnostics-only
    ``[[hooks.PreToolUse]]`` block. See ``archex.integrations.codex_annotate_hook``
    for the payload the hook reads and the context it returns.
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    block_content: str


@dataclass(frozen=True)
class CursorHookInstallPlan:
    """Install or remove the Cursor ``beforeSubmitPrompt`` hook (M23; prompt-level).

    Unlike the Claude Code hook (matcher-grouped entries under
    ``hooks.PreToolUse``) or the Codex hook (a marker-delimited TOML block
    appended to ``config.toml``), this merges a single-entry array under
    ``hooks.beforeSubmitPrompt`` into a *separate* file, ``hooks.json`` —
    Cursor's own hook config lives apart from ``mcp.json``. Because
    ``beforeSubmitPrompt`` is its own top-level key, distinct from
    ``hooks.beforeReadFile``, "never touches ``beforeReadFile``" is a
    structural property of only ever writing under this one key, rather than
    something a shared matcher regex has to get right. See
    ``archex.integrations.cursor_hook`` for why this ships diagnostics-only
    rather than context injection (Cursor's ``beforeSubmitPrompt`` output
    schema has no context-injection field at all).
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    hook_entry: dict[str, object]


@dataclass(frozen=True)
class ClaudeCodeSessionPrimerInstallPlan:
    """Install or remove the opt-in Claude Code SessionStart primer hook."""

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    hook_entry: dict[str, object]


#: Either hook install plan shape. ``install_client_cmd.py`` treats both
#: uniformly; ``write_hook_install_plan``/``render_hook_install_preview``
#: dispatch on the concrete type.
HookInstallPlan = (
    ClaudeCodeHookInstallPlan | TsHookInstallPlan | CodexHookInstallPlan | CursorHookInstallPlan
)

#: Clients with no post-edit adapter, and the upstream reason. Surfaced by
#: the CLI so an unsupported client gets an explicit refusal rather than a
#: silent no-op or a fabricated success message.
POST_EDIT_UNSUPPORTED_CLIENTS: dict[ClientName, str] = {
    "cursor": (
        "Cursor's afterFileEdit is the only event carrying an edited file path "
        "and its documented schema has no output fields at all, so it cannot "
        "return impact to the agent; postToolUse does document an "
        "additional_context output but documents no path field for its Write "
        "tool input, so edited paths cannot be extracted from it. Archex will "
        "not ship a post-edit adapter that cannot be shown to work."
    ),
}


@dataclass(frozen=True)
class ClaudeCodePostEditHookInstallPlan:
    """Install or remove the Claude Code PostToolUse post-edit hook (R21)."""

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    hook_entry: dict[str, object]


@dataclass(frozen=True)
class CodexPostEditHookInstallPlan:
    """Install or remove the Codex CLI PostToolUse post-edit hook (R21).

    Appends a marker-delimited ``[[hooks.PostToolUse]]`` block to the same
    ``config.toml`` the MCP registration and the search-annotation block
    already write to, using its own marker pair so the three coexist.
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    block_content: str


@dataclass(frozen=True)
class TsPostEditHookInstallPlan:
    """Install or remove the standalone TS post-edit module (R21).

    omp and pi share a ``tool_result`` module; opencode uses a structurally
    different ``tool.execute.after`` plugin that mutates its output argument
    in place. Both are written under a filename distinct from the search
    hook's, so the two surfaces install and remove independently.
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    module_content: str


PostEditHookInstallPlan = (
    ClaudeCodePostEditHookInstallPlan | CodexPostEditHookInstallPlan | TsPostEditHookInstallPlan
)


# ---------------------------------------------------------------------------
# Persistent status surfaces (R23)
# ---------------------------------------------------------------------------

#: Filename of the installed Claude Code status-line renderer. Doubles as the
#: ownership marker: ``statusLine`` is a single settings key rather than a
#: list, so install and remove only ever touch an entry whose command
#: references this filename. Any other configured status line is a user's own
#: and is refused rather than replaced.
STATUSLINE_SCRIPT_FILENAME = "archex-statusline.sh"

#: Marker comment inside the script, so an installed file is identifiable as
#: archex-owned even after it is moved or copied.
STATUSLINE_SCRIPT_MARKER = "archex:statusline"

#: How often Claude Code re-runs the command in addition to its event-driven
#: repaints. The renderer's freshness and age segments are time-based, so
#: without a timer a snapshot would appear fresh through an idle session.
STATUSLINE_REFRESH_INTERVAL_SECONDS = 10


@dataclass(frozen=True)
class ClaudeCodeStatuslineInstallPlan:
    """Install or remove the Claude Code status-line renderer (R23).

    Two artifacts, one plan: the renderer script beside the client's other
    archex-owned files, and the ``statusLine`` entry in the same
    ``settings.json`` the hook installers merge into.
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    script_path: Path
    action: HookAction
    script_content: str
    statusline_entry: dict[str, object]


#: Status key the extension registers under. Both hosts sort footer statuses
#: by key and render them inline, so the key is user-visible ordering.
TS_STATUS_KEY = "archex"

#: Clients with no persistent status surface, and the upstream reason.
#: Surfaced by the CLI so an unsupported client is refused explicitly rather
#: than silently no-op'ing or being given a transient stand-in.
STATUSLINE_UNSUPPORTED_CLIENTS: dict[ClientName, str] = {
    "opencode": (
        "OpenCode's plugin surface exposes tool and chat hooks plus observable "
        "TUI events, whose only status-shaped member is the transient "
        "tui.toast.show. A toast disappears, so it cannot carry a persistent "
        "freshness indicator; archex ships no adapter it cannot show to work."
    ),
    "codex": (
        "The Codex CLI exposes no status-line configuration and no hook output "
        "field that renders persistently; its hooks surface text only as "
        "additional_context on a tool event."
    ),
    "cursor": (
        "Cursor's configuration surface is hooks only, with no status or "
        "footer API for an extension to write into."
    ),
}


@dataclass(frozen=True)
class TsStatusInstallPlan:
    """Install or remove the omp/Pi status extension module (R23).

    Unlike the Claude Code status line -- a command the client re-runs per
    repaint -- this is a module the host already has loaded. It reads the
    snapshot in-process with `readFileSync` and pushes the rendered line
    through `ctx.ui.setStatus`, so a repaint launches nothing at all.
    """

    client: ClientName
    scope: ClientScope
    target_path: Path
    action: HookAction
    module_content: str


StatusInstallPlan = ClaudeCodeStatuslineInstallPlan | TsStatusInstallPlan
