"""Client install/bootstrap helpers with an explicit public surface."""

from __future__ import annotations

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
from archex.client_setup.io import write_config_atomic
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
    write_client_install_plan,
)
from archex.client_setup.statusline import (
    build_statusline_install_plan,
    render_statusline_install_preview,
    statusline_interpreter,
    write_statusline_install_plan,
)
from archex.client_setup.statusline_script import render_statusline_script
from archex.client_setup.statusline_ts import render_ts_status_module
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
