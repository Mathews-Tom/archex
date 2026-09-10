"""Claude Code `PostToolUse` hook: bounded post-edit impact feedback (R21).

Upstream contract this adapter is built against (verified against
`https://code.claude.com/docs/en/hooks.md`, not secondary docs):

- `PostToolUse` fires after a tool has already run, so it cannot block or
  fail the edit: the documented exit-2 behavior for this event only surfaces
  stderr to the model, and every other nonzero exit is non-blocking. That is
  the whole reason this event was chosen over any pre-edit one.
- Input carries `cwd`, `tool_name`, `tool_input`, and `tool_response`. The
  file tools put an absolute path in `tool_input.file_path`
  (`notebook_path` for `NotebookEdit`), and `Write`'s documented response is
  `{"filePath": ..., "success": true}`.
- Output supports `hookSpecificOutput.additionalContext`, the same field the
  existing `PreToolUse` search hook uses.
- Matcher values are regex-searched against the tool name, which is why the
  installed matcher `Edit|Write` also selects `MultiEdit` and `NotebookEdit`;
  `AUGMENTED_TOOLS` below is the authoritative runtime filter.

This module is also the shared subprocess backend for the TypeScript
adapters (oh-my-pi, Pi, OpenCode). They translate their own post-execution
event into this Claude-shaped payload and read `additionalContext` back,
exactly as the search hook's TS modules already do, so no client but Claude
Code needs to exist in Python and no second impact engine is created.

Failure discipline: every path exits 0 with no output rather than raising.
A payload for a non-edit tool, a failed edit, a repository archex does not
manage, a stale generation, a timeout, or an internal error all degrade to
silence plus a line on the existing diagnostics log
(`ARCHEX_HOOK_DIAGNOSTICS_LOG`, default `~/.archex/hook-diagnostics.log`).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, cast

from archex.integrations.hook import log_diagnostic
from archex.post_edit import PostEditOutcome, build_event, record_edit
from archex.post_edit.impact import synchronize_and_report_with_timeout
from archex.project import ProjectState

POST_EDIT_EVENT_NAME = "PostToolUse"

#: Claude Code tool names this adapter treats as a file edit. Authoritative
#: runtime filter; `POST_EDIT_MATCHER` is only the installed pre-filter.
AUGMENTED_TOOLS: frozenset[str] = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})

#: Installed `matcher` value. Claude Code regex-searches the matcher against
#: the tool name, so this selects every name in `AUGMENTED_TOOLS` and nothing
#: that reads or searches.
POST_EDIT_MATCHER = "Edit|Write"

#: Payload fields that can carry an edited path, in the order upstream
#: documents them for the tools above.
_PATH_FIELDS = ("file_path", "notebook_path", "filePath")

#: Clients whose TypeScript shim invokes this module as a subprocess and
#: declares itself via `archex_client`. An allowlist, so an arbitrary string
#: on the wire cannot end up in a receipt.
SHIM_CLIENTS: frozenset[str] = frozenset({"omp", "pi", "opencode"})


def main() -> None:
    """Read one Claude Code hook payload and exit successfully on every path.

    The flush is inside the guard because `os._exit` does not flush buffers:
    if a client closed the read end early, an unguarded flush would raise
    `BrokenPipeError`, skip `os._exit(0)`, and exit non-zero with a
    traceback.
    """
    try:
        _main_impl()
        sys.stdout.flush()
    except BaseException as exc:  # noqa: BLE001 - the non-blocking contract requires this
        log_diagnostic("post_edit_unhandled_exception", detail=repr(exc))
    # Skip interpreter teardown so an abandoned refresh thread cannot delay
    # the agent, mirroring `archex.integrations.hook`.
    os._exit(0)


def _main_impl() -> None:
    payload = parse_payload(sys.stdin.read())
    if payload is None:
        return
    result = handle_post_tool_use(payload)
    if result is not None:
        sys.stdout.write(json.dumps(result))


def parse_payload(raw: str) -> dict[str, Any] | None:
    """Parse a hook payload, logging and discarding anything malformed."""
    if not raw.strip():
        log_diagnostic("post_edit_malformed_payload", detail="empty stdin")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        log_diagnostic("post_edit_malformed_payload", detail=repr(exc))
        return None
    if not isinstance(payload, dict):
        log_diagnostic(
            "post_edit_malformed_payload",
            detail=f"expected object, got {type(payload).__name__}",
        )
        return None
    return cast("dict[str, Any]", payload)


def handle_post_tool_use(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Core handler: a parsed `PostToolUse` payload in, hook output or None."""
    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str) or tool_name not in AUGMENTED_TOOLS:
        return None
    if _edit_failed(payload.get("tool_response")):
        log_diagnostic(
            "post_edit_unsuccessful_edit",
            detail=f"tool_name={tool_name}",
            cwd=_cwd_of(payload),
        )
        return None

    context = run_post_edit_cycle(
        client=_client_of(payload),
        tool_name=tool_name,
        cwd=_cwd_of(payload),
        raw_paths=extract_paths(payload),
    )
    if context is None:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": POST_EDIT_EVENT_NAME,
            "additionalContext": context,
        }
    }


def extract_paths(payload: dict[str, Any]) -> list[object]:
    """Collect every candidate edited path the payload carries."""
    found: list[object] = []
    for section_key in ("tool_input", "tool_response"):
        section = payload.get(section_key)
        if not isinstance(section, dict):
            continue
        typed = cast("dict[str, Any]", section)
        for field in _PATH_FIELDS:
            value = typed.get(field)
            if value is not None:
                found.append(value)
    return found


def _edit_failed(tool_response: object) -> bool:
    """Whether the payload explicitly marks the edit as unsuccessful.

    `PostToolUse` fires after the tool ran, and upstream does not guarantee a
    `success` field for every tool, so absence is treated as success. An
    explicit `success: false` or an `error` value is honoured, which is what
    keeps a failed write from marking the index dirty.
    """
    if not isinstance(tool_response, dict):
        return False
    typed = cast("dict[str, Any]", tool_response)
    if typed.get("success") is False:
        return True
    return bool(typed.get("error"))


def _cwd_of(payload: dict[str, Any]) -> str:
    cwd = payload.get("cwd")
    return cwd if isinstance(cwd, str) and cwd else os.getcwd()


def _client_of(payload: dict[str, Any]) -> str:
    """Resolve which client this payload came from.

    Claude Code sends no client field, so its absence means claude-code. The
    TypeScript shims for oh-my-pi, Pi, and OpenCode translate their own
    event into this same Claude-shaped payload and set `archex_client`, so
    the receipt attributes the impact to the host that actually ran the
    edit rather than to Claude Code.
    """
    declared = payload.get("archex_client")
    if isinstance(declared, str) and declared in SHIM_CLIENTS:
        return declared
    return "claude-code"


def run_post_edit_cycle(
    *, client: str, tool_name: str, cwd: str, raw_paths: list[object]
) -> str | None:
    """Record the edit, synchronize, and return a fresh impact block or None.

    Shared by every supported client: the Claude Code handler above, the
    Codex adapter, and the TypeScript shims that invoke this module as a
    subprocess. It is the single place a post-edit event becomes recorded
    state and, if and only if the resulting generation is fresh, text.
    """
    if not raw_paths:
        return None
    try:
        project = ProjectState.resolve(cwd)
    except (ValueError, OSError) as exc:
        log_diagnostic("post_edit_project_error", detail=str(exc), cwd=cwd)
        return None
    if not project.initialized():
        log_diagnostic("post_edit_project_uninitialized", detail=str(project.repo_root), cwd=cwd)
        return None

    event, rejected = build_event(
        client=client,
        tool_name=tool_name,
        repo_root=project.repo_root,
        raw_paths=raw_paths,
    )
    if rejected:
        log_diagnostic(
            "post_edit_path_rejected",
            detail=f"{len(rejected)} path(s) outside the repository or unusable",
            cwd=cwd,
        )
    if not event.paths:
        return None
    record_edit(project.repo_root, event)

    feedback = synchronize_and_report_with_timeout(project.repo_root, client=client)
    if feedback.outcome is not PostEditOutcome.EMITTED:
        log_diagnostic(
            "post_edit_withheld",
            detail=f"outcome={feedback.outcome.value} detail={feedback.detail or ''}",
            cwd=cwd,
        )
        return None
    return feedback.text


if __name__ == "__main__":
    main()
