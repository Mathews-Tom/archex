"""Codex CLI `PostToolUse` hook: bounded post-edit impact feedback (R21).

Unlike M21's `PreToolUse` adapter, which had to ship diagnostics-only
because Codex has no Grep/Glob-equivalent tool-call event to scope
augmentation to, the post-edit event is fully supported. Verified against
`openai/codex@main`, not secondary docs:

- `codex-rs/hooks/src/schema.rs` declares `HookEventNameWire::PostToolUse`.
  `PostToolUseCommandInput` carries `cwd`, `tool_name`, `tool_input`, and
  `tool_response`; `PostToolUseHookSpecificOutputWire` carries
  `additional_context`, serialized camelCase as `additionalContext` under
  `hookSpecificOutput` — the same shape Claude Code uses.
- `codex-rs/core/src/tools/hook_names.rs` pins the canonical edit tool name
  `apply_patch`, with `Write` and `Edit` accepted only as matcher aliases.
  The serialized `tool_name` is always `apply_patch`.
- `codex-rs/core/src/tools/handlers/apply_patch.rs` builds the post-tool-use
  payload as `tool_input = {"command": <raw patch text>}`. Edited paths are
  therefore not a structured field but the patch body itself, which
  `codex-rs/apply-patch/src/parser.rs` pins exactly: `*** Add File: `,
  `*** Update File: `, `*** Delete File: `, and `*** Move to: ` headers
  between `*** Begin Patch` and `*** End Patch`. `_patch_paths` below parses
  precisely those markers and nothing else.

Every code path exits 0 writing `{}` — Codex's output schema defaults every
field, so an empty object means "no decision". A malformed payload, a
non-`apply_patch` tool, an unparsable patch, a stale index, a timeout, or an
internal error all degrade to that empty object plus a diagnostics line.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, cast

from archex.integrations.hook import log_diagnostic
from archex.integrations.post_edit_hook import POST_EDIT_EVENT_NAME, run_post_edit_cycle

#: Codex's canonical serialized tool name for file edits.
EDIT_TOOL_NAME = "apply_patch"

#: `matcher` regex the installer writes into `config.toml`'s
#: `[[hooks.PostToolUse]]` table. Anchored on the canonical name rather than
#: the `Write`/`Edit` compatibility aliases, so the installed config and this
#: module's runtime filter cannot drift.
POST_EDIT_MATCHER = f"^{EDIT_TOOL_NAME}$"

_BEGIN_PATCH_MARKER = "*** Begin Patch"
_END_PATCH_MARKER = "*** End Patch"
_ADD_FILE_MARKER = "*** Add File: "
_UPDATE_FILE_MARKER = "*** Update File: "
_DELETE_FILE_MARKER = "*** Delete File: "
_MOVE_TO_MARKER = "*** Move to: "
_PATH_MARKERS = (
    _ADD_FILE_MARKER,
    _UPDATE_FILE_MARKER,
    _DELETE_FILE_MARKER,
    _MOVE_TO_MARKER,
)

#: Codex's hook output for "no decision"; every field defaults.
_NO_DECISION: dict[str, object] = {}


def main() -> None:
    """Entry point for `python -m archex.integrations.codex_post_edit_hook`.

    Both the fallback write and the flush sit inside the guard: `os._exit`
    does not flush buffers, and an unguarded write to a pipe the client
    already closed would skip `os._exit(0)` and exit non-zero.
    """
    try:
        _main_impl()
        sys.stdout.flush()
    except BaseException as exc:  # noqa: BLE001 - the non-blocking contract requires this
        log_diagnostic("codex_post_edit_unhandled_exception", detail=repr(exc))
        try:
            sys.stdout.write(json.dumps(_NO_DECISION))
            sys.stdout.flush()
        except OSError:
            pass  # the client closed the pipe; exiting 0 is still the contract
    os._exit(0)


def _main_impl() -> None:
    payload = _parse_codex_payload(sys.stdin.read())
    result = _NO_DECISION if payload is None else handle_codex_post_tool_use(payload)
    sys.stdout.write(json.dumps(result))


def _parse_codex_payload(raw: str) -> dict[str, Any] | None:
    if not raw.strip():
        log_diagnostic("codex_post_edit_malformed_payload", detail="empty stdin")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        log_diagnostic("codex_post_edit_malformed_payload", detail=repr(exc))
        return None
    if not isinstance(payload, dict):
        log_diagnostic(
            "codex_post_edit_malformed_payload",
            detail=f"expected object, got {type(payload).__name__}",
        )
        return None
    return cast("dict[str, Any]", payload)


def handle_codex_post_tool_use(payload: dict[str, Any]) -> dict[str, object]:
    """Core handler: a parsed Codex `PostToolUse` payload in, hook output out."""
    if payload.get("tool_name") != EDIT_TOOL_NAME:
        return _NO_DECISION
    cwd = payload.get("cwd")
    resolved_cwd = cwd if isinstance(cwd, str) and cwd else os.getcwd()

    paths = _patch_paths(payload.get("tool_input"))
    if not paths:
        log_diagnostic(
            "codex_post_edit_no_paths",
            detail="apply_patch payload carried no parsable file marker",
            cwd=resolved_cwd,
        )
        return _NO_DECISION

    context = run_post_edit_cycle(
        client="codex",
        tool_name=EDIT_TOOL_NAME,
        cwd=resolved_cwd,
        raw_paths=list(paths),
    )
    if context is None:
        return _NO_DECISION
    return {
        "hookSpecificOutput": {
            "hookEventName": POST_EDIT_EVENT_NAME,
            "additionalContext": context,
        }
    }


def _patch_paths(tool_input: object) -> list[str]:
    """Extract edited paths from a Codex V4A `apply_patch` command payload.

    Marker matching mirrors upstream's own mode machine in
    `codex-rs/apply-patch/src/streaming_parser.rs`, which is not uniform:
    outside an update hunk it matches headers on ``line.trim()``, so an
    indented ``*** Begin Patch`` or ``*** Delete File:`` is accepted; inside
    one it switches to ``line.trim_end()``, because there a leading ``' '``,
    ``'+'``, or ``'-'`` marks a body line.

    Reproducing both rules is what makes this parser neither too strict nor
    too loose. Matching everything on the full trim would let an indented
    context line reproducing a marker inject a path the patch never touched,
    or truncate the scan at a context ``*** End Patch``. Matching everything
    at column zero would silently drop the indented headers upstream
    accepts, skipping impact for a real edit.
    """
    if not isinstance(tool_input, dict):
        return []
    command = cast("dict[str, Any]", tool_input).get("command")
    if not isinstance(command, str) or _BEGIN_PATCH_MARKER not in command:
        return []

    paths: list[str] = []
    inside = False
    in_update_hunk = False
    for raw_line in command.splitlines():
        # Upstream's UpdateFile mode uses trim_end; every other mode trims
        # both ends.
        line = raw_line.rstrip() if in_update_hunk else raw_line.strip()
        if not inside:
            if line == _BEGIN_PATCH_MARKER:
                inside = True
            continue
        if line == _END_PATCH_MARKER:
            break
        for marker in _PATH_MARKERS:
            if line.startswith(marker):
                candidate = line[len(marker) :].strip()
                if candidate:
                    paths.append(candidate)
                # `*** Move to:` only ever appears while already inside an
                # update hunk, so it must not reset the mode.
                if marker != _MOVE_TO_MARKER:
                    in_update_hunk = marker == _UPDATE_FILE_MARKER
                break
    return paths


if __name__ == "__main__":
    main()
