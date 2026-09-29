"""Codex CLI `PostToolUse` hook: annotate search results with the code they hit.

Installed by `archex install-client codex --hooks` on `HOOK_MATCHER`
(`^Bash$`) and invoked as `python -m archex.integrations.codex_annotate_hook`.
It reads the hook payload, takes `tool_response` as the search result text,
asks `archex.annotate` (through `annotate_hook.run_request`, in-process) which
indexed code unit each hit falls in, and prints

    {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                            "additionalContext": "<annotation lines>"}}

Codex records `additionalContext` as a separate developer message in the
conversation next to the tool result, so the original result reaches the model
unchanged; the hook never asks Codex to block or rewrite anything.

Payload facts, checked against Codex 0.153.4 (`rust-v0.153.4`):

- Shell commands run through `exec_command`, whose `PostToolUse` tool name is
  the fixed `Bash` (`codex-rs/core/src/tools/handlers/unified_exec.rs`,
  `post_unified_exec_tool_use_payload`).
- `tool_input` is `{"command": "<the cmd string>"}`. The model-facing `workdir`
  argument is not forwarded, though the command's relative paths are relative to
  it. Codex has already written the `exec_command` call, arguments included, to
  the session rollout (`transcript_path`) when the hook runs, so `workdir` is
  read back from the tail of that file by `tool_use_id` (`recover_workdir`);
  without it the base is the payload's top-level `cwd` (the session's working
  directory), which is what Codex uses when the model sets no `workdir`.
- `tool_response` is the command's output as a JSON string, not an object
  (`codex-rs/core/src/tools/context.rs`, `post_tool_use_response`). Codex sends
  no `PostToolUse` for a command that is still running, and none for a failed
  tool call.

The exit, ledger, budget and fast-exit contract is the shared runner's
(`archex.integrations.post_tool_use_annotate`); this module supplies the host
name, the `tool_response` mapping, and the `workdir` recovery.
"""

from __future__ import annotations

import json
import os
from typing import Final, cast

from archex.annotate import HOST_TOOLS
from archex.integrations.post_tool_use_annotate import PostToolUseHost
from archex.integrations.post_tool_use_annotate import main as run_hook

HOST: Final = "codex"

#: Installed `matcher` value: a regex Codex matches against the tool name,
#: built from the tools `archex.annotate` treats as searches on this host so
#: the installed config and the annotate core's tool table cannot drift apart.
HOOK_MATCHER = "^" + "$|^".join(HOST_TOOLS[HOST]) + "$"

#: How much of the rollout's end is searched for the call. The call is written
#: just before the tool runs, so it sits behind only the events of that one call.
_TRANSCRIPT_TAIL_BYTES: Final = 1 << 20


def response_text(tool: str, tool_input: dict[str, object], response: object) -> str | None:
    """The command output a Codex `PostToolUse` payload carries, or `None` if unusable."""
    del tool, tool_input  # the shell output is the whole response, whatever the command
    return response if isinstance(response, str) else None


def recover_workdir(payload: dict[str, object], tool_input: dict[str, object]) -> dict[str, object]:
    """`tool_input` with the call's `workdir`, when the session rollout records one."""
    if isinstance(tool_input.get("workdir"), str):
        return tool_input
    transcript = payload.get("transcript_path")
    call_id = payload.get("tool_use_id")
    cwd = payload.get("cwd")
    if not (isinstance(transcript, str) and isinstance(call_id, str) and isinstance(cwd, str)):
        return tool_input
    workdir = _rollout_workdir(transcript, call_id)
    if not workdir:
        return tool_input
    return {**tool_input, "workdir": os.path.join(cwd, workdir)}


def _rollout_workdir(path: str, call_id: str) -> str | None:
    """The `workdir` argument of the `exec_command` call `call_id`, or `None`."""
    try:
        with open(path, "rb") as rollout:
            rollout.seek(0, os.SEEK_END)
            rollout.seek(max(0, rollout.tell() - _TRANSCRIPT_TAIL_BYTES))
            tail = rollout.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if call_id not in line:
            continue
        try:
            record: object = json.loads(line)
        except json.JSONDecodeError:
            continue  # the first line of the tail may be cut mid-record
        item = _function_call(record)
        if item is None or item.get("call_id") != call_id:
            continue
        arguments = item.get("arguments")
        try:
            parsed: object = json.loads(arguments) if isinstance(arguments, str) else None
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        workdir = cast("dict[str, object]", parsed).get("workdir")
        return workdir if isinstance(workdir, str) else None
    return None


def _function_call(record: object) -> dict[str, object] | None:
    if not isinstance(record, dict):
        return None
    payload = cast("dict[str, object]", record).get("payload")
    if not isinstance(payload, dict):
        return None
    item = cast("dict[str, object]", payload)
    return item if item.get("type") == "function_call" else None


def main() -> None:
    """Read one hook payload; exit 0 on every path, printing only a success."""
    run_hook(PostToolUseHost(HOST, response_text, recover_workdir))


if __name__ == "__main__":
    main()
