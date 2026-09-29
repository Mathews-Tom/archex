"""Claude Code `PostToolUse` hook: annotate search results with the code they hit.

Installed by `archex install-client claude-code --hooks` on
`HOOK_MATCHER` (`Bash|Grep|Glob`) and invoked as
`python -m archex.integrations.claude_code_annotate_hook`. It reads the hook
payload, maps `tool_response` to the search result text, asks
`archex.annotate` (through `annotate_hook.run_request`, in-process) which
indexed code unit each hit falls in, and prints

    {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                            "additionalContext": "<annotation lines>"}}

Claude Code adds `additionalContext` beside the tool result, so the original
result reaches the model unchanged; the hook never uses `updatedToolOutput`.

Result text per tool (payloads recorded at a live Claude Code 2.1.285 hook):

- `Bash`: `tool_response.stdout`.
- `Grep`: `tool_response.content` when `tool_input.output_mode` is
  ``content``; otherwise `tool_response.filenames`, one per line.
- `Glob`: `tool_response.filenames`, one per line.

Contract:

- Exit 0 on every path. Stdout carries the JSON above on success and nothing
  otherwise; failures go to the diagnostics log
  (`ARCHEX_HOOK_DIAGNOSTICS_LOG`, default `~/.archex/hook-diagnostics.log`).
- One ledger line per *search* call (`ARCHEX_ANNOTATION_LEDGER`, one schema for
  every host, see `annotate_hook.append_ledger`), keyed by `tool_use_id`.
  A shell command that is not a search leaves no ledger line and no output.
- A hard wall-clock guard (`ARCHEX_HOOK_TIMEOUT_SECONDS`, 0.5 s by default,
  counted from this module's `main`) ends the process with exit 0 and no
  stdout once the budget is spent, however far the annotation got.
- A shell command that is not a search exits before the index, the tokenizer,
  or pydantic is imported: this module and everything it imports at top level
  are standard library plus `archex.annotate`, which stays light until a call
  proves to be a search.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Final, cast

from archex.annotate import HOST_TOOLS, classify_call
from archex.integrations.annotate_hook import AnnotateRequest, append_ledger, run_request
from archex.integrations.diagnostics import hook_timeout_seconds, log_diagnostic

HOST: Final = "claude-code"
HOOK_EVENT_NAME: Final = "PostToolUse"

#: Installed `matcher` value: exactly the tools `archex.annotate` treats as
#: searches on this host, so the installed config and the annotate core's own
#: tool table cannot drift apart.
HOOK_MATCHER = "|".join(HOST_TOOLS[HOST])


class _Budget:
    """The wall-clock guard: whichever of the timer and the caller settles first wins."""

    def __init__(self, seconds: float) -> None:
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._settled = False
        self._search: tuple[str, str | None] | None = None
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._started) * 1000

    def note_search(self, tool: str, tool_call_id: str | None) -> None:
        with self._lock:
            self._search = (tool, tool_call_id)

    def settle(self) -> bool:
        """Claim the right to write output; `False` once the timer has fired."""
        with self._lock:
            if self._settled:
                return False
            self._settled = True
        self._timer.cancel()
        return True

    def _expire(self) -> None:
        with self._lock:
            if self._settled:
                return
            self._settled = True
            search = self._search
        log_diagnostic("annotate_timeout", detail=f"budget spent after {self.elapsed_ms():.0f} ms")
        if search is not None:
            append_ledger(
                host=HOST,
                tool=search[0],
                tool_call_id=search[1],
                latency_ms=self.elapsed_ms(),
                reason="timeout",
            )
        os._exit(0)


def main() -> None:
    """Read one hook payload; exit 0 on every path, printing only a success."""
    budget = _Budget(hook_timeout_seconds())
    try:
        output = _handle(sys.stdin.read(), budget)
        if output is not None:
            sys.stdout.write(output)
        sys.stdout.flush()
    except BaseException as exc:  # noqa: BLE001 - the non-blocking contract requires this
        log_diagnostic("annotate_unhandled_exception", detail=repr(exc))
    # `os._exit` skips interpreter teardown and cannot be delayed by a worker
    # still running past the budget; the flush above already ran inside the guard.
    os._exit(0)


def _handle(raw: str, budget: _Budget) -> str | None:
    payload = parse_payload(raw)
    if payload is None:
        return None
    tool = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if not isinstance(tool, str) or not isinstance(tool_input, dict):
        log_diagnostic("malformed_payload", detail="tool_name or tool_input missing or mistyped")
        return None
    tool_input = cast("dict[str, object]", tool_input)
    if isinstance(classify_call(HOST, tool, tool_input), str):
        return None  # not a search: no output, no ledger line, no heavy imports

    tool_use_id = payload.get("tool_use_id")
    tool_call_id = tool_use_id if isinstance(tool_use_id, str) else None
    cwd_raw = payload.get("cwd")
    cwd = cwd_raw if isinstance(cwd_raw, str) and cwd_raw else os.getcwd()
    budget.note_search(tool, tool_call_id)
    try:
        text = response_text(tool, tool_input, payload.get("tool_response"))
        if text is None:
            detail = f"tool={tool} tool_response unusable"
            log_diagnostic("malformed_payload", detail=detail, cwd=cwd)
            _settle_ledger(budget, tool, tool_call_id, reason="malformed_tool_response")
            return None
        result = run_request(AnnotateRequest(HOST, tool, tool_input, text, cwd))
    except Exception as exc:  # noqa: BLE001 - fail open: no stdout, diagnostics only
        log_diagnostic("annotate_error", detail=repr(exc), cwd=cwd)
        _settle_ledger(budget, tool, tool_call_id, reason="internal_error")
        return None

    if not budget.settle():
        return None
    append_ledger(
        host=HOST,
        tool=tool,
        tool_call_id=tool_call_id,
        latency_ms=budget.elapsed_ms(),
        annotation=result,
    )
    if not result.annotated or not result.text:
        return None
    return json.dumps(
        {"hookSpecificOutput": {"hookEventName": HOOK_EVENT_NAME, "additionalContext": result.text}}
    )


def _settle_ledger(budget: _Budget, tool: str, tool_call_id: str | None, *, reason: str) -> None:
    """Write the ledger line for a search call that failed before annotation."""
    if budget.settle():
        append_ledger(
            host=HOST,
            tool=tool,
            tool_call_id=tool_call_id,
            latency_ms=budget.elapsed_ms(),
            reason=reason,
        )


def parse_payload(raw: str) -> dict[str, object] | None:
    """Parse a hook payload, logging and discarding anything malformed."""
    if not raw.strip():
        log_diagnostic("malformed_payload", detail="empty stdin")
        return None
    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        log_diagnostic("malformed_payload", detail=f"invalid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        log_diagnostic("malformed_payload", detail="payload is not a JSON object")
        return None
    return cast("dict[str, object]", payload)


def response_text(tool: str, tool_input: dict[str, object], response: object) -> str | None:
    """The search result text a `PostToolUse` payload carries, or `None` if unusable."""
    if not isinstance(response, dict):
        return None
    fields = cast("dict[str, object]", response)
    if tool == "Bash":
        stdout = fields.get("stdout")
        return stdout if isinstance(stdout, str) else None
    if tool == "Grep" and tool_input.get("output_mode") == "content":
        content = fields.get("content")
        return content if isinstance(content, str) else None
    filenames = fields.get("filenames")
    if isinstance(filenames, list):
        names = cast("list[object]", filenames)
        if all(isinstance(name, str) for name in names):
            return "\n".join(cast("list[str]", names))
    return None


if __name__ == "__main__":
    main()
