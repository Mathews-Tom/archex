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

The exit, ledger, budget and fast-exit contract is the shared runner's
(`archex.integrations.post_tool_use_annotate`); this module only supplies the
host name and the `tool_response` mapping.
"""

from __future__ import annotations

from typing import Final, cast

from archex.annotate import HOST_TOOLS
from archex.integrations.post_tool_use_annotate import PostToolUseHost
from archex.integrations.post_tool_use_annotate import main as run_hook

HOST: Final = "claude-code"

#: Installed `matcher` value: exactly the tools `archex.annotate` treats as
#: searches on this host, so the installed config and the annotate core's own
#: tool table cannot drift apart.
HOOK_MATCHER = "|".join(HOST_TOOLS[HOST])


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


def main() -> None:
    """Read one hook payload; exit 0 on every path, printing only a success."""
    run_hook(PostToolUseHost(HOST, response_text))


if __name__ == "__main__":
    main()
