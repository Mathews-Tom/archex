"""Host-neutral annotation entry point for hook adapters.

`python -m archex.integrations.annotate_hook` reads one JSON request
``{host, tool, input, text, cwd}`` on stdin (``host`` defaults to ``omp``),
prints the decision record as one JSON line — the annotation text plus
``eligible``, ``annotated``, ``units_hit``, ``units_rendered``, ``tokens``,
``freshness``, ``reason``, ``format``, ``index_revision``, ``capped``, and
``out_of_range_hits`` — and exits 0 on every path. A failure prints nothing
and writes one line to the hook diagnostics log.

The omp/Pi extension module and the OpenCode plugin spawn this process for
every call their tool filter passes, including every shell command, so it
imports only the standard library and `archex.annotate` (itself cheap until a
call proves to be a search) before deciding. The Claude Code and Codex hooks
call `run_request` in-process.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from archex.annotate import annotate
from archex.integrations.diagnostics import log_diagnostic

if TYPE_CHECKING:
    from archex.annotate import Annotation

#: Declines that are normal traffic, not faults worth a diagnostics line.
QUIET_DECLINES = frozenset(
    {
        "no_hits",
        "not_search_command",
        "unsupported_tool",
        "all_units_visible",
        "count_output",
    }
)


@dataclass(frozen=True)
class AnnotateRequest:
    host: str
    tool: str
    tool_input: dict[str, object]
    text: str
    cwd: str


def parse_request(raw: str) -> AnnotateRequest:
    """Validate a ``{host, tool, input, text, cwd}`` envelope; raises `ValueError`."""
    envelope_obj: object = json.loads(raw)
    if not isinstance(envelope_obj, dict):
        raise ValueError("request is not a JSON object")
    envelope = cast("dict[str, object]", envelope_obj)
    host = envelope.get("host", "omp")
    tool = envelope.get("tool")
    tool_input = envelope.get("input", {})
    text = envelope.get("text", "")
    cwd = envelope.get("cwd") or "."
    if not isinstance(host, str) or not isinstance(tool, str):
        raise ValueError("request fields 'host' and 'tool' must be strings")
    if not isinstance(tool_input, dict):
        raise ValueError("request field 'input' is not a JSON object")
    if not isinstance(text, str) or not isinstance(cwd, str):
        raise ValueError("request fields 'text' and 'cwd' must be strings")
    return AnnotateRequest(host, tool, cast("dict[str, object]", tool_input), text, cwd)


def run_request(request: AnnotateRequest) -> Annotation:
    """Annotate one request, logging any decline that is not routine traffic."""
    result = annotate(
        request.tool, request.tool_input, request.text, request.cwd, host=request.host
    )
    if result.reason is not None and result.reason not in QUIET_DECLINES:
        detail = (
            f"host={request.host} tool={request.tool} reason={result.reason} "
            f"freshness={result.freshness} format={result.format}"
        )
        log_diagnostic("annotate_declined", detail=detail, cwd=request.cwd)
    return result


def main() -> None:
    cwd: str | None = None
    try:
        request = parse_request(sys.stdin.read())
        cwd = request.cwd
        record = run_request(request).as_record()
    except Exception as exc:  # noqa: BLE001 - fail open: no stdout, diagnostics only
        log_diagnostic("annotate_error", detail=repr(exc), cwd=cwd)
        return
    sys.stdout.write(json.dumps(record, sort_keys=True) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
