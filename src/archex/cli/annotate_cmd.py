"""CLI annotate subcommand: append indexed code-unit facts to a search result.

Also runnable as `python -m archex.cli.annotate_cmd`, which is how the omp/Pi
hook module calls it: that entry skips importing every other subcommand, which
alone costs more than the hook's latency budget.
"""

from __future__ import annotations

import json
import sys
from typing import cast

import click

from archex.annotate import ANNOTATE_TOOLS, Annotation, annotate
from archex.integrations.hook import log_diagnostic

#: Declines that are normal traffic, not faults worth a diagnostics line.
_QUIET_DECLINES = frozenset(
    {"no_hits", "not_search_command", "unsupported_tool", "all_units_visible"}
)


def _load_request(
    raw_stdin: str,
    tool: str | None,
    input_json: str | None,
    cwd: str | None,
    stdin_json: bool,
) -> tuple[str, dict[str, object], str, str]:
    tool_input_obj: object
    if stdin_json:
        envelope_obj: object = json.loads(raw_stdin)
        if not isinstance(envelope_obj, dict):
            raise ValueError("stdin envelope is not a JSON object")
        envelope = cast("dict[str, object]", envelope_obj)
        tool = tool or str(envelope.get("tool", ""))
        tool_input_obj = envelope.get("input", {})
        text_obj = envelope.get("text", "")
        cwd = cwd or str(envelope.get("cwd") or ".")
        if not isinstance(text_obj, str):
            raise ValueError("stdin envelope field 'text' is not a string")
        text = text_obj
    else:
        tool_input_obj = json.loads(input_json) if input_json else {}
        text = raw_stdin
    if not isinstance(tool_input_obj, dict):
        raise ValueError("tool input is not a JSON object")
    if tool not in ANNOTATE_TOOLS:
        raise ValueError(f"unsupported tool: {tool!r}")
    return tool, cast("dict[str, object]", tool_input_obj), text, cwd or "."


@click.command("annotate")
@click.option(
    "--tool",
    type=click.Choice(ANNOTATE_TOOLS),
    default=None,
    help="Host tool that produced the result on stdin.",
)
@click.option("--input-json", default=None, help="The tool's input arguments as a JSON object.")
@click.option("--cwd", default=None, help="Directory the tool ran in (default: current).")
@click.option(
    "--stdin-json",
    is_flag=True,
    default=False,
    help="Read {tool, input, text, cwd} as one JSON object from stdin.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="text: annotation lines only; json: the lines plus the decision record.",
)
def annotate_cmd(
    tool: str | None,
    input_json: str | None,
    cwd: str | None,
    stdin_json: bool,
    output_format: str,
) -> None:
    """Annotate a grep/glob/bash-search result with the code units it hits.

    Reads the tool's model-facing result text on stdin and prints one line per
    distinct indexed code unit containing a hit. Prints nothing when the index
    is not fresh, the output format is unrecognised, or anything fails; the
    reason goes to the hook diagnostics log. Always exits 0.
    """
    result: Annotation | None = None
    effective_cwd = cwd or "."
    try:
        request = _load_request(sys.stdin.read(), tool, input_json, cwd, stdin_json)
        effective_cwd = request[3]
        result = annotate(*request)
    except Exception as exc:  # noqa: BLE001 - fail open: no stdout, diagnostics only
        log_diagnostic("annotate_error", detail=repr(exc), cwd=effective_cwd)
        return
    if result.reason is not None and result.reason not in _QUIET_DECLINES:
        detail = f"reason={result.reason} freshness={result.freshness} format={result.format}"
        log_diagnostic("annotate_declined", detail=detail, cwd=effective_cwd)
    if output_format == "json":
        click.echo(json.dumps(result.as_record(), sort_keys=True))
    elif result.annotated:
        click.echo(result.text)


if __name__ == "__main__":
    annotate_cmd()
