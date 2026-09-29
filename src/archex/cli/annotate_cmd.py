"""CLI annotate subcommand: append indexed code-unit facts to a search result.

Hook adapters do not go through this command: they call
`archex.integrations.annotate_hook`, which skips importing click and every
other subcommand.
"""

from __future__ import annotations

import json
import sys
from typing import cast

import click

from archex.annotate import ANNOTATE_HOSTS, HOST_TOOLS, Annotation
from archex.integrations.annotate_hook import AnnotateRequest, parse_request, run_request
from archex.integrations.diagnostics import log_diagnostic

_TOOL_HELP = "; ".join(f"{host}: {', '.join(tools)}" for host, tools in HOST_TOOLS.items())


def _load_request(
    raw_stdin: str,
    host: str,
    tool: str | None,
    input_json: str | None,
    cwd: str | None,
    stdin_json: bool,
) -> AnnotateRequest:
    if stdin_json:
        request = parse_request(raw_stdin)
        return AnnotateRequest(
            request.host,
            tool or request.tool,
            request.tool_input,
            request.text,
            cwd or request.cwd,
        )
    if tool is None:
        raise ValueError("--tool is required unless --stdin-json supplies it")
    tool_input_obj: object = json.loads(input_json) if input_json else {}
    if not isinstance(tool_input_obj, dict):
        raise ValueError("tool input is not a JSON object")
    return AnnotateRequest(
        host, tool, cast("dict[str, object]", tool_input_obj), raw_stdin, cwd or "."
    )


@click.command("annotate")
@click.option(
    "--host",
    type=click.Choice(ANNOTATE_HOSTS),
    default="omp",
    show_default=True,
    help="Agent host whose tool produced the result on stdin.",
)
@click.option(
    "--tool",
    default=None,
    help=f"The host's name for the tool that produced the result ({_TOOL_HELP}).",
)
@click.option("--input-json", default=None, help="The tool's input arguments as a JSON object.")
@click.option("--cwd", default=None, help="Directory the tool ran in (default: current).")
@click.option(
    "--stdin-json",
    is_flag=True,
    default=False,
    help="Read {host, tool, input, text, cwd} as one JSON object from stdin.",
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
    host: str,
    tool: str | None,
    input_json: str | None,
    cwd: str | None,
    stdin_json: bool,
    output_format: str,
) -> None:
    """Annotate a grep/glob/shell-search result with the code units it hits.

    Reads the tool's result text on stdin and prints one line per distinct
    indexed code unit containing a hit. Prints nothing when the call is not a
    search, the index is not fresh, the output format is unrecognised, or
    anything fails; the reason goes to the hook diagnostics log. Always exits 0.
    """
    result: Annotation
    try:
        request = _load_request(sys.stdin.read(), host, tool, input_json, cwd, stdin_json)
        result = run_request(request)
    except Exception as exc:  # noqa: BLE001 - fail open: no stdout, diagnostics only
        log_diagnostic("annotate_error", detail=repr(exc), cwd=cwd or ".")
        return
    if output_format == "json":
        click.echo(json.dumps(result.as_record(), sort_keys=True))
    elif result.annotated:
        click.echo(result.text)
