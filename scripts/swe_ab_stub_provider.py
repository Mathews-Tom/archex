"""Local model-API stub (chat-completions and Anthropic Messages) for no-spend rehearsals.

Replays a scripted conversation over Server-Sent Events so the whole cell flow
— omp, the annotation hook, the session parser, the validator — runs without a
hosted model. The reply to each request is chosen by how many tool results the
request already carries, so the stub keeps no state between requests and one
instance serves any number of sequential cells.

Every request body is written to ``--capture-dir`` (``request-0001.json``,
...): the system prompt and the advertised tool list a real provider would
receive. Stage 0 reads those captures for its isolation checks.

Script steps (JSON list), one per model turn:

* ``{"tool": "grep", "args": {"pattern": "hash_password"}}`` — a tool call;
* ``{"tool": "edit", "replace": ["old text", "new text"]}`` — an omp ``edit``
  call that rewrites the first line containing ``old text`` in the file the
  most recent ``read`` returned, addressed by that read's hashline anchor;
* ``{"text": "done"}`` — a final answer, which ends the session;
* ``{"http_error": 429, "message": "usage limit reached", "retry_after": 600}`` — an HTTP error
  answered to every request that reaches this step, as a subscription rate limit or quota
  block would be (a step at index 0 blocks before the first tool call; a later one, mid-run).

The same server also speaks the Anthropic Messages protocol, so Claude Code can
be driven against it with ``ANTHROPIC_BASE_URL=http://127.0.0.1:<port>`` and a
placeholder ``ANTHROPIC_AUTH_TOKEN``: ``POST .../v1/messages`` streams the
scripted turn (a ``tool_use`` block for a ``tool`` step, a text block for a
``text`` step, keyed on the number of ``tool_result`` blocks in the request,
with ``args`` passed through as the tool input), ``POST .../count_tokens``
returns an estimate, and a request that advertises no tools (Claude Code's
title and summary side requests) gets a text reply without consuming a turn.

It also speaks the OpenAI Responses protocol for Codex CLI: point a
``[model_providers.<id>]`` table with ``base_url = "http://127.0.0.1:<port>/v1"``
and ``wire_api = "responses"`` at it and ``POST .../responses`` streams the
scripted turn (a ``function_call`` item for a ``tool`` step, e.g. Codex's
``exec_command`` with ``{"cmd": "rg -n x"}``; an assistant message for a ``text``
step), keyed on the number of ``function_call_output`` items in the request.

Usage: ``python scripts/swe_ab_stub_provider.py --port 47811 --script s.json
--capture-dir /tmp/capture``. Cells driven through it record
``provider_endpoint_overridden: true`` and are refused for publication.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

_READ_HEADER = re.compile(r"^\[(?P<path>[^\]#]+)#(?P<tag>[0-9A-F]{4})\]$", re.MULTILINE)
_READ_LINE = re.compile(r"^(?P<line>\d+)[:|]\s?(?P<text>.*)$")


def _chunk(turn: int, delta: dict[str, Any], finish: str | None) -> dict[str, Any]:
    return {
        "id": f"stub-{turn}",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "stub-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _last_read(messages: list[dict[str, Any]]) -> tuple[str, str, list[str]] | None:
    """Path, anchor tag, and numbered lines of the latest read result in the request."""
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        header = _READ_HEADER.search(text)
        if header is not None:
            body = text[header.end() :].lstrip("\n").split("\n")
            return header.group("path"), header.group("tag"), body
    return None


def _edit_arguments(step: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    read = _last_read(messages)
    if read is None:
        raise ValueError("an edit step needs an earlier read result")
    path, tag, lines = read
    old, new = cast("list[str]", step["replace"])
    for raw in lines:
        match = _READ_LINE.match(raw)
        if match is not None and old in match.group("text"):
            number = int(match.group("line"))
            replacement = match.group("text").replace(old, new)
            return {"input": f"[{path}#{tag}]\nPUT {number}.={number}:\n+{replacement}\n"}
    raise ValueError(f"no line containing {old!r} in the last read of {path}")


class _Handler(BaseHTTPRequestHandler):
    script: list[dict[str, Any]] = []
    capture_dir: Path = Path()
    counter = 0
    lock = threading.Lock()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        return

    def do_GET(self) -> None:
        self._json({"object": "list", "data": [{"id": "stub-model", "object": "model"}]})

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = cast("dict[str, Any]", json.loads(self.rfile.read(length) or b"{}"))
        if self.path.split("?", 1)[0].endswith("/count_tokens"):
            self._json({"input_tokens": len(json.dumps(body)) // 4})
            return
        with _Handler.lock:
            _Handler.counter += 1
            number = _Handler.counter
        (self.capture_dir / f"request-{number:04d}.json").write_text(
            json.dumps(body, indent=1), encoding="utf-8"
        )
        route = self.path.split("?", 1)[0]
        if route.endswith("/messages"):
            self._anthropic(body)
            return
        if route.endswith("/responses"):
            self._responses(body)
            return
        messages = cast("list[dict[str, Any]]", body.get("messages") or [])
        turn = sum(1 for message in messages if message.get("role") == "tool")
        step = self.script[min(turn, len(self.script) - 1)]
        if "http_error" in step:
            self._error(step)
            return
        if "tool" in step:
            tool = str(step["tool"])
            arguments = (
                _edit_arguments(step, messages)
                if "replace" in step
                else cast("dict[str, Any]", step.get("args", {}))
            )
            call = {
                "index": 0,
                "id": f"call_{turn}",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(arguments)},
            }
            delta: dict[str, Any] = {"role": "assistant", "tool_calls": [call]}
            finish = "tool_calls"
        else:
            delta = {"role": "assistant", "content": str(step["text"])}
            finish = "stop"
        prompt_tokens = len(json.dumps(messages)) // 4
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 20,
            "total_tokens": prompt_tokens + 20,
        }
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        events: list[dict[str, Any]] = [
            _chunk(turn, delta, None),
            _chunk(turn, {}, finish),
            {**_chunk(turn, {}, None), "choices": [], "usage": usage},
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _anthropic(self, body: dict[str, Any]) -> None:
        """Answer an Anthropic Messages request (Claude Code) over SSE.

        The turn is the number of ``tool_result`` blocks already in the request.
        A request that advertises no tools (title or summary side requests) gets
        a short text reply and does not advance the script.
        """
        messages = cast("list[dict[str, Any]]", body.get("messages") or [])
        turn = sum(
            1
            for message in messages
            if isinstance(message.get("content"), list)
            for block in cast("list[dict[str, Any]]", message["content"])
            if block.get("type") == "tool_result"
        )
        step: dict[str, Any] = (
            self.script[min(turn, len(self.script) - 1)] if body.get("tools") else {"text": "stub"}
        )
        if "tool" in step:
            block: dict[str, Any] = {
                "type": "tool_use",
                "id": f"toolu_stub_{turn:04d}",
                "name": str(step["tool"]),
                "input": {},
            }
            delta: dict[str, Any] = {
                "type": "input_json_delta",
                "partial_json": json.dumps(step.get("args", {})),
            }
            stop_reason = "tool_use"
        else:
            block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": str(step["text"])}
            stop_reason = "end_turn"
        input_tokens = len(json.dumps(messages)) // 4
        message: dict[str, Any] = {
            "id": f"msg_stub_{turn:04d}",
            "type": "message",
            "role": "assistant",
            "model": str(body.get("model") or "stub-model"),
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 1},
        }
        events: list[tuple[str, dict[str, Any]]] = [
            ("message_start", {"type": "message_start", "message": message}),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": block},
            ),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": 20},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for name, event in events:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()

    def _responses(self, body: dict[str, Any]) -> None:
        """Answer an OpenAI Responses request (Codex CLI) over SSE.

        The turn is the number of ``function_call_output`` items already in the
        request ``input`` (Codex resends the whole history each turn). A request
        that advertises no tools gets a short text reply and does not advance the
        script. A ``tool`` step becomes a ``function_call`` output item whose
        ``arguments`` are the step's ``args``; a ``text`` step becomes an
        assistant message.
        """
        items = cast("list[dict[str, Any]]", body.get("input") or [])
        turn = sum(1 for item in items if item.get("type") == "function_call_output")
        step: dict[str, Any] = (
            self.script[min(turn, len(self.script) - 1)] if body.get("tools") else {"text": "stub"}
        )
        if "tool" in step:
            item: dict[str, Any] = {
                "type": "function_call",
                "id": f"fc_stub_{turn:04d}",
                "call_id": f"call_stub_{turn:04d}",
                "name": str(step["tool"]),
                "arguments": json.dumps(step.get("args", {})),
                "status": "completed",
            }
        else:
            item = {
                "type": "message",
                "id": f"msg_stub_{turn:04d}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": str(step["text"]), "annotations": []}],
            }
        input_tokens = len(json.dumps(items)) // 4
        response_id = f"resp_stub_{turn:04d}"
        events: list[dict[str, Any]] = [
            {"type": "response.created", "response": {"id": response_id}},
            {"type": "response.output_item.done", "item": item},
            {
                "type": "response.completed",
                "response": {
                    "id": response_id,
                    "usage": {
                        "input_tokens": input_tokens,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 20,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": input_tokens + 20,
                    },
                },
            },
        ]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()

    def _error(self, step: dict[str, Any]) -> None:
        """Answer the request with an HTTP error, as a subscription rate limit or quota block."""
        payload = json.dumps(
            {
                "error": {
                    "type": "rate_limit_error",
                    "message": str(step.get("message", "rate limited")),
                }
            }
        ).encode()
        self.send_response(int(step["http_error"]))
        self.send_header("content-type", "application/json")
        if "retry_after" in step:
            self.send_header("retry-after", str(step["retry_after"]))
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    _Handler.script = cast("list[dict[str, Any]]", json.loads(args.script.read_text()))
    _Handler.capture_dir = args.capture_dir
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"stub listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
