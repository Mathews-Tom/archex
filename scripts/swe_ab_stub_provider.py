"""Local OpenAI-compatible chat-completions stub for no-spend SWE A/B rehearsals.

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
* ``{"text": "done"}`` — a final answer, which ends the session.

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
        with _Handler.lock:
            _Handler.counter += 1
            number = _Handler.counter
        (self.capture_dir / f"request-{number:04d}.json").write_text(
            json.dumps(body, indent=1), encoding="utf-8"
        )
        messages = cast("list[dict[str, Any]]", body.get("messages") or [])
        turn = sum(1 for message in messages if message.get("role") == "tool")
        step = self.script[min(turn, len(self.script) - 1)]
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
