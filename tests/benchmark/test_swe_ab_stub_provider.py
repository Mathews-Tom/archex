"""The stub's Anthropic Messages and OpenAI Responses modes: the turn each host gets per request."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "swe_ab_stub_provider.py"


@pytest.fixture
def stub_url(tmp_path: Path) -> Iterator[str]:
    spec = importlib.util.spec_from_file_location("swe_ab_stub_provider", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handler = module._Handler  # pyright: ignore[reportPrivateUsage]
    handler.script = [
        {"tool": "Bash", "args": {"command": "rg -n hash_password", "description": "search"}},
        {"text": "done"},
    ]
    handler.capture_dir = tmp_path
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _post(url: str, path: str, body: dict[str, Any]) -> str:
    request = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.read().decode()


def _events(sse: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in sse.splitlines()
        if line.startswith("data: ")
    ]


def _tool_result_message() -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_stub_0000", "content": "x"}],
    }


def test_first_turn_is_the_scripted_tool_use_and_the_next_is_the_final_text(stub_url: str) -> None:
    tools = [{"name": "Bash"}]
    first = _events(
        _post(
            stub_url,
            "/v1/messages",
            {"tools": tools, "messages": [{"role": "user", "content": "go"}]},
        )
    )
    block = next(e["content_block"] for e in first if e["type"] == "content_block_start")
    delta = next(e["delta"] for e in first if e["type"] == "content_block_delta")
    stop = next(e["delta"]["stop_reason"] for e in first if e["type"] == "message_delta")
    assert block["type"] == "tool_use"
    assert block["name"] == "Bash"
    assert json.loads(delta["partial_json"]) == {
        "command": "rg -n hash_password",
        "description": "search",
    }
    assert stop == "tool_use"

    second = _events(
        _post(
            stub_url,
            "/v1/messages",
            {
                "tools": tools,
                "messages": [{"role": "user", "content": "go"}, _tool_result_message()],
            },
        )
    )
    assert next(e["delta"]["text"] for e in second if e["type"] == "content_block_delta") == "done"
    assert (
        next(e["delta"]["stop_reason"] for e in second if e["type"] == "message_delta")
        == "end_turn"
    )


def test_request_without_tools_gets_text_and_does_not_advance_the_script(stub_url: str) -> None:
    side = _events(
        _post(stub_url, "/v1/messages", {"messages": [{"role": "user", "content": "title?"}]})
    )
    assert (
        next(e["content_block"]["type"] for e in side if e["type"] == "content_block_start")
        == "text"
    )

    real = _events(
        _post(
            stub_url,
            "/v1/messages",
            {"tools": [{"name": "Bash"}], "messages": [{"role": "user", "content": "go"}]},
        )
    )
    assert (
        next(e["content_block"]["type"] for e in real if e["type"] == "content_block_start")
        == "tool_use"
    )


def test_count_tokens_answers_json(stub_url: str) -> None:
    body = json.loads(
        _post(
            stub_url, "/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "go"}]}
        )
    )
    assert body["input_tokens"] > 0


# --- OpenAI Responses mode (Codex CLI) -------------------------------------------


def _responses_items(sse: str) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Event types in order, the `output_item.done` item, and the `response.completed` response."""
    events = _events(sse)
    item = next(e["item"] for e in events if e["type"] == "response.output_item.done")
    completed = next(e["response"] for e in events if e["type"] == "response.completed")
    return [e["type"] for e in events], item, completed


def test_responses_first_turn_is_the_scripted_function_call_and_the_next_is_the_final_text(
    stub_url: str,
) -> None:
    tools = [{"type": "function", "name": "exec_command"}]
    order, call, completed = _responses_items(
        _post(stub_url, "/v1/responses", {"tools": tools, "input": [{"type": "message"}]})
    )
    assert order == ["response.created", "response.output_item.done", "response.completed"]
    assert call["type"] == "function_call"
    assert call["name"] == "Bash"
    assert call["call_id"] == "call_stub_0000"
    assert json.loads(call["arguments"]) == {
        "command": "rg -n hash_password",
        "description": "search",
    }
    assert completed["usage"]["input_tokens"] > 0

    _, final, _ = _responses_items(
        _post(
            stub_url,
            "/v1/responses",
            {
                "tools": tools,
                "input": [
                    {"type": "message"},
                    call,
                    {"type": "function_call_output", "call_id": call["call_id"], "output": "x"},
                ],
            },
        )
    )
    assert final["type"] == "message"
    assert final["content"][0]["text"] == "done"


def test_responses_request_without_tools_gets_text_and_does_not_advance_the_script(
    stub_url: str,
) -> None:
    _, side, _ = _responses_items(_post(stub_url, "/v1/responses", {"input": []}))
    assert side["type"] == "message"

    _, real, _ = _responses_items(
        _post(stub_url, "/v1/responses", {"tools": [{"name": "x"}], "input": []})
    )
    assert real["type"] == "function_call"


def test_http_error_body_is_returned_verbatim_without_a_retry_after_header(tmp_path: Path) -> None:
    body = {
        "error": {
            "code": "model_capacity_exhausted",
            "message": "No capacity for this model without displacing an active one.",
            "type": "rate_limit_error",
        }
    }
    spec = importlib.util.spec_from_file_location("swe_ab_stub_provider", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handler = module._Handler  # pyright: ignore[reportPrivateUsage]
    handler.script = [{"http_error": 429, "body": body}]
    handler.capture_dir = tmp_path
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
            data=json.dumps({"tools": [{"type": "function"}], "messages": []}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
    assert caught.value.code == 429
    assert json.loads(caught.value.read()) == body
    assert caught.value.headers.get("retry-after") is None
