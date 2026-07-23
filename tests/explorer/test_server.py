"""Browser-level HTTP tests: real requests against a real loopback server.

Exercises the explorer purely through HTTP, the way a browser would -- these
are the "artifact-only browser tests" and "security tests" M5's stack
requires (loopback binding, session token, CSP, `Host` validation, offline
rendering).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.client import HTTPResponse
from pathlib import Path

import pytest

from archex.explorer.loader import ExplorerData, load_explorer_data
from archex.explorer.server import ExplorerSecurityError, ExplorerServer, create_server
from archex.graph_artifact import (
    ArchGraph,
    GraphExportMetadata,
    GraphNode,
    GraphNodeType,
    GraphProject,
)


def _artifact_json(path: Path, source_revision: str = "deadbeef") -> Path:
    payload = {
        "schema_version": {"value": "1.0.0"},
        "archex_version": "0.22.0",
        "generated_at": "2026-07-24T00:00:00Z",
        "source_identity": "acme/widget",
        "source_root": "/repo",
        "source_revision": source_revision,
        "working_tree_fingerprint": "fp",
        "index_generation": "gen1",
        "index_schema_version": "1",
        "chunker_revision": "c1",
        "config_fingerprint": "cfg1",
        "diff": {
            "base_ref": "main",
            "changed_files": [{"path": "a.py", "status": "M", "handle": "file:a.py"}],
            "changed_files_total": 1,
        },
    }
    artifact_path = path / "artifact.json"
    artifact_path.write_text(json.dumps(payload))
    return artifact_path


@pytest.fixture
def explorer_data(tmp_path: Path) -> ExplorerData:
    return load_explorer_data(_artifact_json(tmp_path))


@pytest.fixture
def running_server(explorer_data: ExplorerData) -> Iterator[ExplorerServer]:
    server = create_server(explorer_data, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(url: str) -> HTTPResponse:
    return urllib.request.urlopen(url, timeout=5)  # noqa: S310


def test_root_without_token_is_forbidden(running_server: ExplorerServer) -> None:
    base = f"http://127.0.0.1:{running_server.server_address[1]}/"

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(base)

    assert exc_info.value.code == 403


def test_root_with_valid_token_renders_and_sets_session_cookie(
    running_server: ExplorerServer,
) -> None:
    response = _get(running_server.url)

    assert response.status == 200
    body = response.read().decode("utf-8")
    assert "acme/widget" in body
    assert response.headers.get("Set-Cookie", "").startswith("archex_session=")


def test_session_cookie_authorizes_subsequent_requests_without_token(
    running_server: ExplorerServer,
) -> None:
    first = _get(running_server.url)
    cookie = first.headers["Set-Cookie"].split(";", 1)[0]
    port = running_server.server_address[1]

    request = urllib.request.Request(  # noqa: S310
        f"http://127.0.0.1:{port}/view/diff", headers={"Cookie": cookie}
    )
    response = urllib.request.urlopen(request, timeout=5)  # noqa: S310

    assert response.status == 200
    assert "a.py" in response.read().decode("utf-8")


def test_invalid_token_is_forbidden(running_server: ExplorerServer) -> None:
    port = running_server.server_address[1]

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(f"http://127.0.0.1:{port}/?token=wrong-token")

    assert exc_info.value.code == 403


def test_unknown_view_is_not_found(running_server: ExplorerServer) -> None:
    port = running_server.server_address[1]

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(f"http://127.0.0.1:{port}/view/does-not-exist?token={running_server.token}")

    assert exc_info.value.code == 404


def test_post_is_method_not_allowed(running_server: ExplorerServer) -> None:
    port = running_server.server_address[1]
    request = urllib.request.Request(  # noqa: S310
        f"http://127.0.0.1:{port}/?token={running_server.token}", data=b"", method="POST"
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request, timeout=5)  # noqa: S310

    assert exc_info.value.code == 405


def test_server_refuses_to_bind_non_loopback_host(explorer_data: ExplorerData) -> None:
    with pytest.raises(ExplorerSecurityError):
        create_server(explorer_data, host="0.0.0.0")  # noqa: S104


def test_forbidden_response_does_not_leak_artifact_content(
    running_server: ExplorerServer,
) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _get(f"http://127.0.0.1:{running_server.server_address[1]}/")

    body = exc_info.value.read().decode("utf-8")
    assert "acme/widget" not in body
    assert "a.py" not in body


def _graph() -> ArchGraph:
    return ArchGraph(
        project=GraphProject(name="widget", total_files=1),
        metadata=GraphExportMetadata(archex_version="0.22.0"),
        nodes=[GraphNode(id="file:a.py", type=GraphNodeType.FILE, label="a.py", module="pkg")],
    )


@pytest.fixture
def explorer_data_with_graph(tmp_path: Path) -> ExplorerData:
    return ExplorerData(
        artifact=load_explorer_data(_artifact_json(tmp_path)).artifact, graph=_graph()
    )


@pytest.fixture
def running_server_with_graph(explorer_data_with_graph: ExplorerData) -> Iterator[ExplorerServer]:
    server = create_server(explorer_data_with_graph, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_valid_response_carries_csp_and_hardening_headers(running_server: ExplorerServer) -> None:
    response = _get(running_server.url)

    csp = response.headers.get("Content-Security-Policy", "")
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Referrer-Policy") == "no-referrer"
    assert response.headers.get("Cache-Control") == "no-store"


def test_session_cookie_is_hardened(running_server: ExplorerServer) -> None:
    response = _get(running_server.url)

    cookie = response.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "Path=/" in cookie


def test_wrong_host_header_is_rejected(running_server: ExplorerServer) -> None:
    port = running_server.server_address[1]
    request = urllib.request.Request(  # noqa: S310
        f"http://127.0.0.1:{port}/?token={running_server.token}",
        headers={"Host": "evil.example.com"},
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(request, timeout=5)  # noqa: S310

    assert exc_info.value.code == 400


def test_localhost_alias_host_header_is_accepted(running_server: ExplorerServer) -> None:
    port = running_server.server_address[1]
    request = urllib.request.Request(  # noqa: S310
        f"http://127.0.0.1:{port}/?token={running_server.token}",
        headers={"Host": f"localhost:{port}"},
    )

    response = urllib.request.urlopen(request, timeout=5)  # noqa: S310

    assert response.status == 200


def test_module_map_view_is_reachable(running_server: ExplorerServer) -> None:
    port = running_server.server_address[1]

    response = _get(f"http://127.0.0.1:{port}/view/modules?token={running_server.token}")

    assert response.status == 200
    assert "No graph artifact provided" in response.read().decode("utf-8")


def test_receipt_view_is_reachable(running_server: ExplorerServer) -> None:
    port = running_server.server_address[1]

    response = _get(f"http://127.0.0.1:{port}/view/receipt?token={running_server.token}")

    assert response.status == 200
    assert "Receipt" in response.read().decode("utf-8")


def test_health_view_is_reachable(running_server: ExplorerServer) -> None:
    port = running_server.server_address[1]

    response = _get(f"http://127.0.0.1:{port}/view/health?token={running_server.token}")

    assert response.status == 200
    assert "Index Health" in response.read().decode("utf-8")


def test_module_map_view_renders_with_graph(running_server_with_graph: ExplorerServer) -> None:
    port = running_server_with_graph.server_address[1]

    response = _get(f"http://127.0.0.1:{port}/view/modules?token={running_server_with_graph.token}")

    assert response.status == 200
    assert "pkg" in response.read().decode("utf-8")


def test_neighborhood_view_finds_seed_with_graph(
    running_server_with_graph: ExplorerServer,
) -> None:
    port = running_server_with_graph.server_address[1]

    response = _get(
        f"http://127.0.0.1:{port}/view/neighborhood"
        f"?node=file:a.py&token={running_server_with_graph.token}"
    )

    assert response.status == 200
    assert "file:a.py" in response.read().decode("utf-8")
