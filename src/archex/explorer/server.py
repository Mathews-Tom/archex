"""Loopback-only HTTP server rendering the local explorer.

PR-1 binds hardcoded to `127.0.0.1`/`::1` and requires a per-process session
token on every request (see `archex.explorer.security`). CSP response
headers and `Host` header validation are added by a follow-up hardening
change; until then this server is loopback-reachable-only, GET-only, and
serves nothing without a valid token.
"""

from __future__ import annotations

import logging
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from archex.explorer.render import render_diff_page, render_error_page, render_page
from archex.explorer.security import (
    SESSION_COOKIE_NAME,
    TOKEN_QUERY_PARAM,
    generate_token,
    is_authorized,
    token_from_query,
    token_matches,
)
from archex.explorer.viewmodel import build_diff_view, build_manifest_view

if TYPE_CHECKING:
    from archex.explorer.loader import ExplorerData

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})

logger = logging.getLogger(__name__)


class ExplorerSecurityError(ValueError):
    """Raised when the explorer server is asked to bind an unsafe address."""


class ExplorerServer(ThreadingHTTPServer):
    """A loopback-only, GET-only, session-token-gated `AnalysisArtifactV1` viewer."""

    daemon_threads = True
    allow_reuse_address = True
    data: ExplorerData
    token: str

    def __init__(
        self,
        data: ExplorerData,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        token: str | None = None,
    ) -> None:
        if host not in LOOPBACK_HOSTS:
            raise ExplorerSecurityError(
                f"refusing to bind non-loopback host {host!r}; "
                f"only {sorted(LOOPBACK_HOSTS)} are allowed"
            )
        self.data = data
        self.token = token or generate_token()
        self.address_family = socket.AF_INET6 if host == "::1" else socket.AF_INET
        super().__init__((host, port), _ExplorerRequestHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        display_host = f"[{host}]" if ":" in str(host) else str(host)
        return f"http://{display_host}:{port}/?{TOKEN_QUERY_PARAM}={self.token}"


def create_server(
    data: ExplorerData,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
) -> ExplorerServer:
    return ExplorerServer(data, host=host, port=port, token=token)


class _ExplorerRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logger.debug("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        self._handle_get()

    def do_POST(self) -> None:  # noqa: N802
        self._reject(405, "Method Not Allowed", "This explorer serves read-only GET requests.")

    def do_PUT(self) -> None:  # noqa: N802
        self._reject(405, "Method Not Allowed", "This explorer serves read-only GET requests.")

    def do_DELETE(self) -> None:  # noqa: N802
        self._reject(405, "Method Not Allowed", "This explorer serves read-only GET requests.")

    def do_PATCH(self) -> None:  # noqa: N802
        self._reject(405, "Method Not Allowed", "This explorer serves read-only GET requests.")

    def _handle_get(self) -> None:
        server = self._explorer_server()
        split = urlsplit(self.path)
        if not is_authorized(
            query_string=split.query,
            cookie_header=self.headers.get("Cookie"),
            expected_token=server.token,
        ):
            self._reject(403, "Forbidden", "A valid session token is required.")
            return

        route = split.path
        manifest = build_manifest_view(server.data)
        if route in ("/", ""):
            body = render_page("archex explorer", manifest, self._index_body())
        elif route == "/view/diff":
            body = render_diff_page(manifest, build_diff_view(server.data))
        else:
            self._reject(404, "Not Found", f"No view at {route!r}.")
            return

        self._respond(200, body, presented_token=token_from_query(split.query), server=server)

    def _explorer_server(self) -> ExplorerServer:
        server = self.server
        assert isinstance(server, ExplorerServer)
        return server

    def _index_body(self) -> str:
        return '<h2>Views</h2>\n<ul><li><a href="/view/diff">Diff Review</a></li></ul>'

    def _respond(
        self,
        status: int,
        html_body: str,
        *,
        presented_token: str | None,
        server: ExplorerServer,
    ) -> None:
        payload = html_body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        if presented_token is not None and token_matches(presented_token, server.token):
            self.send_header("Set-Cookie", f"{SESSION_COOKIE_NAME}={presented_token}; Path=/")
        self.end_headers()
        self.wfile.write(payload)

    def _reject(self, status: int, title: str, message: str) -> None:
        payload = render_error_page(status, title, message).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
