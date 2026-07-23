"""Loopback-only HTTP server rendering the local explorer.

Binds hardcoded to `127.0.0.1`/`::1` and requires a per-process session
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
from urllib.parse import parse_qs, urlsplit

from archex.explorer.render import (
    NAV_ITEMS,
    render_diff_page,
    render_error_page,
    render_health_page,
    render_module_map_page,
    render_neighborhood_page,
    render_page,
    render_receipt_page,
)
from archex.explorer.security import (
    SESSION_COOKIE_NAME,
    TOKEN_QUERY_PARAM,
    generate_token,
    is_authorized,
    token_from_query,
    token_matches,
)
from archex.explorer.viewmodel import (
    DEFAULT_NEIGHBORHOOD_DEPTH,
    DEFAULT_NEIGHBORHOOD_LIMIT,
    build_diff_view,
    build_health_view,
    build_manifest_view,
    build_module_map_view,
    build_neighborhood_view,
    build_receipt_view,
)

if TYPE_CHECKING:
    from archex.explorer.loader import ExplorerData
    from archex.explorer.viewmodel import ManifestView, NeighborhoodView
    from archex.graph_query import GraphDirection

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
        params = parse_qs(split.query)
        manifest = build_manifest_view(server.data)
        body = self._render_route(route, params, server, manifest)
        if body is None:
            self._reject(404, "Not Found", f"No view at {route!r}.")
            return

        self._respond(200, body, presented_token=token_from_query(split.query), server=server)

    def _render_route(
        self,
        route: str,
        params: dict[str, list[str]],
        server: ExplorerServer,
        manifest: ManifestView,
    ) -> str | None:
        if route in ("/", ""):
            return render_page("archex explorer", manifest, self._index_body())
        if route == "/view/diff":
            return render_diff_page(manifest, build_diff_view(server.data))
        if route == "/view/modules":
            return render_module_map_page(manifest, build_module_map_view(server.data))
        if route == "/view/receipt":
            return render_receipt_page(manifest, build_receipt_view(server.data))
        if route == "/view/health":
            return render_health_page(manifest, build_health_view(server.data))
        if route == "/view/neighborhood":
            return render_neighborhood_page(manifest, self._neighborhood_view(params, server))
        return None

    def _neighborhood_view(
        self, params: dict[str, list[str]], server: ExplorerServer
    ) -> NeighborhoodView:
        query = params.get("node", [None])[0]
        direction_value = params.get("direction", ["both"])[0]
        direction: GraphDirection = (
            direction_value  # type: ignore[assignment]
            if direction_value in ("both", "out", "in")
            else "both"
        )
        depth = _parse_positive_int(params.get("depth", [None])[0], DEFAULT_NEIGHBORHOOD_DEPTH)
        limit = _parse_positive_int(params.get("limit", [None])[0], DEFAULT_NEIGHBORHOOD_LIMIT)
        return build_neighborhood_view(
            server.data, query, direction=direction, depth=depth, limit=limit
        )

    def _explorer_server(self) -> ExplorerServer:
        server = self.server
        assert isinstance(server, ExplorerServer)
        return server

    def _index_body(self) -> str:
        items = "".join(f'<li><a href="{path}">{label}</a></li>' for label, path in NAV_ITEMS)
        return f"<h2>Views</h2>\n<ul>{items}</ul>"

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


def _parse_positive_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 1 else default
