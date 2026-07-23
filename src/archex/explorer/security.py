"""Session, `Host`, and response-header security controls for the local explorer.

The explorer requires a per-process session token on every request (query
string `?token=` or the `archex_session` cookie the server sets once a
valid token is presented, hardened with `HttpOnly`/`SameSite=Strict`).
Every response also carries a restrictive Content-Security-Policy and
related hardening headers, and every request's `Host` header is validated
against the server's own bind address/port to defend against DNS
rebinding -- see `archex.explorer.server` for where these are enforced.
"""

from __future__ import annotations

import hmac
import secrets
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

SESSION_COOKIE_NAME = "archex_session"
TOKEN_QUERY_PARAM = "token"


def generate_token() -> str:
    """Generate a fresh, unguessable per-process session token."""
    return secrets.token_urlsafe(32)


def token_matches(candidate: str | None, expected: str) -> bool:
    """Constant-time comparison; `candidate` is untrusted client input."""
    if not candidate:
        return False
    return hmac.compare_digest(candidate, expected)


def token_from_query(query_string: str) -> str | None:
    values = parse_qs(query_string).get(TOKEN_QUERY_PARAM)
    return values[0] if values else None


def token_from_cookie_header(cookie_header: str | None) -> str | None:
    if not cookie_header:
        return None
    jar = SimpleCookie()
    jar.load(cookie_header)
    morsel = jar.get(SESSION_COOKIE_NAME)
    return morsel.value if morsel is not None else None


def is_authorized(*, query_string: str, cookie_header: str | None, expected_token: str) -> bool:
    """True when either the query token or the session cookie matches."""
    return token_matches(token_from_query(query_string), expected_token) or token_matches(
        token_from_cookie_header(cookie_header), expected_token
    )


CSP_HEADER_VALUE = (
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "img-src 'self'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)

SECURITY_RESPONSE_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CSP_HEADER_VALUE,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def session_cookie_header(token: str) -> str:
    """A hardened `Set-Cookie` value: HttpOnly, SameSite=Strict, path-scoped."""
    return f"{SESSION_COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Strict"


def allowed_host_headers(bind_host: str, port: int) -> frozenset[str]:
    """The exact `Host` header values this server accepts for BIND_HOST:PORT.

    A browser may send either the literal bind address or the `localhost`
    alias; anything else -- including a bind address with a different port,
    or an attacker-controlled DNS name that happens to resolve to this
    loopback address -- is a DNS-rebinding attempt and must be rejected.
    """
    if bind_host == "127.0.0.1":
        return frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})
    if bind_host == "::1":
        return frozenset({f"[::1]:{port}", f"localhost:{port}"})
    raise ValueError(f"unsupported bind host {bind_host!r}")


def is_valid_host_header(host_header: str | None, allowed: frozenset[str]) -> bool:
    return host_header is not None and host_header in allowed
