"""Session-token authorization for the local explorer.

The explorer requires a per-process session token on every request (query
string `?token=` or the `archex_session` cookie the server sets once a valid
token is presented). Loopback binding, CSP, and `Host` header validation are
layered on in `archex.explorer.server`; this module owns only credential
generation and constant-time comparison so neither concern leaks timing
information about the token.
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
