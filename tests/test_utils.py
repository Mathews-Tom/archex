"""Tests for archex.utils.resolve_source."""

from __future__ import annotations

import pytest

from archex.utils import resolve_source


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/user/repo.git",
        "http://example.com/repo.git",
        "HTTP://example.com/repo.git",
        "HTTPS://example.com/repo.git",
        "Https://example.com/repo.git",
    ],
)
def test_resolve_source_recognizes_http_urls_case_insensitively(url: str) -> None:
    """MCP tool callers may pass a non-lowercase scheme; it must still resolve to a URL."""
    source = resolve_source(url)

    assert source.url == url
    assert source.local_path is None


@pytest.mark.parametrize(
    "path_or_url",
    [
        "/home/user/local-repo",
        "./relative/path",
        "example.com:some/repo.git",
        "git@github.com:user/repo.git",
        "ssh://git@github.com/user/repo.git",
        "ext::sh -c touch /tmp/pwned",
    ],
)
def test_resolve_source_treats_non_http_values_as_local_path(path_or_url: str) -> None:
    """Anything that isn't an http(s) URL is handed to the local-path branch, which
    open_local() then validates (and rejects if it isn't a real repo directory) —
    it must never be misclassified as a URL and handed to clone_repo().
    """
    source = resolve_source(path_or_url)

    assert source.local_path == path_or_url
    assert source.url is None
