"""Shared utilities for archex CLI and integrations."""

from __future__ import annotations

from archex.acquire import is_remote_url
from archex.models import RepoSource


def resolve_source(path_or_url: str) -> RepoSource:
    """Build a RepoSource from a local path or HTTP(S) URL string."""
    if is_remote_url(path_or_url):
        return RepoSource(url=path_or_url)
    return RepoSource(local_path=path_or_url)


def printable(text: str) -> str:
    """Replace control characters with `?` before echoing text to a terminal.

    Used for values that come from the filesystem rather than from archex —
    a directory name can legitimately contain an escape sequence, and a
    terminal would interpret it rather than display it.
    """
    return "".join(char if char.isprintable() or char == " " else "?" for char in text)
