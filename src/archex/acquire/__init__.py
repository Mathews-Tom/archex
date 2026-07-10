"""Repository acquisition: clone, fetch, and prepare source trees for analysis."""

from __future__ import annotations

from archex.acquire.discovery import discover_files
from archex.acquire.git import clone_repo, is_remote_url
from archex.acquire.local import open_local

__all__ = ["clone_repo", "discover_files", "is_remote_url", "open_local"]
