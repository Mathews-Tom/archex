"""archex — architecture extraction and analysis toolkit."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from archex.api import analyze, compare, query, record_usage_event

    __version__: str

# Repo-local state is published under POSIX advisory locks (`fcntl.flock`), and
# CI covers Linux and macOS only. Fail at the package boundary with a remedy
# instead of a bare `ModuleNotFoundError: fcntl` from deep inside `archex.api`.
if sys.platform == "win32":
    raise ImportError(
        "archex supports Linux and macOS only; on Windows, install and run it inside WSL."
    )

# `__version__` resolves lazily too: `importlib.metadata` alone costs ~20 ms,
# a third of the 60 ms a hook may spend deciding that a call is not a search.

__all__ = ["analyze", "query", "compare", "record_usage_event", "__version__"]

_LAZY_API_EXPORTS = frozenset({"analyze", "compare", "query", "record_usage_event"})


def __getattr__(name: str) -> Any:
    """Lazily resolve the `archex.api` re-exports.

    `archex.api` pulls in the full parse/index/retrieval pipeline (tree-sitter
    grammars, embedders, graph analysis). Importing it eagerly here would make
    even `import archex.index.store` pay that cost, which matters for
    latency-sensitive entry points like `archex.integrations.claude_code_annotate_hook`
    (the Claude Code PostToolUse hook, invoked as a subprocess under a ~500ms
    budget). Deferring the import keeps plain submodule imports cheap while
    `from archex import query` (and friends) keep working unchanged.
    """
    if name in _LAZY_API_EXPORTS:
        from archex import api

        return getattr(api, name)
    if name == "__version__":
        from importlib.metadata import version

        return version("archex")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
