"""archex — architecture extraction and analysis toolkit."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from archex.api import analyze, compare, query, record_usage_event

__version__ = version("archex")

__all__ = ["analyze", "query", "compare", "record_usage_event", "__version__"]

_LAZY_API_EXPORTS = frozenset({"analyze", "compare", "query", "record_usage_event"})


def __getattr__(name: str) -> Any:
    """Lazily resolve the `archex.api` re-exports.

    `archex.api` pulls in the full parse/index/retrieval pipeline (tree-sitter
    grammars, embedders, graph analysis). Importing it eagerly here would make
    even `import archex.index.store` pay that cost, which matters for
    latency-sensitive entry points like `archex.integrations.hook` (the M19
    Claude Code PreToolUse hook, invoked as a subprocess under a ~500ms
    budget). Deferring the import keeps plain submodule imports cheap while
    `from archex import query` (and friends) keep working unchanged.
    """
    if name in _LAZY_API_EXPORTS:
        from archex import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
