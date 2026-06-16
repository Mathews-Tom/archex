"""archex — architecture extraction and analysis toolkit."""

from __future__ import annotations

from importlib.metadata import version

__version__ = version("archex")

from archex.api import analyze, compare, query, record_usage_event

__all__ = ["analyze", "query", "compare", "record_usage_event", "__version__"]
