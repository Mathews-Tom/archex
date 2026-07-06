"""Renderers package: XML, JSON, Markdown, and (optional) TOON output formatters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.serve.renderers.json import render_json
from archex.serve.renderers.markdown import render_markdown
from archex.serve.renderers.xml import render_xml

if TYPE_CHECKING:
    from archex.serve.renderers.toon import render_toon as render_toon

__all__ = ["render_json", "render_markdown", "render_toon", "render_xml"]


def __getattr__(name: str) -> object:
    """Lazily import `render_toon` so the optional `toons` dependency is
    only required when this attribute is actually accessed, keeping the
    package import (and thus every other renderer) extras-independent.
    """
    if name == "render_toon":
        from archex.serve.renderers.toon import render_toon

        return render_toon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
