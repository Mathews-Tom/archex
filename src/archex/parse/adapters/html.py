"""HTML STRUCTURED-tier adapter."""

from __future__ import annotations

from archex.parse.adapters.structured import StructuredAdapter


class HtmlAdapter(StructuredAdapter):
    """Extract HTML element outlines without claiming programming symbols."""

    _language_id = "html"
