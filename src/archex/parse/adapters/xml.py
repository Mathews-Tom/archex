"""XML STRUCTURED-tier adapter (generic, dialect-agnostic)."""

from __future__ import annotations

from archex.parse.adapters.structured import StructuredAdapter


class XmlAdapter(StructuredAdapter):
    """Outline-only adapter for well-formed XML with no dialect assumptions.

    Generic XML has no universally native cross-file reference syntax --
    unlike HTML's `src`/`href` or CSS's `@import`/`url()`, an attribute that
    looks like a reference (e.g. `ref="other.xml"`) is dialect-specific
    convention, not XML grammar. Claiming it as a cross-reference here would
    invent semantics the format itself does not define. Dialect-specific
    mechanisms (Maven POM `<dependency>`, Android manifest components,
    XInclude, ...) are a separate concern layered on top of this adapter,
    not a generic-XML feature; see the XML dialect-plugin milestone.

    This adapter therefore inherits `StructuredAdapter.extract_references`'s
    empty-list default and only contributes the element outline.
    """

    _language_id = "xml"
