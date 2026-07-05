"""XML STRUCTURED-tier adapter (generic, dialect-agnostic)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.parse.adapters.structured import StructuredAdapter
from archex.parse.adapters.xml_maven import (
    extract_maven_dependencies,
    is_maven_pom,
    resolve_maven_dependency,
)

if TYPE_CHECKING:
    from archex.models import ImportStatement


class XmlAdapter(StructuredAdapter):
    """Outline-only adapter for well-formed XML with no dialect assumptions,
    plus the one dialect wired in today: Maven's `pom.xml`.

    Generic XML has no universally native cross-file reference syntax --
    unlike HTML's `src`/`href` or CSS's `@import`/`url()`, an attribute that
    looks like a reference (e.g. `ref="other.xml"`) is dialect-specific
    convention, not XML grammar. Claiming it as a cross-reference here would
    invent semantics the format itself does not define. `extract_references`
    therefore stays empty for anything that is not a recognized dialect.

    `pom.xml` is the one dialect implemented here (`xml_maven.py`): Maven's
    `<dependency>` declarations are extracted and resolved to sibling repo
    modules where determinable. Other XML dialects (Android manifest,
    Spring beans, XInclude, ...) remain unimplemented -- see the XML
    dialect-plugin milestone.
    """

    _language_id = "xml"

    def extract_references(
        self, tree: object, source: bytes, file_path: str
    ) -> list[ImportStatement]:
        if not is_maven_pom(file_path, tree, source):
            return []
        return extract_maven_dependencies(tree, source, file_path)

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        return resolve_maven_dependency(imp, file_map)
