"""LSAP data models — pure Pydantic, zero external dependencies."""

from __future__ import annotations

from pydantic import BaseModel


class HoverInfo(BaseModel):
    """Type and documentation information from an LSP hover request."""

    type_signature: str = ""
    documentation: str = ""
    raw_content: str = ""


class ReferenceLocation(BaseModel):
    """A single reference location returned by LSP references request."""

    file_path: str
    line: int
    character: int = 0
    context_line: str = ""


class DefinitionLocation(BaseModel):
    """A symbol definition location returned by LSP definition request."""

    file_path: str
    line: int
    character: int = 0
    context_line: str = ""


class LSAPEnrichment(BaseModel):
    """Aggregated LSP enrichment data attached to a SymbolSource."""

    hover: HoverInfo | None = None
    references: list[ReferenceLocation] = []
    definition: DefinitionLocation | None = None
    reference_count: int = 0
