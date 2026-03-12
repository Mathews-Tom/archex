"""Typed result containers for the unified artifact pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.models import (
        CodeChunk,
        DiscoveredFile,
        Edge,
        ImportStatement,
        ParsedFile,
    )


@dataclass
class ArtifactBundle:
    """Complete output of the parse → import-resolve → chunk pipeline."""

    files: list[DiscoveredFile]
    parsed_files: list[ParsedFile]
    resolved_imports: dict[str, list[ImportStatement]]
    chunks: list[CodeChunk]
    edges: list[Edge]
    sources: dict[str, bytes] = field(default_factory=lambda: {}, repr=False)
