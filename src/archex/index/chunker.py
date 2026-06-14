"""Backward-compatibility re-exports — canonical module is archex.pipeline.chunker."""

# pyright: reportPrivateUsage=false
from archex.pipeline.chunker import (  # noqa: F401
    ASTChunker,
    CastChunker,
    Chunker,
    _format_import,
    _import_relevant,
    _merge_small_chunks,
    build_breadcrumbs,
    create_chunker,
    expand_identifiers,
)

__all__ = [
    "ASTChunker",
    "CastChunker",
    "Chunker",
    "_format_import",
    "_import_relevant",
    "_merge_small_chunks",
    "build_breadcrumbs",
    "create_chunker",
    "expand_identifiers",
]
