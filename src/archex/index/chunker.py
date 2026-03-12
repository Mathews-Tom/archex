"""Backward-compatibility re-exports — canonical module is archex.pipeline.chunker."""

from archex.pipeline.chunker import (  # noqa: F401
    ASTChunker,
    Chunker,
    _format_import,
    _import_relevant,
    _merge_small_chunks,
    expand_identifiers,
)

__all__ = [
    "ASTChunker",
    "Chunker",
    "_format_import",
    "_import_relevant",
    "_merge_small_chunks",
    "expand_identifiers",
]
