"""Shared internal pipeline stages for acquire/parse/index orchestration."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from archex.acquire import discover_files
from archex.exceptions import ParseError
from archex.models import EdgeKind
from archex.parse import (
    TreeSitterEngine,
    build_file_map,
    extract_symbols,
    parse_imports,
    resolve_imports,
)
from archex.pipeline.chunker import ASTChunker

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from archex.models import (
        ChunkSurrogate,
        CodeChunk,
        Config,
        DiscoveredFile,
        Edge,
        ImportStatement,
        IndexConfig,
        ParsedFile,
    )
    from archex.parse.adapters import LanguageAdapter
    from archex.pipeline.chunker import Chunker
    from archex.pipeline.models import ArtifactBundle


_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass
class ParseArtifacts:
    files: list[DiscoveredFile]
    parsed_files: list[ParsedFile]
    resolved_imports: dict[str, list[ImportStatement]]


def parse_repository(
    repo_path: Path,
    config: Config,
    adapters: dict[str, LanguageAdapter],
) -> ParseArtifacts:
    """Run discover + parse + import resolution for a repo path."""
    files = discover_files(
        repo_path,
        languages=config.languages,
        max_file_size=config.max_file_size,
    )
    engine = TreeSitterEngine()
    parsed_files = extract_symbols(
        files, engine, adapters, parallel=config.parallel, strict=config.strict
    )
    import_map = parse_imports(
        files,
        engine,
        adapters,
        parallel=config.parallel,
        strict=config.strict,
    )
    file_map = build_file_map(files)
    file_languages = {f.path: f.language for f in files}
    resolved_map = resolve_imports(import_map, file_map, adapters, file_languages)
    return ParseArtifacts(files=files, parsed_files=parsed_files, resolved_imports=resolved_map)


def build_chunks(
    files: list[DiscoveredFile],
    parsed_files: list[ParsedFile],
    index_config: IndexConfig,
    chunker: Chunker | None = None,
    *,
    strict: bool = False,
) -> list[CodeChunk]:
    """Build code chunks from parsed files with source-bytes hydration."""
    file_chunker: Chunker = chunker if chunker is not None else ASTChunker(config=index_config)
    sources: dict[str, bytes] = {}
    for f in files:
        try:
            sources[f.path] = Path(f.absolute_path).read_bytes()
        except OSError as err:
            logger.warning("Failed to read file for chunking: %s", f.absolute_path)
            if strict:
                raise ParseError(f"Failed to read file: {f.absolute_path}") from err
            continue
    return file_chunker.chunk_files(parsed_files, sources)


def _surrogate_identifier_anchors(content: str, limit: int = 8) -> list[str]:
    """Extract a bounded identifier set for surrogate lexical grounding."""
    anchors: list[str] = []
    for match in _IDENTIFIER_RE.findall(content):
        if match not in anchors:
            anchors.append(match)
        if len(anchors) >= limit:
            break
    return anchors


def build_chunk_surrogates(
    chunks: list[CodeChunk],
    *,
    version: str = "v1",
) -> list[ChunkSurrogate]:
    """Build deterministic retrieval surrogates for semantic routing."""
    from archex.models import ChunkSurrogate

    surrogates: list[ChunkSurrogate] = []
    for chunk in chunks:
        fields = [
            f"path: {chunk.file_path}",
            f"language: {chunk.language}",
            f"lines: {chunk.start_line}-{chunk.end_line}",
        ]
        if chunk.symbol_kind is not None:
            fields.append(f"kind: {chunk.symbol_kind}")
        if chunk.symbol_name:
            fields.append(f"symbol: {chunk.symbol_name}")
        if chunk.qualified_name:
            fields.append(f"qualified: {chunk.qualified_name}")
        if chunk.signature:
            fields.append(f"signature: {chunk.signature}")
        if chunk.visibility:
            fields.append(f"visibility: {chunk.visibility}")
        if chunk.imports_context:
            fields.append(f"imports: {chunk.imports_context}")
        if chunk.docstring:
            fields.append(f"doc: {chunk.docstring.strip()}")
        anchors = _surrogate_identifier_anchors(chunk.content)
        if anchors:
            fields.append(f"anchors: {' '.join(anchors)}")
        surrogates.append(
            ChunkSurrogate(
                chunk_id=chunk.id,
                file_path=chunk.file_path,
                surrogate_text="\n".join(fields),
                surrogate_version=version,
            )
        )
    return surrogates


def _read_sources(
    files: list[DiscoveredFile],
    *,
    strict: bool = False,
) -> dict[str, bytes]:
    """Read source bytes for discovered files."""
    sources: dict[str, bytes] = {}
    for f in files:
        try:
            sources[f.path] = Path(f.absolute_path).read_bytes()
        except OSError as err:
            logger.warning("Failed to read file for chunking: %s", f.absolute_path)
            if strict:
                raise ParseError(f"Failed to read file: {f.absolute_path}") from err
            continue
    return sources


def _build_edges(
    resolved_imports: dict[str, list[ImportStatement]],
) -> list[Edge]:
    """Build dependency edges from resolved import map."""
    from archex.models import Edge

    return [
        Edge(
            source=file_path,
            target=imp.resolved_path,
            kind=EdgeKind.IMPORTS,
            location=f"{file_path}:{imp.line}",
        )
        for file_path, imps in resolved_imports.items()
        for imp in imps
        if imp.resolved_path is not None
    ]


def produce_artifacts(
    repo_path: Path,
    config: Config,
    adapters: dict[str, LanguageAdapter],
    index_config: IndexConfig | None = None,
    *,
    strict: bool = False,
) -> ArtifactBundle:
    """Run the full parse → import-resolve → chunk pipeline and return all artifacts.

    This is the unified entry point for artifact production. It composes
    parse_repository() and build_chunks() into a single call that also
    produces dependency edges and retains source bytes.

    Args:
        repo_path: Root of the repository to process.
        config: Discovery and parsing configuration.
        adapters: Language-specific TreeSitter adapters.
        index_config: Chunking parameters (defaults to IndexConfig()).
        strict: Raise on file-read errors instead of skipping.

    Returns:
        ArtifactBundle with files, parsed_files, resolved_imports, chunks,
        edges, and source bytes.
    """
    from archex.models import IndexConfig as IndexConfigModel
    from archex.pipeline.models import ArtifactBundle as Bundle

    effective_index_config = index_config or IndexConfigModel()

    # Stage 1: parse + import resolution
    artifacts = parse_repository(repo_path, config, adapters)

    # Stage 2: read source bytes
    sources = _read_sources(artifacts.files, strict=strict)

    # Stage 3: chunk
    chunker: Chunker = ASTChunker(config=effective_index_config)
    chunks = chunker.chunk_files(artifacts.parsed_files, sources)

    # Stage 4: build dependency edges
    edges = _build_edges(artifacts.resolved_imports)

    return Bundle(
        files=artifacts.files,
        parsed_files=artifacts.parsed_files,
        resolved_imports=artifacts.resolved_imports,
        chunks=chunks,
        edges=edges,
        sources=sources,
    )
