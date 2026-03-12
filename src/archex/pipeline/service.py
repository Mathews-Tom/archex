"""Shared internal pipeline stages for acquire/parse/index orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from archex.acquire import discover_files
from archex.exceptions import ParseError
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
        CodeChunk,
        Config,
        DiscoveredFile,
        ImportStatement,
        IndexConfig,
        ParsedFile,
    )
    from archex.parse.adapters import LanguageAdapter
    from archex.pipeline.chunker import Chunker


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
