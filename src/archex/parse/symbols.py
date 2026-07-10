"""Symbol extraction utilities: qualified name resolution and visibility inference."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from archex.exceptions import ParseError
from archex.languages import UNKNOWN_LANGUAGE_ID, UNKNOWN_LINE_WINDOW
from archex.models import ChunkRange, ParsedFile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from archex.models import DiscoveredFile, ImportStatement
    from archex.parse.adapters.base import LanguageAdapter
    from archex.parse.engine import TreeSitterEngine


def _count_lines(source: bytes) -> int:
    return source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)


@dataclass(frozen=True)
class ParsedExtraction:
    parsed_files: list[ParsedFile]
    imports_by_path: dict[str, list[ImportStatement]]


def _extract_chunk_ranges(
    adapter: LanguageAdapter,
    tree: object,
    source: bytes,
    relative_path: str,
) -> list[ChunkRange]:
    extract_chunk_ranges = getattr(adapter, "extract_chunk_ranges", None)
    if not callable(extract_chunk_ranges):
        return []
    extractor = cast("Callable[[object, bytes, str], list[ChunkRange]]", extract_chunk_ranges)
    return extractor(tree, source, relative_path)


def _line_window_ranges(total_lines: int) -> list[ChunkRange]:
    return [
        ChunkRange(start_line=start, end_line=min(start + UNKNOWN_LINE_WINDOW - 1, total_lines))
        for start in range(1, total_lines + 1, UNKNOWN_LINE_WINDOW)
    ]


def _parse_unknown_file(absolute_path: str, relative_path: str) -> ParsedFile:
    source = Path(absolute_path).read_bytes()
    line_count = _count_lines(source)
    return ParsedFile(
        path=relative_path,
        language=UNKNOWN_LANGUAGE_ID,
        chunk_ranges=_line_window_ranges(line_count),
        lines=line_count,
    )


def _parse_with_adapter(
    *,
    absolute_path: str,
    relative_path: str,
    language: str,
    engine: TreeSitterEngine,
    adapter: LanguageAdapter,
) -> ParsedFile:
    tree = engine.parse_file(absolute_path, language)
    source = Path(absolute_path).read_bytes()
    symbols = adapter.extract_symbols(tree, source, relative_path)
    return ParsedFile(
        path=relative_path,
        language=language,
        symbols=symbols,
        chunk_ranges=_extract_chunk_ranges(adapter, tree, source, relative_path),
        lines=_count_lines(source),
    )


def _parse_with_adapter_and_imports(
    *,
    absolute_path: str,
    relative_path: str,
    language: str,
    engine: TreeSitterEngine,
    adapter: LanguageAdapter,
) -> tuple[ParsedFile, list[ImportStatement]]:
    tree = engine.parse_file(absolute_path, language)
    source = Path(absolute_path).read_bytes()
    symbols = adapter.extract_symbols(tree, source, relative_path)
    parsed = ParsedFile(
        path=relative_path,
        language=language,
        symbols=symbols,
        chunk_ranges=_extract_chunk_ranges(adapter, tree, source, relative_path),
        lines=_count_lines(source),
    )
    return parsed, adapter.parse_imports(tree, source, relative_path)


def _extract_symbols_sequential(
    files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: Mapping[str, LanguageAdapter],
    strict: bool = False,
) -> list[ParsedFile]:
    results: list[ParsedFile] = []
    errors: list[tuple[str, ParseError]] = []
    for discovered_file in files:
        adapter = adapters.get(discovered_file.language)
        if adapter is None:
            if discovered_file.language != UNKNOWN_LANGUAGE_ID:
                continue
            results.append(_parse_unknown_file(discovered_file.absolute_path, discovered_file.path))
            continue
        try:
            results.append(
                _parse_with_adapter(
                    absolute_path=str(discovered_file.absolute_path),
                    relative_path=discovered_file.path,
                    language=discovered_file.language,
                    engine=engine,
                    adapter=adapter,
                )
            )
        except ParseError as e:
            errors.append((discovered_file.path, e))
            logger.warning("Skipping %s due to parse error: %s", discovered_file.path, e)
    if errors and strict:
        paths = ", ".join(p for p, _ in errors)
        raise ParseError(f"Sequential parsing failed for {len(errors)} file(s): {paths}")
    return results


def _parse_file_worker(absolute_path: str, relative_path: str, language: str) -> ParsedFile | None:
    """Worker function for parallel parsing — creates its own engine and adapter."""
    from archex.parse.adapters import ADAPTERS
    from archex.parse.engine import TreeSitterEngine as _Engine

    adapter_class = ADAPTERS.get(language)
    if adapter_class is None:
        if language != UNKNOWN_LANGUAGE_ID:
            return None
        return _parse_unknown_file(absolute_path, relative_path)

    engine = _Engine()
    return _parse_with_adapter(
        absolute_path=absolute_path,
        relative_path=relative_path,
        language=language,
        engine=engine,
        adapter=adapter_class(),
    )


def _extract_symbols_and_imports_sequential(
    files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: Mapping[str, LanguageAdapter],
    strict: bool = False,
) -> ParsedExtraction:
    parsed_files: list[ParsedFile] = []
    imports_by_path: dict[str, list[ImportStatement]] = {}
    errors: list[tuple[str, ParseError]] = []
    for discovered_file in files:
        adapter = adapters.get(discovered_file.language)
        if adapter is None:
            if discovered_file.language != UNKNOWN_LANGUAGE_ID:
                continue
            parsed = _parse_unknown_file(discovered_file.absolute_path, discovered_file.path)
            parsed_files.append(parsed)
            imports_by_path[discovered_file.path] = []
            continue
        try:
            parsed, imports = _parse_with_adapter_and_imports(
                absolute_path=str(discovered_file.absolute_path),
                relative_path=discovered_file.path,
                language=discovered_file.language,
                engine=engine,
                adapter=adapter,
            )
        except ParseError as e:
            errors.append((discovered_file.path, e))
            logger.warning("Skipping %s due to parse error: %s", discovered_file.path, e)
            continue
        parsed_files.append(parsed)
        imports_by_path[discovered_file.path] = imports
    if errors and strict:
        paths = ", ".join(p for p, _ in errors)
        raise ParseError(f"Sequential parsing failed for {len(errors)} file(s): {paths}")
    if len(errors) == len(files) and files:
        paths = ", ".join(p for p, _ in errors[:5])
        raise ParseError(
            f"Total parser failure: all {len(errors)} files failed to parse. First 5: {paths}"
        )
    return ParsedExtraction(parsed_files=parsed_files, imports_by_path=imports_by_path)


def _parse_artifacts_worker(
    absolute_path: str, relative_path: str, language: str
) -> tuple[ParsedFile, list[ImportStatement]] | None:
    """Worker function for parallel parsing of symbols and imports."""
    from archex.parse.adapters import ADAPTERS
    from archex.parse.engine import TreeSitterEngine as _Engine

    adapter_class = ADAPTERS.get(language)
    if adapter_class is None:
        if language != UNKNOWN_LANGUAGE_ID:
            return None
        return _parse_unknown_file(absolute_path, relative_path), []

    engine = _Engine()
    return _parse_with_adapter_and_imports(
        absolute_path=absolute_path,
        relative_path=relative_path,
        language=language,
        engine=engine,
        adapter=adapter_class(),
    )


def extract_symbols(
    files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: Mapping[str, LanguageAdapter],
    parallel: bool = False,
    strict: bool = False,
) -> list[ParsedFile]:
    """Extract symbols from all discovered files using the appropriate language adapter.

    Files whose language has no registered adapter are skipped, except unknown
    text files, which receive deterministic line-window chunk ranges.
    When parallel=True and len(files) > 10, uses ProcessPoolExecutor for concurrency.
    """
    eligible = [
        f
        for f in files
        if adapters.get(f.language) is not None or f.language == UNKNOWN_LANGUAGE_ID
    ]

    if parallel and len(files) > 10:
        try:
            results: list[ParsedFile] = []
            errors: list[tuple[str, Exception]] = []
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _parse_file_worker,
                        str(f.absolute_path),
                        f.path,
                        f.language,
                    )
                    for f in eligible
                ]
                for fut, f in zip(futures, eligible, strict=True):
                    try:
                        result = fut.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        errors.append((f.path, e))
            if errors and strict:
                paths = ", ".join(p for p, _ in errors)
                raise ParseError(f"Parallel parsing failed for {len(errors)} file(s): {paths}")
            if len(errors) == len(eligible) and eligible:
                paths = ", ".join(p for p, _ in errors[:5])
                raise ParseError(
                    f"Total worker failure: all {len(errors)} files failed to parse. "
                    f"First 5: {paths}"
                )
            return results
        except ParseError:
            raise
        except Exception as e:
            if strict:
                raise ParseError(f"Parallel parsing failed: {e}") from e
            logger.error("Parallel executor failed, falling back to sequential: %s", e)
    return _extract_symbols_sequential(eligible, engine, adapters, strict=strict)


def extract_symbols_and_imports(
    files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: Mapping[str, LanguageAdapter],
    parallel: bool = False,
    strict: bool = False,
) -> ParsedExtraction:
    """Extract parsed files and imports with one tree-sitter parse per file."""
    eligible = [
        f
        for f in files
        if adapters.get(f.language) is not None or f.language == UNKNOWN_LANGUAGE_ID
    ]

    if parallel and len(files) > 10:
        try:
            parsed_files: list[ParsedFile] = []
            imports_by_path: dict[str, list[ImportStatement]] = {}
            errors: list[tuple[str, Exception]] = []
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _parse_artifacts_worker,
                        str(f.absolute_path),
                        f.path,
                        f.language,
                    )
                    for f in eligible
                ]
                for fut, f in zip(futures, eligible, strict=True):
                    try:
                        result = fut.result()
                        if result is not None:
                            parsed, imports = result
                            parsed_files.append(parsed)
                            imports_by_path[f.path] = imports
                    except Exception as e:
                        errors.append((f.path, e))
            if errors and strict:
                paths = ", ".join(p for p, _ in errors)
                raise ParseError(f"Parallel parsing failed for {len(errors)} file(s): {paths}")
            if len(errors) == len(eligible) and eligible:
                paths = ", ".join(p for p, _ in errors[:5])
                raise ParseError(
                    f"Total worker failure: all {len(errors)} files failed to parse. "
                    f"First 5: {paths}"
                )
            return ParsedExtraction(parsed_files=parsed_files, imports_by_path=imports_by_path)
        except ParseError:
            raise
        except Exception as e:
            if strict:
                raise ParseError(f"Parallel parsing failed: {e}") from e
            logger.error("Parallel executor failed, falling back to sequential: %s", e)

    return _extract_symbols_and_imports_sequential(eligible, engine, adapters, strict=strict)
