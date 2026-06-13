"""Symbol extraction utilities: qualified name resolution and visibility inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, cast

from archex.exceptions import ParseError
from archex.models import ChunkRange, ParsedFile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from archex.models import DiscoveredFile
    from archex.parse.adapters.base import LanguageAdapter
    from archex.parse.engine import TreeSitterEngine


def _count_lines(source: bytes) -> int:
    return source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)


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
    extract_chunk_ranges = getattr(adapter, "extract_chunk_ranges", None)
    chunk_ranges: list[ChunkRange] = []
    if callable(extract_chunk_ranges):
        extractor = cast("Callable[[object, bytes, str], list[ChunkRange]]", extract_chunk_ranges)
        chunk_ranges = extractor(tree, source, relative_path)
    return ParsedFile(
        path=relative_path,
        language=language,
        symbols=symbols,
        chunk_ranges=chunk_ranges,
        lines=_count_lines(source),
    )


def _extract_symbols_sequential(
    files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: Mapping[str, LanguageAdapter],
) -> list[ParsedFile]:
    results: list[ParsedFile] = []
    for discovered_file in files:
        adapter = adapters.get(discovered_file.language)
        if adapter is None:
            continue
        results.append(
            _parse_with_adapter(
                absolute_path=str(discovered_file.absolute_path),
                relative_path=discovered_file.path,
                language=discovered_file.language,
                engine=engine,
                adapter=adapter,
            )
        )
    return results


def _parse_file_worker(absolute_path: str, relative_path: str, language: str) -> ParsedFile | None:
    """Worker function for parallel parsing — creates its own engine and adapter."""
    from archex.parse.adapters import ADAPTERS
    from archex.parse.engine import TreeSitterEngine as _Engine

    engine = _Engine()
    adapter_class = ADAPTERS.get(language)
    if adapter_class is None:
        return None

    return _parse_with_adapter(
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

    Files whose language has no registered adapter are skipped.
    When parallel=True and len(files) > 10, uses ProcessPoolExecutor for concurrency.
    """
    eligible = [f for f in files if adapters.get(f.language) is not None]

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
            return results
        except ParseError:
            raise
        except Exception as e:
            if strict:
                raise ParseError(f"Parallel parsing failed: {e}") from e
            logger.error("Parallel executor failed, falling back to sequential: %s", e)

    return _extract_symbols_sequential(eligible, engine, adapters)
