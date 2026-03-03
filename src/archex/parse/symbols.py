"""Symbol extraction utilities: qualified name resolution and visibility inference."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from archex.exceptions import ParseError
from archex.models import ParsedFile

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from archex.models import DiscoveredFile
    from archex.parse.adapters.base import LanguageAdapter
    from archex.parse.engine import TreeSitterEngine


def _parse_file_worker(absolute_path: str, relative_path: str, language: str) -> ParsedFile | None:
    """Worker function for parallel parsing — creates its own engine and adapter."""
    from archex.parse.adapters import ADAPTERS
    from archex.parse.engine import TreeSitterEngine as _Engine

    engine = _Engine()
    adapter_class = ADAPTERS.get(language)
    if adapter_class is None:
        return None

    adapter = adapter_class()
    tree = engine.parse_file(absolute_path, language)

    with open(absolute_path, "rb") as fh:
        source = fh.read()

    symbols = adapter.extract_symbols(tree, source, relative_path)
    line_count = source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)

    return ParsedFile(
        path=relative_path,
        language=language,
        symbols=symbols,
        lines=line_count,
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

    results_seq: list[ParsedFile] = []

    for f in eligible:
        adapter = adapters.get(f.language)
        if adapter is None:
            continue

        tree = engine.parse_file(f.absolute_path, f.language)

        with open(f.absolute_path, "rb") as fh:
            source = fh.read()

        symbols = adapter.extract_symbols(tree, source, f.path)
        line_count = source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)

        results_seq.append(
            ParsedFile(
                path=f.path,
                language=f.language,
                symbols=symbols,
                lines=line_count,
            )
        )

    return results_seq
