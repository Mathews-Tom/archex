"""Symbol extraction utilities: qualified name resolution and visibility inference."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.models import ParsedFile

if TYPE_CHECKING:
    from collections.abc import Mapping

    from archex.models import DiscoveredFile
    from archex.parse.adapters.base import LanguageAdapter
    from archex.parse.engine import TreeSitterEngine


def extract_symbols(
    files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: Mapping[str, LanguageAdapter],
) -> list[ParsedFile]:
    """Extract symbols from all discovered files using the appropriate language adapter.

    Files whose language has no registered adapter are skipped.
    """
    results: list[ParsedFile] = []

    for f in files:
        adapter = adapters.get(f.language)
        if adapter is None:
            continue

        tree = engine.parse_file(f.absolute_path, f.language)

        with open(f.absolute_path, "rb") as fh:
            source = fh.read()

        symbols = adapter.extract_symbols(tree, source, f.path)
        line_count = source.count(b"\n") + (1 if source and not source.endswith(b"\n") else 0)

        results.append(
            ParsedFile(
                path=f.path,
                language=f.language,
                symbols=symbols,
                lines=line_count,
            )
        )

    return results
