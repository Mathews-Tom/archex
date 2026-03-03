"""Import resolution: parse import statements and resolve to absolute module paths."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from archex.exceptions import ParseError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from archex.models import DiscoveredFile, ImportStatement
    from archex.parse.adapters.base import LanguageAdapter
    from archex.parse.engine import TreeSitterEngine


def _parse_imports_worker(
    absolute_path: str, relative_path: str, language: str
) -> tuple[str, list[ImportStatement]] | None:
    """Worker function for parallel import parsing — creates its own engine and adapter."""
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

    imports = adapter.parse_imports(tree, source, relative_path)
    return (relative_path, imports)


def parse_imports(
    files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: Mapping[str, LanguageAdapter],
    parallel: bool = False,
    strict: bool = False,
) -> dict[str, list[ImportStatement]]:
    """Parse imports from all files. Returns mapping of file_path → list[ImportStatement].

    When parallel=True and len(files) > 10, uses ProcessPoolExecutor for concurrency.
    """
    eligible = [f for f in files if adapters.get(f.language) is not None]

    if parallel and len(files) > 10:
        try:
            result: dict[str, list[ImportStatement]] = {}
            errors: list[tuple[str, Exception]] = []
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _parse_imports_worker,
                        str(f.absolute_path),
                        f.path,
                        f.language,
                    )
                    for f in eligible
                ]
                for fut, f in zip(futures, eligible, strict=True):
                    try:
                        entry = fut.result()
                        if entry is not None:
                            path, imports = entry
                            result[path] = imports
                    except Exception as e:
                        errors.append((f.path, e))
            if errors and strict:
                paths = ", ".join(p for p, _ in errors)
                raise ParseError(
                    f"Parallel import parsing failed for {len(errors)} file(s): {paths}"
                )
            return result
        except ParseError:
            raise
        except Exception as e:
            if strict:
                raise ParseError(f"Parallel import parsing failed: {e}") from e
            logger.error("Parallel executor failed, falling back to sequential: %s", e)

    result_seq: dict[str, list[ImportStatement]] = {}

    for f in files:
        adapter = adapters.get(f.language)
        if adapter is None:
            continue

        tree = engine.parse_file(f.absolute_path, f.language)

        with open(f.absolute_path, "rb") as fh:
            source = fh.read()

        imports = adapter.parse_imports(tree, source, f.path)
        result_seq[f.path] = imports

    return result_seq


def resolve_imports(
    import_map: dict[str, list[ImportStatement]],
    file_map: dict[str, str],
    adapters: Mapping[str, LanguageAdapter],
    file_languages: dict[str, str],
) -> dict[str, list[ImportStatement]]:
    """Resolve import paths in-place. Returns the same dict with resolved_path set."""
    for file_path, imports in import_map.items():
        language_id = file_languages.get(file_path)
        if language_id is None:
            continue
        adapter = adapters.get(language_id)
        if adapter is None:
            continue

        for imp in imports:
            resolved = adapter.resolve_import(imp, file_map)
            imp.resolved_path = resolved

    return import_map


def build_file_map(files: list[DiscoveredFile]) -> dict[str, str]:
    """Build module_path → file_path mapping for internal import resolution.

    Examples:
      "models.py"          → {"models": "models.py"}
      "services/auth.py"   → {"services.auth": "services/auth.py"}
      "services/__init__.py" → {"services": "services/__init__.py"}
    """
    file_map: dict[str, str] = {}

    for f in files:
        path = f.path
        # Normalize path separators
        normalized = path.replace(os.sep, "/")

        if normalized.endswith("/__init__.py"):
            # Package init: key is the package path as dotted module
            package_path = normalized[: -len("/__init__.py")]
            module_key = package_path.replace("/", ".")
        elif normalized.endswith(".py"):
            module_key = normalized[:-3].replace("/", ".")
        else:
            # Non-Python files: use path without extension as key
            base, _ = os.path.splitext(normalized)
            module_key = base.replace("/", ".")

        file_map[module_key] = path

    return file_map
