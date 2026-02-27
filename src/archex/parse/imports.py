"""Import resolution: parse import statements and resolve to absolute module paths."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from archex.models import DiscoveredFile, ImportStatement
    from archex.parse.adapters.base import LanguageAdapter
    from archex.parse.engine import TreeSitterEngine


def parse_imports(
    files: list[DiscoveredFile],
    engine: TreeSitterEngine,
    adapters: Mapping[str, LanguageAdapter],
) -> dict[str, list[ImportStatement]]:
    """Parse imports from all files. Returns mapping of file_path → list[ImportStatement]."""
    result: dict[str, list[ImportStatement]] = {}

    for f in files:
        adapter = adapters.get(f.language)
        if adapter is None:
            continue

        tree = engine.parse_file(f.absolute_path, f.language)

        with open(f.absolute_path, "rb") as fh:
            source = fh.read()

        imports = adapter.parse_imports(tree, source, f.path)
        result[f.path] = imports

    return result


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
