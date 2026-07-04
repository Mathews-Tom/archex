"""Shared STRUCTURED-tier adapter base.

STRUCTURED languages expose outline chunks and native file references without
claiming programming symbols. Concrete adapters override ``extract_references``
for language-native reference syntax and inherit the chunk-node-driven outline
logic used by chunk-only languages.
"""

from __future__ import annotations

import os
from typing import Any, final

from archex.languages import LanguageSupport, get_language_support
from archex.models import (
    ChunkRange,
    DiscoveredFile,
    ImportStatement,
    LanguageTier,
    Symbol,
    Visibility,
)


class StructuredAdapter:
    """Base adapter for outline-plus-reference languages with no code symbols."""

    _language_id: str = ""

    def __init__(self) -> None:
        self._support()

    def _support(self) -> LanguageSupport:
        support = get_language_support(self._language_id)
        if support is None:
            raise ValueError(f"STRUCTURED adapter language {self._language_id!r} is not registered")
        if support.tier is not LanguageTier.STRUCTURED:
            raise ValueError(
                f"STRUCTURED adapter language {self._language_id!r} is registered as {support.tier}"
            )
        return support

    @property
    def language_id(self) -> str:
        return self._language_id

    @property
    def file_extensions(self) -> list[str]:
        return list(self._support().extensions)

    @property
    def tree_sitter_name(self) -> str:
        return self._support().pack_name

    @final
    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        return []

    def extract_references(
        self, tree: object, source: bytes, file_path: str
    ) -> list[ImportStatement]:
        return []

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        return self.extract_references(tree, source, file_path)

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        if imp.is_relative or _is_relative_reference(imp.module):
            return _resolve_relative_reference(imp.module, imp.file_path, file_map)
        return _resolve_direct_reference(imp.module, file_map)

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        return []

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        return Visibility.PUBLIC

    def extract_chunk_ranges(self, tree: object, source: bytes, file_path: str) -> list[ChunkRange]:
        support = self._support()

        root = _root_node(tree)
        ranges: list[ChunkRange] = []
        for node in _walk_named(root):
            if node.type not in support.chunk_node_types:
                continue
            start_line = int(node.start_point[0]) + 1
            end_row = int(node.end_point[0])
            end_column = int(node.end_point[1])
            end_line = end_row + 1 if end_column > 0 else end_row
            if end_line < start_line:
                continue
            ranges.append(ChunkRange(start_line=start_line, end_line=end_line))
        return _deduplicate_ranges(ranges)


def make_structured_adapter(language_id: str) -> type[StructuredAdapter]:
    class _Adapter(StructuredAdapter):
        _language_id = language_id

    _Adapter.__name__ = f"{language_id.title().replace('_', '')}StructuredAdapter"
    return _Adapter


def _resolve_direct_reference(module: str, file_map: dict[str, str]) -> str | None:
    normalized = os.path.normpath(module).replace(os.sep, "/")
    values = {os.path.normpath(path).replace(os.sep, "/"): path for path in file_map.values()}
    if normalized in values:
        return values[normalized]
    if _is_path_reference(normalized) and normalized in file_map:
        return file_map[normalized]
    return None


def _is_relative_reference(reference: str) -> bool:
    return reference.startswith("./") or reference.startswith("../")


def _is_path_reference(reference: str) -> bool:
    return "/" in reference or bool(os.path.splitext(reference)[1])


def _resolve_relative_reference(
    module: str,
    file_path: str,
    file_map: dict[str, str],
) -> str | None:
    base_dir = os.path.dirname(file_path)
    candidate = os.path.normpath(os.path.join(base_dir, module)).replace(os.sep, "/")
    return _resolve_direct_reference(candidate, file_map)


def _root_node(tree: object) -> Any:
    parsed_tree: Any = tree
    return parsed_tree.root_node


def _walk_named(node: Any) -> list[Any]:
    children = [child for child in node.children if child.is_named]
    result: list[Any] = []
    for child in children:
        result.append(child)
        result.extend(_walk_named(child))
    return result


def _deduplicate_ranges(ranges: list[ChunkRange]) -> list[ChunkRange]:
    ranges.sort(key=lambda item: (item.start_line, item.end_line))
    result: list[ChunkRange] = []
    last_end = 0
    for item in ranges:
        if item.start_line <= last_end:
            continue
        result.append(item)
        last_end = item.end_line
    return result
