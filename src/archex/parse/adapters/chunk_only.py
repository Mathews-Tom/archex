"""Chunk-only adapter for grammars without symbol/import extraction."""

from __future__ import annotations

from typing import Any

from archex.languages import get_language_support
from archex.models import ChunkRange, DiscoveredFile, ImportStatement, Symbol, Visibility


class ChunkOnlyAdapter:
    """Parse a language for AST chunk ranges without claiming symbols or imports."""

    _language_id: str = ""

    @property
    def language_id(self) -> str:
        return self._language_id

    @property
    def file_extensions(self) -> list[str]:
        support = get_language_support(self._language_id)
        return list(support.extensions) if support is not None else []

    @property
    def tree_sitter_name(self) -> str:
        support = get_language_support(self._language_id)
        return support.pack_name if support is not None else self._language_id

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        return []

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        return []

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        return None

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        return []

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        return Visibility.PUBLIC

    def extract_chunk_ranges(self, tree: object, source: bytes, file_path: str) -> list[ChunkRange]:
        support = get_language_support(self._language_id)
        if support is None or not support.chunk_node_types:
            return []

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


def make_chunk_only_adapter(language_id: str) -> type[ChunkOnlyAdapter]:
    class _Adapter(ChunkOnlyAdapter):
        _language_id = language_id

    _Adapter.__name__ = f"{language_id.title().replace('_', '')}ChunkOnlyAdapter"
    return _Adapter


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
