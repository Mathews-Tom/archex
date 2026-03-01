"""Shared tree-sitter node accessor helpers.

All tree-sitter node access is confined to these helpers so adapter modules
stay fully typed despite the absence of tree-sitter type stubs.
"""

from __future__ import annotations

from typing import Any


def ts_text(node: object, source: bytes) -> str:
    """Extract UTF-8 text from a tree-sitter node."""
    n: Any = node
    return source[n.start_byte : n.end_byte].decode("utf-8", errors="replace")


def ts_type(node: object) -> str:
    n: Any = node
    return str(n.type)


def ts_children(node: object) -> list[object]:
    n: Any = node
    return list(n.children)


def ts_named_children(node: object) -> list[object]:
    n: Any = node
    return list(n.named_children)


def ts_field(node: object, field: str) -> object | None:
    """Return the child node for a named field."""
    n: Any = node
    result: object | None = n.child_by_field_name(field)
    return result


def ts_start_line(node: object) -> int:
    n: Any = node
    return int(n.start_point[0]) + 1


def ts_end_line(node: object) -> int:
    n: Any = node
    return int(n.end_point[0]) + 1
