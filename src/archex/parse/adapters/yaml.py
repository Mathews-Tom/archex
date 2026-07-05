"""YAML STRUCTURED-tier adapter."""

from __future__ import annotations

from typing import Any

from archex.models import ImportStatement
from archex.parse.adapters.structured import StructuredAdapter


class YamlAdapter(StructuredAdapter):
    """Extract YAML document outlines and anchor/alias cross-references.

    YAML's only native cross-reference mechanism is the anchor (`&name`) /
    alias (`*name`) pair, and it is intra-document by construction -- YAML
    has no native cross-file import syntax. An alias only counts as a
    correctly extracted reference when a matching anchor is actually
    defined somewhere in the same document; an alias with no matching
    anchor is dropped rather than reported as an unverifiable reference.
    Because every surfaced reference is confirmed intra-document,
    `resolve_import` always resolves back to the alias's own containing
    file.
    """

    _language_id = "yaml"

    def extract_references(
        self, tree: object, source: bytes, file_path: str
    ) -> list[ImportStatement]:
        root = _root_node(tree)
        named_nodes = _walk_named(root)
        anchor_names = {
            name
            for node in named_nodes
            if node.type == "anchor"
            for name in (_named_child_text(node, "anchor_name", source),)
            if name is not None
        }

        references: list[ImportStatement] = []
        for node in named_nodes:
            if node.type != "alias":
                continue
            alias_name = _named_child_text(node, "alias_name", source)
            if alias_name is None or alias_name not in anchor_names:
                continue
            references.append(
                ImportStatement(
                    module=alias_name,
                    file_path=file_path,
                    line=int(node.start_point[0]) + 1,
                    is_relative=False,
                )
            )
        return references

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        return file_map.get(imp.file_path)


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


def _named_child_text(node: Any, child_type: str, source: bytes) -> str | None:
    for child in node.children:
        if child.type == child_type:
            return _node_text(child, source)
    return None


def _node_text(node: Any, source: bytes) -> str:
    return source[int(node.start_byte) : int(node.end_byte)].decode("utf-8", errors="replace")
