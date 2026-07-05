"""HTML STRUCTURED-tier adapter."""

from __future__ import annotations

import posixpath
from typing import Any
from urllib.parse import urlsplit

from archex.models import ImportStatement
from archex.parse.adapters.structured import StructuredAdapter

_REFERENCE_ATTRIBUTES = {
    "a": "href",
    "img": "src",
    "link": "href",
    "script": "src",
}
_SKIPPED_SCHEMES = {"data", "http", "https", "javascript", "mailto", "tel"}


class HtmlAdapter(StructuredAdapter):
    """Extract HTML element outlines and local file references."""

    _language_id = "html"

    def extract_references(
        self, tree: object, source: bytes, file_path: str
    ) -> list[ImportStatement]:
        references: list[ImportStatement] = []
        root = _root_node(tree)
        for node in _walk_named(root):
            if node.type not in {"start_tag", "self_closing_tag"}:
                continue
            tag_name = _tag_name(node, source)
            if tag_name is None:
                continue
            attribute_name = _REFERENCE_ATTRIBUTES.get(tag_name)
            if attribute_name is None:
                continue
            raw_target = _attribute_value(node, attribute_name, source)
            if raw_target is None:
                continue
            target = _local_reference_path(raw_target)
            if target is None:
                continue
            references.append(
                ImportStatement(
                    module=target,
                    file_path=file_path,
                    line=int(node.start_point[0]) + 1,
                    is_relative=not target.startswith("/"),
                )
            )
        return references

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        module = _normalize_path(imp.module)
        if module.startswith("/"):
            return _resolve_html_reference(module.lstrip("/"), file_map)

        base_dir = posixpath.dirname(imp.file_path)
        candidates: list[str] = []
        if base_dir:
            candidates.append(_normalize_path(posixpath.join(base_dir, module)))
        candidates.append(module)

        for candidate in candidates:
            resolved = _resolve_html_reference(candidate, file_map)
            if resolved is not None:
                return resolved
        return None


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


def _tag_name(node: Any, source: bytes) -> str | None:
    for child in node.children:
        if child.type == "tag_name":
            return _node_text(child, source).lower()
    return None


def _attribute_value(node: Any, attribute_name: str, source: bytes) -> str | None:
    for child in node.children:
        if child.type != "attribute":
            continue
        name: str | None = None
        value: str | None = None
        for attr_child in child.children:
            if attr_child.type == "attribute_name":
                name = _node_text(attr_child, source).lower()
            elif attr_child.type in {"quoted_attribute_value", "attribute_value"}:
                value = _unquote_attribute_value(_node_text(attr_child, source))
        if name == attribute_name:
            return value
    return None


def _node_text(node: Any, source: bytes) -> str:
    return source[int(node.start_byte) : int(node.end_byte)].decode("utf-8", errors="replace")


def _unquote_attribute_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _local_reference_path(reference: str) -> str | None:
    stripped = reference.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("//"):
        return None

    parsed = urlsplit(stripped)
    if parsed.scheme.lower() in _SKIPPED_SCHEMES or parsed.netloc:
        return None
    if not parsed.path:
        return None
    return parsed.path


def _normalize_path(path: str) -> str:
    normalized = posixpath.normpath(path)
    if normalized == ".":
        return ""
    if path.startswith("/") and not normalized.startswith("/"):
        return f"/{normalized}"
    return normalized


def _resolve_html_reference(candidate: str, file_map: dict[str, str]) -> str | None:
    normalized = _normalize_path(candidate)
    if normalized in file_map:
        return file_map[normalized]
    module_key, _ = posixpath.splitext(normalized)
    dotted_key = module_key.replace("/", ".")
    if dotted_key in file_map:
        return file_map[dotted_key]
    return None
