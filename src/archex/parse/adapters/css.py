"""CSS STRUCTURED-tier adapter."""

from __future__ import annotations

import posixpath
from typing import Any
from urllib.parse import urlsplit

from archex.models import ImportStatement
from archex.parse.adapters.structured import StructuredAdapter

_SKIPPED_SCHEMES = {"data", "http", "https", "mailto", "tel"}


class CssAdapter(StructuredAdapter):
    """Extract CSS rule outlines and native `@import`/`url()` references.

    CSS's native cross-file reference mechanisms are the `@import` at-rule
    (either a bare string or a `url(...)` argument) and the `url()`
    function wherever it appears in a declaration value (`background`,
    `background-image`, ...). Both forms resolve to the same grammar node
    -- a `call_expression` whose `function_name` is `url` -- so a single
    generic walk covers both `@import url(...)` and property-value
    `url(...)` references; a plain `@import "target.css";` with no `url()`
    wrapper is handled separately since it has no `call_expression` child.
    """

    _language_id = "css"

    def extract_references(
        self, tree: object, source: bytes, file_path: str
    ) -> list[ImportStatement]:
        references: list[ImportStatement] = []
        root = _root_node(tree)
        for node in _walk_named(root):
            if node.type == "call_expression":
                raw_target = _url_call_argument(node, source)
            elif node.type == "import_statement":
                raw_target = _direct_import_string(node, source)
            else:
                continue
            if raw_target is None:
                continue
            target = _local_reference_target(raw_target)
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
            return _resolve_css_reference(module.lstrip("/"), file_map)

        base_dir = posixpath.dirname(imp.file_path)
        candidates: list[str] = []
        if base_dir:
            candidates.append(_normalize_path(posixpath.join(base_dir, module)))
        candidates.append(module)

        for candidate in candidates:
            resolved = _resolve_css_reference(candidate, file_map)
            if resolved is not None:
                return resolved
        return None


def _url_call_argument(node: Any, source: bytes) -> str | None:
    function_name = _child_text(node, "function_name", source)
    if function_name is None or function_name.lower() != "url":
        return None
    arguments = next((child for child in node.children if child.type == "arguments"), None)
    if arguments is None:
        return None
    return _string_or_plain_value(arguments, source)


def _direct_import_string(node: Any, source: bytes) -> str | None:
    for child in node.children:
        if child.type == "string_value":
            return _string_content(child, source)
    return None


def _string_or_plain_value(arguments: Any, source: bytes) -> str | None:
    for child in arguments.children:
        if child.type == "string_value":
            return _string_content(child, source)
        if child.type == "plain_value":
            return _node_text(child, source)
    return None


def _string_content(node: Any, source: bytes) -> str:
    for child in node.children:
        if child.type == "string_content":
            return _node_text(child, source)
    return ""


def _local_reference_target(destination: str) -> str | None:
    stripped = destination.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("//"):
        return None

    parsed = urlsplit(stripped)
    if parsed.scheme.lower() in _SKIPPED_SCHEMES or parsed.netloc:
        return None
    if not parsed.path:
        return None
    return parsed.path


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


def _child_text(node: Any, child_type: str, source: bytes) -> str | None:
    for child in node.children:
        if child.type == child_type:
            return _node_text(child, source)
    return None


def _node_text(node: Any, source: bytes) -> str:
    return source[int(node.start_byte) : int(node.end_byte)].decode("utf-8", errors="replace")


def _normalize_path(path: str) -> str:
    normalized = posixpath.normpath(path)
    if normalized == ".":
        return ""
    if path.startswith("/") and not normalized.startswith("/"):
        return f"/{normalized}"
    return normalized


def _resolve_css_reference(candidate: str, file_map: dict[str, str]) -> str | None:
    normalized = _normalize_path(candidate)
    if normalized in file_map:
        return file_map[normalized]
    module_key, _ = posixpath.splitext(normalized)
    dotted_key = module_key.replace("/", ".")
    if dotted_key in file_map:
        return file_map[dotted_key]
    return None
