"""Markdown STRUCTURED-tier adapter."""

from __future__ import annotations

import posixpath
from typing import Any
from urllib.parse import urlsplit

from tree_sitter import Parser
from tree_sitter_language_pack import get_language

from archex.models import ImportStatement
from archex.parse.adapters.structured import StructuredAdapter

_SKIPPED_SCHEMES = {"data", "http", "https", "mailto", "tel"}

_inline_parser: Parser | None = None


def _inline_language_parser() -> Parser:
    """Lazily build the `markdown_inline` sub-grammar parser.

    `tree-sitter-markdown` (the project's `markdown` grammar) deliberately
    leaves inline content -- links, images, emphasis -- as an opaque
    `inline` node; recovering its structure requires reparsing that node's
    byte span with the companion `markdown_inline` grammar, both bundled in
    the already-required `tree-sitter-language-pack`. This mirrors the
    upstream project's own two-grammar design rather than inventing a link
    syntax of its own.
    """
    global _inline_parser
    if _inline_parser is None:
        _inline_parser = Parser(get_language("markdown_inline"))
    return _inline_parser


class MarkdownAdapter(StructuredAdapter):
    """Extract Markdown section outlines and native link/section-anchor references.

    Markdown's only native cross-reference mechanisms are its link forms
    (inline `[text](target)`/`![alt](target)`, reference-style
    `[text][label]` resolved against a `[label]: target` definition) and
    section-anchor fragments (`#heading`) inside those same link targets.
    A fragment-only target is intra-document (Markdown has no separate
    heading-anchor symbol to claim), so it resolves back to its own
    containing file; everything else resolves like a normal local file
    reference.
    """

    _language_id = "markdown"

    def extract_references(
        self, tree: object, source: bytes, file_path: str
    ) -> list[ImportStatement]:
        root = _root_node(tree)
        named_nodes = _walk_named(root)
        definitions: dict[str, str] = {}
        references: list[ImportStatement] = []

        for node in named_nodes:
            if node.type != "link_reference_definition":
                continue
            label_text = _child_text(node, "link_label", source)
            destination_text = _child_text(node, "link_destination", source)
            if label_text is None or destination_text is None:
                continue
            destination = _unwrap_destination(destination_text)
            definitions[_normalize_label(_unwrap_label(label_text))] = destination
            reference = _build_reference(destination, file_path, int(node.start_point[0]) + 1)
            if reference is not None:
                references.append(reference)

        for node in named_nodes:
            if node.type != "inline":
                continue
            references.extend(_extract_inline_references(node, source, file_path, definitions))

        references.sort(key=lambda imp: imp.line)
        return references

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        if imp.module.startswith("#"):
            return file_map.get(imp.file_path)

        module = _normalize_path(imp.module)
        if module.startswith("/"):
            return _resolve_markdown_reference(module.lstrip("/"), file_map)

        base_dir = posixpath.dirname(imp.file_path)
        candidates: list[str] = []
        if base_dir:
            candidates.append(_normalize_path(posixpath.join(base_dir, module)))
        candidates.append(module)

        for candidate in candidates:
            resolved = _resolve_markdown_reference(candidate, file_map)
            if resolved is not None:
                return resolved
        return None


def _extract_inline_references(
    inline_node: Any, source: bytes, file_path: str, definitions: dict[str, str]
) -> list[ImportStatement]:
    segment = source[int(inline_node.start_byte) : int(inline_node.end_byte)]
    inline_tree = _inline_language_parser().parse(segment)
    base_row = int(inline_node.start_point[0])

    references: list[ImportStatement] = []
    for node in _walk_named(inline_tree.root_node):
        destination: str | None
        if node.type in {"inline_link", "image"}:
            raw = _child_text(node, "link_destination", segment)
            destination = _unwrap_destination(raw) if raw is not None else None
        elif node.type in {"full_reference_link", "shortcut_link"}:
            label_raw = _child_text(node, "link_label", segment)
            text_raw = _child_text(node, "link_text", segment)
            label_source = label_raw if label_raw is not None else text_raw
            destination = (
                definitions.get(_normalize_label(_unwrap_label(label_source)))
                if label_source is not None
                else None
            )
        else:
            continue
        if destination is None:
            continue
        line = base_row + int(node.start_point[0]) + 1
        reference = _build_reference(destination, file_path, line)
        if reference is not None:
            references.append(reference)
    return references


def _build_reference(destination: str, file_path: str, line: int) -> ImportStatement | None:
    target = _local_reference_target(destination)
    if target is None:
        return None
    is_relative = not target.startswith("#") and not target.startswith("/")
    return ImportStatement(module=target, file_path=file_path, line=line, is_relative=is_relative)


def _local_reference_target(destination: str) -> str | None:
    stripped = destination.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        return stripped
    if stripped.startswith("//"):
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


def _unwrap_destination(text: str) -> str:
    if len(text) >= 2 and text[0] == "<" and text[-1] == ">":
        return text[1:-1]
    return text


def _unwrap_label(text: str) -> str:
    if len(text) >= 2 and text[0] == "[" and text[-1] == "]":
        return text[1:-1]
    return text


def _normalize_label(text: str) -> str:
    return " ".join(text.split()).casefold()


def _normalize_path(path: str) -> str:
    normalized = posixpath.normpath(path)
    if normalized == ".":
        return ""
    if path.startswith("/") and not normalized.startswith("/"):
        return f"/{normalized}"
    return normalized


def _resolve_markdown_reference(candidate: str, file_map: dict[str, str]) -> str | None:
    normalized = _normalize_path(candidate)
    if normalized in file_map:
        return file_map[normalized]
    module_key, _ = posixpath.splitext(normalized)
    dotted_key = module_key.replace("/", ".")
    if dotted_key in file_map:
        return file_map[dotted_key]
    return None
