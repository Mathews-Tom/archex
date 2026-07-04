"""Ruby parse adapter: extract symbols and imports from .rb files using tree-sitter."""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Any

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.ts_node import (
    ts_end_line as _end_line,
)
from archex.parse.adapters.ts_node import (
    ts_field as _field,
)
from archex.parse.adapters.ts_node import (
    ts_named_children as _named_children,
)
from archex.parse.adapters.ts_node import (
    ts_start_line as _start_line,
)
from archex.parse.adapters.ts_node import (
    ts_text as _text,
)
from archex.parse.adapters.ts_node import (
    ts_type as _type,
)

# ---------------------------------------------------------------------------
# Constant / qualification helpers
# ---------------------------------------------------------------------------


def _constant_path(node: object, source: bytes) -> str:
    """Return a Ruby constant or scope-resolution path with '.' separators."""
    parts: list[str] = []

    def collect(current: object) -> None:
        current_type = _type(current)
        if current_type == "constant":
            parts.append(_text(current, source))
            return
        if current_type == "scope_resolution":
            for child in _named_children(current):
                if _type(child) in {"constant", "scope_resolution"}:
                    collect(child)

    collect(node)
    if parts:
        return ".".join(parts)
    return _text(node, source).replace("::", ".")


def _qualify(parent_name: str | None, name: str) -> str:
    if not parent_name:
        return name
    if name.startswith(f"{parent_name}."):
        return name
    return f"{parent_name}.{name}"


def _short_name(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Signature / visibility helpers
# ---------------------------------------------------------------------------


_VISIBILITY_MARKERS: dict[str, Visibility] = {
    "public": Visibility.PUBLIC,
    "protected": Visibility.INTERNAL,
    "private": Visibility.PRIVATE,
}


def _method_signature(node: object, source: bytes, name: str, singleton: bool) -> str:
    params = _field(node, "parameters")
    params_text = _text(params, source) if params is not None else "()"
    receiver = "self." if singleton else ""
    return f"def {receiver}{name}{params_text}"


def _call_method_name(node: object, source: bytes) -> str | None:
    method_node = _field(node, "method")
    if method_node is None:
        return None
    return _text(method_node, source)


def _visibility_marker(node: object, source: bytes) -> Visibility | None:
    if _type(node) == "identifier":
        return _VISIBILITY_MARKERS.get(_text(node, source))
    if _type(node) == "call":
        method_name = _call_method_name(node, source)
        if method_name is not None:
            return _VISIBILITY_MARKERS.get(method_name)
    return None


def _symbol_argument_names(node: object | None, source: bytes) -> list[str]:
    if node is None:
        return []
    names: list[str] = []
    node_type = _type(node)
    if node_type == "simple_symbol":
        names.append(_text(node, source).lstrip(":"))
    elif node_type == "string_content":
        names.append(_text(node, source))
    for child in _named_children(node):
        names.extend(_symbol_argument_names(child, source))
    return names


def _method_argument_nodes(node: object | None) -> list[object]:
    if node is None:
        return []
    methods: list[object] = []
    if _type(node) in {"method", "singleton_method"}:
        methods.append(node)
        return methods
    for child in _named_children(node):
        methods.extend(_method_argument_nodes(child))
    return methods


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------


def _extract_method(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str | None,
    visibility: Visibility,
    force_singleton: bool = False,
) -> Symbol | None:
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    singleton = force_singleton or _type(node) == "singleton_method"
    qualified_parent = parent_name
    if singleton:
        object_node = _field(node, "object")
        if object_node is not None and _type(object_node) != "self":
            qualified_parent = _constant_path(object_node, source)
    qualified_name = _qualify(qualified_parent, name)
    return Symbol(
        name=name,
        qualified_name=qualified_name,
        kind=SymbolKind.METHOD,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=visibility,
        signature=_method_signature(node, source, name, singleton),
        parent=qualified_parent,
    )


def _extract_constant(
    node: object, source: bytes, file_path: str, parent_name: str
) -> Symbol | None:
    left = _field(node, "left")
    if left is None or _type(left) != "constant":
        return None
    name = _text(left, source)
    return Symbol(
        name=name,
        qualified_name=_qualify(parent_name, name),
        kind=SymbolKind.CONSTANT,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=Visibility.PUBLIC,
        parent=parent_name,
    )


def _extract_module_or_class(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str | None,
) -> list[Symbol]:
    name_node = _field(node, "name")
    if name_node is None:
        return []

    declared_name = _constant_path(name_node, source)
    qualified_name = _qualify(parent_name, declared_name)
    kind = SymbolKind.MODULE if _type(node) == "module" else SymbolKind.CLASS
    symbols = [
        Symbol(
            name=_short_name(declared_name),
            qualified_name=qualified_name,
            kind=kind,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=Visibility.PUBLIC,
            parent=parent_name,
        )
    ]

    body = _field(node, "body")
    if body is not None:
        symbols.extend(_extract_body_symbols(body, source, file_path, qualified_name))
    return symbols


def _apply_retroactive_visibility(
    symbols: list[Symbol], parent_name: str | None, method_names: list[str], visibility: Visibility
) -> None:
    if not method_names:
        return
    names = set(method_names)
    for index, symbol in enumerate(symbols):
        if (
            symbol.kind == SymbolKind.METHOD
            and symbol.parent == parent_name
            and symbol.name in names
        ):
            symbols[index] = symbol.model_copy(update={"visibility": visibility})


def _extract_visibility_call_symbols(
    node: object,
    source: bytes,
    file_path: str,
    parent_name: str | None,
    visibility: Visibility,
) -> list[Symbol]:
    symbols: list[Symbol] = []
    arguments = _field(node, "arguments")
    for method_node in _method_argument_nodes(arguments):
        method = _extract_method(method_node, source, file_path, parent_name, visibility)
        if method is not None:
            symbols.append(method)
    return symbols


def _extract_body_symbols(
    body: object,
    source: bytes,
    file_path: str,
    parent_name: str | None,
    force_singleton_methods: bool = False,
) -> list[Symbol]:
    symbols: list[Symbol] = []
    method_visibility = Visibility.PUBLIC

    for child in _named_children(body):
        child_type = _type(child)
        marker = _visibility_marker(child, source)
        if marker is not None:
            if child_type == "call":
                call_symbols = _extract_visibility_call_symbols(
                    child, source, file_path, parent_name, marker
                )
                method_names = _symbol_argument_names(_field(child, "arguments"), source)
                if call_symbols:
                    symbols.extend(call_symbols)
                if method_names:
                    _apply_retroactive_visibility(symbols, parent_name, method_names, marker)
                    continue
                if call_symbols:
                    continue
                method_visibility = marker
                continue
            method_visibility = marker
            continue

        if child_type in {"module", "class"}:
            symbols.extend(_extract_module_or_class(child, source, file_path, parent_name))
        elif child_type in {"method", "singleton_method"}:
            method = _extract_method(
                child,
                source,
                file_path,
                parent_name,
                method_visibility,
                force_singleton=force_singleton_methods,
            )
            if method is not None:
                symbols.append(method)
        elif child_type == "singleton_class":
            singleton_body = _field(child, "body")
            singleton_parent = parent_name
            receiver = _field(child, "value")
            if receiver is not None and _type(receiver) != "self":
                singleton_parent = _constant_path(receiver, source)
            if singleton_body is not None:
                symbols.extend(
                    _extract_body_symbols(
                        singleton_body,
                        source,
                        file_path,
                        singleton_parent,
                        force_singleton_methods=True,
                    )
                )
        elif child_type == "assignment" and parent_name is not None:
            constant = _extract_constant(child, source, file_path, parent_name)
            if constant is not None:
                symbols.append(constant)

    return symbols


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------


_REQUIRE_METHODS: frozenset[str] = frozenset({"require", "require_relative"})


def _first_string_content(node: object, source: bytes) -> str | None:
    if _type(node) == "string_content":
        return _text(node, source)
    for child in _named_children(node):
        value = _first_string_content(child, source)
        if value is not None:
            return value
    return None


def _parse_require_call(node: object, source: bytes, file_path: str) -> ImportStatement | None:
    method_node = _field(node, "method")
    arguments_node = _field(node, "arguments")
    if method_node is None or arguments_node is None:
        return None

    method_name = _text(method_node, source)
    if method_name not in _REQUIRE_METHODS:
        return None

    module = _first_string_content(arguments_node, source)
    if module is None or not module:
        return None

    return ImportStatement(
        module=module,
        symbols=[],
        file_path=file_path,
        line=_start_line(node),
        is_relative=method_name == "require_relative",
    )


def _walk_require_calls(root: object, source: bytes, file_path: str) -> list[ImportStatement]:
    imports: list[ImportStatement] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if _type(node) == "call":
            imp = _parse_require_call(node, source, file_path)
            if imp is not None:
                imports.append(imp)
        stack.extend(reversed(_named_children(node)))
    return imports


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------


def _with_rb_suffix(module: str) -> str:
    return module if module.endswith(".rb") else f"{module}.rb"


def _normalize_path(path: str) -> str:
    return os.path.normpath(path).replace(os.sep, "/")


def _file_map_path(file_map_key: str, file_map_value: str) -> tuple[str, str]:
    return file_map_key.replace(os.sep, "/"), _normalize_path(file_map_value)


def _lookup_file_map_path(candidate: str, file_map: dict[str, str]) -> str | None:
    normalized = _normalize_path(candidate)
    for key, value in file_map.items():
        normalized_key, normalized_value = _file_map_path(key, value)
        if normalized in {normalized_key, normalized_value}:
            return value
    return None


def _resolve_relative_require(imp: ImportStatement, file_map: dict[str, str]) -> str | None:
    source_dir = PurePosixPath(imp.file_path).parent
    candidate = str((source_dir / _with_rb_suffix(imp.module)).as_posix())
    return _lookup_file_map_path(candidate, file_map)


def _resolve_load_path_require(module: str, file_map: dict[str, str]) -> str | None:
    candidates = {module, _with_rb_suffix(module)}
    for rel_path, abs_path in file_map.items():
        normalized_key, normalized_value = _file_map_path(rel_path, abs_path)
        suffix_match = any(
            normalized_key.endswith(f"/{c}") or normalized_value.endswith(f"/{c}")
            for c in candidates
        )
        if normalized_key in candidates or normalized_value in candidates or suffix_match:
            return abs_path
    return None


def _resolve_ruby_import(imp: ImportStatement, file_map: dict[str, str]) -> str | None:
    if imp.is_relative:
        return _resolve_relative_require(imp, file_map)
    return _resolve_load_path_require(imp.module, file_map)


# ---------------------------------------------------------------------------
# Entry point detection
# ---------------------------------------------------------------------------


_ENTRY_BASENAMES: frozenset[str] = frozenset(
    {"app.rb", "main.rb", "server.rb", "config.ru", "Rakefile"}
)
_ENTRY_MARKERS: tuple[str, ...] = ("#!/usr/bin/env ruby", "#!/usr/bin/ruby")


# ---------------------------------------------------------------------------
# RubyAdapter
# ---------------------------------------------------------------------------


class RubyAdapter:
    """Language adapter for Ruby source files.

    `qualified_name` uses Archex's uniform '.' separator rather than Ruby's
    native '::'. Import resolution is conservative: `require_relative` resolves
    against the importing file's directory, while load-path `require` only
    resolves when the requested path directly matches a local file suffix.
    External gems such as `json` and `set` therefore stay unresolved instead of
    being guessed from unrelated same-basename files.
    """

    @property
    def language_id(self) -> str:
        return "ruby"

    @property
    def file_extensions(self) -> list[str]:
        return [".rb"]

    @property
    def tree_sitter_name(self) -> str:
        return "ruby"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract module, class, method, singleton-method, and constant symbols."""
        t: Any = tree
        root: object = t.root_node
        return _extract_body_symbols(root, source, file_path, None)

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract `require` and `require_relative` import calls."""
        t: Any = tree
        root: object = t.root_node
        return _walk_require_calls(root, source, file_path)

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve a Ruby require to a local file, or None if it is external."""
        return _resolve_ruby_import(imp, file_map)

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect conventional Ruby application entry points and shebang scripts."""
        entry_points: list[str] = []
        for f in files:
            if os.path.basename(f.path) in _ENTRY_BASENAMES:
                entry_points.append(f.path)
                continue
            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read(128)
            except OSError:
                continue
            if any(content.startswith(marker) for marker in _ENTRY_MARKERS):
                entry_points.append(f.path)
        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Return the symbol's stored visibility (set during extraction)."""
        return symbol.visibility
