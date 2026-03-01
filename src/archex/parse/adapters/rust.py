"""Rust parse adapter: extract symbols and imports from .rs files using tree-sitter."""

from __future__ import annotations

import os
from typing import Any

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters.ts_node import (
    ts_children as _children,
)
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
# Visibility helpers
# ---------------------------------------------------------------------------


def _get_visibility(node: object) -> Visibility:
    """Return the Visibility for a node based on its visibility_modifier child."""
    for child in _children(node):
        if _type(child) == "visibility_modifier":
            return _parse_visibility_modifier(child)
    return Visibility.PRIVATE


def _parse_visibility_modifier(node: object) -> Visibility:
    """Parse a visibility_modifier node: pub, pub(crate), pub(super)."""
    children = _children(node)
    # Simple `pub` — just the keyword, no parenthesized qualifier
    if len(children) <= 1:
        return Visibility.PUBLIC
    # pub(crate) or pub(super) — has parenthesized children
    for child in children:
        ct = _type(child)
        if ct in ("crate", "self", "super"):
            return Visibility.INTERNAL
    return Visibility.PUBLIC


# ---------------------------------------------------------------------------
# Symbol extraction helpers
# ---------------------------------------------------------------------------


def _extract_function(
    node: object, source: bytes, file_path: str, parent: str | None = None
) -> Symbol | None:
    """Extract a Symbol from a function_item node."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    vis = _get_visibility(node)
    kind = SymbolKind.METHOD if parent else SymbolKind.FUNCTION
    qualified = f"{parent}.{name}" if parent else name

    # Build signature
    params_node = _field(node, "parameters")
    ret_node = _field(node, "return_type")
    sig_parts = [f"fn {name}"]
    if params_node:
        sig_parts.append(_text(params_node, source))
    else:
        sig_parts.append("()")
    if ret_node:
        sig_parts.append(f" -> {_text(ret_node, source)}")
    signature = "".join(sig_parts)

    return Symbol(
        name=name,
        qualified_name=qualified,
        kind=kind,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        signature=signature,
        parent=parent,
    )


def _extract_struct(node: object, source: bytes, file_path: str) -> Symbol | None:
    """Extract a Symbol from a struct_item node."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    vis = _get_visibility(node)
    return Symbol(
        name=name,
        qualified_name=name,
        kind=SymbolKind.TYPE,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
    )


def _extract_enum(node: object, source: bytes, file_path: str) -> Symbol | None:
    """Extract a Symbol from an enum_item node."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    vis = _get_visibility(node)
    return Symbol(
        name=name,
        qualified_name=name,
        kind=SymbolKind.ENUM,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
    )


def _extract_trait(node: object, source: bytes, file_path: str) -> list[Symbol]:
    """Extract a trait Symbol plus its method declarations."""
    name_node = _field(node, "name")
    if name_node is None:
        return []
    name = _text(name_node, source)
    vis = _get_visibility(node)
    symbols: list[Symbol] = [
        Symbol(
            name=name,
            qualified_name=name,
            kind=SymbolKind.INTERFACE,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
        )
    ]
    # Extract trait method declarations from body
    body = _field(node, "body")
    if body is not None:
        for child in _named_children(body):
            ct = _type(child)
            if ct == "function_signature_item":
                method_sym = _extract_trait_method(child, source, file_path, name)
                if method_sym:
                    symbols.append(method_sym)
            elif ct == "function_item":
                # Default method implementation in trait
                method_sym = _extract_function(child, source, file_path, parent=name)
                if method_sym:
                    symbols.append(method_sym)
    return symbols


def _extract_trait_method(
    node: object, source: bytes, file_path: str, trait_name: str
) -> Symbol | None:
    """Extract a method declaration from a trait body."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    qualified = f"{trait_name}.{name}"
    return Symbol(
        name=name,
        qualified_name=qualified,
        kind=SymbolKind.METHOD,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=Visibility.PUBLIC,
        parent=trait_name,
    )


def _extract_impl_block(node: object, source: bytes, file_path: str) -> list[Symbol]:
    """Extract method symbols from an impl block."""
    symbols: list[Symbol] = []
    # Get the type being implemented
    type_node = _field(node, "type")
    if type_node is None:
        return symbols
    type_name = _text(type_node, source)

    body = _field(node, "body")
    if body is None:
        return symbols

    for child in _named_children(body):
        if _type(child) == "function_item":
            sym = _extract_function(child, source, file_path, parent=type_name)
            if sym:
                symbols.append(sym)
    return symbols


def _extract_const(node: object, source: bytes, file_path: str) -> Symbol | None:
    """Extract a Symbol from a const_item node."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    vis = _get_visibility(node)
    return Symbol(
        name=name,
        qualified_name=name,
        kind=SymbolKind.CONSTANT,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
    )


def _extract_static(node: object, source: bytes, file_path: str) -> Symbol | None:
    """Extract a Symbol from a static_item node."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    vis = _get_visibility(node)
    return Symbol(
        name=name,
        qualified_name=name,
        kind=SymbolKind.CONSTANT,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
    )


def _extract_type_alias(node: object, source: bytes, file_path: str) -> Symbol | None:
    """Extract a Symbol from a type_item node."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    vis = _get_visibility(node)
    return Symbol(
        name=name,
        qualified_name=name,
        kind=SymbolKind.TYPE,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
    )


def _extract_macro(node: object, source: bytes, file_path: str) -> Symbol | None:
    """Extract a Symbol from a macro_definition node."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    return Symbol(
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=Visibility.PUBLIC,
    )


def _walk_source_file(root: object, source: bytes, file_path: str) -> list[Symbol]:
    """Walk the source_file root and collect all symbols."""
    symbols: list[Symbol] = []

    for child in _named_children(root):
        ct = _type(child)

        if ct == "function_item":
            sym = _extract_function(child, source, file_path)
            if sym:
                symbols.append(sym)
        elif ct == "struct_item":
            sym = _extract_struct(child, source, file_path)
            if sym:
                symbols.append(sym)
        elif ct == "enum_item":
            sym = _extract_enum(child, source, file_path)
            if sym:
                symbols.append(sym)
        elif ct == "trait_item":
            symbols.extend(_extract_trait(child, source, file_path))
        elif ct == "impl_item":
            symbols.extend(_extract_impl_block(child, source, file_path))
        elif ct == "const_item":
            sym = _extract_const(child, source, file_path)
            if sym:
                symbols.append(sym)
        elif ct == "static_item":
            sym = _extract_static(child, source, file_path)
            if sym:
                symbols.append(sym)
        elif ct == "type_item":
            sym = _extract_type_alias(child, source, file_path)
            if sym:
                symbols.append(sym)
        elif ct == "macro_definition":
            sym = _extract_macro(child, source, file_path)
            if sym:
                symbols.append(sym)

    return symbols


# ---------------------------------------------------------------------------
# Import parsing helpers
# ---------------------------------------------------------------------------


def _parse_use_declaration(node: object, source: bytes, file_path: str) -> list[ImportStatement]:
    """Parse a use_declaration node into ImportStatement(s)."""
    line = _start_line(node)
    argument = _field(node, "argument")
    if argument is None:
        return []

    results: list[ImportStatement] = []
    _collect_use_paths(argument, source, file_path, line, "", results)
    return results


def _collect_use_paths(
    node: object,
    source: bytes,
    file_path: str,
    line: int,
    prefix: str,
    results: list[ImportStatement],
) -> None:
    """Recursively collect import paths from a use tree node."""
    ct = _type(node)

    if ct == "scoped_identifier":
        full_path = _text(node, source)
        module, _, symbol = full_path.rpartition("::")
        is_relative = _is_relative_path(full_path)
        results.append(
            ImportStatement(
                module=module if module else full_path,
                symbols=[symbol] if module else [],
                file_path=file_path,
                line=line,
                is_relative=is_relative,
            )
        )

    elif ct == "use_as_clause":
        path_node = _field(node, "path")
        alias_node = _field(node, "alias")
        if path_node:
            full_path = _text(path_node, source)
            module, _, symbol = full_path.rpartition("::")
            alias = _text(alias_node, source) if alias_node else None
            is_relative = _is_relative_path(full_path)
            results.append(
                ImportStatement(
                    module=module if module else full_path,
                    symbols=[symbol] if module else [],
                    alias=alias,
                    file_path=file_path,
                    line=line,
                    is_relative=is_relative,
                )
            )

    elif ct == "use_list":
        for child in _named_children(node):
            _collect_use_paths(child, source, file_path, line, prefix, results)

    elif ct == "scoped_use_list":
        path_node = _field(node, "path")
        list_node = _field(node, "list")
        base = _text(path_node, source) if path_node else prefix
        if list_node:
            symbols: list[str] = []
            for child in _named_children(list_node):
                child_ct = _type(child)
                if child_ct == "identifier":
                    symbols.append(_text(child, source))
                elif child_ct == "scoped_identifier":
                    nested_path = _text(child, source)
                    nested_mod, _, nested_sym = nested_path.rpartition("::")
                    full_mod = f"{base}::{nested_mod}" if nested_mod else base
                    is_relative = _is_relative_path(full_mod)
                    results.append(
                        ImportStatement(
                            module=full_mod,
                            symbols=[nested_sym],
                            file_path=file_path,
                            line=line,
                            is_relative=is_relative,
                        )
                    )
                elif child_ct == "self":
                    symbols.append("self")
                elif child_ct == "use_as_clause":
                    path_child = _field(child, "path")
                    if path_child:
                        symbols.append(_text(path_child, source))
                elif child_ct in ("scoped_use_list", "use_wildcard"):
                    _collect_use_paths(child, source, file_path, line, base, results)

            if symbols:
                is_relative = _is_relative_path(base)
                results.append(
                    ImportStatement(
                        module=base,
                        symbols=symbols,
                        file_path=file_path,
                        line=line,
                        is_relative=is_relative,
                    )
                )

    elif ct == "use_wildcard":
        # use some_path::*
        full = _text(node, source)
        module = full.rsplit("::*", 1)[0] if "::" in full else prefix
        is_relative = _is_relative_path(module)
        results.append(
            ImportStatement(
                module=module,
                symbols=["*"],
                file_path=file_path,
                line=line,
                is_relative=is_relative,
            )
        )

    elif ct == "identifier":
        name = _text(node, source)
        module = prefix if prefix else name
        syms = [name] if prefix else []
        is_relative = _is_relative_path(module)
        results.append(
            ImportStatement(
                module=module,
                symbols=syms,
                file_path=file_path,
                line=line,
                is_relative=is_relative,
            )
        )

    elif ct == "self":
        module = prefix if prefix else "self"
        is_relative = True
        results.append(
            ImportStatement(
                module=module,
                symbols=["self"],
                file_path=file_path,
                line=line,
                is_relative=is_relative,
            )
        )


def _is_relative_path(path: str) -> bool:
    """Check if a Rust use path is crate-relative."""
    return path.startswith(("crate", "super", "self"))


# ---------------------------------------------------------------------------
# Import resolution helpers
# ---------------------------------------------------------------------------


def _resolve_rust_path(
    module: str, symbols: list[str], file_path: str, file_map: dict[str, str]
) -> str | None:
    """Resolve a Rust use path to a file in the project."""
    if not _is_relative_path(module):
        return None

    parts: list[str]
    if module.startswith("crate"):
        rest = module.removeprefix("crate").lstrip(":")
        parts = rest.split("::") if rest else []
    elif module.startswith("super"):
        rest = module.removeprefix("super").lstrip(":")
        dir_path = os.path.dirname(file_path)
        parent = os.path.dirname(dir_path)
        parts_rest = rest.split("::") if rest else []
        for depth in range(len(parts_rest), 0, -1):
            candidate_rel = os.path.join(parent, *parts_rest[:depth]) + ".rs"
            for val in file_map.values():
                if val.endswith(candidate_rel) or val == candidate_rel:
                    return val
        return None
    elif module.startswith("self"):
        rest = module.removeprefix("self").lstrip(":")
        dir_path = os.path.dirname(file_path)
        parts_rest = rest.split("::") if rest else []
        for depth in range(len(parts_rest), 0, -1):
            candidate_rel = os.path.join(dir_path, *parts_rest[:depth]) + ".rs"
            for val in file_map.values():
                if val.endswith(candidate_rel) or val == candidate_rel:
                    return val
        return None
    else:
        return None

    # For crate:: paths, try matching against file_map values
    for depth in range(len(parts), 0, -1):
        segment = os.sep.join(parts[:depth])
        candidate_file = segment + ".rs"
        candidate_mod = os.path.join(segment, "mod.rs")
        candidate_lib = os.path.join(segment, "lib.rs")

        for val in file_map.values():
            normalized = val.replace("/", os.sep)
            if normalized.endswith(candidate_file):
                return val
            if normalized.endswith(candidate_mod):
                return val
            if normalized.endswith(candidate_lib):
                return val

    return None


# ---------------------------------------------------------------------------
# RustAdapter
# ---------------------------------------------------------------------------


class RustAdapter:
    """Language adapter for Rust source files."""

    @property
    def language_id(self) -> str:
        return "rust"

    @property
    def file_extensions(self) -> list[str]:
        return [".rs"]

    @property
    def tree_sitter_name(self) -> str:
        return "rust"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract all symbols from a Rust parse tree."""
        t: Any = tree
        root: object = t.root_node
        return _walk_source_file(root, source, file_path)

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all use declarations from a Rust parse tree."""
        t: Any = tree
        root: object = t.root_node
        imports: list[ImportStatement] = []

        for child in _named_children(root):
            if _type(child) == "use_declaration":
                imports.extend(_parse_use_declaration(child, source, file_path))

        return imports

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve a Rust use path to a file, or None for external crates."""
        return _resolve_rust_path(imp.module, imp.symbols, imp.file_path, file_map)

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect entry points: files with fn main() or lib.rs."""
        entry_points: list[str] = []

        for f in files:
            basename = os.path.basename(f.path)
            if basename == "lib.rs":
                entry_points.append(f.path)
                continue

            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue

            if "fn main()" in content:
                entry_points.append(f.path)

        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Return the symbol's stored visibility (set during extraction)."""
        return symbol.visibility
