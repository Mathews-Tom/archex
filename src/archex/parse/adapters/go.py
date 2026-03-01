"""Go parse adapter: extract symbols and imports from .go files using tree-sitter."""

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
# Visibility: Go exports via uppercase first letter
# ---------------------------------------------------------------------------


def _classify_go_name(name: str) -> Visibility:
    if not name:
        return Visibility.PRIVATE
    if name[0].isupper():
        return Visibility.PUBLIC
    return Visibility.PRIVATE


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def _get_func_signature(node: object, source: bytes, name: str) -> str:
    """Build a minimal function signature: func Name(params) result."""
    params_node = _field(node, "parameters")
    result_node = _field(node, "result")
    params = _text(params_node, source) if params_node else "()"
    sig = f"func {name}{params}"
    if result_node:
        sig += f" {_text(result_node, source)}"
    return sig


def _get_method_signature(node: object, source: bytes, name: str, receiver: str) -> str:
    """Build a method signature: func (recv) Name(params) result."""
    params_node = _field(node, "parameters")
    result_node = _field(node, "result")
    params = _text(params_node, source) if params_node else "()"
    sig = f"func ({receiver}) {name}{params}"
    if result_node:
        sig += f" {_text(result_node, source)}"
    return sig


# ---------------------------------------------------------------------------
# Receiver extraction
# ---------------------------------------------------------------------------


def _extract_receiver_type(recv_node: object, source: bytes) -> str:
    """Extract the base type name from a method receiver parameter list.

    Handles both value receivers `(x Type)` and pointer receivers `(x *Type)`.
    Returns the bare type name without pointer star.
    """
    for param in _named_children(recv_node):
        type_node = _field(param, "type")
        if type_node is None:
            continue
        if _type(type_node) == "pointer_type":
            for child in _named_children(type_node):
                if _type(child) == "type_identifier":
                    return _text(child, source)
        elif _type(type_node) == "type_identifier":
            return _text(type_node, source)
    return ""


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------


def _extract_functions(root: object, source: bytes, file_path: str) -> list[Symbol]:
    """Extract top-level function declarations."""
    symbols: list[Symbol] = []
    for child in _children(root):
        if _type(child) != "function_declaration":
            continue
        name_node = _field(child, "name")
        if name_node is None:
            continue
        name = _text(name_node, source)
        sig = _get_func_signature(child, source, name)
        symbols.append(
            Symbol(
                name=name,
                qualified_name=name,
                kind=SymbolKind.FUNCTION,
                file_path=file_path,
                start_line=_start_line(child),
                end_line=_end_line(child),
                visibility=_classify_go_name(name),
                signature=sig,
            )
        )
    return symbols


def _extract_methods(root: object, source: bytes, file_path: str) -> list[Symbol]:
    """Extract method declarations (functions with receivers)."""
    symbols: list[Symbol] = []
    for child in _children(root):
        if _type(child) != "method_declaration":
            continue
        name_node = _field(child, "name")
        recv_node = _field(child, "receiver")
        if name_node is None or recv_node is None:
            continue
        name = _text(name_node, source)
        receiver_type = _extract_receiver_type(recv_node, source)
        recv_text = _text(recv_node, source).strip("()")
        sig = _get_method_signature(child, source, name, recv_text)
        qualified = f"{receiver_type}.{name}" if receiver_type else name
        symbols.append(
            Symbol(
                name=name,
                qualified_name=qualified,
                kind=SymbolKind.METHOD,
                file_path=file_path,
                start_line=_start_line(child),
                end_line=_end_line(child),
                visibility=_classify_go_name(name),
                signature=sig,
                parent=receiver_type or None,
            )
        )
    return symbols


def _extract_type_declarations(root: object, source: bytes, file_path: str) -> list[Symbol]:
    """Extract type declarations: structs, interfaces, type aliases."""
    symbols: list[Symbol] = []
    for child in _children(root):
        if _type(child) != "type_declaration":
            continue
        for spec in _named_children(child):
            spec_type = _type(spec)
            if spec_type not in ("type_spec", "type_alias"):
                continue
            name_node = _field(spec, "name")
            if name_node is None:
                continue
            name = _text(name_node, source)
            type_node = _field(spec, "type")
            kind = SymbolKind.TYPE
            if type_node is not None:
                tn = _type(type_node)
                if tn == "interface_type":
                    kind = SymbolKind.INTERFACE
            symbols.append(
                Symbol(
                    name=name,
                    qualified_name=name,
                    kind=kind,
                    file_path=file_path,
                    start_line=_start_line(spec),
                    end_line=_end_line(spec),
                    visibility=_classify_go_name(name),
                )
            )
    return symbols


def _extract_const_var(root: object, source: bytes, file_path: str) -> list[Symbol]:
    """Extract const and var declarations."""
    symbols: list[Symbol] = []
    for child in _children(root):
        ct = _type(child)
        if ct == "const_declaration":
            kind = SymbolKind.CONSTANT
        elif ct == "var_declaration":
            kind = SymbolKind.VARIABLE
        else:
            continue
        _collect_specs(child, source, file_path, kind, symbols)
    return symbols


def _collect_specs(
    decl: object,
    source: bytes,
    file_path: str,
    kind: SymbolKind,
    out: list[Symbol],
) -> None:
    """Collect const_spec / var_spec from a declaration, handling grouped blocks."""
    for spec in _named_children(decl):
        st = _type(spec)
        if st in ("const_spec", "var_spec"):
            name_node = _field(spec, "name")
            if name_node is None:
                continue
            name = _text(name_node, source)
            out.append(
                Symbol(
                    name=name,
                    qualified_name=name,
                    kind=kind,
                    file_path=file_path,
                    start_line=_start_line(spec),
                    end_line=_end_line(spec),
                    visibility=_classify_go_name(name),
                )
            )
        elif st in ("const_spec_list", "var_spec_list"):
            for inner in _named_children(spec):
                if _type(inner) in ("const_spec", "var_spec"):
                    name_node = _field(inner, "name")
                    if name_node is None:
                        continue
                    name = _text(name_node, source)
                    out.append(
                        Symbol(
                            name=name,
                            qualified_name=name,
                            kind=kind,
                            file_path=file_path,
                            start_line=_start_line(inner),
                            end_line=_end_line(inner),
                            visibility=_classify_go_name(name),
                        )
                    )


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------


def _strip_quotes(s: str) -> str:
    """Remove surrounding double quotes from a Go import path."""
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


def _parse_import_spec(spec: object, source: bytes, file_path: str, line: int) -> ImportStatement:
    """Parse a single import_spec node into an ImportStatement."""
    name_node = _field(spec, "name")
    path_node = _field(spec, "path")
    path_str = _strip_quotes(_text(path_node, source)) if path_node else ""
    alias: str | None = None
    if name_node is not None:
        alias_text = _text(name_node, source)
        if alias_text not in ("_", "."):
            alias = alias_text
    return ImportStatement(
        module=path_str,
        alias=alias,
        file_path=file_path,
        line=line,
        is_relative=False,
    )


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------


def _resolve_go_import(module: str, file_map: dict[str, str]) -> str | None:
    """Resolve a Go import path to a local file path.

    Go imports use package paths. We match trailing path segments of the import
    against directory components of files in the file map.
    """
    parts = module.split("/")
    pkg_name = parts[-1]

    for key, abs_path in file_map.items():
        dir_path = os.path.dirname(key)
        dir_parts = dir_path.replace("\\", "/").split("/")
        if len(dir_parts) >= len(parts) and dir_parts[-len(parts) :] == parts:
            return abs_path
        if dir_parts and dir_parts[-1] == pkg_name:
            return abs_path

    return None


# ---------------------------------------------------------------------------
# GoAdapter
# ---------------------------------------------------------------------------


class GoAdapter:
    """Language adapter for Go source files."""

    @property
    def language_id(self) -> str:
        return "go"

    @property
    def file_extensions(self) -> list[str]:
        return [".go"]

    @property
    def tree_sitter_name(self) -> str:
        return "go"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract all symbols from a Go parse tree."""
        t: Any = tree
        root: object = t.root_node

        symbols: list[Symbol] = []
        symbols.extend(_extract_type_declarations(root, source, file_path))
        symbols.extend(_extract_functions(root, source, file_path))
        symbols.extend(_extract_methods(root, source, file_path))
        symbols.extend(_extract_const_var(root, source, file_path))
        return symbols

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all import statements from a Go parse tree."""
        t: Any = tree
        root: object = t.root_node
        imports: list[ImportStatement] = []

        for child in _children(root):
            if _type(child) != "import_declaration":
                continue
            line = _start_line(child)
            for ic in _named_children(child):
                ic_type = _type(ic)
                if ic_type == "import_spec":
                    imports.append(_parse_import_spec(ic, source, file_path, line))
                elif ic_type == "import_spec_list":
                    for spec in _named_children(ic):
                        if _type(spec) == "import_spec":
                            imports.append(
                                _parse_import_spec(spec, source, file_path, _start_line(spec))
                            )
        return imports

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve a Go import to a local file path, or None if external."""
        return _resolve_go_import(imp.module, file_map)

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect Go entry points: files with `package main` and `func main()`."""
        entry_points: list[str] = []
        for f in files:
            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            if "package main" in content and "func main()" in content:
                entry_points.append(f.path)
        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Classify visibility: uppercase first letter -> PUBLIC, else -> PRIVATE."""
        return _classify_go_name(symbol.name)
