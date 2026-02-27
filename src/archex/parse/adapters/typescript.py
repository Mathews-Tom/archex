"""TypeScript/JavaScript parse adapter: extract symbols and imports using tree-sitter."""

from __future__ import annotations

import os
from typing import Any

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility

# ---------------------------------------------------------------------------
# Thin accessor layer for tree-sitter Node (no type stubs available).
# ---------------------------------------------------------------------------


def _text(node: object, source: bytes) -> str:
    n: Any = node
    return source[n.start_byte : n.end_byte].decode("utf-8", errors="replace")


def _type(node: object) -> str:
    n: Any = node
    return str(n.type)


def _children(node: object) -> list[object]:
    n: Any = node
    return list(n.children)


def _field(node: object, field: str) -> object | None:
    n: Any = node
    result: object | None = n.child_by_field_name(field)
    return result


def _start_line(node: object) -> int:
    n: Any = node
    return int(n.start_point[0]) + 1


def _end_line(node: object) -> int:
    n: Any = node
    return int(n.end_point[0]) + 1


def _named_children(node: object) -> list[object]:
    n: Any = node
    return list(n.named_children)


# ---------------------------------------------------------------------------
# Visibility helpers
# ---------------------------------------------------------------------------


def _classify_visibility(exported: bool) -> Visibility:
    return Visibility.PUBLIC if exported else Visibility.PRIVATE


# ---------------------------------------------------------------------------
# Symbol extraction helpers
# ---------------------------------------------------------------------------


def _extract_class_body_members(
    class_name: str,
    body: object,
    source: bytes,
    file_path: str,
) -> list[Symbol]:
    """Extract method symbols from a class body."""
    symbols: list[Symbol] = []
    for child in _named_children(body):
        ct = _type(child)
        if ct not in ("method_definition", "public_field_definition"):
            continue
        name_node = _field(child, "name")
        if name_node is None:
            continue
        method_name = _text(name_node, source)
        qualified = f"{class_name}.{method_name}"
        symbols.append(
            Symbol(
                name=method_name,
                qualified_name=qualified,
                kind=SymbolKind.METHOD,
                file_path=file_path,
                start_line=_start_line(child),
                end_line=_end_line(child),
                visibility=Visibility.PUBLIC,
                parent=class_name,
            )
        )
    return symbols


def _walk_program(root: object, source: bytes, file_path: str) -> list[Symbol]:
    """Walk the program node and collect all top-level symbols."""
    symbols: list[Symbol] = []

    for child in _named_children(root):
        ct = _type(child)

        # export_statement wraps the actual declaration
        if ct == "export_statement":
            inner = _unwrap_export(child, source, file_path, exported=True)
            symbols.extend(inner)
            continue

        # Top-level declarations without export
        syms = _extract_declaration(child, source, file_path, exported=False)
        symbols.extend(syms)

    return symbols


def _unwrap_export(
    export_node: object, source: bytes, file_path: str, exported: bool
) -> list[Symbol]:
    """Extract symbols from an export_statement node."""
    symbols: list[Symbol] = []
    for child in _named_children(export_node):
        ct = _type(child)
        # export { x } from '...' or export { x } — skip, these are re-exports handled in imports
        if ct == "export_clause":
            continue
        syms = _extract_declaration(child, source, file_path, exported=exported)
        symbols.extend(syms)
    return symbols


def _extract_declaration(
    node: object, source: bytes, file_path: str, exported: bool
) -> list[Symbol]:
    """Extract symbols from a declaration node."""
    ct = _type(node)
    symbols: list[Symbol] = []
    vis = _classify_visibility(exported)

    if ct == "function_declaration":
        name_node = _field(node, "name")
        if name_node is not None:
            name = _text(name_node, source)
            symbols.append(
                Symbol(
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.FUNCTION,
                    file_path=file_path,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    visibility=vis,
                )
            )

    elif ct == "class_declaration":
        name_node = _field(node, "name")
        if name_node is not None:
            class_name = _text(name_node, source)
            symbols.append(
                Symbol(
                    name=class_name,
                    qualified_name=class_name,
                    kind=SymbolKind.CLASS,
                    file_path=file_path,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    visibility=vis,
                )
            )
            body = _field(node, "body")
            if body is not None:
                symbols.extend(_extract_class_body_members(class_name, body, source, file_path))

    elif ct == "interface_declaration":
        name_node = _field(node, "name")
        if name_node is not None:
            name = _text(name_node, source)
            symbols.append(
                Symbol(
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.INTERFACE,
                    file_path=file_path,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    visibility=vis,
                )
            )

    elif ct == "type_alias_declaration":
        name_node = _field(node, "name")
        if name_node is not None:
            name = _text(name_node, source)
            symbols.append(
                Symbol(
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.TYPE,
                    file_path=file_path,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    visibility=vis,
                )
            )

    elif ct == "enum_declaration":
        name_node = _field(node, "name")
        if name_node is not None:
            name = _text(name_node, source)
            symbols.append(
                Symbol(
                    name=name,
                    qualified_name=name,
                    kind=SymbolKind.ENUM,
                    file_path=file_path,
                    start_line=_start_line(node),
                    end_line=_end_line(node),
                    visibility=vis,
                )
            )

    elif ct in ("lexical_declaration", "variable_declaration"):
        # const/let/var declarations — possibly multiple declarators
        for decl_child in _named_children(node):
            if _type(decl_child) != "variable_declarator":
                continue
            name_node = _field(decl_child, "name")
            if name_node is None:
                continue
            name = _text(name_node, source)
            # Determine CONSTANT vs VARIABLE based on keyword
            kind = _const_or_variable(node, source)
            symbols.append(
                Symbol(
                    name=name,
                    qualified_name=name,
                    kind=kind,
                    file_path=file_path,
                    start_line=_start_line(decl_child),
                    end_line=_end_line(decl_child),
                    visibility=vis,
                )
            )

    return symbols


def _const_or_variable(decl_node: object, source: bytes) -> SymbolKind:
    """Return CONSTANT for const declarations, VARIABLE otherwise."""
    for child in _children(decl_node):
        if _type(child) == "const":
            return SymbolKind.CONSTANT
        raw = _text(child, source)
        if raw == "const":
            return SymbolKind.CONSTANT
    return SymbolKind.VARIABLE


# ---------------------------------------------------------------------------
# Import parsing helpers
# ---------------------------------------------------------------------------


def _parse_import_statement_node(
    node: object, source: bytes, file_path: str
) -> list[ImportStatement]:
    """Parse an import_statement node."""
    line = _start_line(node)
    results: list[ImportStatement] = []

    # Find the module source (string node)
    module_str = ""
    symbols: list[str] = []
    alias: str | None = None
    is_relative = False

    for child in _named_children(node):
        ct = _type(child)
        if ct == "string":
            raw = _text(child, source).strip("'\"")
            module_str = raw
            is_relative = raw.startswith(".")
        elif ct == "import_clause":
            symbols, alias = _parse_import_clause(child, source)
        elif ct == "namespace_import":
            # import * as X from '...'
            for nc in _named_children(child):
                if _type(nc) == "identifier":
                    alias = _text(nc, source)
            symbols = ["*"]

    if module_str:
        results.append(
            ImportStatement(
                module=module_str,
                symbols=symbols,
                alias=alias,
                file_path=file_path,
                line=line,
                is_relative=is_relative,
            )
        )
    return results


def _parse_import_clause(clause: object, source: bytes) -> tuple[list[str], str | None]:
    """Parse import clause: { x, y as z } or default import."""
    symbols: list[str] = []
    alias: str | None = None

    for child in _named_children(clause):
        ct = _type(child)
        if ct == "identifier":
            # default import: import Foo from '...'
            symbols.append(_text(child, source))
        elif ct == "named_imports":
            for spec in _named_children(child):
                if _type(spec) == "import_specifier":
                    name_node = _field(spec, "name")
                    alias_node = _field(spec, "alias")
                    if name_node is not None:
                        symbols.append(_text(name_node, source))
                    if alias_node is not None and alias is None:
                        alias = _text(alias_node, source)
        elif ct == "namespace_import":
            for nc in _named_children(child):
                if _type(nc) == "identifier":
                    alias = _text(nc, source)
            symbols = ["*"]

    return symbols, alias


def _parse_export_statement_imports(
    node: object, source: bytes, file_path: str
) -> list[ImportStatement]:
    """Parse re-export statements: export { x } from './mod'."""
    line = _start_line(node)
    results: list[ImportStatement] = []

    module_str = ""
    is_relative = False
    symbols: list[str] = []

    for child in _named_children(node):
        ct = _type(child)
        if ct == "string":
            raw = _text(child, source).strip("'\"")
            module_str = raw
            is_relative = raw.startswith(".")
        elif ct == "export_clause":
            for spec in _named_children(child):
                if _type(spec) == "export_specifier":
                    name_node = _field(spec, "name")
                    if name_node is not None:
                        symbols.append(_text(name_node, source))

    if module_str:
        results.append(
            ImportStatement(
                module=module_str,
                symbols=symbols,
                file_path=file_path,
                line=line,
                is_relative=is_relative,
            )
        )
    return results


def _parse_call_expression_require(
    node: object, source: bytes, file_path: str
) -> ImportStatement | None:
    """Detect CommonJS require() calls."""
    # call_expression: function=identifier("require"), arguments
    func_node = _field(node, "function")
    if func_node is None:
        return None
    if _type(func_node) != "identifier" or _text(func_node, source) != "require":
        return None
    args_node = _field(node, "arguments")
    if args_node is None:
        return None
    for arg in _named_children(args_node):
        if _type(arg) == "string":
            raw = _text(arg, source).strip("'\"")
            return ImportStatement(
                module=raw,
                file_path=file_path,
                line=_start_line(node),
                is_relative=raw.startswith("."),
            )
    return None


def _collect_requires(root: object, source: bytes, file_path: str) -> list[ImportStatement]:
    """Recursively walk tree looking for require() calls."""
    results: list[ImportStatement] = []
    stack: list[object] = list(_named_children(root))
    while stack:
        node = stack.pop()
        if _type(node) == "call_expression":
            imp = _parse_call_expression_require(node, source, file_path)
            if imp is not None:
                results.append(imp)
        stack.extend(_named_children(node))
    return results


# ---------------------------------------------------------------------------
# Import resolution helpers
# ---------------------------------------------------------------------------

_TS_EXTENSIONS = [".ts", ".tsx", ".js", ".jsx"]


def _normalize_module(module: str) -> str:
    """Strip .js extension (TS convention: import from './foo.js' → './foo')."""
    if module.endswith(".js") or module.endswith(".jsx"):
        return module[: module.rfind(".")]
    return module


def _resolve_relative(module: str, file_path: str, file_map: dict[str, str]) -> str | None:
    """Resolve a relative module specifier against the importing file."""
    norm = _normalize_module(module)
    file_dir = os.path.dirname(file_path)
    base = os.path.normpath(os.path.join(file_dir, norm))

    # Build a set of all paths (keys and values) for O(1) lookup
    all_values: set[str] = set(file_map.values())

    # Try direct file matches (base + extension) against values
    for ext in _TS_EXTENSIONS:
        candidate = base + ext
        if candidate in all_values:
            return candidate

    # Try index file against values
    for ext in _TS_EXTENSIONS:
        candidate = os.path.join(base, f"index{ext}")
        if candidate in all_values:
            return candidate

    # Try matching against keys in file_map (relative paths) — key → value
    for key, val in file_map.items():
        norm_key = os.path.normpath(key)
        if norm_key == base:
            return val
        for ext in _TS_EXTENSIONS:
            if norm_key == base + ext:
                return val
            # Strip extension from key and compare
            if key.endswith(ext) and os.path.normpath(key[: -len(ext)]) == base:
                return val

    # Try index files by key
    for key, val in file_map.items():
        norm_key = os.path.normpath(key)
        for ext in _TS_EXTENSIONS:
            index_candidate = os.path.normpath(os.path.join(base, f"index{ext}"))
            if norm_key == index_candidate:
                return val

    return None


# ---------------------------------------------------------------------------
# TypeScriptAdapter
# ---------------------------------------------------------------------------


class TypeScriptAdapter:
    """Language adapter for TypeScript and JavaScript source files."""

    @property
    def language_id(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> list[str]:
        return [".ts", ".tsx", ".js", ".jsx"]

    @property
    def tree_sitter_name(self) -> str:
        return "typescript"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract symbols by walking the parse tree."""
        t: Any = tree
        root: object = t.root_node
        return _walk_program(root, source, file_path)

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all import statements from a TypeScript/JavaScript parse tree."""
        t: Any = tree
        root: object = t.root_node
        imports: list[ImportStatement] = []

        for child in _named_children(root):
            ct = _type(child)
            if ct == "import_statement":
                imports.extend(_parse_import_statement_node(child, source, file_path))
            elif ct == "export_statement":
                # Re-exports: export { x } from './mod'
                imports.extend(_parse_export_statement_imports(child, source, file_path))

        # Collect CommonJS require() calls
        imports.extend(_collect_requires(root, source, file_path))
        return imports

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve an import to an absolute file path, or None if external."""
        if imp.is_relative:
            return _resolve_relative(imp.module, imp.file_path, file_map)
        # Absolute/external — not resolvable to a local file in the general case
        norm = _normalize_module(imp.module)
        if norm in file_map:
            return file_map[norm]
        return None

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect entry points: index.ts/js, main.ts/js, or files with export default."""
        entry_points: list[str] = []
        entry_names = {"index.ts", "index.js", "index.tsx", "main.ts", "main.js", "main.tsx"}

        for f in files:
            basename = os.path.basename(f.path)
            if basename in entry_names:
                entry_points.append(f.path)
                continue

            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue

            if "export default" in content:
                entry_points.append(f.path)

        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Return the symbol's stored visibility (set during extraction)."""
        return symbol.visibility
