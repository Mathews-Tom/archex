"""PHP parse adapter: extract symbols and imports from .php files using tree-sitter."""

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


def _map_php_visibility(
    node: object, source: bytes, default: Visibility = Visibility.PUBLIC
) -> Visibility:
    """Extract visibility from a declaration's `visibility_modifier` child.

    PHP has no package-private concept: members without an explicit modifier
    (and all top-level declarations, which carry no visibility modifier at
    all) default to public.
    """
    for child in _children(node):
        if _type(child) == "visibility_modifier":
            text = _text(child, source).strip().lower()
            if text == "public":
                return Visibility.PUBLIC
            if text == "protected":
                return Visibility.INTERNAL
            if text == "private":
                return Visibility.PRIVATE
    return default


# ---------------------------------------------------------------------------
# Namespace / qualification helpers
# ---------------------------------------------------------------------------


def _namespace_text(node: object, source: bytes) -> str:
    """Render a namespace_name node's `\\`-joined source text as a `.`-joined prefix."""
    return _text(node, source).replace("\\", ".")


def _qualify(namespace: str | None, name: str) -> str:
    return f"{namespace}.{name}" if namespace else name


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def _get_return_type(node: object, source: bytes) -> str:
    """Extract the return type from a method_declaration or function_definition."""
    return_node = _field(node, "return_type")
    if return_node is not None:
        return _text(return_node, source)
    return "mixed"


def _build_signature(node: object, source: bytes, name: str, return_type: str) -> str:
    params_node = _field(node, "parameters")
    params = _text(params_node, source) if params_node is not None else "()"
    return f"function {name}{params}: {return_type}"


# ---------------------------------------------------------------------------
# Constructor property promotion (PHP 8.0+)
# ---------------------------------------------------------------------------


def _extract_promoted_properties(
    params_node: object | None,
    source: bytes,
    file_path: str,
    parent_name: str,
    location: tuple[int, int],
) -> list[Symbol]:
    """Extract constructor-promoted properties as VARIABLE members of the class.

    A promoted parameter (``__construct(private readonly int $id)``) both
    declares and initializes a property; it has no declaration site of its
    own, so it is anchored at the constructor's line range.
    """
    if params_node is None:
        return []
    start_line, end_line = location
    symbols: list[Symbol] = []
    for param in _named_children(params_node):
        if _type(param) != "property_promotion_parameter":
            continue
        name_node = _field(param, "name")
        if name_node is None:
            continue
        name = _text(name_node, source).lstrip("$")
        vis = _map_php_visibility(param, source, Visibility.PUBLIC)
        symbols.append(
            Symbol(
                name=name,
                qualified_name=f"{parent_name}.{name}",
                kind=SymbolKind.VARIABLE,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                visibility=vis,
                parent=parent_name,
            )
        )
    return symbols


# ---------------------------------------------------------------------------
# Class / interface / trait member extraction
# ---------------------------------------------------------------------------


def _extract_method(node: object, source: bytes, file_path: str, parent_name: str) -> Symbol | None:
    """Extract a method_declaration node (including constructors)."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    return_type = _get_return_type(node, source)
    signature = _build_signature(node, source, name, return_type)
    vis = _map_php_visibility(node, source, Visibility.PUBLIC)
    return Symbol(
        name=name,
        qualified_name=f"{parent_name}.{name}",
        kind=SymbolKind.METHOD,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        signature=signature,
        parent=parent_name,
    )


def _extract_property_symbols(
    node: object, source: bytes, file_path: str, parent_name: str
) -> list[Symbol]:
    """Extract property_declaration symbols (may declare multiple variables)."""
    vis = _map_php_visibility(node, source, Visibility.PUBLIC)
    symbols: list[Symbol] = []
    for prop in _named_children(node):
        if _type(prop) != "property_element":
            continue
        name_node = _field(prop, "name")
        if name_node is None:
            continue
        name = _text(name_node, source).lstrip("$")
        symbols.append(
            Symbol(
                name=name,
                qualified_name=f"{parent_name}.{name}",
                kind=SymbolKind.VARIABLE,
                file_path=file_path,
                start_line=_start_line(node),
                end_line=_end_line(node),
                visibility=vis,
                parent=parent_name,
            )
        )
    return symbols


def _extract_const_symbols(
    node: object, source: bytes, file_path: str, parent_name: str
) -> list[Symbol]:
    """Extract const_declaration symbols (may declare multiple constants)."""
    vis = _map_php_visibility(node, source, Visibility.PUBLIC)
    symbols: list[Symbol] = []
    for const_el in _named_children(node):
        if _type(const_el) != "const_element":
            continue
        name_nodes = [c for c in _named_children(const_el) if _type(c) == "name"]
        if not name_nodes:
            continue
        name = _text(name_nodes[0], source)
        symbols.append(
            Symbol(
                name=name,
                qualified_name=f"{parent_name}.{name}",
                kind=SymbolKind.CONSTANT,
                file_path=file_path,
                start_line=_start_line(node),
                end_line=_end_line(node),
                visibility=vis,
                parent=parent_name,
            )
        )
    return symbols


def _extract_enum_case(
    node: object, source: bytes, file_path: str, parent_name: str
) -> Symbol | None:
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    return Symbol(
        name=name,
        qualified_name=f"{parent_name}.{name}",
        kind=SymbolKind.CONSTANT,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=Visibility.PUBLIC,
        parent=parent_name,
    )


def _extract_body_members(
    body: object, source: bytes, file_path: str, parent_name: str
) -> list[Symbol]:
    """Extract all member symbols from a class/interface/trait declaration_list."""
    symbols: list[Symbol] = []
    for child in _named_children(body):
        ct = _type(child)
        if ct == "method_declaration":
            method = _extract_method(child, source, file_path, parent_name)
            if method is None:
                continue
            symbols.append(method)
            symbols.extend(
                _extract_promoted_properties(
                    _field(child, "parameters"),
                    source,
                    file_path,
                    parent_name,
                    (method.start_line, method.end_line),
                )
            )
        elif ct == "property_declaration":
            symbols.extend(_extract_property_symbols(child, source, file_path, parent_name))
        elif ct == "const_declaration":
            symbols.extend(_extract_const_symbols(child, source, file_path, parent_name))
    return symbols


# ---------------------------------------------------------------------------
# Top-level declaration extraction
# ---------------------------------------------------------------------------
#
# PHP has no named nested type declarations (no inner classes/interfaces/
# traits), so class/interface/trait/enum symbols are always extracted at
# namespace scope, never nested under a `parent_name`.


def _extract_type_symbols(
    node: object, source: bytes, file_path: str, namespace: str | None, kind: SymbolKind
) -> list[Symbol]:
    """Extract a class/interface/trait declaration and its members.

    Traits have no dedicated ``SymbolKind`` (the model intentionally adds no
    new kinds for this tranche); they are reported as ``CLASS``, the closest
    structural fit — a trait's body has the same member shape as a class.
    """
    name_node = _field(node, "name")
    if name_node is None:
        return []
    name = _text(name_node, source)
    qualified = _qualify(namespace, name)
    vis = _map_php_visibility(node, source, Visibility.PUBLIC)
    symbols: list[Symbol] = [
        Symbol(
            name=name,
            qualified_name=qualified,
            kind=kind,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            parent=None,
        )
    ]
    body = _field(node, "body")
    if body is not None:
        symbols.extend(_extract_body_members(body, source, file_path, qualified))
    return symbols


def _extract_enum_symbols(
    node: object, source: bytes, file_path: str, namespace: str | None
) -> list[Symbol]:
    name_node = _field(node, "name")
    if name_node is None:
        return []
    name = _text(name_node, source)
    qualified = _qualify(namespace, name)
    symbols: list[Symbol] = [
        Symbol(
            name=name,
            qualified_name=qualified,
            kind=SymbolKind.ENUM,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=Visibility.PUBLIC,
            parent=None,
        )
    ]
    body = _field(node, "body")
    if body is None:
        return symbols
    for child in _named_children(body):
        ct = _type(child)
        if ct == "enum_case":
            case_symbol = _extract_enum_case(child, source, file_path, qualified)
            if case_symbol is not None:
                symbols.append(case_symbol)
        elif ct == "method_declaration":
            method = _extract_method(child, source, file_path, qualified)
            if method is not None:
                symbols.append(method)
    return symbols


def _extract_function_symbol(
    node: object, source: bytes, file_path: str, namespace: str | None
) -> Symbol | None:
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    return_type = _get_return_type(node, source)
    signature = _build_signature(node, source, name, return_type)
    return Symbol(
        name=name,
        qualified_name=_qualify(namespace, name),
        kind=SymbolKind.FUNCTION,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=Visibility.PUBLIC,
        signature=signature,
        parent=None,
    )


def _extract_top_level_declarations(
    nodes: list[object], source: bytes, file_path: str, namespace: str | None
) -> list[Symbol]:
    symbols: list[Symbol] = []
    for node in nodes:
        ct = _type(node)
        if ct == "class_declaration":
            symbols.extend(
                _extract_type_symbols(node, source, file_path, namespace, SymbolKind.CLASS)
            )
        elif ct == "interface_declaration":
            symbols.extend(
                _extract_type_symbols(node, source, file_path, namespace, SymbolKind.INTERFACE)
            )
        elif ct == "trait_declaration":
            symbols.extend(
                _extract_type_symbols(node, source, file_path, namespace, SymbolKind.CLASS)
            )
        elif ct == "enum_declaration":
            symbols.extend(_extract_enum_symbols(node, source, file_path, namespace))
        elif ct == "function_definition":
            fn = _extract_function_symbol(node, source, file_path, namespace)
            if fn is not None:
                symbols.append(fn)
    return symbols


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------


def _clause_module(clause: object, source: bytes, prefix: str | None) -> str | None:
    """Build the fully backslash-qualified module path for one use clause."""
    for child in _named_children(clause):
        if _type(child) in ("qualified_name", "name"):
            leaf = _text(child, source)
            return f"{prefix}\\{leaf}" if prefix else leaf
    return None


def _clause_alias(clause: object, source: bytes) -> str | None:
    alias_node = _field(clause, "alias")
    return _text(alias_node, source) if alias_node is not None else None


def _parse_use_declaration(node: object, source: bytes, file_path: str) -> list[ImportStatement]:
    """Parse a namespace_use_declaration into one ImportStatement per imported name.

    Handles simple (`use A\\B;`), aliased (`use A\\B as C;`), grouped
    (`use A\\{B, C};`), and `use function`/`use const` forms — each imported
    name becomes its own ImportStatement since, unlike a Python `from`
    import, each PHP use-clause name is an independent, separately
    resolvable class/function/constant.
    """
    line = _start_line(node)
    prefix_node: object | None = None
    group: object | None = None
    for child in _children(node):
        ct = _type(child)
        if ct == "namespace_name":
            prefix_node = child
        elif ct == "namespace_use_group":
            group = child

    clauses: list[object]
    prefix: str | None
    if group is not None:
        prefix = _text(prefix_node, source) if prefix_node is not None else None
        clauses = [c for c in _named_children(group) if _type(c) == "namespace_use_clause"]
    else:
        prefix = None
        clauses = [c for c in _named_children(node) if _type(c) == "namespace_use_clause"]

    imports: list[ImportStatement] = []
    for clause in clauses:
        module = _clause_module(clause, source, prefix)
        if module is None:
            continue
        imports.append(
            ImportStatement(
                module=module.replace("\\", "."),
                alias=_clause_alias(clause, source),
                file_path=file_path,
                line=line,
                is_relative=False,
            )
        )
    return imports


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------


def _resolve_php_import(module: str, file_map: dict[str, str]) -> str | None:
    """Resolve a PHP `use` import to a local .php file, or None if external.

    Matches the imported class/function/constant's basename against files in
    *file_map*, scoring candidates by how many trailing namespace segments
    match the candidate's directory structure (PSR-4-style). A namespaced
    import with zero directory overlap is treated as external; an
    unnamespaced (global) import falls back to a basename-only match.
    """
    parts = [p for p in module.split(".") if p]
    if not parts:
        return None
    class_name = parts[-1]
    namespace_parts = parts[:-1]
    target_file = f"{class_name}.php"

    candidates: list[tuple[int, str]] = []
    for key, abs_path in file_map.items():
        if os.path.basename(key) != target_file:
            continue
        dir_path = os.path.dirname(key).replace("\\", "/")
        dir_segments = [s for s in dir_path.split("/") if s]

        score = 0
        for i, part in enumerate(reversed(namespace_parts)):
            idx = len(dir_segments) - 1 - i
            if idx >= 0 and dir_segments[idx] == part:
                score += 1
            else:
                break

        if namespace_parts and score == 0:
            continue

        candidates.append((score, abs_path))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Entry point detection
# ---------------------------------------------------------------------------

_ENTRY_BASENAMES: frozenset[str] = frozenset({"index.php", "cli.php", "console.php", "artisan"})
_ENTRY_MARKERS: tuple[str, ...] = ("#!/usr/bin/env php", "php_sapi_name(")


# ---------------------------------------------------------------------------
# PHPAdapter
# ---------------------------------------------------------------------------


class PHPAdapter:
    """Language adapter for PHP source files."""

    @property
    def language_id(self) -> str:
        return "php"

    @property
    def file_extensions(self) -> list[str]:
        return [".php"]

    @property
    def tree_sitter_name(self) -> str:
        return "php"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract all symbols from a PHP parse tree.

        Tracks the active namespace across semicolon-style
        (`namespace App\\Models;`) siblings and recurses into brace-style
        (`namespace App\\Legacy { ... }`) bodies.
        """
        t: Any = tree
        root: object = t.root_node
        symbols: list[Symbol] = []
        namespace: str | None = None

        for node in _named_children(root):
            if _type(node) == "namespace_definition":
                name_node = _field(node, "name")
                ns = _namespace_text(name_node, source) if name_node is not None else None
                body = _field(node, "body")
                if body is not None:
                    symbols.extend(
                        _extract_top_level_declarations(
                            _named_children(body), source, file_path, ns
                        )
                    )
                else:
                    namespace = ns
            else:
                symbols.extend(
                    _extract_top_level_declarations([node], source, file_path, namespace)
                )

        return symbols

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all use-import declarations from a PHP parse tree."""
        t: Any = tree
        root: object = t.root_node
        imports: list[ImportStatement] = []

        for node in _named_children(root):
            ct = _type(node)
            if ct == "namespace_use_declaration":
                imports.extend(_parse_use_declaration(node, source, file_path))
            elif ct == "namespace_definition":
                body = _field(node, "body")
                if body is not None:
                    for child in _named_children(body):
                        if _type(child) == "namespace_use_declaration":
                            imports.extend(_parse_use_declaration(child, source, file_path))

        return imports

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve a PHP `use` import to a local file, or None if external."""
        return _resolve_php_import(imp.module, file_map)

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect PHP entry points: front controllers, CLI scripts, shebang scripts."""
        entry_points: list[str] = []
        for f in files:
            if os.path.basename(f.path) in _ENTRY_BASENAMES:
                entry_points.append(f.path)
                continue
            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            if any(marker in content for marker in _ENTRY_MARKERS):
                entry_points.append(f.path)
        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Return the symbol's stored visibility (set during extraction)."""
        return symbol.visibility
