"""C parse adapter: extract symbols and imports from .c/.h files using tree-sitter."""

from __future__ import annotations

import os
import re
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
# Top-level declaration walking
# ---------------------------------------------------------------------------
#
# C headers overwhelmingly wrap their declarations in constructs that nest
# children rather than flattening them to top-level siblings of
# translation_unit: preprocessor conditionals (#ifdef/#if/#elif/#else, for
# platform guards and include guards) and `extern "C" { ... }` linkage blocks
# (for C++ interop). A naive one-level scan of root.children would silently
# miss the majority of real-world .h file content. Symbol extraction and
# #include parsing both walk through this same helper so a conditionally
# compiled declaration or include is never missed by one and found by the
# other.

_CONDITIONAL_TYPES = frozenset({"preproc_ifdef", "preproc_if", "preproc_elif", "preproc_else"})


def _toplevel_declarations(node: object) -> list[object]:
    """Flatten declarations nested inside preprocessor conditionals and
    `extern "C"` linkage blocks into one top-level sequence."""
    result: list[object] = []
    for child in _named_children(node):
        ctype = _type(child)
        if ctype in _CONDITIONAL_TYPES:
            result.extend(_toplevel_declarations(child))
        elif ctype == "linkage_specification":
            body = _field(child, "body")
            if body is not None:
                result.extend(_toplevel_declarations(body))
        else:
            result.append(child)
    return result


# ---------------------------------------------------------------------------
# Visibility: C has no naming convention, only the `static` storage class
# ---------------------------------------------------------------------------


def _is_static(node: object, source: bytes) -> bool:
    """True if `node` carries a `static` storage_class_specifier child.

    C's public/private distinction is a real language semantic (internal
    vs. external linkage), not a project convention like Go's uppercase
    letter -- so this cannot be recomputed from a symbol's name alone.
    """
    for child in _named_children(node):
        if _type(child) == "storage_class_specifier" and _text(child, source) == "static":
            return True
    return False


# ---------------------------------------------------------------------------
# Function extraction: definitions and prototypes share one code path
# ---------------------------------------------------------------------------


def _unwrap_function_declarator(declarator: object | None) -> object | None:
    """Unwrap leading pointer_declarator layers (pointer return types) to
    find the underlying function_declarator, or None if `declarator` does
    not describe a function at all (e.g. a plain variable or a
    function-pointer variable, whose function_declarator wraps a
    parenthesized_declarator instead of a bare identifier -- rejected by
    the caller, not here)."""
    node = declarator
    while node is not None and _type(node) == "pointer_declarator":
        node = _field(node, "declarator")
    if node is not None and _type(node) == "function_declarator":
        return node
    return None


def _function_name(func_declarator: object, source: bytes) -> str | None:
    """Return the function's name, or None if this is a function-*pointer*
    declarator (`int (*fp)(int)`) rather than a function declaration --
    those wrap a parenthesized_declarator instead of a plain identifier."""
    inner = _field(func_declarator, "declarator")
    if inner is not None and _type(inner) == "identifier":
        return _text(inner, source)
    return None


def _function_signature(node: object, declarator: object, source: bytes) -> str:
    """Slice the source from the declaration's start through the end of its
    outermost declarator, excluding the body/semicolon -- covers storage
    class, return type, name, and parameters exactly as written."""
    n: Any = node
    d: Any = declarator
    text = source[n.start_byte : d.end_byte].decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", text).strip()


def _extract_function(node: object, source: bytes, file_path: str) -> Symbol | None:
    declarator = _field(node, "declarator")
    if declarator is None:
        return None
    func_declarator = _unwrap_function_declarator(declarator)
    if func_declarator is None:
        return None
    name = _function_name(func_declarator, source)
    if not name:
        return None
    return Symbol(
        name=name,
        qualified_name=name,
        kind=SymbolKind.FUNCTION,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=Visibility.PRIVATE if _is_static(node, source) else Visibility.PUBLIC,
        signature=_function_signature(node, declarator, source),
    )


def _extract_functions(root: object, source: bytes, file_path: str) -> list[Symbol]:
    """Extract both function definitions and function prototype
    declarations -- a .h file's public API is almost entirely prototypes,
    so limiting extraction to function_definition would report zero
    functions for every header."""
    symbols: list[Symbol] = []
    for node in _toplevel_declarations(root):
        ntype = _type(node)
        if ntype not in ("function_definition", "declaration"):
            continue
        symbol = _extract_function(node, source, file_path)
        if symbol is not None:
            symbols.append(symbol)
    return symbols


# ---------------------------------------------------------------------------
# Struct extraction: bare, typedef-anonymous, typedef-named, K&R combined
# ---------------------------------------------------------------------------


def _struct_symbol(
    spec: object,
    source: bytes,
    file_path: str,
    line_node: object,
    name_override: str | None = None,
) -> Symbol | None:
    """Build a TYPE symbol from a struct_specifier, or None for a
    forward declaration / bare type reference (no `body` field) or an
    anonymous struct with no typedef alias to name it."""
    body = _field(spec, "body")
    if body is None:
        return None
    name = name_override
    if name is None:
        name_node = _field(spec, "name")
        if name_node is not None:
            name = _text(name_node, source)
    if not name:
        return None
    return Symbol(
        name=name,
        qualified_name=name,
        kind=SymbolKind.TYPE,
        file_path=file_path,
        start_line=_start_line(line_node),
        end_line=_end_line(line_node),
        visibility=Visibility.PUBLIC,
    )


def _extract_structs(root: object, source: bytes, file_path: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    for node in _toplevel_declarations(root):
        ntype = _type(node)
        if ntype == "struct_specifier":
            # struct Point { ... }; -- bare top-level definition.
            symbol = _struct_symbol(node, source, file_path, line_node=node)
            if symbol is not None:
                symbols.append(symbol)
        elif ntype == "declaration":
            # struct Point { ... } origin; -- K&R combined struct + variable.
            type_field = _field(node, "type")
            if type_field is not None and _type(type_field) == "struct_specifier":
                symbol = _struct_symbol(type_field, source, file_path, line_node=node)
                if symbol is not None:
                    symbols.append(symbol)
        elif ntype == "type_definition":
            # typedef struct { ... } Size;  /  typedef struct Rect { ... } Rect;
            type_field = _field(node, "type")
            declarator = _field(node, "declarator")
            if (
                type_field is not None
                and _type(type_field) == "struct_specifier"
                and declarator is not None
                and _type(declarator) == "type_identifier"
            ):
                alias = _text(declarator, source)
                symbol = _struct_symbol(
                    type_field, source, file_path, line_node=node, name_override=alias
                )
                if symbol is not None:
                    symbols.append(symbol)
    return symbols


# ---------------------------------------------------------------------------
# #include parsing
# ---------------------------------------------------------------------------


def _parse_include(node: object, source: bytes, file_path: str) -> ImportStatement | None:
    path_node = _field(node, "path")
    if path_node is None:
        return None
    path_type = _type(path_node)
    if path_type == "string_literal":
        module = _text(path_node, source).strip('"')
        is_relative = True
    elif path_type == "system_lib_string":
        module = _text(path_node, source).strip("<>")
        is_relative = False
    else:
        return None
    return ImportStatement(
        module=module,
        file_path=file_path,
        line=_start_line(node),
        is_relative=is_relative,
    )


def _parse_includes(root: object, source: bytes, file_path: str) -> list[ImportStatement]:
    imports: list[ImportStatement] = []
    for node in _toplevel_declarations(root):
        if _type(node) != "preproc_include":
            continue
        imp = _parse_include(node, source, file_path)
        if imp is not None:
            imports.append(imp)
    return imports


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------


def _resolve_c_include(imp: ImportStatement, file_map: dict[str, str]) -> str | None:
    """Resolve a quoted #include to a local file, or None for angle-bracket
    (system/library) includes, which are never local by convention."""
    if not imp.is_relative:
        return None

    file_dir = os.path.dirname(imp.file_path)
    candidate = os.path.normpath(os.path.join(file_dir, imp.module))
    values = set(file_map.values())
    if candidate in values:
        return candidate

    # Fallback: match by basename anywhere in the project. Real C projects
    # commonly reference headers via a compiler -I search path rather than
    # strictly alongside the including file -- a build-system detail this
    # adapter has no visibility into, so basename matching is the best
    # available heuristic (the same class of tradeoff Go's resolver makes
    # for package-path matching).
    target_basename = os.path.basename(imp.module)
    for value in file_map.values():
        if os.path.basename(value) == target_basename:
            return value
    return None


# ---------------------------------------------------------------------------
# Entry point detection
# ---------------------------------------------------------------------------

# Matches a `main` function *definition* (has a body), not a prototype
# (`int main(void);`, terminated by `;` rather than `{`) or an unrelated call.
_MAIN_DEFINITION = re.compile(r"\bmain\s*\([^;{}]*\)\s*\{")


# ---------------------------------------------------------------------------
# CAdapter
# ---------------------------------------------------------------------------


class CAdapter:
    """Language adapter for C source and header files."""

    @property
    def language_id(self) -> str:
        return "c"

    @property
    def file_extensions(self) -> list[str]:
        return [".c", ".h"]

    @property
    def tree_sitter_name(self) -> str:
        return "c"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract all symbols from a C parse tree: functions (definitions
        and prototypes) and structs (bare, typedef-anonymous, typedef-named)."""
        t: Any = tree
        root: object = t.root_node
        symbols: list[Symbol] = []
        symbols.extend(_extract_structs(root, source, file_path))
        symbols.extend(_extract_functions(root, source, file_path))
        return symbols

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all #include directives from a C parse tree."""
        t: Any = tree
        root: object = t.root_node
        return _parse_includes(root, source, file_path)

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve a quoted #include to a local file, or None if external."""
        return _resolve_c_include(imp, file_map)

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect C entry points: .c files containing a `main` function
        definition. Headers are excluded -- a `main` definition in a header
        would produce a multiple-definition link error in any real project."""
        entry_points: list[str] = []
        for f in files:
            if not f.path.endswith(".c"):
                continue
            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            if _MAIN_DEFINITION.search(content):
                entry_points.append(f.path)
        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Return the symbol's stored visibility (set during extraction from
        the `static` storage class -- not recomputable from the name alone)."""
        return symbol.visibility
