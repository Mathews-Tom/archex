"""C++ parse adapter: extract symbols and imports from .cc/.cpp/.cxx/.hpp/.hh/.hxx
files using tree-sitter."""

from __future__ import annotations

import re
from typing import Any

from archex.models import Symbol, SymbolKind, Visibility
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
# Top-level declaration flattening
# ---------------------------------------------------------------------------
#
# Three C++ constructs wrap declarations that must be treated as their
# enclosing scope's direct members rather than opaque containers:
#
# - Preprocessor conditionals (#ifdef/#if/#elif/#else) -- same idiom as C,
#   most commonly the `#ifdef __cplusplus extern "C" { ... } #endif` guard
#   real C++ headers use to stay includable from C.
# - `extern "C" { ... }` linkage-specification blocks (and their unbraced,
#   single-declaration form `extern "C" void f();`) -- C++-to-C interop.
# - `template <...> ...` wrappers around a templated function or class --
#   the template parameter list carries no symbol information the adapter
#   needs, so the templated declaration underneath is unwrapped and treated
#   like any other function/class at that scope.
#
# Symbol extraction and #include parsing both flatten through the same
# helper so a conditionally compiled declaration or include is never missed
# by one and found by the other -- same discipline as the C adapter's
# `_toplevel_declarations`, extended for the two C++-only wrapper kinds.

_CONDITIONAL_TYPES = frozenset({"preproc_ifdef", "preproc_if", "preproc_elif", "preproc_else"})


def _unwrap_template_declaration(node: object) -> object | None:
    """A `template_declaration` wraps exactly one templated declaration
    (function_definition, a `;`-terminated class_specifier, etc.) after its
    `template_parameter_list` child. Returns that inner node."""
    for child in _named_children(node):
        if _type(child) != "template_parameter_list":
            return child
    return None


def _flatten_declarations(node: object) -> list[object]:
    """Flatten preprocessor conditionals, extern "C" linkage blocks, and
    template wrappers into one top-level sequence of real declarations."""
    result: list[object] = []
    for child in _named_children(node):
        ctype = _type(child)
        if ctype in _CONDITIONAL_TYPES:
            result.extend(_flatten_declarations(child))
        elif ctype == "linkage_specification":
            body = _field(child, "body")
            if body is None:
                continue
            if _type(body) == "declaration_list":
                result.extend(_flatten_declarations(body))
            else:
                # Unbraced form: extern "C" void f(); -- `body` IS the
                # single wrapped declaration, not a container of them.
                result.append(body)
        elif ctype == "template_declaration":
            inner = _unwrap_template_declaration(child)
            if inner is not None:
                result.append(inner)
        else:
            result.append(child)
    return result


# ---------------------------------------------------------------------------
# Scope-path qualification helpers
# ---------------------------------------------------------------------------
#
# `scope_path` is a list of namespace/class name segments accumulated while
# recursing (e.g. ["geo", "shapes"] or ["geo", "Point"]). Qualified names are
# always "."-joined -- including for explicit/partial template
# specializations, whose own "name" is the full templated-type text (e.g.
# "Pair<int>"). This keeps every C++ qualified name compatible with
# `archex.precision._get_parent_qname`'s generic dotted-name parent lookup,
# which is shared across every language adapter.


def _scope_qname(scope_path: list[str]) -> str | None:
    return ".".join(scope_path) if scope_path else None


def _qualify(scope_path: list[str], name: str) -> str:
    return ".".join([*scope_path, name]) if scope_path else name


def _map_access_specifier(text: str, current: Visibility) -> Visibility:
    if text == "public":
        return Visibility.PUBLIC
    if text == "private":
        return Visibility.PRIVATE
    if text == "protected":
        return Visibility.INTERNAL
    return current


# ---------------------------------------------------------------------------
# Function/method extraction -- definitions and prototypes share one path
# ---------------------------------------------------------------------------
#
# A function-shaped declarator's *inner* declarator tells us what kind of
# callable this is, independent of whether it has a body:
#
# - `field_identifier` / `destructor_name` / `operator_name` -- these only
#   ever occur inside a class/struct body: a plain method, destructor, or
#   operator overload declared (or inline-defined) as a member.
# - `identifier` -- a free function at namespace/file scope, OR (inside a
#   class body) an inline constructor, whose declarator is indistinguishable
#   from a free function's except for the `in_class` context it was found in.
#
# Out-of-class member definitions (`Point::getX() { ... }`, the .cpp half of
# a header/impl split) use a third declarator shape, `qualified_identifier`
# -- handled separately once header/impl pairing is in scope.

_STATIC_KEYWORD = "static"


def _is_static_free_function(node: object, source: bytes) -> bool:
    """True if `node` carries a `static` storage_class_specifier -- internal
    linkage, the same real C++ semantic C's adapter checks for."""
    for child in _named_children(node):
        if _type(child) == "storage_class_specifier" and _text(child, source) == _STATIC_KEYWORD:
            return True
    return False


def _unwrap_function_declarator(declarator: object | None) -> object | None:
    """Unwrap leading pointer_declarator layers (pointer return types) to
    find the underlying function_declarator, or None if `declarator` does
    not describe a function at all."""
    node = declarator
    while node is not None and _type(node) == "pointer_declarator":
        node = _field(node, "declarator")
    if node is not None and _type(node) == "function_declarator":
        return node
    return None


def _function_signature(node: object, declarator: object, source: bytes) -> str:
    """Slice the source from the declaration's start through the end of its
    outermost declarator, excluding the body/semicolon."""
    n: Any = node
    d: Any = declarator
    text = source[n.start_byte : d.end_byte].decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", text).strip()


def _extract_function_like(
    node: object,
    source: bytes,
    file_path: str,
    scope_path: list[str],
    in_class: bool,
    default_visibility: Visibility,
) -> Symbol | None:
    """Extract a FUNCTION or METHOD symbol from a function_definition, or a
    bodyless `declaration`/`field_declaration` sharing the same declarator
    shape (a prototype -- headers are mostly prototypes)."""
    declarator = _field(node, "declarator")
    if declarator is None:
        return None
    func_declarator = _unwrap_function_declarator(declarator)
    if func_declarator is None:
        return None
    inner = _field(func_declarator, "declarator")
    if inner is None:
        return None
    inner_type = _type(inner)

    if inner_type in ("field_identifier", "destructor_name", "operator_name"):
        if not in_class:
            return None
        name = _text(inner, source)
        parent = _scope_qname(scope_path)
        kind = SymbolKind.METHOD
    elif inner_type == "identifier":
        name = _text(inner, source)
        parent = _scope_qname(scope_path)
        kind = SymbolKind.METHOD if in_class else SymbolKind.FUNCTION
    else:
        return None

    if not name:
        return None
    visibility = default_visibility
    if kind == SymbolKind.FUNCTION and _is_static_free_function(node, source):
        visibility = Visibility.PRIVATE
    return Symbol(
        name=name,
        qualified_name=f"{parent}.{name}" if parent else name,
        kind=kind,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=visibility,
        signature=_function_signature(node, declarator, source),
        parent=parent,
    )


# ---------------------------------------------------------------------------
# Data member extraction
# ---------------------------------------------------------------------------


def _unwrap_field_identifier(node: object) -> object | None:
    """Unwrap pointer_declarator/reference_declarator layers on a data
    member's declarator to find its field_identifier. `reference_declarator`
    exposes no `declarator` field (unlike `pointer_declarator`), so its
    identifier is found positionally instead."""
    while node is not None and _type(node) in ("pointer_declarator", "reference_declarator"):
        if _type(node) == "pointer_declarator":
            node = _field(node, "declarator")
        else:
            named = _named_children(node)
            node = named[-1] if named else None
    if node is not None and _type(node) == "field_identifier":
        return node
    return None


def _extract_field_symbols(
    node: object, source: bytes, file_path: str, parent: str | None, visibility: Visibility
) -> list[Symbol]:
    """Extract VARIABLE/CONSTANT symbols from a field_declaration whose
    declarator(s) are plain data members (`int x_, y_;`), not functions --
    a single field_declaration may declare more than one member. A member
    is CONSTANT when it carries both `static` and a const-ish qualifier
    (`const`/`constexpr`); everything else is a VARIABLE."""
    tokens = {_text(c, source) for c in _children(node)}
    is_constant = _STATIC_KEYWORD in tokens and ("const" in tokens or "constexpr" in tokens)
    kind = SymbolKind.CONSTANT if is_constant else SymbolKind.VARIABLE

    symbols: list[Symbol] = []
    for child in _children(node):
        target = child if _type(child) == "field_identifier" else _unwrap_field_identifier(child)
        if target is None:
            continue
        name = _text(target, source)
        symbols.append(
            Symbol(
                name=name,
                qualified_name=f"{parent}.{name}" if parent else name,
                kind=kind,
                file_path=file_path,
                start_line=_start_line(node),
                end_line=_end_line(node),
                visibility=visibility,
                parent=parent,
            )
        )
    return symbols


# ---------------------------------------------------------------------------
# Class/struct extraction -- default visibility, nested types, member walk
# ---------------------------------------------------------------------------


def _class_or_struct_name(node: object, source: bytes) -> str | None:
    """Return the class/struct's own name, or the full templated-type text
    for an explicit/partial specialization (e.g. 'Pair<int>' -- genuinely
    distinct from the primary template's plain 'Pair', by construction, not
    by disambiguation), or None for an anonymous class/struct."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    return _text(name_node, source).replace("::", ".")


def _extract_type_decl(
    node: object,
    source: bytes,
    file_path: str,
    scope_path: list[str],
    default_visibility: Visibility,
) -> list[Symbol]:
    """Build a CLASS (class_specifier) or TYPE (struct_specifier) symbol
    plus its recursively extracted members, or [] for a forward declaration
    (no `body` field) or an anonymous class/struct with no name to give it."""
    body = _field(node, "body")
    if body is None:
        return []
    name = _class_or_struct_name(node, source)
    if not name:
        return []
    kind = SymbolKind.CLASS if _type(node) == "class_specifier" else SymbolKind.TYPE
    parent = _scope_qname(scope_path)
    own_symbol = Symbol(
        name=name,
        qualified_name=_qualify(scope_path, name),
        kind=kind,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=default_visibility,
        parent=parent,
    )
    # C++ default member access: private for `class`, public for `struct` --
    # a real language semantic (like C's `static` storage class), not a
    # naming convention, so it cannot be recomputed from a member's name.
    member_default_visibility = Visibility.PUBLIC if kind == SymbolKind.TYPE else Visibility.PRIVATE
    members = _extract_scope_members(
        _flatten_declarations(body),
        source,
        file_path,
        [*scope_path, name],
        True,
        member_default_visibility,
    )
    return [own_symbol, *members]


# ---------------------------------------------------------------------------
# Namespace extraction
# ---------------------------------------------------------------------------


def _namespace_name(node: object, source: bytes) -> str | None:
    """Return the namespace's name -- a plain `namespace_identifier`, or the
    full `::`-joined text of a C++17 nested-namespace-definition
    (`namespace geo::shapes { ... }`), converted to '.' -- or None for an
    anonymous namespace (`namespace { ... }`)."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    return _text(name_node, source).replace("::", ".")


def _extract_namespace(
    node: object, source: bytes, file_path: str, scope_path: list[str]
) -> list[Symbol]:
    name = _namespace_name(node, source)
    body = _field(node, "body")
    if body is None:
        return []
    inner_scope = [*scope_path, name] if name else scope_path
    # An anonymous namespace gives its direct members internal linkage --
    # the modern-C++-preferred equivalent of a top-level `static` -- without
    # affecting the access-specifier-governed visibility of members nested
    # deeper inside a class declared within it.
    inner_default = Visibility.PUBLIC if name else Visibility.PRIVATE
    members = _extract_scope_members(
        _flatten_declarations(body), source, file_path, inner_scope, False, inner_default
    )
    if name is None:
        return members
    own = Symbol(
        name=name.rsplit(".", 1)[-1],
        qualified_name=_qualify(scope_path, name),
        kind=SymbolKind.MODULE,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=Visibility.PUBLIC,
        parent=_scope_qname(scope_path),
    )
    return [own, *members]


# ---------------------------------------------------------------------------
# Unified scope-member dispatcher -- shared by namespace/file scope and
# class/struct body scope
# ---------------------------------------------------------------------------


def _extract_scope_members(
    nodes_flat: list[object],
    source: bytes,
    file_path: str,
    scope_path: list[str],
    in_class: bool,
    default_visibility: Visibility,
) -> list[Symbol]:
    symbols: list[Symbol] = []
    visibility = default_visibility
    for node in nodes_flat:
        ntype = _type(node)
        if ntype == "namespace_definition":
            if in_class:
                continue  # namespaces cannot nest inside a class
            symbols.extend(_extract_namespace(node, source, file_path, scope_path))
        elif ntype == "access_specifier":
            visibility = _map_access_specifier(_text(node, source), visibility)
        elif ntype in ("class_specifier", "struct_specifier"):
            symbols.extend(_extract_type_decl(node, source, file_path, scope_path, visibility))
        elif ntype == "function_definition":
            sym = _extract_function_like(node, source, file_path, scope_path, in_class, visibility)
            if sym is not None:
                symbols.append(sym)
        elif ntype in ("declaration", "field_declaration"):
            type_field = _field(node, "type")
            is_nested_type = type_field is not None and _type(type_field) in (
                "class_specifier",
                "struct_specifier",
            )
            if is_nested_type:
                # A nested type declaration (no declarator) or a K&R
                # combined struct-definition-plus-variable
                # (`struct Foo { ... } var;`, declarator present) -- either
                # way the type itself is the symbol; a trailing variable
                # name is not.
                symbols.extend(
                    _extract_type_decl(type_field, source, file_path, scope_path, visibility)
                )
                continue
            sym = _extract_function_like(node, source, file_path, scope_path, in_class, visibility)
            if sym is not None:
                symbols.append(sym)
            elif ntype == "field_declaration" and in_class:
                parent = _scope_qname(scope_path)
                symbols.extend(_extract_field_symbols(node, source, file_path, parent, visibility))
    return symbols


# ---------------------------------------------------------------------------
# CppAdapter
# ---------------------------------------------------------------------------


class CppAdapter:
    """Language adapter for C++ source and header files."""

    @property
    def language_id(self) -> str:
        return "cpp"

    @property
    def file_extensions(self) -> list[str]:
        return [".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx"]

    @property
    def tree_sitter_name(self) -> str:
        return "cpp"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract all symbols from a C++ parse tree: namespaces, classes,
        structs (including nested types and template specializations),
        functions, and data members. Out-of-class (header/impl-split)
        member definitions are handled separately."""
        t: Any = tree
        root: object = t.root_node
        flat = _flatten_declarations(root)
        return _extract_scope_members(flat, source, file_path, [], False, Visibility.PUBLIC)
