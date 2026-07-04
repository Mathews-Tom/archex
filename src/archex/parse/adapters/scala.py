"""Scala parse adapter: extract symbols and imports from .scala/.sc files using tree-sitter."""

from __future__ import annotations

import os
from typing import Any

from archex.models import DiscoveredFile, ImportStatement, Symbol, SymbolKind, Visibility
from archex.parse.adapters._jvm_helpers import map_jvm_visibility, resolve_jvm_import
from archex.parse.adapters.ts_node import ts_children as _children
from archex.parse.adapters.ts_node import ts_end_line as _end_line
from archex.parse.adapters.ts_node import ts_field as _field
from archex.parse.adapters.ts_node import ts_named_children as _named_children
from archex.parse.adapters.ts_node import ts_start_line as _start_line
from archex.parse.adapters.ts_node import ts_text as _text
from archex.parse.adapters.ts_node import ts_type as _type

# ---------------------------------------------------------------------------
# Qualification helpers
# ---------------------------------------------------------------------------


def _qualify(parent: str | None, name: str) -> str:
    return f"{parent}.{name}" if parent else name


# ---------------------------------------------------------------------------
# Modifier helpers
# ---------------------------------------------------------------------------


def _extract_visibility(node: object, default: Visibility = Visibility.PUBLIC) -> Visibility:
    """Extract visibility from a node's `modifiers` -> `access_modifier` child.

    Scala declarations default to PUBLIC when no modifier is present. Only
    `private`/`protected` affect visibility (bare or qualified, e.g.
    `private[this]`, `protected[com.example]` -- the bracketed qualifier
    doesn't change the PRIVATE/INTERNAL mapping). Other modifier keywords
    (`override`, `implicit`, `lazy`, `abstract`, `final`, `sealed`, `case`)
    are unnamed tokens directly under `modifiers` with no wrapping node and
    never affect visibility.
    """
    for child in _children(node):
        if _type(child) != "modifiers":
            continue
        for mod in _children(child):
            if _type(mod) != "access_modifier":
                continue
            for tok in _children(mod):
                tok_type = _type(tok)
                if tok_type in ("private", "protected"):
                    return map_jvm_visibility(tok_type, default=default)
    return default


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def _build_function_signature(node: object, source: bytes, name: str) -> str:
    """Build a `def name(params)(params...): ReturnType` signature string.

    Curried functions declare multiple sibling `parameters` nodes (one per
    parameter list); `child_by_field_name` only returns the first, so all
    parameter-list children are collected positionally instead.
    """
    param_groups = [c for c in _named_children(node) if _type(c) == "parameters"]
    params_text = "".join(_text(p, source) for p in param_groups) if param_groups else "()"

    return_type = _field(node, "return_type")
    if return_type is not None:
        return f"def {name}{params_text}: {_text(return_type, source)}"
    return f"def {name}{params_text}"


# ---------------------------------------------------------------------------
# Member extraction
# ---------------------------------------------------------------------------


def _extract_method(node: object, source: bytes, file_path: str, parent_name: str) -> Symbol | None:
    """Extract a member `function_definition`/`function_declaration` as METHOD."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    vis = _extract_visibility(node)
    sig = _build_function_signature(node, source, name)

    return Symbol(
        name=name,
        qualified_name=_qualify(parent_name, name),
        kind=SymbolKind.METHOD,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        signature=sig,
        parent=parent_name,
    )


def _extract_field(
    node: object, source: bytes, file_path: str, parent_name: str, kind: SymbolKind
) -> list[Symbol]:
    """Extract a member `val_definition`/`var_definition` as CONSTANT/VARIABLE.

    Only the simple-binding form (`pattern` field is a plain `identifier`)
    is extracted; destructuring patterns (`val (a, b) = ...`,
    `val Some(x) = ...`) have no single name to report and are skipped.
    """
    pattern_node = _field(node, "pattern")
    if pattern_node is None or _type(pattern_node) != "identifier":
        return []
    name = _text(pattern_node, source)
    vis = _extract_visibility(node)

    return [
        Symbol(
            name=name,
            qualified_name=_qualify(parent_name, name),
            kind=kind,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            parent=parent_name,
        )
    ]


def _extract_body_members(
    body: object, source: bytes, file_path: str, parent_name: str
) -> list[Symbol]:
    """Extract all member symbols from a class/object/trait `template_body`."""
    symbols: list[Symbol] = []

    for child in _named_children(body):
        ct = _type(child)

        if ct in ("function_definition", "function_declaration"):
            sym = _extract_method(child, source, file_path, parent_name)
            if sym is not None:
                symbols.append(sym)
        elif ct == "val_definition":
            symbols.extend(
                _extract_field(child, source, file_path, parent_name, SymbolKind.CONSTANT)
            )
        elif ct == "var_definition":
            symbols.extend(
                _extract_field(child, source, file_path, parent_name, SymbolKind.VARIABLE)
            )
        elif ct in ("class_definition", "object_definition"):
            symbols.extend(
                _extract_type_symbols(child, source, file_path, SymbolKind.CLASS, parent_name)
            )
        elif ct == "trait_definition":
            symbols.extend(
                _extract_type_symbols(child, source, file_path, SymbolKind.INTERFACE, parent_name)
            )

    return symbols


def _extract_type_symbols(
    node: object,
    source: bytes,
    file_path: str,
    kind: SymbolKind,
    parent_name: str | None = None,
) -> list[Symbol]:
    """Extract a class/object/trait declaration and its members.

    `class_definition` and `object_definition` both report `SymbolKind.CLASS`
    (matching the existing Kotlin `object_declaration -> CLASS` precedent —
    the model has no dedicated singleton kind); `trait_definition` reports
    `SymbolKind.INTERFACE` (the model has no dedicated trait kind, and a
    Scala trait is structurally closest to an interface with default
    methods). A `class Foo` / `object Foo` companion pair therefore shares
    both `qualified_name` and `kind`; the resulting `symbol_id` collision is
    already resolved by `pipeline/chunker.py::_disambiguate_symbol_ids`,
    the same mechanism used for same-name/kind overloads in other languages.
    """
    name_node = _field(node, "name")
    if name_node is None:
        return []
    name = _text(name_node, source)
    qualified = _qualify(parent_name, name)
    vis = _extract_visibility(node)

    symbols: list[Symbol] = [
        Symbol(
            name=name,
            qualified_name=qualified,
            kind=kind,
            file_path=file_path,
            start_line=_start_line(node),
            end_line=_end_line(node),
            visibility=vis,
            parent=parent_name,
        )
    ]

    body = _field(node, "body")
    if body is not None:
        symbols.extend(_extract_body_members(body, source, file_path, qualified))

    return symbols


def _extract_top_level_function(
    node: object, source: bytes, file_path: str, namespace: str | None
) -> Symbol | None:
    """Extract a package-level (non-member) `function_definition` as FUNCTION."""
    name_node = _field(node, "name")
    if name_node is None:
        return None
    name = _text(name_node, source)
    vis = _extract_visibility(node)
    sig = _build_function_signature(node, source, name)

    return Symbol(
        name=name,
        qualified_name=_qualify(namespace, name),
        kind=SymbolKind.FUNCTION,
        file_path=file_path,
        start_line=_start_line(node),
        end_line=_end_line(node),
        visibility=vis,
        signature=sig,
        parent=None,
    )


def _extract_top_level(
    nodes: list[object],
    source: bytes,
    file_path: str,
    namespace: str | None,
    symbols: list[Symbol],
) -> None:
    """Walk top-level (or package-body) declarations, tracking the active package.

    Scala allows both a single `package a.b` (semicolon-style, no body:
    extends the *ambient* namespace for every following sibling, including
    chained consecutive `package` statements) and `package a.b { ... }`
    (brace-style, with a `body` field: nests its declarations, which are
    NOT top-level `compilation_unit` children and must be recursed into
    explicitly, the same shape as PHP's brace-style namespace handling).
    """
    active_namespace = namespace

    for node in nodes:
        ct = _type(node)

        if ct == "package_clause":
            name_node = _field(node, "name")
            pkg_name = _text(name_node, source) if name_node is not None else None
            body = _field(node, "body")
            if body is not None:
                extended = _qualify(active_namespace, pkg_name) if pkg_name else active_namespace
                _extract_top_level(_named_children(body), source, file_path, extended, symbols)
            elif pkg_name:
                active_namespace = _qualify(active_namespace, pkg_name)
        elif ct in ("class_definition", "object_definition"):
            symbols.extend(
                _extract_type_symbols(node, source, file_path, SymbolKind.CLASS, active_namespace)
            )
        elif ct == "trait_definition":
            symbols.extend(
                _extract_type_symbols(
                    node, source, file_path, SymbolKind.INTERFACE, active_namespace
                )
            )
        elif ct in ("function_definition", "function_declaration"):
            fn = _extract_top_level_function(node, source, file_path, active_namespace)
            if fn is not None:
                symbols.append(fn)


# ---------------------------------------------------------------------------
# Import parsing
# ---------------------------------------------------------------------------


def _parse_scala_import(node: object, source: bytes, file_path: str) -> list[ImportStatement]:
    """Parse a single `import_declaration` into one or more ImportStatements.

    A dotted import path is a flat sequence of sibling `identifier` children
    (no wrapping path node), followed optionally by a `namespace_wildcard`
    (`import a.b._`) or a `namespace_selectors` group
    (`import a.b.{C, D => E, _}`). Each name in a selector group becomes its
    own ImportStatement (mirroring PHP's grouped `use` handling) since each
    is an independently resolvable class/object/trait, unlike a Python
    `from` import.
    """
    line = _start_line(node)
    parts: list[str] = []
    is_wildcard = False
    selectors: object | None = None

    for child in _named_children(node):
        ct = _type(child)
        if ct == "identifier":
            parts.append(_text(child, source))
        elif ct == "namespace_wildcard":
            is_wildcard = True
        elif ct == "namespace_selectors":
            selectors = child

    if not parts:
        return []
    base = ".".join(parts)

    if selectors is not None:
        imports: list[ImportStatement] = []
        for sel in _named_children(selectors):
            st = _type(sel)
            if st == "identifier":
                name = _text(sel, source)
                imports.append(
                    ImportStatement(
                        module=f"{base}.{name}", file_path=file_path, line=line, is_relative=False
                    )
                )
            elif st == "arrow_renamed_identifier":
                sel_name_node = _field(sel, "name")
                if sel_name_node is None:
                    continue
                sel_alias_node = _field(sel, "alias")
                name = _text(sel_name_node, source)
                alias = _text(sel_alias_node, source) if sel_alias_node is not None else None
                imports.append(
                    ImportStatement(
                        module=f"{base}.{name}",
                        alias=alias,
                        file_path=file_path,
                        line=line,
                        is_relative=False,
                    )
                )
            elif st == "namespace_wildcard":
                imports.append(
                    ImportStatement(
                        module=f"{base}._", file_path=file_path, line=line, is_relative=False
                    )
                )
        return imports

    if is_wildcard:
        return [
            ImportStatement(module=f"{base}._", file_path=file_path, line=line, is_relative=False)
        ]

    return [ImportStatement(module=base, file_path=file_path, line=line, is_relative=False)]


def _collect_imports(
    nodes: list[object], source: bytes, file_path: str, imports: list[ImportStatement]
) -> None:
    """Walk top-level (or package-body) declarations, collecting imports.

    Recurses into brace-style `package` bodies the same way `_extract_top_level`
    does, so an `import` nested inside `package a.b { ... }` is not missed.
    """
    for node in nodes:
        ct = _type(node)
        if ct == "import_declaration":
            imports.extend(_parse_scala_import(node, source, file_path))
        elif ct == "package_clause":
            body = _field(node, "body")
            if body is not None:
                _collect_imports(_named_children(body), source, file_path, imports)


# ---------------------------------------------------------------------------
# Entry point detection
# ---------------------------------------------------------------------------

_ENTRY_BASENAMES: frozenset[str] = frozenset({"Main.scala", "App.scala", "Boot.scala"})
_ENTRY_MARKERS: tuple[str, ...] = ("extends App", "def main(args: Array[String]")


# ---------------------------------------------------------------------------
# ScalaAdapter
# ---------------------------------------------------------------------------


class ScalaAdapter:
    """Language adapter for Scala source files.

    Import resolution reuses the JVM package-to-directory-segment-scoring
    heuristic (`resolve_jvm_import`, extensions=(".scala",)) shared with
    Java/Kotlin, since Scala's package/import semantics are structurally
    identical. Unlike Java, Scala doesn't enforce one top-level declaration
    per file, so an import naming a non-primary declaration inside a
    multi-declaration file will not resolve -- conservative by design,
    consistent with the Ruby adapter's "stay unresolved instead of being
    guessed" philosophy.
    """

    @property
    def language_id(self) -> str:
        return "scala"

    @property
    def file_extensions(self) -> list[str]:
        return [".scala", ".sc"]

    @property
    def tree_sitter_name(self) -> str:
        return "scala"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract class, object, trait, and top-level function symbols from a Scala parse tree."""
        t: Any = tree
        root: object = t.root_node
        symbols: list[Symbol] = []
        _extract_top_level(_named_children(root), source, file_path, None, symbols)
        return symbols

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all import declarations from a Scala parse tree."""
        t: Any = tree
        root: object = t.root_node
        imports: list[ImportStatement] = []
        _collect_imports(_named_children(root), source, file_path, imports)
        return imports

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve a Scala import to a local .scala file, or None if external or a wildcard."""
        if imp.module.endswith("._"):
            return None
        return resolve_jvm_import(imp.module, file_map, extensions=(".scala",))

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect Scala entry points: `extends App`, explicit `main`, and conventional filenames."""
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
