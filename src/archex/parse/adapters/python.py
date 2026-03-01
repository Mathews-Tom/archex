"""Python parse adapter: extract symbols and imports from .py files using tree-sitter."""

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
    ts_start_line as _start_line,
)
from archex.parse.adapters.ts_node import (
    ts_text as _text,
)
from archex.parse.adapters.ts_node import (
    ts_type as _type,
)


def _parent(node: object) -> object | None:
    n: Any = node
    result: object | None = n.parent
    return result


def _start_byte(node: object) -> int:
    n: Any = node
    return int(n.start_byte)


def _get_docstring(node: object, source: bytes) -> str | None:
    """Return docstring from a function/class body, if present."""
    body = _field(node, "body")
    if body is None:
        return None
    children = _children(body)
    if not children:
        return None
    first = children[0]
    if _type(first) == "expression_statement":
        expr_children = _children(first)
        if expr_children and _type(expr_children[0]) in ("string", "concatenated_string"):
            raw = _text(expr_children[0], source)
            for q in ('"""', "'''", '"', "'"):
                if raw.startswith(q) and raw.endswith(q) and len(raw) >= 2 * len(q):
                    return raw[len(q) : -len(q)].strip()
    return None


def _get_decorators(node: object, source: bytes) -> list[str]:
    """Extract decorators from the parent decorated_definition, if any."""
    parent = _parent(node)
    if parent is None:
        return []
    if _type(parent) != "decorated_definition":
        return []
    result: list[str] = []
    for child in _children(parent):
        if _type(child) == "decorator":
            raw = _text(child, source).lstrip("@").split("(")[0].strip()
            result.append(raw)
    return result


def _get_signature(node: object, source: bytes, name: str) -> str | None:
    """Build a minimal function signature string."""
    params_node = _field(node, "parameters")
    return_node = _field(node, "return_type")
    params = _text(params_node, source) if params_node else "()"
    if return_node:
        return f"def {name}{params} -> {_text(return_node, source)}"
    return f"def {name}{params}"


def _walk_classes(root: object, source: bytes, file_path: str) -> tuple[list[Symbol], set[int]]:
    """Walk the AST and collect class symbols and their method start_bytes."""
    class_symbols: list[Symbol] = []
    method_bytes: set[int] = set()

    for child in _children(root):
        node = child
        # Unwrap decorated_definition
        if _type(node) == "decorated_definition":
            inner = _field(node, "definition")
            if inner is None:
                continue
            node = inner

        if _type(node) != "class_definition":
            continue

        name_node = _field(node, "name")
        if name_node is None:
            continue

        class_name = _text(name_node, source)
        decorators = _get_decorators(node, source)
        docstring = _get_docstring(node, source)
        vis = _classify_name(class_name)

        class_symbols.append(
            Symbol(
                name=class_name,
                qualified_name=class_name,
                kind=SymbolKind.CLASS,
                file_path=file_path,
                start_line=_start_line(node),
                end_line=_end_line(node),
                visibility=vis,
                decorators=decorators,
                docstring=docstring,
            )
        )

        body = _field(node, "body")
        if body is None:
            continue

        for body_child in _children(body):
            method_node = body_child
            if _type(method_node) == "decorated_definition":
                inner = _field(method_node, "definition")
                if inner is None:
                    continue
                method_node = inner

            if _type(method_node) != "function_definition":
                continue

            method_name_node = _field(method_node, "name")
            if method_name_node is None:
                continue

            method_bytes.add(_start_byte(method_node))
            method_name = _text(method_name_node, source)
            qualified = f"{class_name}.{method_name}"
            m_decorators = _get_decorators(method_node, source)
            m_docstring = _get_docstring(method_node, source)
            m_sig = _get_signature(method_node, source, method_name)
            m_vis = _classify_name(method_name)

            class_symbols.append(
                Symbol(
                    name=method_name,
                    qualified_name=qualified,
                    kind=SymbolKind.METHOD,
                    file_path=file_path,
                    start_line=_start_line(method_node),
                    end_line=_end_line(method_node),
                    visibility=m_vis,
                    signature=m_sig,
                    docstring=m_docstring,
                    decorators=m_decorators,
                    parent=class_name,
                )
            )

    return class_symbols, method_bytes


def _walk_functions(
    root: object, source: bytes, file_path: str, skip_bytes: set[int]
) -> list[Symbol]:
    """Walk the AST and collect top-level function symbols."""
    func_symbols: list[Symbol] = []

    for child in _children(root):
        node = child
        if _type(node) == "decorated_definition":
            inner = _field(node, "definition")
            if inner is None:
                continue
            node = inner

        if _type(node) != "function_definition":
            continue

        if _start_byte(node) in skip_bytes:
            continue

        name_node = _field(node, "name")
        if name_node is None:
            continue

        func_name = _text(name_node, source)
        decorators = _get_decorators(node, source)
        docstring = _get_docstring(node, source)
        sig = _get_signature(node, source, func_name)
        vis = _classify_name(func_name)

        func_symbols.append(
            Symbol(
                name=func_name,
                qualified_name=func_name,
                kind=SymbolKind.FUNCTION,
                file_path=file_path,
                start_line=_start_line(node),
                end_line=_end_line(node),
                visibility=vis,
                signature=sig,
                docstring=docstring,
                decorators=decorators,
            )
        )

    return func_symbols


def _classify_name(name: str) -> Visibility:
    if name.startswith("__") and name.endswith("__"):
        return Visibility.PUBLIC
    if name.startswith("_"):
        return Visibility.PRIVATE
    return Visibility.PUBLIC


def _parse_import_statement(node: object, source: bytes, file_path: str) -> list[ImportStatement]:
    """Parse an import_statement node (e.g., `import os` or `import os as operating_system`)."""
    line = _start_line(node)
    results: list[ImportStatement] = []

    for child in _children(node):
        ct = _type(child)
        if ct == "dotted_name":
            module = _text(child, source)
            results.append(
                ImportStatement(module=module, file_path=file_path, line=line, is_relative=False)
            )
        elif ct == "aliased_import":
            name_node = _field(child, "name")
            alias_node = _field(child, "alias")
            if name_node:
                module = _text(name_node, source)
                alias = _text(alias_node, source) if alias_node else None
                results.append(
                    ImportStatement(
                        module=module,
                        alias=alias,
                        file_path=file_path,
                        line=line,
                        is_relative=False,
                    )
                )

    return results


def _parse_from_import_statement(
    node: object, source: bytes, file_path: str
) -> ImportStatement | None:
    """Parse an import_from_statement node.

    Handles: from X import Y, from . import X, from ..pkg import Y.
    """
    line = _start_line(node)
    children = _children(node)

    # Phase 1: find module reference (dotted_name or relative_import) before "import"
    module_str = ""
    is_relative = False
    past_from = False
    past_module = False

    symbols: list[str] = []
    alias: str | None = None

    for child in children:
        ct = _type(child)

        if ct == "from":
            past_from = True
            continue

        if past_from and not past_module:
            if ct == "dotted_name":
                module_str = _text(child, source)
                past_module = True
                continue
            if ct == "relative_import":
                # relative_import children: import_prefix (dots) and optional dotted_name
                dots = 0
                rel_mod = ""
                for rc in _children(child):
                    rct = _type(rc)
                    if rct == "import_prefix":
                        dots = len(_text(rc, source))
                    elif rct == "dotted_name":
                        rel_mod = _text(rc, source)
                is_relative = True
                module_str = "." * dots + rel_mod
                past_module = True
                continue

        if ct == "import":
            past_module = True  # ensure past_module is set even without explicit module
            continue

        # After "import" keyword: collect imported names
        if past_module:
            if ct == "dotted_name":
                symbols.append(_text(child, source))
            elif ct == "aliased_import":
                name_node = _field(child, "name")
                alias_node = _field(child, "alias")
                if name_node:
                    symbols.append(_text(name_node, source))
                    if alias_node and not alias:
                        alias = _text(alias_node, source)
            elif ct == "wildcard_import":
                symbols.append("*")

    return ImportStatement(
        module=module_str,
        symbols=symbols,
        alias=alias,
        file_path=file_path,
        line=line,
        is_relative=is_relative,
    )


class PythonAdapter:
    """Language adapter for Python source files."""

    @property
    def language_id(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> list[str]:
        return [".py"]

    @property
    def tree_sitter_name(self) -> str:
        return "python"

    def extract_symbols(self, tree: object, source: bytes, file_path: str) -> list[Symbol]:
        """Extract functions, classes, and methods by walking the parse tree."""
        t: Any = tree
        root: object = t.root_node

        class_symbols, method_bytes = _walk_classes(root, source, file_path)
        func_symbols = _walk_functions(root, source, file_path, method_bytes)

        return class_symbols + func_symbols

    def parse_imports(self, tree: object, source: bytes, file_path: str) -> list[ImportStatement]:
        """Extract all import statements by walking the top-level module children."""
        t: Any = tree
        root: object = t.root_node

        imports: list[ImportStatement] = []

        for child in _children(root):
            ct = _type(child)
            if ct == "import_statement":
                imports.extend(_parse_import_statement(child, source, file_path))
            elif ct == "import_from_statement":
                imp = _parse_from_import_statement(child, source, file_path)
                if imp is not None:
                    imports.append(imp)
            elif ct == "if_statement":
                # Handle TYPE_CHECKING guard: if TYPE_CHECKING: from X import Y
                body = _field(child, "consequence")
                if body is not None:
                    for body_child in _children(body):
                        bct = _type(body_child)
                        if bct == "import_from_statement":
                            imp = _parse_from_import_statement(body_child, source, file_path)
                            if imp is not None:
                                imports.append(imp)
                        elif bct == "import_statement":
                            imports.extend(_parse_import_statement(body_child, source, file_path))

        return imports

    def resolve_import(self, imp: ImportStatement, file_map: dict[str, str]) -> str | None:
        """Resolve an import to a file path, or None for external imports."""
        if imp.is_relative:
            module = imp.module
            dots = len(module) - len(module.lstrip("."))
            rel_module = module[dots:]

            file_dir = os.path.dirname(imp.file_path)
            base_dir = file_dir
            for _ in range(dots - 1):
                base_dir = os.path.dirname(base_dir)

            if rel_module:
                candidate_module = rel_module.replace(".", os.sep)
                candidate_path = os.path.join(base_dir, candidate_module + ".py")
                if candidate_path in file_map.values():
                    return candidate_path
                init_path = os.path.join(base_dir, candidate_module, "__init__.py")
                if init_path in file_map.values():
                    return init_path
            else:
                init_path = os.path.join(base_dir, "__init__.py")
                if init_path in file_map.values():
                    return init_path
            return None

        module = imp.module
        if module in file_map:
            return file_map[module]

        parts = module.split(".")
        for i in range(len(parts), 0, -1):
            key = ".".join(parts[:i])
            if key in file_map:
                return file_map[key]

        return None

    def detect_entry_points(self, files: list[DiscoveredFile]) -> list[str]:
        """Detect entry points from file names and content."""
        entry_points: list[str] = []

        for f in files:
            basename = os.path.basename(f.path)
            if basename == "__main__.py":
                entry_points.append(f.path)
                continue

            try:
                with open(f.absolute_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue

            if 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content:
                entry_points.append(f.path)
                continue

            if "def main(" in content:
                entry_points.append(f.path)

        return entry_points

    def classify_visibility(self, symbol: Symbol) -> Visibility:
        """Classify visibility: __dunder__ → PUBLIC, _prefix → PRIVATE, else → PUBLIC."""
        return _classify_name(symbol.name)
