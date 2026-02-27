"""Interface surface extraction: identify and document public API boundaries."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from archex.models import (
    Interface,
    Parameter,
    ParsedFile,
    SymbolKind,
    SymbolRef,
    Visibility,
)

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph


# ---------------------------------------------------------------------------
# Signature parsing helpers
# ---------------------------------------------------------------------------

_PARAM_RE = re.compile(
    r"^\s*(?P<name>\w+)"
    r"(?:\s*:\s*(?P<annotation>[^=,)]+?))?"
    r"(?:\s*=\s*(?P<default>[^,)]+?))?"
    r"\s*$"
)


def _parse_parameters(sig: str) -> list[Parameter]:
    """Extract parameters from a function signature string like 'def foo(a: int, b=1)'."""
    # Extract the params substring between the first ( and matching )
    paren_start = sig.find("(")
    paren_end = sig.rfind(")")
    if paren_start == -1 or paren_end == -1:
        return []

    raw_params = sig[paren_start + 1 : paren_end].strip()
    if not raw_params:
        return []

    params: list[Parameter] = []
    # Split on commas — careful with nested brackets (generics)
    depth = 0
    current: list[str] = []
    parts: list[str] = []
    for char in raw_params:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())

    for part in parts:
        part = part.strip()
        if not part or part in ("*", "/", "**kwargs", "*args"):
            continue

        # Handle *args and **kwargs
        is_var_positional = part.startswith("*") and not part.startswith("**")
        is_var_keyword = part.startswith("**")
        if is_var_positional:
            part = part[1:]
        elif is_var_keyword:
            part = part[2:]

        m = _PARAM_RE.match(part)
        if m is None:
            continue

        name = m.group("name")
        annotation = (m.group("annotation") or "").strip() or None
        default_raw = (m.group("default") or "").strip() or None

        # self/cls are not real parameters for the interface surface
        if name in ("self", "cls"):
            continue

        params.append(
            Parameter(
                name=name,
                type_annotation=annotation,
                default=default_raw,
                required=default_raw is None and not is_var_positional and not is_var_keyword,
            )
        )

    return params


def _parse_return_type(sig: str) -> str | None:
    """Extract return type annotation from a signature string."""
    arrow_pos = sig.rfind("->")
    if arrow_pos == -1:
        return None
    return sig[arrow_pos + 2 :].strip()


# ---------------------------------------------------------------------------
# used_by computation
# ---------------------------------------------------------------------------


def _build_used_by(graph: DependencyGraph) -> dict[str, list[str]]:
    """Map each file path to the list of files that import it."""
    used_by: dict[str, list[str]] = {}
    for edge in graph.file_edges():
        target = edge.target
        source = edge.source
        if target not in used_by:
            used_by[target] = []
        if source not in used_by[target]:
            used_by[target].append(source)
    return used_by


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_TOP_LEVEL_KINDS = {SymbolKind.FUNCTION, SymbolKind.CLASS, SymbolKind.INTERFACE}


def extract_interfaces(
    parsed_files: list[ParsedFile],
    graph: DependencyGraph,
) -> list[Interface]:
    """Extract public top-level symbols as interface surface with used_by and parsed signatures.

    Enhancements over profile.py's _extract_interfaces:
    - Excludes methods (only top-level symbols)
    - Excludes private symbols (starting with _)
    - Populates used_by from the dependency graph
    - Parses parameters and return types from signatures
    """
    used_by_map = _build_used_by(graph)
    interfaces: list[Interface] = []

    for pf in parsed_files:
        for sym in pf.symbols:
            if sym.visibility != Visibility.PUBLIC:
                continue
            if sym.kind not in _TOP_LEVEL_KINDS:
                continue
            # Exclude methods (they have a parent class)
            if sym.parent is not None:
                continue
            # Exclude private names (redundant but explicit)
            if sym.name.startswith("_") and not (
                sym.name.startswith("__") and sym.name.endswith("__")
            ):
                continue

            ref = SymbolRef(
                name=sym.name,
                qualified_name=sym.qualified_name,
                file_path=sym.file_path,
                kind=sym.kind,
            )

            signature = sym.signature or sym.name
            parameters = _parse_parameters(signature) if sym.signature else []
            return_type = _parse_return_type(signature) if sym.signature else None
            used_by = used_by_map.get(pf.path, [])

            interfaces.append(
                Interface(
                    symbol=ref,
                    signature=signature,
                    parameters=parameters,
                    return_type=return_type,
                    docstring=sym.docstring,
                    used_by=used_by,
                )
            )

    return interfaces
