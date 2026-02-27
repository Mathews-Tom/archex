"""Module boundary detection: cluster files into logical modules and score cohesion."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx.algorithms.community import louvain_communities

from archex.models import Module, ParsedFile, SymbolKind, SymbolRef, Visibility

if TYPE_CHECKING:
    from archex.index.graph import DependencyGraph


def _build_nx_graph(graph: DependencyGraph, parsed_files: list[ParsedFile]) -> Any:
    """Build an undirected nx.Graph from DependencyGraph edges for community detection."""
    g: Any = nx.Graph()  # type: ignore[type-arg]
    for pf in parsed_files:
        g.add_node(pf.path)  # type: ignore[misc]
    for edge in graph.file_edges():
        g.add_edge(edge.source, edge.target)  # type: ignore[misc]
    return g


def _infer_module_name(files: list[str]) -> str:
    """Infer a module name from the longest common path prefix of a file set."""
    if not files:
        return "unknown"
    if len(files) == 1:
        # Use directory name or stem
        path = files[0]
        parent = os.path.dirname(path)
        if parent:
            return os.path.basename(parent) or os.path.splitext(os.path.basename(path))[0]
        return os.path.splitext(os.path.basename(path))[0]

    # Find common prefix at directory level
    dirs = [os.path.dirname(f) for f in files]
    common = os.path.commonprefix(dirs)
    # Strip trailing path separator
    common = common.rstrip(os.sep).rstrip("/")
    if common:
        return os.path.basename(common) or common
    # Fallback: stem of the first file
    return os.path.splitext(os.path.basename(files[0]))[0]


def _cohesion_score(community: set[str], g: Any) -> float:
    """Calculate cohesion = internal edges / max possible edges for the subgraph."""
    n = len(community)
    if n < 2:
        return 1.0
    subgraph: Any = g.subgraph(community)  # type: ignore[misc]
    internal: int = subgraph.number_of_edges()  # type: ignore[misc]
    max_possible = n * (n - 1) / 2
    return round(internal / max_possible, 4)


def _build_module_from_community(
    community: set[str],
    g: Any,
    file_lookup: dict[str, ParsedFile],
) -> Module:
    """Build a Module from a community of file paths."""
    files = sorted(community)
    name = _infer_module_name(files)
    root_path = os.path.commonprefix(files).rstrip(os.sep).rstrip("/") or ""

    # Exports: public top-level symbols (not methods)
    exports: list[SymbolRef] = []
    line_count = 0
    for fpath in files:
        pf = file_lookup.get(fpath)
        if pf is None:
            continue
        line_count += pf.lines
        for sym in pf.symbols:
            if (
                sym.visibility == Visibility.PUBLIC
                and sym.parent is None
                and sym.kind in (SymbolKind.FUNCTION, SymbolKind.CLASS, SymbolKind.INTERFACE)
            ):
                exports.append(
                    SymbolRef(
                        name=sym.name,
                        qualified_name=sym.qualified_name,
                        file_path=sym.file_path,
                        kind=sym.kind,
                    )
                )

    # Internal deps: edges within community
    internal_deps: list[str] = []
    external_deps: list[str] = []
    for fpath in files:
        pf = file_lookup.get(fpath)
        if pf is None:
            continue
        for imp in pf.imports:
            resolved = imp.resolved_path
            if resolved is None:
                module_name = imp.module.split(".")[0] if imp.module else ""
                if module_name and module_name not in external_deps:
                    external_deps.append(module_name)
            elif resolved in community:
                if resolved not in internal_deps:
                    internal_deps.append(resolved)
            else:
                if resolved not in external_deps:
                    external_deps.append(resolved)

    cohesion = _cohesion_score(community, g)

    return Module(
        name=name,
        root_path=root_path,
        files=files,
        exports=exports,
        internal_deps=internal_deps,
        external_deps=external_deps,
        cohesion_score=cohesion,
        file_count=len(files),
        line_count=line_count,
    )


def detect_modules(
    graph: DependencyGraph,
    parsed_files: list[ParsedFile],
) -> list[Module]:
    """Detect module boundaries via Louvain community detection on the file graph.

    Returns one Module per community, with cohesion scores, exports, and deps.
    """
    if not parsed_files:
        return []

    g = _build_nx_graph(graph, parsed_files)
    file_lookup: dict[str, ParsedFile] = {pf.path: pf for pf in parsed_files}

    if g.number_of_nodes() == 0:  # type: ignore[misc]
        return []

    if g.number_of_nodes() == 1:  # type: ignore[misc]  # type: ignore[misc]
        community: set[str] = {str(n) for n in g.nodes()}  # type: ignore[misc]
        return [_build_module_from_community(community, g, file_lookup)]

    try:
        raw_communities: list[Any] = louvain_communities(g, seed=42)  # type: ignore[no-untyped-call]
    except Exception:
        # Fall back: treat all files as one module
        raw_communities = [g.nodes()]  # type: ignore[misc]

    # Cast communities to set[str]
    str_communities: list[set[str]] = [{str(n) for n in c} for c in raw_communities]

    modules: list[Module] = []
    for community in str_communities:
        module = _build_module_from_community(community, g, file_lookup)
        modules.append(module)

    return modules
