"""Dependency graph construction: build and query the file/symbol edge graph."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

import networkx as nx

from archex.models import Edge, EdgeKind, ImportStatement, ParsedFile

if TYPE_CHECKING:
    from pathlib import Path


class DependencyGraph:
    """Wraps two nx.DiGraph instances: one for files, one for symbols."""

    def __init__(self) -> None:
        self._file_graph: nx.DiGraph[str] = nx.DiGraph()  # type: ignore[type-arg]
        self._symbol_graph: nx.DiGraph[str] = nx.DiGraph()  # type: ignore[type-arg]
        self._centrality_cache: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_parsed_files(
        cls,
        parsed_files: list[ParsedFile],
        import_map: dict[str, list[ImportStatement]],
    ) -> DependencyGraph:
        """Build a DependencyGraph from parsed files and their resolved imports."""
        graph = cls()

        # Add file nodes
        for pf in parsed_files:
            graph._file_graph.add_node(pf.path)  # type: ignore[misc]

        # Add symbol nodes
        for pf in parsed_files:
            for sym in pf.symbols:
                graph._symbol_graph.add_node(sym.qualified_name, file_path=pf.path)  # type: ignore[misc]

        # Add file-level edges from resolved imports
        for file_path, imports in import_map.items():
            for imp in imports:
                if imp.resolved_path is not None and graph._file_graph.has_node(  # type: ignore[misc]
                    imp.resolved_path
                ):
                    graph._file_graph.add_edge(  # type: ignore[misc]
                        file_path,
                        imp.resolved_path,
                        kind=EdgeKind.IMPORTS,
                        location=f"{file_path}:{imp.line}",
                    )

        return graph

    def add_co_directory_edges(self) -> int:
        """Add bidirectional edges between files in the same directory.

        For languages where import resolution fails (Go, Rust), files in the
        same directory are implicitly co-dependent — they share the same package
        scope and can reference each other's symbols without import statements.

        Only adds edges where none exist yet.  Returns the number of edges added.
        """
        from collections import defaultdict

        dir_groups: defaultdict[str, list[str]] = defaultdict(list)
        for node in self._file_graph.nodes():  # type: ignore[misc]
            path = str(node)
            directory = path.rsplit("/", 1)[0] if "/" in path else ""
            dir_groups[directory].append(path)

        added = 0
        for files in dir_groups.values():
            if len(files) < 2:
                continue
            for i, src in enumerate(files):
                for tgt in files[i + 1 :]:
                    if not self._file_graph.has_edge(src, tgt):  # type: ignore[misc]
                        self._file_graph.add_edge(  # type: ignore[misc]
                            src, tgt, kind=EdgeKind.CO_DIRECTORY
                        )
                        added += 1
                    if not self._file_graph.has_edge(tgt, src):  # type: ignore[misc]
                        self._file_graph.add_edge(  # type: ignore[misc]
                            tgt, src, kind=EdgeKind.CO_DIRECTORY
                        )
                        added += 1

        if added > 0:
            self._centrality_cache = None

        return added

    @classmethod
    def from_edges(cls, edges: list[Edge]) -> DependencyGraph:
        """Reconstruct a file-level DependencyGraph from Edge objects."""
        graph = cls()
        for edge in edges:
            graph._file_graph.add_edge(  # type: ignore[misc]
                edge.source,
                edge.target,
                kind=edge.kind,
                location=edge.location,
            )
        return graph

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def add_file_node(self, path: str) -> None:
        """Add a file node to the file-level graph."""
        self._file_graph.add_node(path)  # type: ignore[misc]
        self._centrality_cache = None

    def add_file_edge(self, source: str, target: str, kind: str = "imports") -> None:
        """Add an edge to the file-level graph."""
        self._file_graph.add_edge(source, target, kind=kind)  # type: ignore[misc]
        self._centrality_cache = None

    def update_files(
        self,
        removed_paths: set[str],
        new_edges: list[Edge],
    ) -> None:
        """Incrementally update the file graph for changed files.

        Removes all nodes (and their incident edges) for files in removed_paths,
        then adds new edges (implicitly creating nodes). Invalidates centrality cache.

        Args:
            removed_paths: File paths to remove (modified + deleted files).
                Modified files are removed then re-added via new_edges.
            new_edges: Edges from re-parsed and newly added files.
        """
        for path in removed_paths:
            if self._file_graph.has_node(path):  # type: ignore[misc]
                self._file_graph.remove_node(path)  # type: ignore[misc]
            nodes_to_remove = [
                n
                for n in self._symbol_graph.nodes()  # type: ignore[misc]
                if self._symbol_graph.nodes[n].get("file_path") == path  # type: ignore[misc]
            ]
            for n in nodes_to_remove:
                self._symbol_graph.remove_node(n)  # type: ignore[misc]

        for edge in new_edges:
            self._file_graph.add_edge(  # type: ignore[misc]
                edge.source,
                edge.target,
                kind=edge.kind,
                location=edge.location,
            )

        self._centrality_cache = None

    @property
    def file_count(self) -> int:
        return self._file_graph.number_of_nodes()  # type: ignore[misc]

    @property
    def file_edge_count(self) -> int:
        return self._file_graph.number_of_edges()  # type: ignore[misc]

    @property
    def symbol_count(self) -> int:
        return self._symbol_graph.number_of_nodes()  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def file_edges(self) -> list[Edge]:
        """Return Edge objects for all file-level edges."""
        edges: list[Edge] = []
        for src, tgt, data in self._file_graph.edges(data=True):  # type: ignore[misc]
            src_str = str(src)
            tgt_str = str(tgt)
            edge_data: dict[str, Any] = data  # type: ignore[assignment]
            edges.append(
                Edge(
                    source=src_str,
                    target=tgt_str,
                    kind=edge_data.get("kind", EdgeKind.IMPORTS),
                    location=edge_data.get("location"),
                )
            )
        return edges

    def neighborhood(self, node: str, hops: int = 1) -> set[str]:
        """BFS in both directions up to N hops from node."""
        if not self._file_graph.has_node(node):  # type: ignore[misc]
            return set()

        visited: set[str] = set()
        frontier: set[str] = {node}

        for _ in range(hops):
            next_frontier: set[str] = set()
            for n in frontier:
                for pred in self._file_graph.predecessors(n):  # type: ignore[misc]
                    pred_str = str(pred)
                    if pred_str not in visited and pred_str != node:
                        next_frontier.add(pred_str)
                for succ in self._file_graph.successors(n):  # type: ignore[misc]
                    succ_str = str(succ)
                    if succ_str not in visited and succ_str != node:
                        next_frontier.add(succ_str)
            visited |= frontier
            frontier = next_frontier - visited

        visited |= frontier
        visited.discard(node)
        return visited

    def imports_of(self, node: str) -> set[str]:
        """Return files that *node* directly imports (successors in the directed graph)."""
        if not self._file_graph.has_node(node):  # type: ignore[misc]
            return set()
        return {str(s) for s in self._file_graph.successors(node)}  # type: ignore[misc]

    def imported_by(self, node: str) -> set[str]:
        """Return files that directly import *node* (predecessors in the directed graph)."""
        if not self._file_graph.has_node(node):  # type: ignore[misc]
            return set()
        return {str(p) for p in self._file_graph.predecessors(node)}  # type: ignore[misc]

    def structural_centrality(self) -> dict[str, float]:
        """Return PageRank centrality scores for all file nodes (lazily cached)."""
        if self._centrality_cache is not None:
            return self._centrality_cache
        if self._file_graph.number_of_nodes() == 0:  # type: ignore[misc]
            return {}
        raw: dict[Any, float] = nx.pagerank(self._file_graph)  # type: ignore[misc]
        self._centrality_cache = {str(k): v for k, v in raw.items()}
        return self._centrality_cache

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_sqlite(self, db_path: str | Path) -> None:
        """Persist file nodes and edges to SQLite."""
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY)")
            cur.execute(
                "CREATE TABLE IF NOT EXISTS edges "
                "(source TEXT, target TEXT, kind TEXT, location TEXT)"
            )
            cur.execute("DELETE FROM files")
            cur.execute("DELETE FROM edges")

            for node in self._file_graph.nodes():  # type: ignore[misc]
                cur.execute("INSERT INTO files (path) VALUES (?)", (str(node),))

            for src, tgt, data in self._file_graph.edges(data=True):  # type: ignore[misc]
                edge_data: dict[str, Any] = data  # type: ignore[assignment]
                cur.execute(
                    "INSERT INTO edges (source, target, kind, location) VALUES (?, ?, ?, ?)",
                    (
                        str(src),
                        str(tgt),
                        edge_data.get("kind", EdgeKind.IMPORTS),
                        edge_data.get("location"),
                    ),
                )

            conn.commit()
        finally:
            conn.close()

    @classmethod
    def from_sqlite(cls, db_path: str | Path) -> DependencyGraph:
        """Restore a DependencyGraph from SQLite."""
        graph = cls()
        conn = sqlite3.connect(str(db_path))
        try:
            cur = conn.cursor()
            for (path,) in cur.execute("SELECT path FROM files"):
                graph._file_graph.add_node(str(path))  # type: ignore[misc]
            for src, tgt, kind, location in cur.execute(
                "SELECT source, target, kind, location FROM edges"
            ):
                graph._file_graph.add_edge(str(src), str(tgt), kind=kind, location=location)  # type: ignore[misc]
        finally:
            conn.close()
        return graph
