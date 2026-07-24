"""Dependency graph construction: build and query the file/symbol edge graph."""

from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

import networkx as nx

from archex.integrations.semantic.models import SemanticEdgeKind
from archex.models import Edge, EdgeConfidence, EdgeKind, ImportStatement, ParsedFile

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.semantic.models import SemanticEdgeEvidence

_CO_DIRECTORY_CONFIDENCE_SCORE = 0.6
_CO_DIRECTORY_EVIDENCE = [
    "same directory package scope; added only when no direct import edge exists"
]

# Directories at or below this file count get full pairwise co-directory
# edges (every file linked to every other, both directions). Above it,
# each file only links to a bounded forward window of neighbors instead —
# a flat directory of 500 files would otherwise add ~250k edges (O(n^2));
# windowing bounds growth to O(n * window) while preserving local
# same-directory connectivity for languages that rely on this fallback.
_CO_DIRECTORY_DENSE_THRESHOLD = 50
_CO_DIRECTORY_WINDOW_SIZE = 20

_SEMANTIC_EDGE_KIND: dict[SemanticEdgeKind, EdgeKind] = {
    SemanticEdgeKind.DEFINITION: EdgeKind.SEMANTIC_DEFINITION,
    SemanticEdgeKind.REFERENCE: EdgeKind.SEMANTIC_REFERENCE,
    SemanticEdgeKind.IMPLEMENTATION: EdgeKind.SEMANTIC_IMPLEMENTATION,
}


def _resolved_import_evidence(file_path: str, imported_module: str, line: int) -> list[str]:
    return [f"resolved import {imported_module!r} at {file_path}:{line}"]


def _edge_attrs(
    *,
    kind: EdgeKind | str,
    location: str | None = None,
    confidence: EdgeConfidence = EdgeConfidence.EXTRACTED,
    confidence_score: float = 1.0,
    evidence: list[str] | None = None,
    provider: str | None = None,
    provider_version: str | None = None,
) -> dict[str, object]:
    return {
        "kind": kind,
        "location": location,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "evidence": list(evidence or []),
        "provider": provider,
        "provider_version": provider_version,
        "traversable": confidence != EdgeConfidence.AMBIGUOUS,
    }


def _edge_from_data(source: object, target: object, data: dict[str, Any]) -> Edge:
    return Edge(
        source=str(source),
        target=str(target),
        kind=data.get("kind", EdgeKind.IMPORTS),
        location=data.get("location"),
        confidence=data.get("confidence", EdgeConfidence.EXTRACTED),
        confidence_score=float(data.get("confidence_score", 1.0)),
        evidence=list(data.get("evidence", [])),
        provider=data.get("provider"),
        provider_version=data.get("provider_version"),
    )


def _ensure_sqlite_edge_confidence_columns(conn: sqlite3.Connection) -> None:
    edge_columns = {row[1] for row in conn.execute("PRAGMA table_info(edges)")}
    edge_migrations = {
        "confidence": "ALTER TABLE edges ADD COLUMN confidence TEXT DEFAULT 'extracted'",
        "confidence_score": "ALTER TABLE edges ADD COLUMN confidence_score REAL DEFAULT 1.0",
        "evidence": "ALTER TABLE edges ADD COLUMN evidence TEXT DEFAULT '[]'",
        "provider": "ALTER TABLE edges ADD COLUMN provider TEXT",
        "provider_version": "ALTER TABLE edges ADD COLUMN provider_version TEXT",
    }
    migrated = False
    for column, statement in edge_migrations.items():
        if column not in edge_columns:
            conn.execute(statement)
            migrated = True
    if migrated:
        conn.commit()


class DependencyGraph:
    """Wraps two nx.DiGraph instances: one for files, one for symbols."""

    def __init__(self) -> None:
        self._file_graph: nx.DiGraph[str] = nx.DiGraph()  # type: ignore[type-arg]
        self._symbol_graph: nx.DiGraph[str] = nx.DiGraph()  # type: ignore[type-arg]
        self._centrality_cache: dict[str, float] | None = None

    def _invalidate_centrality_caches(self) -> None:
        self._centrality_cache = None

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
                        **_edge_attrs(
                            kind=EdgeKind.IMPORTS,
                            location=f"{file_path}:{imp.line}",
                            evidence=_resolved_import_evidence(file_path, imp.module, imp.line),
                        ),
                    )
                else:
                    graph._file_graph.add_node(f"unresolved://{imp.module}")  # type: ignore[misc]
                    graph._file_graph.add_edge(  # type: ignore[misc]
                        file_path,
                        f"unresolved://{imp.module}",
                        **_edge_attrs(
                            kind=EdgeKind.IMPORTS,
                            location=f"{file_path}:{imp.line}",
                            confidence=EdgeConfidence.AMBIGUOUS,
                            evidence=[
                                f"unresolved import {imp.module!r} at {file_path}:{imp.line}"
                            ],
                        ),
                    )

        return graph

    def add_co_directory_edges(self) -> int:
        """Add bidirectional edges between files in the same directory.

        For languages where import resolution fails (Go, Rust), files in the
        same directory are implicitly co-dependent — they share the same package
        scope and can reference each other's symbols without import statements.

        Directories with more than `_CO_DIRECTORY_DENSE_THRESHOLD` files skip
        full pairwise edge generation in favor of a bounded per-file window
        (`_CO_DIRECTORY_WINDOW_SIZE`), so a large flat directory cannot produce
        a near-complete graph.

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
            dense = len(files) <= _CO_DIRECTORY_DENSE_THRESHOLD
            for i, src in enumerate(files):
                targets = (
                    files[i + 1 :] if dense else files[i + 1 : i + 1 + _CO_DIRECTORY_WINDOW_SIZE]
                )
                for tgt in targets:
                    if not self._file_graph.has_edge(src, tgt):  # type: ignore[misc]
                        self._file_graph.add_edge(  # type: ignore[misc]
                            src,
                            tgt,
                            **_edge_attrs(
                                kind=EdgeKind.CO_DIRECTORY,
                                confidence=EdgeConfidence.HEURISTIC,
                                confidence_score=_CO_DIRECTORY_CONFIDENCE_SCORE,
                                evidence=_CO_DIRECTORY_EVIDENCE,
                            ),
                        )
                        added += 1
                    if not self._file_graph.has_edge(tgt, src):  # type: ignore[misc]
                        self._file_graph.add_edge(  # type: ignore[misc]
                            tgt,
                            src,
                            **_edge_attrs(
                                kind=EdgeKind.CO_DIRECTORY,
                                confidence=EdgeConfidence.HEURISTIC,
                                confidence_score=_CO_DIRECTORY_CONFIDENCE_SCORE,
                                evidence=_CO_DIRECTORY_EVIDENCE,
                            ),
                        )
                        added += 1

        if added > 0:
            self._invalidate_centrality_caches()

        return added

    def add_semantic_edges(self, evidence: list[SemanticEdgeEvidence]) -> int:
        """Add conditional SCIP/LSP semantic edges (M6) to the file graph.

        Only connects source/target file paths that already exist as file
        nodes — semantic evidence enriches the syntax graph's existing
        corpus, it never introduces a file the syntax pass didn't already
        discover. Every added edge carries its provider, provider version,
        and confidence, and is a distinct EdgeKind from every syntax edge.

        The underlying graph allows at most one edge per (source, target)
        file pair, so a pair that already has ANY edge — syntax or a prior
        semantic edge from this same call — is left untouched rather than
        overwritten; syntax evidence authority is never replaced, and a
        second semantic relationship between the same two files with the
        same direction is skipped rather than silently clobbering the
        first. Returns the number of edges added.
        """
        added = 0
        for item in evidence:
            source_path = item.source.file_path
            target_path = item.target.file_path
            if not self._file_graph.has_node(source_path):  # type: ignore[misc]
                continue
            if not self._file_graph.has_node(target_path):  # type: ignore[misc]
                continue
            if self._file_graph.has_edge(source_path, target_path):  # type: ignore[misc]
                continue
            self._file_graph.add_edge(  # type: ignore[misc]
                source_path,
                target_path,
                **_edge_attrs(
                    kind=_SEMANTIC_EDGE_KIND[item.kind],
                    location=(
                        f"{source_path}:{item.source.line} -> {target_path}:{item.target.line}"
                    ),
                    confidence=EdgeConfidence.EXTRACTED,
                    confidence_score=item.confidence,
                    evidence=[
                        f"{item.provider.value} v{item.provider_version} {item.kind.value}: "
                        f"{source_path}:{item.source.line} -> {target_path}:{item.target.line}"
                    ],
                    provider=item.provider.value,
                    provider_version=item.provider_version,
                ),
            )
            added += 1

        if added > 0:
            self._invalidate_centrality_caches()

        return added

    @classmethod
    def from_edges(cls, edges: list[Edge]) -> DependencyGraph:
        """Reconstruct a file-level DependencyGraph from Edge objects."""
        graph = cls()
        for edge in edges:
            graph._file_graph.add_edge(  # type: ignore[misc]
                edge.source,
                edge.target,
                **_edge_attrs(
                    kind=edge.kind,
                    location=edge.location,
                    confidence=edge.confidence,
                    confidence_score=edge.confidence_score,
                    evidence=edge.evidence,
                    provider=edge.provider,
                    provider_version=edge.provider_version,
                ),
            )
        return graph

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def add_file_node(self, path: str) -> None:
        """Add a file node to the file-level graph."""
        self._file_graph.add_node(path)  # type: ignore[misc]
        self._invalidate_centrality_caches()

    def add_file_edge(self, source: str, target: str, kind: str = "imports") -> None:
        """Add an edge to the file-level graph."""
        self._file_graph.add_edge(source, target, **_edge_attrs(kind=kind))  # type: ignore[misc]
        self._invalidate_centrality_caches()

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
                **_edge_attrs(
                    kind=edge.kind,
                    location=edge.location,
                    confidence=edge.confidence,
                    confidence_score=edge.confidence_score,
                    evidence=edge.evidence,
                    provider=edge.provider,
                    provider_version=edge.provider_version,
                ),
            )

        self._invalidate_centrality_caches()

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
            edge_data: dict[str, Any] = data  # type: ignore[assignment]
            edges.append(_edge_from_data(src, tgt, edge_data))
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
                for pred, _, data in self._file_graph.in_edges(n, data=True):  # type: ignore[misc]
                    if data.get("traversable", True):
                        pred_str = str(pred)
                        if pred_str not in visited and pred_str != node:
                            next_frontier.add(pred_str)
                for _, succ, data in self._file_graph.out_edges(n, data=True):  # type: ignore[misc]
                    if data.get("traversable", True):
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
        return {
            str(succ)
            for _, succ, data in self._file_graph.out_edges(node, data=True)  # type: ignore[misc]
            if data.get("traversable", True)
        }

    def imported_by(self, node: str) -> set[str]:
        """Return files that directly import *node* (predecessors in the directed graph)."""
        if not self._file_graph.has_node(node):  # type: ignore[misc]
            return set()
        return {
            str(pred)
            for pred, _, data in self._file_graph.in_edges(node, data=True)  # type: ignore[misc]
            if data.get("traversable", True)
        }

    def structural_centrality(self) -> dict[str, float]:
        """Return PageRank centrality scores for all file nodes (lazily cached)."""
        if self._centrality_cache is not None:
            return self._centrality_cache
        if self._file_graph.number_of_nodes() == 0:  # type: ignore[misc]
            return {}
        traversable_edges = [
            (src, tgt)
            for src, tgt, data in self._file_graph.edges(data=True)  # type: ignore[misc]
            if data.get("traversable", True)
        ]
        graph = self._file_graph.edge_subgraph(traversable_edges).copy()  # type: ignore[misc]
        graph.add_nodes_from(self._file_graph.nodes())  # type: ignore[misc]
        raw: dict[Any, float] = nx.pagerank(graph)  # type: ignore[misc]
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
                "(source TEXT, target TEXT, kind TEXT, location TEXT, "
                "confidence TEXT, confidence_score REAL, evidence TEXT, "
                "provider TEXT, provider_version TEXT)"
            )
            _ensure_sqlite_edge_confidence_columns(conn)
            cur.execute("DELETE FROM files")
            cur.execute("DELETE FROM edges")

            for node in self._file_graph.nodes():  # type: ignore[misc]
                cur.execute("INSERT INTO files (path) VALUES (?)", (str(node),))

            for src, tgt, data in self._file_graph.edges(data=True):  # type: ignore[misc]
                edge_data: dict[str, Any] = data  # type: ignore[assignment]
                cur.execute(
                    "INSERT INTO edges "
                    "(source, target, kind, location, confidence, confidence_score, evidence, "
                    "provider, provider_version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(src),
                        str(tgt),
                        edge_data.get("kind", EdgeKind.IMPORTS),
                        edge_data.get("location"),
                        edge_data.get("confidence", EdgeConfidence.EXTRACTED),
                        float(edge_data.get("confidence_score", 1.0)),
                        json.dumps(edge_data.get("evidence", []), sort_keys=True),
                        edge_data.get("provider"),
                        edge_data.get("provider_version"),
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
            _ensure_sqlite_edge_confidence_columns(conn)
            for (path,) in cur.execute("SELECT path FROM files"):
                graph._file_graph.add_node(str(path))  # type: ignore[misc]
            for (
                src,
                tgt,
                kind,
                location,
                confidence,
                confidence_score,
                evidence,
                provider,
                provider_version,
            ) in cur.execute(
                "SELECT source, target, kind, location, confidence, confidence_score, evidence, "
                "provider, provider_version FROM edges"
            ):
                graph._file_graph.add_edge(  # type: ignore[misc]
                    str(src),
                    str(tgt),
                    **_edge_attrs(
                        kind=EdgeKind(kind),
                        location=location,
                        confidence=EdgeConfidence(confidence),
                        confidence_score=float(confidence_score),
                        evidence=json.loads(str(evidence)),
                        provider=provider,
                        provider_version=provider_version,
                    ),
                )
        finally:
            conn.close()
        return graph
