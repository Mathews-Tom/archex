"""Graph-native structural query API for exported architecture graphs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from archex.graph_artifact import (
    ArchGraph,
    GraphEdge,
    GraphNode,
    build_arch_graph_from_store,
    file_node_id,
    load_arch_graph,
    node_id_for_path,
)
from archex.index.store import IndexStore

_T = TypeVar("_T")

GraphDirection = Literal["out", "in", "both"]
NodeMatchKind = Literal["id", "path", "label", "fuzzy"]

DEFAULT_GRAPH_LIMIT = 25
DEFAULT_HUB_DEGREE = 50


class GraphQueryError(ValueError):
    """Raised when a graph query cannot be answered."""


class GraphNodeSummary(BaseModel):
    id: str
    type: str
    label: str
    path: str | None = None
    language: str | None = None
    symbol_kind: str | None = None
    module: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    description: str = ""
    degree: int = 0


class GraphEdgeSummary(BaseModel):
    source: GraphNodeSummary
    target: GraphNodeSummary
    type: str
    label: str = ""
    location: str | None = None
    confidence: str
    confidence_score: float
    evidence: list[str] = Field(default_factory=list)


class GraphNodeLookupResult(BaseModel):
    query: str
    matches: list[GraphNodeSummary]
    match_kind: NodeMatchKind | None = None
    truncated: bool = False
    omitted: int = 0


class GraphNeighborsResult(BaseModel):
    query: str
    seed: GraphNodeSummary
    direction: GraphDirection
    depth: int
    edges: list[GraphEdgeSummary]
    traversed_nodes: list[GraphNodeSummary]
    hubs: list[GraphNodeSummary]
    truncated: bool = False
    omitted_edges: int = 0


class GraphPathResult(BaseModel):
    source_query: str
    target_query: str
    source: GraphNodeSummary | None
    target: GraphNodeSummary | None
    nodes: list[GraphNodeSummary]
    edges: list[GraphEdgeSummary]
    direction: GraphDirection
    found: bool
    avoided_hubs: list[GraphNodeSummary]
    truncated: bool = False


class GraphStatsResult(BaseModel):
    project: str
    nodes: int
    edges: int
    files: int
    languages: dict[str, int]
    edge_types: dict[str, int]
    max_degree: int
    hubs: list[GraphNodeSummary]


class GraphHubsResult(BaseModel):
    hubs: list[GraphNodeSummary]
    threshold: int
    limit: int
    truncated: bool = False
    omitted: int = 0


@dataclass(frozen=True)
class _AdjacentEdge:
    edge: GraphEdge
    neighbor_id: str


class GraphQuery:
    """Reusable handle for deterministic structural queries over an architecture graph."""

    def __init__(self, graph: ArchGraph, *, hub_degree: int = DEFAULT_HUB_DEGREE) -> None:
        if hub_degree < 1:
            raise GraphQueryError("hub_degree must be at least 1")
        self.graph = graph
        self.hub_degree = hub_degree
        self._nodes = _dedupe_nodes(graph.nodes)
        self._nodes_by_id = {node.id: node for node in self._nodes}
        self._outgoing = self._build_adjacency("out")
        self._incoming = self._build_adjacency("in")
        self._degree_by_id = {
            node.id: len(self._outgoing.get(node.id, ())) + len(self._incoming.get(node.id, ()))
            for node in self._nodes
        }
        self._path_to_ids = self._build_path_index()
        self._label_to_ids = self._build_label_index()

    @classmethod
    def from_artifact(cls, path: str | Path, *, hub_degree: int = DEFAULT_HUB_DEGREE) -> GraphQuery:
        return cls(load_arch_graph(Path(path)), hub_degree=hub_degree)

    @classmethod
    def from_store(
        cls,
        store: IndexStore,
        *,
        repo_root: Path | None = None,
        hub_degree: int = DEFAULT_HUB_DEGREE,
    ) -> GraphQuery:
        return cls(build_arch_graph_from_store(store, repo_root=repo_root), hub_degree=hub_degree)

    @classmethod
    def from_index(
        cls,
        index_path: str | Path,
        *,
        repo_root: Path | None = None,
        hub_degree: int = DEFAULT_HUB_DEGREE,
    ) -> GraphQuery:
        path = Path(index_path)
        if not path.exists():
            raise GraphQueryError(f"Index path does not exist: {path}")
        with IndexStore(path) as store:
            return cls.from_store(store, repo_root=repo_root, hub_degree=hub_degree)

    def lookup(self, query: str, *, limit: int = DEFAULT_GRAPH_LIMIT) -> GraphNodeLookupResult:
        self._validate_limit(limit)
        node, match_kind = self._resolve_exact(query)
        if node is not None:
            return GraphNodeLookupResult(
                query=query,
                matches=[self._summarize_node(node)],
                match_kind=match_kind,
            )

        needle = _normalize_lookup(query).casefold()
        matches = [
            node
            for node in self._nodes
            if needle
            and (
                needle in node.id.casefold()
                or needle in node.label.casefold()
                or needle in (node.path or "").casefold()
            )
        ]
        matches = sorted(matches, key=self._node_sort_key)
        selected, omitted = _truncate(matches, limit)
        return GraphNodeLookupResult(
            query=query,
            matches=[self._summarize_node(node) for node in selected],
            match_kind="fuzzy" if selected else None,
            truncated=omitted > 0,
            omitted=omitted,
        )

    def neighbors(
        self,
        query: str,
        *,
        direction: GraphDirection = "both",
        depth: int = 1,
        limit: int = DEFAULT_GRAPH_LIMIT,
    ) -> GraphNeighborsResult:
        self._validate_direction(direction)
        self._validate_limit(limit)
        if depth < 1:
            raise GraphQueryError("depth must be at least 1")
        seed = self._require_one_node(query)
        seed_id = seed.id
        seen_nodes = {seed_id}
        hubs: set[str] = set()
        selected_edges: list[GraphEdge] = []
        omitted_edges = 0
        queue: deque[tuple[str, int]] = deque([(seed_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            is_seed = current_id == seed_id
            if not is_seed and self._is_hub(current_id):
                hubs.add(current_id)
                continue
            adjacent = self._adjacent(current_id, direction)
            remaining = limit - len(selected_edges)
            if remaining <= 0:
                omitted_edges += len(adjacent)
                continue
            selected, omitted = _truncate(adjacent, remaining)
            omitted_edges += omitted
            for item in selected:
                selected_edges.append(item.edge)
                if item.neighbor_id not in seen_nodes:
                    seen_nodes.add(item.neighbor_id)
                    if self._is_hub(item.neighbor_id) and item.neighbor_id != seed_id:
                        hubs.add(item.neighbor_id)
                    else:
                        queue.append((item.neighbor_id, current_depth + 1))

        traversed = sorted(
            (self._nodes_by_id[node_id] for node_id in seen_nodes),
            key=self._node_sort_key,
        )
        return GraphNeighborsResult(
            query=query,
            seed=self._summarize_node(seed),
            direction=direction,
            depth=depth,
            edges=[self._summarize_edge(edge) for edge in selected_edges],
            traversed_nodes=[self._summarize_node(node) for node in traversed],
            hubs=[self._summarize_node(self._nodes_by_id[node_id]) for node_id in sorted(hubs)],
            truncated=omitted_edges > 0,
            omitted_edges=omitted_edges,
        )

    def shortest_path(
        self,
        source_query: str,
        target_query: str,
        *,
        direction: GraphDirection = "both",
        max_edges: int = 100,
    ) -> GraphPathResult:
        self._validate_direction(direction)
        self._validate_limit(max_edges)
        source = self._require_one_node(source_query)
        target = self._require_one_node(target_query)
        source_id = source.id
        target_id = target.id
        avoided_hubs: set[str] = set()
        parents: dict[str, tuple[str, GraphEdge]] = {}
        visited = {source_id}
        queue: deque[str] = deque([source_id])
        expanded_edges = 0
        truncated = False

        while queue:
            current_id = queue.popleft()
            if current_id == target_id:
                break
            if current_id not in {source_id, target_id} and self._is_hub(current_id):
                avoided_hubs.add(current_id)
                continue
            for item in self._adjacent(current_id, direction):
                expanded_edges += 1
                if expanded_edges > max_edges:
                    truncated = True
                    queue.clear()
                    break
                neighbor_id = item.neighbor_id
                if neighbor_id in visited:
                    continue
                if (
                    neighbor_id != target_id
                    and neighbor_id != source_id
                    and self._is_hub(neighbor_id)
                ):
                    avoided_hubs.add(neighbor_id)
                    continue
                visited.add(neighbor_id)
                parents[neighbor_id] = (current_id, item.edge)
                queue.append(neighbor_id)
        if target_id not in visited:
            return GraphPathResult(
                source_query=source_query,
                target_query=target_query,
                source=self._summarize_node(source),
                target=self._summarize_node(target),
                nodes=[],
                edges=[],
                direction=direction,
                found=False,
                avoided_hubs=[
                    self._summarize_node(self._nodes_by_id[node_id])
                    for node_id in sorted(avoided_hubs)
                ],
                truncated=truncated,
            )

        path_node_ids = [target_id]
        path_edges: list[GraphEdge] = []
        cursor = target_id
        while cursor != source_id:
            parent_id, edge = parents[cursor]
            path_edges.append(edge)
            path_node_ids.append(parent_id)
            cursor = parent_id
        path_node_ids.reverse()
        path_edges.reverse()
        return GraphPathResult(
            source_query=source_query,
            target_query=target_query,
            source=self._summarize_node(source),
            target=self._summarize_node(target),
            nodes=[self._summarize_node(self._nodes_by_id[node_id]) for node_id in path_node_ids],
            edges=[self._summarize_edge(edge) for edge in path_edges],
            direction=direction,
            found=True,
            avoided_hubs=[
                self._summarize_node(self._nodes_by_id[node_id]) for node_id in sorted(avoided_hubs)
            ],
            truncated=truncated,
        )

    def stats(self, *, hub_limit: int = 10) -> GraphStatsResult:
        self._validate_limit(hub_limit)
        edge_types: dict[str, int] = {}
        for edge in self.graph.edges:
            edge_types[edge.type.value] = edge_types.get(edge.type.value, 0) + 1
        max_degree = max(self._degree_by_id.values(), default=0)
        return GraphStatsResult(
            project=self.graph.project.name,
            nodes=len(self._nodes),
            edges=len(self.graph.edges),
            files=self.graph.project.total_files,
            languages=self.graph.project.languages,
            edge_types=dict(sorted(edge_types.items())),
            max_degree=max_degree,
            hubs=self.hubs(limit=hub_limit).hubs,
        )

    def hubs(
        self,
        *,
        limit: int = DEFAULT_GRAPH_LIMIT,
        threshold: int | None = None,
    ) -> GraphHubsResult:
        self._validate_limit(limit)
        effective_threshold = threshold if threshold is not None else self.hub_degree
        if effective_threshold < 1:
            raise GraphQueryError("threshold must be at least 1")
        hubs = [
            self._nodes_by_id[node_id]
            for node_id, degree in self._degree_by_id.items()
            if degree >= effective_threshold
        ]
        hubs = sorted(
            hubs,
            key=lambda node: (-self._degree_by_id[node.id], node.type.value, node.id),
        )
        selected, omitted = _truncate(hubs, limit)
        return GraphHubsResult(
            hubs=[self._summarize_node(node) for node in selected],
            threshold=effective_threshold,
            limit=limit,
            truncated=omitted > 0,
            omitted=omitted,
        )

    def _build_adjacency(
        self,
        direction: Literal["out", "in"],
    ) -> dict[str, tuple[_AdjacentEdge, ...]]:
        adjacency: dict[str, list[_AdjacentEdge]] = {node.id: [] for node in self._nodes}
        for edge in self.graph.edges:
            if edge.source not in self._nodes_by_id or edge.target not in self._nodes_by_id:
                continue
            if direction == "out":
                adjacency[edge.source].append(_AdjacentEdge(edge=edge, neighbor_id=edge.target))
            else:
                adjacency[edge.target].append(_AdjacentEdge(edge=edge, neighbor_id=edge.source))
        return {
            node_id: tuple(sorted(edges, key=self._adjacent_sort_key))
            for node_id, edges in adjacency.items()
        }

    def _build_path_index(self) -> dict[str, tuple[str, ...]]:
        index: dict[str, list[str]] = {}
        for node in self._nodes:
            if node.path is not None:
                index.setdefault(_normalize_lookup(node.path), []).append(node.id)
        return {path: tuple(sorted(ids)) for path, ids in index.items()}

    def _build_label_index(self) -> dict[str, tuple[str, ...]]:
        index: dict[str, list[str]] = {}
        for node in self._nodes:
            index.setdefault(node.label, []).append(node.id)
        return {label: tuple(sorted(ids)) for label, ids in index.items()}

    def _resolve_exact(self, query: str) -> tuple[GraphNode | None, NodeMatchKind | None]:
        normalized = _normalize_lookup(query)
        if query in self._nodes_by_id:
            return self._nodes_by_id[query], "id"
        node_id = node_id_for_path(normalized)
        if node_id in self._nodes_by_id:
            return self._nodes_by_id[node_id], "path"
        file_id = file_node_id(normalized)
        if file_id in self._nodes_by_id:
            return self._nodes_by_id[file_id], "path"
        path_ids = self._path_to_ids.get(normalized)
        if path_ids is not None and len(path_ids) == 1:
            return self._nodes_by_id[path_ids[0]], "path"
        label_ids = self._label_to_ids.get(query)
        if label_ids is not None and len(label_ids) == 1:
            return self._nodes_by_id[label_ids[0]], "label"
        return None, None

    def _require_one_node(self, query: str) -> GraphNode:
        exact, _match_kind = self._resolve_exact(query)
        if exact is not None:
            return exact
        lookup = self.lookup(query, limit=2)
        if not lookup.matches:
            raise GraphQueryError(f"No graph node matches {query!r}")
        if len(lookup.matches) > 1:
            ids = ", ".join(match.id for match in lookup.matches)
            raise GraphQueryError(f"Graph node query {query!r} is ambiguous: {ids}")
        return self._nodes_by_id[lookup.matches[0].id]

    def _adjacent(self, node_id: str, direction: GraphDirection) -> tuple[_AdjacentEdge, ...]:
        if direction == "out":
            return self._outgoing.get(node_id, ())
        if direction == "in":
            return self._incoming.get(node_id, ())
        return tuple(
            sorted(
                self._outgoing.get(node_id, ()) + self._incoming.get(node_id, ()),
                key=self._adjacent_sort_key,
            )
        )

    def _is_hub(self, node_id: str) -> bool:
        return self._degree_by_id[node_id] >= self.hub_degree

    def _summarize_node(self, node: GraphNode) -> GraphNodeSummary:
        return GraphNodeSummary(
            id=node.id,
            type=node.type.value,
            label=node.label,
            path=node.path,
            language=node.language,
            symbol_kind=node.symbol_kind,
            module=node.module,
            line_start=node.line_start,
            line_end=node.line_end,
            description=node.description,
            degree=self._degree_by_id[node.id],
        )

    def _summarize_edge(self, edge: GraphEdge) -> GraphEdgeSummary:
        return GraphEdgeSummary(
            source=self._summarize_node(self._nodes_by_id[edge.source]),
            target=self._summarize_node(self._nodes_by_id[edge.target]),
            type=edge.type.value,
            label=edge.label,
            location=edge.location,
            confidence=edge.confidence.value,
            confidence_score=edge.confidence_score,
            evidence=list(edge.evidence),
        )

    def _node_sort_key(self, node: GraphNode) -> tuple[str, str, str]:
        return (node.path or "", node.type.value, node.id)

    def _adjacent_sort_key(self, item: _AdjacentEdge) -> tuple[str, str, str, str]:
        edge = item.edge
        return (edge.source, edge.target, edge.type.value, edge.location or "")

    @staticmethod
    def _validate_direction(direction: GraphDirection) -> None:
        if direction not in {"out", "in", "both"}:
            raise GraphQueryError(f"Unsupported graph direction {direction!r}")

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if limit < 1:
            raise GraphQueryError("limit must be at least 1")


def _dedupe_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    selected: dict[str, GraphNode] = {}
    for node in sorted(nodes, key=_dedupe_node_sort_key):
        selected.setdefault(node.id, node)
    return list(selected.values())


def _dedupe_node_sort_key(node: GraphNode) -> tuple[str, str, str, int, int]:
    return (
        node.type.value,
        node.id,
        node.path or "",
        node.line_start or 0,
        node.line_end or 0,
    )


def _normalize_lookup(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def _truncate(items: list[_T] | tuple[_T, ...], limit: int) -> tuple[list[_T], int]:
    selected = list(items[:limit])
    return selected, max(0, len(items) - limit)
