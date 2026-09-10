"""Pure, deterministic view models projected from `ExplorerData`.

Every builder in this module is a pure function over already-loaded
artifacts (`archex.explorer.loader.ExplorerData`): no file I/O, no network
access, no repository indexing, and no new graph-edge construction. Bounded
list fields mirror the `*_total` convention `archex.report.artifact` and
`archex.graph_query` already use, so every view can show "N of TOTAL" rather
than silently truncating.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from archex.graph_artifact import GraphEdgeType
from archex.graph_query import DEFAULT_GRAPH_LIMIT, GraphQuery, GraphQueryError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from archex.explorer.loader import ExplorerData
    from archex.graph_query import GraphDirection, GraphEdgeSummary, GraphNodeSummary
    from archex.report.artifact import EvidenceLocation

MAX_DIFF_FILE_ROWS = 100
MAX_SYMBOL_CANDIDATE_ROWS = 100
MAX_INTERFACE_CANDIDATE_ROWS = 100
MAX_TEST_CANDIDATE_ROWS = 100
MAX_UNSUPPORTED_FILE_ROWS = 100


@dataclass(frozen=True)
class ManifestView:
    """The cross-cutting provenance banner every explorer page renders.

    Satisfies M5's acceptance that "all views display artifact provenance,
    freshness/completeness, exclusions, unknowns, and evidence paths" without
    duplicating the full receipt (see `ReceiptView`).
    """

    source_identity: str
    source_revision: str
    archex_version: str
    schema_version: str
    generated_at: str
    freshness: str
    completeness: str
    confidence: str
    redaction_mode: str
    has_graph: bool
    excluded_total: int
    unknown_total: int
    evidence_count: int


def build_manifest_view(data: ExplorerData) -> ManifestView:
    artifact = data.artifact
    return ManifestView(
        source_identity=artifact.source_identity,
        source_revision=artifact.source_revision,
        archex_version=artifact.archex_version,
        schema_version=artifact.schema_version.value,
        generated_at=artifact.generated_at,
        freshness=artifact.freshness.value,
        completeness=artifact.completeness.value,
        confidence=artifact.confidence.value,
        redaction_mode=artifact.redaction_mode.value,
        has_graph=data.graph is not None,
        excluded_total=sum(artifact.excluded_counts.values()),
        unknown_total=sum(artifact.unknown_counts.values()),
        evidence_count=len(artifact.evidence_locations),
    )


@dataclass(frozen=True)
class DiffHunkRow:
    start_line: int
    end_line: int


@dataclass(frozen=True)
class DiffFileRow:
    path: str
    status: str
    handle: str
    old_path: str | None
    hunks: list[DiffHunkRow]


@dataclass(frozen=True)
class SymbolCandidateRow:
    handle: str
    file_path: str
    label: str
    symbol_kind: str | None
    start_line: int
    end_line: int
    risk_level: str
    confidence: str
    signals: list[str]


@dataclass(frozen=True)
class InterfaceCandidateRow:
    path: str
    symbol_id: str
    handle: str
    confidence: str


@dataclass(frozen=True)
class TestCandidateRow:
    path: str
    handle: str
    reason: str
    confidence: str


@dataclass(frozen=True)
class UnsupportedFileRow:
    path: str
    reason: str


@dataclass(frozen=True)
class DiffView:
    base_ref: str
    base_resolved_sha: str
    head_ref: str
    risk_level: str
    risk_reasons: list[str]

    changed_files: list[DiffFileRow]
    changed_files_total: int
    symbol_candidates: list[SymbolCandidateRow]
    symbol_candidates_total: int
    affected_interfaces: list[InterfaceCandidateRow]
    affected_interfaces_total: int
    test_candidates: list[TestCandidateRow]
    test_candidates_total: int
    unsupported_files: list[UnsupportedFileRow]
    unsupported_files_total: int


def build_diff_view(data: ExplorerData) -> DiffView:
    diff = data.artifact.diff
    return DiffView(
        base_ref=diff.base_ref,
        base_resolved_sha=diff.base_resolved_sha,
        head_ref=diff.head_ref,
        risk_level=diff.risk_level.value,
        risk_reasons=list(diff.risk_reasons),
        changed_files=[
            DiffFileRow(
                path=change.path,
                status=change.status,
                handle=change.handle,
                old_path=change.old_path,
                hunks=[
                    DiffHunkRow(start_line=hunk.start_line, end_line=hunk.end_line)
                    for hunk in change.hunks
                ],
            )
            for change in diff.changed_files[:MAX_DIFF_FILE_ROWS]
        ],
        changed_files_total=diff.changed_files_total,
        symbol_candidates=[
            SymbolCandidateRow(
                handle=candidate.handle,
                file_path=candidate.file_path,
                label=candidate.qualified_name or candidate.symbol_name or "<unnamed>",
                symbol_kind=candidate.symbol_kind,
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                risk_level=candidate.risk_level,
                confidence=candidate.confidence.value,
                signals=list(candidate.signals),
            )
            for candidate in diff.symbol_candidates[:MAX_SYMBOL_CANDIDATE_ROWS]
        ],
        symbol_candidates_total=diff.symbol_candidates_total,
        affected_interfaces=[
            InterfaceCandidateRow(
                path=interface.path,
                symbol_id=interface.symbol_id,
                handle=interface.handle,
                confidence=interface.confidence.value,
            )
            for interface in diff.affected_interfaces[:MAX_INTERFACE_CANDIDATE_ROWS]
        ],
        affected_interfaces_total=diff.affected_interfaces_total,
        test_candidates=[
            TestCandidateRow(
                path=test.path,
                handle=test.handle,
                reason=test.reason,
                confidence=test.confidence.value,
            )
            for test in diff.test_candidates[:MAX_TEST_CANDIDATE_ROWS]
        ],
        test_candidates_total=diff.test_candidates_total,
        unsupported_files=[
            UnsupportedFileRow(path=unsupported.path, reason=unsupported.reason)
            for unsupported in diff.unsupported_files[:MAX_UNSUPPORTED_FILE_ROWS]
        ],
        unsupported_files_total=diff.unsupported_files_total,
    )


def evidence_rows(evidence: list[EvidenceLocation], *, limit: int) -> list[EvidenceLocation]:
    """Shared bounded-slice helper so every view truncates evidence identically."""
    return evidence[:limit]


@dataclass(frozen=True)
class ReceiptView:
    """Is the artifact fresh, complete, and evidenced? (DEVELOPMENT_PLAN M5)."""

    freshness: str
    completeness: str
    confidence: str
    redaction_mode: str
    generated_at: str
    evidence_locations: list[EvidenceLocation]
    evidence_locations_total: int
    excluded_counts: dict[str, int]
    unknown_counts: dict[str, int]


MAX_EVIDENCE_ROWS = 100


def build_receipt_view(data: ExplorerData) -> ReceiptView:
    artifact = data.artifact
    return ReceiptView(
        freshness=artifact.freshness.value,
        completeness=artifact.completeness.value,
        confidence=artifact.confidence.value,
        redaction_mode=artifact.redaction_mode.value,
        generated_at=artifact.generated_at,
        evidence_locations=evidence_rows(artifact.evidence_locations, limit=MAX_EVIDENCE_ROWS),
        evidence_locations_total=len(artifact.evidence_locations),
        excluded_counts=dict(artifact.excluded_counts),
        unknown_counts=dict(artifact.unknown_counts),
    )


@dataclass(frozen=True)
class HealthView:
    """Is this evidence trustworthy? Index/parser/config identity (DEVELOPMENT_PLAN M5)."""

    archex_version: str
    schema_version: str
    index_generation: str
    index_schema_version: str
    chunker_revision: str
    parser_versions: dict[str, str]
    retrieval_profile: str | None
    config_fingerprint: str
    working_tree_fingerprint: str
    producer: str
    producer_version: str


def build_health_view(data: ExplorerData) -> HealthView:
    artifact = data.artifact
    return HealthView(
        archex_version=artifact.archex_version,
        schema_version=artifact.schema_version.value,
        index_generation=artifact.index_generation,
        index_schema_version=artifact.index_schema_version,
        chunker_revision=artifact.chunker_revision,
        parser_versions=dict(artifact.parser_versions),
        retrieval_profile=artifact.retrieval_profile,
        config_fingerprint=artifact.config_fingerprint,
        working_tree_fingerprint=artifact.working_tree_fingerprint,
        producer=artifact.producer,
        producer_version=artifact.producer_version,
    )


MAX_MODULE_ROWS = 200
_UNASSIGNED_MODULE = "(unassigned)"


@dataclass(frozen=True)
class ModuleRow:
    module: str
    node_count: int
    file_count: int
    symbol_count: int
    interface_count: int


@dataclass(frozen=True)
class ModuleMapView:
    """Where should I start? Module-aggregated node counts, not a force graph.

    Default graph presentation is aggregation, never the raw per-node/edge
    graph -- `build_neighborhood_view` is the only view that projects
    individual nodes/edges, and only a bounded neighborhood of them.
    """

    available: bool
    modules: list[ModuleRow]
    modules_total: int


def build_module_map_view(data: ExplorerData, *, limit: int = MAX_MODULE_ROWS) -> ModuleMapView:
    if data.graph is None:
        return ModuleMapView(available=False, modules=[], modules_total=0)

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"node": 0, "file": 0, "symbol": 0, "interface": 0}
    )
    for node in data.graph.nodes:
        module = node.module or _UNASSIGNED_MODULE
        counts[module]["node"] += 1
        if node.type.value == "file":
            counts[module]["file"] += 1
        elif node.type.value == "symbol":
            counts[module]["symbol"] += 1
        elif node.type.value == "interface":
            counts[module]["interface"] += 1

    rows = sorted(
        (
            ModuleRow(
                module=module,
                node_count=stats["node"],
                file_count=stats["file"],
                symbol_count=stats["symbol"],
                interface_count=stats["interface"],
            )
            for module, stats in counts.items()
        ),
        key=lambda row: (-row.node_count, row.module),
    )
    return ModuleMapView(available=True, modules=rows[:limit], modules_total=len(rows))


DEFAULT_NEIGHBORHOOD_DEPTH = 1
DEFAULT_NEIGHBORHOOD_LIMIT = DEFAULT_GRAPH_LIMIT


@dataclass(frozen=True)
class NeighborNodeRow:
    id: str
    type: str
    label: str
    path: str | None
    degree: int


@dataclass(frozen=True)
class NeighborEdgeRow:
    source_id: str
    target_id: str
    type: str
    confidence: str
    orientation: str


@dataclass(frozen=True)
class EdgeTypeFacet:
    """One typed-edge filter checkbox: how many edges carry it, is it selected."""

    type: str
    count: int
    selected: bool


#: Orientation of an edge relative to the seed, derived from the bounded
#: traversal subgraph rather than from any new graph data: `out` runs away
#: from the seed (the seed side depends on the far side), `in` runs toward it
#: (the far side depends on the seed side), and `lateral` connects two nodes
#: the traversal reached at the same distance, where neither endpoint is
#: closer to the seed and no dependency direction relative to the seed exists.
ORIENTATION_OUT = "out"
ORIENTATION_IN = "in"
ORIENTATION_LATERAL = "lateral"


@dataclass(frozen=True)
class NeighborhoodView:
    """What directly depends on this? A bounded traversal, never the full graph."""

    available: bool
    query: str | None
    error: str | None
    seed: NeighborNodeRow | None
    direction: str
    depth: int
    limit: int
    nodes: list[NeighborNodeRow]
    edges: list[NeighborEdgeRow]
    hubs: list[NeighborNodeRow]
    truncated: bool
    omitted_edges: int
    edge_type_facets: list[EdgeTypeFacet]
    selected_edge_types: list[str]
    filtered_edges: int


def _empty_neighborhood(
    *,
    available: bool,
    query: str | None,
    error: str | None,
    direction: str,
    depth: int,
    limit: int,
    selected_edge_types: list[str],
) -> NeighborhoodView:
    return NeighborhoodView(
        available=available,
        query=query,
        error=error,
        seed=None,
        direction=direction,
        depth=depth,
        limit=limit,
        nodes=[],
        edges=[],
        hubs=[],
        truncated=False,
        omitted_edges=0,
        edge_type_facets=[],
        selected_edge_types=selected_edge_types,
        filtered_edges=0,
    )


def normalize_edge_types(requested: Iterable[str] | None) -> list[str]:
    """Keep only recognized `GraphEdgeType` values, sorted and deduplicated.

    An unrecognized value is dropped rather than honored, so a malformed
    filter degrades to "no filter" instead of silently hiding every edge --
    the same convention the depth/limit/direction parameters already use.
    """
    if requested is None:
        return []
    known = {member.value for member in GraphEdgeType}
    return sorted({value for value in requested if value in known})


def build_neighborhood_view(
    data: ExplorerData,
    query: str | None,
    *,
    direction: GraphDirection = "both",
    depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    limit: int = DEFAULT_NEIGHBORHOOD_LIMIT,
    edge_types: Iterable[str] | None = None,
    graph_query: GraphQuery | None = None,
) -> NeighborhoodView:
    selected = normalize_edge_types(edge_types)
    if data.graph is None:
        return _empty_neighborhood(
            available=False,
            query=query,
            error="no graph artifact provided",
            direction=direction,
            depth=depth,
            limit=limit,
            selected_edge_types=selected,
        )
    if not query:
        return _empty_neighborhood(
            available=True,
            query=query,
            error=None,
            direction=direction,
            depth=depth,
            limit=limit,
            selected_edge_types=selected,
        )

    engine = graph_query if graph_query is not None else GraphQuery(data.graph)
    try:
        result = engine.neighbors(query, direction=direction, depth=depth, limit=limit)
    except GraphQueryError as exc:
        return _empty_neighborhood(
            available=True,
            query=query,
            error=str(exc),
            direction=direction,
            depth=depth,
            limit=limit,
            selected_edge_types=selected,
        )

    seed_id = result.seed.id
    distances = _seed_distances(seed_id, result.edges)
    all_rows = _dedupe_edge_rows(_edge_row(edge, distances) for edge in result.edges)
    facets = _edge_type_facets(all_rows, selected)
    rows = [row for row in all_rows if not selected or row.type in selected]
    nodes = _retained_nodes(result.traversed_nodes, rows, seed_id=seed_id, filtered=bool(selected))

    return NeighborhoodView(
        available=True,
        query=query,
        error=None,
        seed=_node_row(result.seed),
        direction=result.direction,
        depth=result.depth,
        limit=limit,
        nodes=nodes,
        edges=rows,
        hubs=[_node_row(node) for node in result.hubs],
        truncated=result.truncated,
        omitted_edges=result.omitted_edges,
        edge_type_facets=facets,
        selected_edge_types=selected,
        filtered_edges=len(all_rows) - len(rows),
    )


def _seed_distances(seed_id: str, edges: list[GraphEdgeSummary]) -> dict[str, int]:
    """Undirected hop distance from the seed, over the traversal's own edges only.

    Pure projection: it reads the bounded edge list the traversal already
    returned and constructs no adjacency the graph does not contain. Nodes the
    bounded edge set does not connect to the seed are simply absent, which the
    caller reads as `lateral`.
    """
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adjacency[edge.source.id].add(edge.target.id)
        adjacency[edge.target.id].add(edge.source.id)

    distances = {seed_id: 0}
    queue: deque[str] = deque([seed_id])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def _dedupe_edge_rows(rows: Iterable[NeighborEdgeRow]) -> list[NeighborEdgeRow]:
    """One display row per distinct edge, in traversal order.

    `GraphQuery.neighbors` walks with `direction="both"` from every frontier
    node, so an edge between two traversed nodes is legitimately selected once
    from each endpoint and appears twice in its `edges` list at depth > 1.
    That is correct for the traversal's own edge budget -- and `truncated` /
    `omitted_edges` are reported against that budget and left untouched here --
    but a rendered table that lists the same relationship twice, and a type
    facet that counts it twice, are display defects. This collapses duplicates
    only for display; it drops no distinct relationship, because the key covers
    every field a row shows.
    """
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[NeighborEdgeRow] = []
    for row in rows:
        key = (row.source_id, row.target_id, row.type, row.confidence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _edge_type_facets(rows: list[NeighborEdgeRow], selected: list[str]) -> list[EdgeTypeFacet]:
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.type] += 1
    chosen = set(selected)
    return [
        EdgeTypeFacet(type=edge_type, count=count, selected=edge_type in chosen)
        for edge_type, count in sorted(counts.items())
    ]


def _retained_nodes(
    traversed: list[GraphNodeSummary],
    rows: list[NeighborEdgeRow],
    *,
    seed_id: str,
    filtered: bool,
) -> list[NeighborNodeRow]:
    """The traversed nodes, narrowed to the seed and the retained edges' endpoints.

    Without a filter this is the traversal's own node set unchanged. With one
    it drops nodes only the hidden edges reached, so the node table never
    claims a relationship the edge table no longer shows.
    """
    if not filtered:
        return [_node_row(node) for node in traversed]
    reachable = {seed_id}
    for row in rows:
        reachable.add(row.source_id)
        reachable.add(row.target_id)
    return [_node_row(node) for node in traversed if node.id in reachable]


def _node_row(node: GraphNodeSummary) -> NeighborNodeRow:
    return NeighborNodeRow(
        id=node.id,
        type=node.type,
        label=node.label,
        path=node.path,
        degree=node.degree,
    )


def _edge_row(edge: GraphEdgeSummary, distances: dict[str, int]) -> NeighborEdgeRow:
    return NeighborEdgeRow(
        source_id=edge.source.id,
        target_id=edge.target.id,
        type=edge.type,
        confidence=edge.confidence,
        orientation=_orientation(edge.source.id, edge.target.id, distances),
    )


def _orientation(source_id: str, target_id: str, distances: dict[str, int]) -> str:
    source_distance = distances.get(source_id)
    target_distance = distances.get(target_id)
    if source_distance is None or target_distance is None:
        return ORIENTATION_LATERAL
    if source_distance < target_distance:
        return ORIENTATION_OUT
    if target_distance < source_distance:
        return ORIENTATION_IN
    return ORIENTATION_LATERAL


MAX_NODE_SEARCH_ROWS = 100


@dataclass(frozen=True)
class NodeSearchRow:
    id: str
    type: str
    label: str
    path: str | None
    module: str | None
    degree: int


@dataclass(frozen=True)
class NodeSearchView:
    """Which node did I mean? Bounded exact-then-fuzzy lookup over the graph."""

    available: bool
    query: str | None
    match_kind: str | None
    limit: int
    matches: list[NodeSearchRow]
    truncated: bool
    omitted: int


def build_node_search_view(
    data: ExplorerData,
    query: str | None,
    *,
    limit: int = MAX_NODE_SEARCH_ROWS,
    graph_query: GraphQuery | None = None,
) -> NodeSearchView:
    """Resolve QUERY to candidate graph nodes via `GraphQuery.lookup`.

    Adds no matcher of its own: `lookup` already implements the exact-then-
    fuzzy id/label/path resolution every other archex graph surface uses, so
    the explorer and `archex graph neighbors` agree on what a query names.
    """
    capped = min(max(limit, 1), MAX_NODE_SEARCH_ROWS)
    if data.graph is None or not query:
        return NodeSearchView(
            available=data.graph is not None,
            query=query,
            match_kind=None,
            limit=capped,
            matches=[],
            truncated=False,
            omitted=0,
        )

    engine = graph_query if graph_query is not None else GraphQuery(data.graph)
    result = engine.lookup(query, limit=capped)
    return NodeSearchView(
        available=True,
        query=query,
        match_kind=result.match_kind,
        limit=capped,
        matches=[
            NodeSearchRow(
                id=node.id,
                type=node.type,
                label=node.label,
                path=node.path,
                module=node.module,
                degree=node.degree,
            )
            for node in result.matches
        ],
        truncated=result.truncated,
        omitted=result.omitted,
    )
