"""Pure, deterministic view models projected from `ExplorerData`.

Every builder in this module is a pure function over already-loaded
artifacts (`archex.explorer.loader.ExplorerData`): no file I/O, no network
access, no repository indexing, and no new graph-edge construction. Bounded
list fields mirror the `*_total` convention `archex.report.artifact` and
`archex.graph_query` already use, so every view can show "N of TOTAL" rather
than silently truncating.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from archex.graph_query import DEFAULT_GRAPH_LIMIT, GraphQuery, GraphQueryError

if TYPE_CHECKING:
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


def build_neighborhood_view(
    data: ExplorerData,
    query: str | None,
    *,
    direction: GraphDirection = "both",
    depth: int = DEFAULT_NEIGHBORHOOD_DEPTH,
    limit: int = DEFAULT_NEIGHBORHOOD_LIMIT,
    graph_query: GraphQuery | None = None,
) -> NeighborhoodView:
    if data.graph is None:
        return NeighborhoodView(
            available=False,
            query=query,
            error="no graph artifact provided",
            seed=None,
            direction=direction,
            depth=depth,
            limit=limit,
            nodes=[],
            edges=[],
            hubs=[],
            truncated=False,
            omitted_edges=0,
        )
    if not query:
        return NeighborhoodView(
            available=True,
            query=query,
            error=None,
            seed=None,
            direction=direction,
            depth=depth,
            limit=limit,
            nodes=[],
            edges=[],
            hubs=[],
            truncated=False,
            omitted_edges=0,
        )

    engine = graph_query if graph_query is not None else GraphQuery(data.graph)
    try:
        result = engine.neighbors(query, direction=direction, depth=depth, limit=limit)
    except GraphQueryError as exc:
        return NeighborhoodView(
            available=True,
            query=query,
            error=str(exc),
            seed=None,
            direction=direction,
            depth=depth,
            limit=limit,
            nodes=[],
            edges=[],
            hubs=[],
            truncated=False,
            omitted_edges=0,
        )

    return NeighborhoodView(
        available=True,
        query=query,
        error=None,
        seed=_node_row(result.seed),
        direction=result.direction,
        depth=result.depth,
        limit=limit,
        nodes=[_node_row(node) for node in result.traversed_nodes],
        edges=[_edge_row(edge) for edge in result.edges],
        hubs=[_node_row(node) for node in result.hubs],
        truncated=result.truncated,
        omitted_edges=result.omitted_edges,
    )


def _node_row(node: GraphNodeSummary) -> NeighborNodeRow:
    return NeighborNodeRow(
        id=node.id,
        type=node.type,
        label=node.label,
        path=node.path,
        degree=node.degree,
    )


def _edge_row(edge: GraphEdgeSummary) -> NeighborEdgeRow:
    return NeighborEdgeRow(
        source_id=edge.source.id,
        target_id=edge.target.id,
        type=edge.type,
        confidence=edge.confidence,
    )
