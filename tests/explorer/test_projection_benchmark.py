"""M5 scale evidence: 10k/100k-node projection tests against declared budgets.

Marked slow (excluded from the default `-m 'not slow'` run) since building a
100k-node synthetic graph and its `GraphQuery` adjacency indices takes real
wall time. Run explicitly with `-m slow`, or via
`scripts/m5_explorer_projection_benchmark.py` for printed evidence.

Budgets are declared against a measured baseline (see the script's module
docstring for the exact numbers) with headroom for CI variance, not
invented thresholds.
"""

from __future__ import annotations

import time

import pytest

from archex.explorer.loader import ExplorerData
from archex.explorer.render import render_module_map_page, render_neighborhood_page
from archex.explorer.server import create_server
from archex.explorer.viewmodel import (
    build_manifest_view,
    build_module_map_view,
    build_neighborhood_view,
)
from archex.graph_artifact import (
    ArchGraph,
    GraphEdge,
    GraphEdgeType,
    GraphExportMetadata,
    GraphNode,
    GraphNodeType,
    GraphProject,
)
from archex.report.artifact import AnalysisArtifactV1, DiffAnalysis, ReportSchemaVersion

# Measured baseline (this repo's reference dev machine, 2026-07-24):
#   10,000 nodes:  server startup ~0.07s, module map view+render ~0.002s,
#                   post-startup neighborhood lookup <0.001s.
#   100,000 nodes: server startup ~1.12s, module map view+render ~0.024s,
#                   post-startup neighborhood lookup <0.001s.
# Budgets carry >2x headroom over the measured baseline.
STARTUP_BUDGET_SECONDS = {10_000: 0.5, 100_000: 3.0}
MODULE_MAP_BUDGET_SECONDS = {10_000: 0.1, 100_000: 0.5}
NEIGHBORHOOD_LOOKUP_BUDGET_SECONDS = {10_000: 0.05, 100_000: 0.2}


def _artifact() -> AnalysisArtifactV1:
    return AnalysisArtifactV1(
        schema_version=ReportSchemaVersion(),
        generated_at="2026-07-24T00:00:00Z",
        source_identity="acme/widget",
        source_root="/repo",
        source_revision="deadbeef",
        working_tree_fingerprint="fp",
        index_generation="gen1",
        index_schema_version="1",
        chunker_revision="c1",
        config_fingerprint="cfg1",
        diff=DiffAnalysis(base_ref="main"),
    )


def synthetic_graph(node_count: int, *, module_count: int = 50) -> ArchGraph:
    """A deterministic ring-connected synthetic graph with NODE_COUNT file nodes."""
    nodes = [
        GraphNode(
            id=f"file:f{i}.py",
            type=GraphNodeType.FILE,
            label=f"f{i}.py",
            module=f"pkg{i % module_count}",
        )
        for i in range(node_count)
    ]
    edges = [
        GraphEdge(
            source=f"file:f{i}.py",
            target=f"file:f{(i + 1) % node_count}.py",
            type=GraphEdgeType.IMPORTS,
        )
        for i in range(node_count)
    ]
    return ArchGraph(
        project=GraphProject(name="widget", total_files=node_count),
        metadata=GraphExportMetadata(archex_version="0.22.0"),
        nodes=nodes,
        edges=edges,
    )


@pytest.mark.slow
@pytest.mark.parametrize("node_count", [10_000, 100_000])
def test_explorer_projection_meets_declared_budgets(node_count: int) -> None:
    graph = synthetic_graph(node_count)
    data = ExplorerData(artifact=_artifact(), graph=graph)

    start = time.perf_counter()
    server = create_server(data, port=0)
    startup_elapsed = time.perf_counter() - start
    try:
        budget = STARTUP_BUDGET_SECONDS[node_count]
        assert startup_elapsed <= budget, (
            f"server startup at {node_count} nodes took {startup_elapsed:.3f}s, budget is {budget}s"
        )

        manifest = build_manifest_view(data)

        start = time.perf_counter()
        module_map = build_module_map_view(data)
        render_module_map_page(manifest, module_map)
        module_map_elapsed = time.perf_counter() - start
        module_map_budget = MODULE_MAP_BUDGET_SECONDS[node_count]
        assert module_map_elapsed <= module_map_budget, (
            f"module map at {node_count} nodes took {module_map_elapsed:.3f}s, "
            f"budget is {module_map_budget}s"
        )
        assert module_map.available is True
        assert module_map.modules_total > 0

        seed_id = graph.nodes[0].id
        start = time.perf_counter()
        neighborhood = build_neighborhood_view(
            data, seed_id, depth=1, limit=25, graph_query=server.graph_query
        )
        render_neighborhood_page(manifest, neighborhood)
        neighborhood_elapsed = time.perf_counter() - start
        neighborhood_budget = NEIGHBORHOOD_LOOKUP_BUDGET_SECONDS[node_count]
        assert neighborhood_elapsed <= neighborhood_budget, (
            f"neighborhood lookup at {node_count} nodes took {neighborhood_elapsed:.3f}s, "
            f"budget is {neighborhood_budget}s"
        )
        assert neighborhood.error is None
        assert neighborhood.seed is not None
    finally:
        server.server_close()
