"""Evidence script: measure explorer projection latency at 10k/100k graph nodes.

M5's DEVELOPMENT_PLAN.md acceptance row requires "10k/100k projection tests
meet declared rendering budgets". Run:

    uv run python scripts/m5_explorer_projection_benchmark.py

Builds a deterministic ring-connected synthetic `ArchGraph` at each node
count (never real repository source -- this is a scale test of the
explorer's own view-model/render/GraphQuery layers, not of parsing or
indexing), then times:

- server startup, including the one-time `GraphQuery` adjacency-index build
  `archex.explorer.server.ExplorerServer` performs so per-request
  neighborhood lookups never re-pay it;
- Module Map view-model construction plus HTML render;
- a post-startup Target Neighborhood lookup (view-model construction plus
  HTML render), which should be dominated by the bounded depth/limit
  traversal, not by graph size.

Exits 1 (after printing every measurement) if any declared budget is
exceeded. Budgets mirror `tests/explorer/test_projection_benchmark.py`
(kept in sync by hand -- both are small, deliberately duplicated rather
than cross-imported so this script stays a self-contained, type-checkable
CLI evidence tool independent of the test tree).
"""

from __future__ import annotations

import time

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

NODE_COUNTS = (10_000, 100_000)

# Measured baseline (this repo's reference dev machine, 2026-07-24):
#   10,000 nodes:  server startup ~0.07s, module map view+render ~0.002s,
#                   post-startup neighborhood lookup <0.001s.
#   100,000 nodes: server startup ~1.12s, module map view+render ~0.024s,
#                   post-startup neighborhood lookup <0.001s.
# Budgets carry >2x headroom over the measured baseline. Kept identical to
# `tests/explorer/test_projection_benchmark.py`'s budgets by hand.
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


def _synthetic_graph(node_count: int, *, module_count: int = 50) -> ArchGraph:
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


def _measure(node_count: int) -> tuple[bool, dict[str, float]]:
    graph = _synthetic_graph(node_count)
    data = ExplorerData(artifact=_artifact(), graph=graph)

    start = time.perf_counter()
    server = create_server(data, port=0)
    startup_elapsed = time.perf_counter() - start
    try:
        manifest = build_manifest_view(data)

        start = time.perf_counter()
        module_map = build_module_map_view(data)
        render_module_map_page(manifest, module_map)
        module_map_elapsed = time.perf_counter() - start

        seed_id = graph.nodes[0].id
        start = time.perf_counter()
        neighborhood = build_neighborhood_view(
            data, seed_id, depth=1, limit=25, graph_query=server.graph_query
        )
        render_neighborhood_page(manifest, neighborhood)
        neighborhood_elapsed = time.perf_counter() - start
    finally:
        server.server_close()

    measurements = {
        "startup_seconds": startup_elapsed,
        "module_map_seconds": module_map_elapsed,
        "neighborhood_seconds": neighborhood_elapsed,
    }
    within_budget = (
        startup_elapsed <= STARTUP_BUDGET_SECONDS[node_count]
        and module_map_elapsed <= MODULE_MAP_BUDGET_SECONDS[node_count]
        and neighborhood_elapsed <= NEIGHBORHOOD_LOOKUP_BUDGET_SECONDS[node_count]
    )
    return within_budget, measurements


def main() -> int:
    print(f"{'nodes':>8}  {'startup':>10}  {'module_map':>12}  {'neighborhood':>14}  budget")
    all_within_budget = True
    for node_count in NODE_COUNTS:
        within_budget, measurements = _measure(node_count)
        all_within_budget = all_within_budget and within_budget
        status = "PASS" if within_budget else "FAIL"
        print(
            f"{node_count:>8,}  "
            f"{measurements['startup_seconds']:>9.3f}s  "
            f"{measurements['module_map_seconds']:>11.3f}s  "
            f"{measurements['neighborhood_seconds']:>13.4f}s  {status}"
        )
        print(
            f"           budgets: startup<={STARTUP_BUDGET_SECONDS[node_count]}s "
            f"module_map<={MODULE_MAP_BUDGET_SECONDS[node_count]}s "
            f"neighborhood<={NEIGHBORHOOD_LOOKUP_BUDGET_SECONDS[node_count]}s"
        )

    if not all_within_budget:
        print("\nFAILED: one or more measurements exceeded its declared budget.")
        return 1
    print("\nPASSED: every measurement is within its declared budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
