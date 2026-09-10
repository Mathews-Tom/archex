"""Artifact-only, size-bounded data loading for the local explorer.

Loads a previously generated `AnalysisArtifactV1` (and, optionally, a
previously generated `ArchGraph`) from disk. This module performs no
repository indexing, source parsing, or graph construction of its own -- it
only validates and deserializes artifacts that `archex report diff` and
`archex graph export` already produced. Neither loaded artifact ever carries
raw source text (see `RedactionMode` and `ArchGraph`'s node/edge shapes), so
the explorer has no source content to redact or leak.

The explorer additionally bounds what it will accept. Every view is already
row-bounded, but the canonical loaders read and validate a whole document
before any view runs, so an artifact far larger than anything archex produces
would be materialized in full and -- once exported as static HTML -- handed to
a browser. The explorer therefore refuses an oversized input outright instead
of degrading, and says which limit was exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from archex.graph_artifact import ArchGraph, GraphArtifactError, load_arch_graph
from archex.report.artifact import AnalysisArtifactV1, ReportArtifactError, load_analysis_artifact

if TYPE_CHECKING:
    from pathlib import Path


MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_GRAPH_BYTES = 128 * 1024 * 1024
MAX_GRAPH_NODES = 200_000
MAX_GRAPH_EDGES = 400_000


class ExplorerDataError(ValueError):
    """Raised when the explorer's input artifacts cannot be loaded."""


@dataclass(frozen=True)
class ExplorerData:
    """The explorer's entire read-only input: one diff artifact, one optional graph."""

    artifact: AnalysisArtifactV1
    graph: ArchGraph | None


def _require_within_byte_budget(path: Path, *, limit: int, label: str) -> None:
    """Reject an oversized file before it is read, not after."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ExplorerDataError(f"Failed to read {label} {path}: {exc}") from exc
    if size > limit:
        raise ExplorerDataError(
            f"{label} {path} is {size} bytes, above the explorer's {limit}-byte limit; "
            f"the explorer refuses to load it rather than render a partial view"
        )


def load_explorer_data(artifact_path: Path, graph_path: Path | None = None) -> ExplorerData:
    """Load ARTIFACT_PATH (required) and GRAPH_PATH (optional) for the explorer.

    Raises `ExplorerDataError` for a missing, unreadable, schema-invalid, or
    oversized artifact -- the explorer never falls back to a partially loaded
    or synthesized state.
    """
    _require_within_byte_budget(artifact_path, limit=MAX_ARTIFACT_BYTES, label="report artifact")
    try:
        artifact = load_analysis_artifact(artifact_path)
    except ReportArtifactError as exc:
        raise ExplorerDataError(str(exc)) from exc

    graph: ArchGraph | None = None
    if graph_path is not None:
        _require_within_byte_budget(graph_path, limit=MAX_GRAPH_BYTES, label="graph artifact")
        try:
            graph = load_arch_graph(graph_path)
        except GraphArtifactError as exc:
            raise ExplorerDataError(str(exc)) from exc
        if len(graph.nodes) > MAX_GRAPH_NODES:
            raise ExplorerDataError(
                f"graph artifact {graph_path} declares {len(graph.nodes)} nodes, above the "
                f"explorer's {MAX_GRAPH_NODES}-node limit"
            )
        if len(graph.edges) > MAX_GRAPH_EDGES:
            raise ExplorerDataError(
                f"graph artifact {graph_path} declares {len(graph.edges)} edges, above the "
                f"explorer's {MAX_GRAPH_EDGES}-edge limit"
            )

    return ExplorerData(artifact=artifact, graph=graph)
