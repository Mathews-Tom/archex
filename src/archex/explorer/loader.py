"""Artifact-only data loading for the local explorer.

Loads a previously generated `AnalysisArtifactV1` (and, optionally, a
previously generated `ArchGraph`) from disk. This module performs no
repository indexing, source parsing, or graph construction of its own -- it
only validates and deserializes artifacts that `archex report diff` and
`archex graph export` already produced. Neither loaded artifact ever carries
raw source text (see `RedactionMode` and `ArchGraph`'s node/edge shapes), so
the explorer has no source content to redact or leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from archex.graph_artifact import ArchGraph, GraphArtifactError, load_arch_graph
from archex.report.artifact import AnalysisArtifactV1, ReportArtifactError, load_analysis_artifact

if TYPE_CHECKING:
    from pathlib import Path


class ExplorerDataError(ValueError):
    """Raised when the explorer's input artifacts cannot be loaded."""


@dataclass(frozen=True)
class ExplorerData:
    """The explorer's entire read-only input: one diff artifact, one optional graph."""

    artifact: AnalysisArtifactV1
    graph: ArchGraph | None


def load_explorer_data(artifact_path: Path, graph_path: Path | None = None) -> ExplorerData:
    """Load ARTIFACT_PATH (required) and GRAPH_PATH (optional) for the explorer.

    Raises `ExplorerDataError` for a missing, unreadable, or schema-invalid
    artifact -- the explorer never falls back to a partially loaded or
    synthesized state.
    """
    try:
        artifact = load_analysis_artifact(artifact_path)
    except ReportArtifactError as exc:
        raise ExplorerDataError(str(exc)) from exc

    graph: ArchGraph | None = None
    if graph_path is not None:
        try:
            graph = load_arch_graph(graph_path)
        except GraphArtifactError as exc:
            raise ExplorerDataError(str(exc)) from exc

    return ExplorerData(artifact=artifact, graph=graph)
