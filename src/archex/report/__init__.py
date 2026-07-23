"""AnalysisArtifactV1: the canonical, versioned diff-review artifact.

Every renderer (JSON, Markdown, static HTML) and the CI delta example project
the same `AnalysisArtifactV1` without reinterpreting it: no source parsing,
edge construction, reranking, or security inference happens downstream of
`build_analysis_artifact`.
"""

from __future__ import annotations
