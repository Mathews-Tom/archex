"""Local, loopback-only explorer that renders `AnalysisArtifactV1`.

The explorer is a read-only projection over artifacts other commands already
produced (`archex report diff`'s `AnalysisArtifactV1`, `archex graph
export`'s `ArchGraph`). It never parses repository source, never constructs
graph edges of its own, and never mutates project state -- see
`archex.explorer.loader` for the artifact-only data boundary and
`archex.explorer.server` for the loopback/session/CSP/Host security controls
enforced on every request.
"""

from __future__ import annotations
