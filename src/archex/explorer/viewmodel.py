"""Pure, deterministic view models projected from `ExplorerData`.

Every builder in this module is a pure function over already-loaded
artifacts (`archex.explorer.loader.ExplorerData`): no file I/O, no network
access, no repository indexing, and no new graph-edge construction. Bounded
list fields mirror the `*_total` convention `archex.report.artifact` and
`archex.graph_query` already use, so every view can show "N of TOTAL" rather
than silently truncating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from archex.explorer.loader import ExplorerData
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
