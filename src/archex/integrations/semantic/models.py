"""Semantic evidence models — pure Pydantic, zero external dependencies.

These types describe conditional, provider-sourced semantic evidence (SCIP
compiler indexes and LSAP/LSP definition/reference/implementation queries)
that is kept structurally distinct from Tree-sitter syntax evidence. Every
edge produced by a provider records which provider produced it, at what
version, with what confidence, and at what evidence location — so a consumer
can always tell semantic evidence apart from syntax evidence and never has to
guess at provenance. Every provider run yields a receipt describing whether
it was available, partially available, or unavailable, with a human-readable
reason; providers never silently fall back to inventing edges instead.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, model_validator


class SemanticProviderName(StrEnum):
    """Identifies which conditional evidence provider produced a record."""

    SCIP = "scip"
    LSP = "lsp"


class SemanticEdgeKind(StrEnum):
    """The LSP/SCIP request family that produced a semantic edge."""

    DEFINITION = "definition"
    REFERENCE = "reference"
    IMPLEMENTATION = "implementation"


class ProviderAvailability(StrEnum):
    """Explicit availability state for a semantic evidence provider run.

    ``UNAVAILABLE``, ``PARTIAL``, and ``STALE`` are first-class outcomes, not
    error paths papered over with an empty result: M6 requires that an
    unusable provider produce one of these states with a reason rather than
    silently contributing zero edges or falling back to heuristic evidence.
    """

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    UNKNOWN = "unknown"


class SemanticEvidenceLocation(BaseModel):
    """A single file position referenced by a provider."""

    file_path: str
    line: int
    character: int = 0
    symbol: str = ""

    @model_validator(mode="after")
    def _validate_position(self) -> SemanticEvidenceLocation:
        if self.line < 0:
            raise ValueError("line must be >= 0")
        if self.character < 0:
            raise ValueError("character must be >= 0")
        return self


class SemanticEdgeEvidence(BaseModel):
    """One provider-sourced semantic relationship between two locations.

    Raw provider output before it is folded into a graph ``Edge``. Kept
    separate from ``archex.models.Edge`` so provider adapters never need to
    import the core graph model, and so the full evidence location detail
    (line/character on both ends) is available for receipt reporting even
    though the graph edge collapses it to file-to-file granularity.
    """

    provider: SemanticProviderName
    provider_version: str
    kind: SemanticEdgeKind
    source: SemanticEvidenceLocation
    target: SemanticEvidenceLocation
    confidence: float

    @model_validator(mode="after")
    def _validate_confidence(self) -> SemanticEdgeEvidence:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return self


class SemanticProviderReceipt(BaseModel):
    """Availability and completeness receipt for one provider run.

    Always produced, whether or not the provider was usable. ``reason`` is
    required whenever ``availability`` is not ``AVAILABLE`` so an unusable
    provider is always explained rather than silently absent.
    """

    provider: SemanticProviderName
    availability: ProviderAvailability
    reason: str = ""
    tool_name: str | None = None
    tool_version: str | None = None
    files_attempted: int = 0
    files_succeeded: int = 0
    evidence_count: int = 0
    collected_at: str = ""

    @model_validator(mode="after")
    def _validate_receipt(self) -> SemanticProviderReceipt:
        if self.availability != ProviderAvailability.AVAILABLE and not self.reason.strip():
            raise ValueError(f"reason is required when availability is {self.availability!r}")
        if self.files_attempted < 0 or self.files_succeeded < 0 or self.evidence_count < 0:
            raise ValueError("counts must be non-negative")
        if self.files_succeeded > self.files_attempted:
            raise ValueError("files_succeeded cannot exceed files_attempted")
        return self
