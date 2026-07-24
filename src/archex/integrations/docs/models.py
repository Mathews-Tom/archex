"""Documentation-graph (conditional doc/ADR/ownership) evidence models (M9) — pure Pydantic.

These types describe conditional, provider-sourced local documentation
evidence: markdown documentation links, architecture-decision-record
provenance, and CODEOWNERS-style ownership records. Structurally distinct
from Tree-sitter syntax evidence and from M6/M7/M8's semantic/runtime/
history evidence -- documentation evidence is never added to
``DependencyGraph`` as an edge and is never described as a code
dependency. A markdown file that mentions a source path is evidence of
association (this file is discussed, owned, or decided about here), never
of an import, call, or reference relationship between two code artifacts.

Every item is revision-bound and every provider run yields an explicit
availability receipt; providers never silently apply stale evidence or
fabricate documentation, ownership, or decision records.

Collection never harvests a remote source: every provider reads only
files already present on local disk under the repository root (markdown
documentation, ADR files, and a CODEOWNERS-style ownership manifest).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, model_validator


class DocEvidenceProviderName(StrEnum):
    """Identifies which conditional documentation-graph evidence provider produced a record."""

    DOC_LINK = "doc_link"
    ADR = "adr"
    OWNERSHIP = "ownership"


class ProviderAvailability(StrEnum):
    """Explicit availability state for a documentation-graph evidence provider run.

    Mirrors ``archex.integrations.semantic.models.ProviderAvailability`` (M6),
    ``archex.integrations.runtime.models.ProviderAvailability`` (M7), and
    ``archex.integrations.history.models.ProviderAvailability`` (M8) in shape
    but is kept independent: every conditional evidence channel is
    separately disableable and separately rollback-able. ``UNAVAILABLE``,
    ``PARTIAL``, and ``STALE`` are first-class outcomes -- an unusable or
    revision-mismatched provider must produce one of these states with a
    reason rather than silently contributing zero records or applying
    evidence collected against a different revision.
    """

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    UNKNOWN = "unknown"


class DocumentationLink(BaseModel):
    """One local markdown document's reference to a source path.

    Never a dependency claim: ``doc_path`` is the markdown file that
    contains the reference and ``target_path`` is the source path it
    mentions, resolved and confirmed to exist under the repository root at
    ``revision`` -- a reference to a path that does not resolve locally is
    never recorded.
    """

    doc_path: str
    target_path: str
    link_text: str
    revision: str

    @model_validator(mode="after")
    def _validate_link(self) -> DocumentationLink:
        if not self.doc_path.strip():
            raise ValueError("doc_path must not be empty")
        if not self.target_path.strip():
            raise ValueError("target_path must not be empty")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        return self


class AdrRecord(BaseModel):
    """One architecture-decision-record's metadata and the source paths it references.

    ``status`` is the ADR's own declared status text (for example
    ``"Accepted"`` or ``"Superseded"``), read verbatim from the document --
    archex never infers or normalizes it into a health signal.
    """

    adr_id: str
    title: str
    status: str
    doc_path: str
    referenced_paths: list[str] = []
    revision: str

    @model_validator(mode="after")
    def _validate_adr(self) -> AdrRecord:
        if not self.adr_id.strip():
            raise ValueError("adr_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if not self.doc_path.strip():
            raise ValueError("doc_path must not be empty")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        return self


class OwnershipRecord(BaseModel):
    """One CODEOWNERS-style path-pattern-to-owner mapping.

    ``source_path`` is the ownership manifest the pattern was read from --
    never inferred from commit authorship or any other heuristic.
    """

    path_pattern: str
    owners: list[str]
    source_path: str
    revision: str

    @model_validator(mode="after")
    def _validate_ownership(self) -> OwnershipRecord:
        if not self.path_pattern.strip():
            raise ValueError("path_pattern must not be empty")
        if not self.owners:
            raise ValueError("owners must not be empty")
        if not self.source_path.strip():
            raise ValueError("source_path must not be empty")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        return self


class DocProviderReceipt(BaseModel):
    """Availability, revision-validity, and completeness receipt for one provider run.

    Always produced, whether or not the provider was usable. ``reason`` is
    required whenever ``availability`` is not ``AVAILABLE`` so an unusable or
    stale provider is always explained rather than silently absent.
    """

    provider: DocEvidenceProviderName
    availability: ProviderAvailability
    reason: str = ""
    expected_revision: str = ""
    observed_revision: str | None = None
    sources_scanned: int = 0
    records_collected: int = 0
    collected_at: str = ""

    @model_validator(mode="after")
    def _validate_receipt(self) -> DocProviderReceipt:
        if self.availability != ProviderAvailability.AVAILABLE and not self.reason.strip():
            raise ValueError(f"reason is required when availability is {self.availability!r}")
        if self.sources_scanned < 0:
            raise ValueError("sources_scanned must be non-negative")
        if self.records_collected < 0:
            raise ValueError("records_collected must be non-negative")
        return self
