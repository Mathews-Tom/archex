"""Repository-memory (conditional history) evidence models (M8) — pure Pydantic.

These types describe conditional, provider-sourced local-history evidence:
commit-level change cards, non-causal temporal-coupling observations, and
operator-authored rationale. Structurally distinct from Tree-sitter syntax
evidence and from M6/M7's semantic/runtime evidence -- history evidence is
never added to ``DependencyGraph`` as an edge, and ``TemporalCouplingObservation``
is deliberately never expressed as a dependency-shaped relationship: two
files that changed together are evidence of association, never of a code
dependency. Every item is revision-bound and every provider run yields an
explicit availability receipt; providers never silently apply stale evidence
or fabricate history.

Collection never harvests a remote source: git log evidence is read from the
local git history already on disk, and commit-message issue/PR references
are extracted as plain local text (never resolved against a remote API).
Operator rationale is supplied entirely out of band by an operator, never
inferred or fetched.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, model_validator


class HistoryEvidenceProviderName(StrEnum):
    """Identifies which conditional repository-memory evidence provider produced a record."""

    GIT_LOG = "git_log"
    OPERATOR_RATIONALE = "operator_rationale"


class ProviderAvailability(StrEnum):
    """Explicit availability state for a repository-memory evidence provider run.

    Mirrors ``archex.integrations.semantic.models.ProviderAvailability`` (M6)
    and ``archex.integrations.runtime.models.ProviderAvailability`` (M7) in
    shape but is kept independent: every conditional evidence channel is
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


class LinkedReference(BaseModel):
    """An issue/PR reference number extracted from a commit message's local text.

    Never resolved against a remote API: this is exactly the substring a
    commit message already contains (for example ``#123`` or ``GH-123``),
    kept as local evidence of an association, not a fetched issue/PR record.
    """

    raw_text: str
    identifier: str

    @model_validator(mode="after")
    def _validate_reference(self) -> LinkedReference:
        if not self.raw_text.strip():
            raise ValueError("raw_text must not be empty")
        if not self.identifier.strip():
            raise ValueError("identifier must not be empty")
        return self


class ChangeCard(BaseModel):
    """One revision-bounded local-history observation for a single commit.

    ``commit_subject`` is the commit's first message line only -- never the
    full commit body or diff -- bounding what local-history evidence can
    expose. ``touched_test_files`` is this card's validated-test link: test
    files changed in the same commit as the rest of ``changed_files``.
    """

    commit_sha: str
    commit_subject: str
    committed_at: str
    changed_files: list[str] = []
    touched_test_files: list[str] = []
    linked_references: list[LinkedReference] = []
    revision: str

    @model_validator(mode="after")
    def _validate_change_card(self) -> ChangeCard:
        if not self.commit_sha.strip():
            raise ValueError("commit_sha must not be empty")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        if not set(self.touched_test_files) <= set(self.changed_files):
            raise ValueError("touched_test_files must be a subset of changed_files")
        return self


class TemporalCouplingObservation(BaseModel):
    """Non-causal co-change frequency between two files across a revision window.

    Never a dependency claim: a consumer must present this only as "changed
    together N times in M considered commits", never as an edge, import, or
    call relationship. Kept structurally distinct from
    ``archex.models.EdgeKind`` -- no member of this model set is ever folded
    into ``DependencyGraph``.
    """

    file_a: str
    file_b: str
    co_change_count: int
    window_commit_count: int
    revision: str

    @model_validator(mode="after")
    def _validate_coupling(self) -> TemporalCouplingObservation:
        if self.file_a == self.file_b:
            raise ValueError("file_a and file_b must be different files")
        if self.co_change_count < 1:
            raise ValueError("co_change_count must be >= 1")
        if self.window_commit_count < self.co_change_count:
            raise ValueError("window_commit_count must be >= co_change_count")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        return self


class OperatorRationale(BaseModel):
    """Operator-authored rationale for one file, supplied out of band.

    Never inferred or generated by archex -- a provider that cannot confirm
    an operator actually wrote this for the current revision must return an
    explicit ``STALE``/``UNAVAILABLE`` receipt with no records instead of
    constructing one.
    """

    target_path: str
    rationale: str
    author: str | None = None
    recorded_at: str
    revision: str

    @model_validator(mode="after")
    def _validate_rationale(self) -> OperatorRationale:
        if not self.target_path.strip():
            raise ValueError("target_path must not be empty")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        return self


class HistoryProviderReceipt(BaseModel):
    """Availability, revision-validity, and completeness receipt for one provider run.

    Always produced, whether or not the provider was usable. ``reason`` is
    required whenever ``availability`` is not ``AVAILABLE`` so an unusable or
    stale provider is always explained rather than silently absent.
    """

    provider: HistoryEvidenceProviderName
    availability: ProviderAvailability
    reason: str = ""
    expected_revision: str = ""
    observed_revision: str | None = None
    window_commit_count: int = 0
    records_collected: int = 0
    collected_at: str = ""

    @model_validator(mode="after")
    def _validate_receipt(self) -> HistoryProviderReceipt:
        if self.availability != ProviderAvailability.AVAILABLE and not self.reason.strip():
            raise ValueError(f"reason is required when availability is {self.availability!r}")
        if self.window_commit_count < 0:
            raise ValueError("window_commit_count must be non-negative")
        if self.records_collected < 0:
            raise ValueError("records_collected must be non-negative")
        return self
