"""Runtime and coverage evidence models (M7) — pure Pydantic, zero external dependencies.

These types describe conditional, provider-sourced runtime evidence (line
coverage collected by ``coverage.py`` and folded-stack sampling profiles)
that is kept structurally distinct from Tree-sitter syntax evidence and from
M6's compiler-grade semantic evidence: it is never added to
``DependencyGraph`` as an edge. Every record is revision-bound -- it carries
the exact git revision it was collected against -- so a consumer can always
tell whether the evidence still applies to the checkout it is being read
against. Every provider run yields a receipt describing whether it was
available, partially available, unavailable, or stale (collected against a
different revision than the one currently being analyzed), with a
human-readable reason; providers never silently apply stale or mismatched
evidence instead.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, model_validator


class RuntimeEvidenceProviderName(StrEnum):
    """Identifies which conditional runtime/coverage evidence provider produced a record."""

    COVERAGE = "coverage"
    RUNTIME_PROFILE = "runtime_profile"


class ProviderAvailability(StrEnum):
    """Explicit availability state for a runtime/coverage evidence provider run.

    Mirrors ``archex.integrations.semantic.models.ProviderAvailability`` (M6)
    in shape but is kept independent: M7's runtime/coverage channel and M6's
    semantic channel are separately disableable and separately
    rollback-able, so their availability vocabularies must not be coupled by
    a shared import. ``UNAVAILABLE``, ``PARTIAL``, and ``STALE`` are
    first-class outcomes, not error paths papered over with an empty
    result -- an unusable or revision-mismatched provider must produce one
    of these states with a reason rather than silently contributing zero
    records or applying evidence collected against a different revision.
    """

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    STALE = "stale"
    UNKNOWN = "unknown"


class CoverageLineRecord(BaseModel):
    """One executed-line observation for a single source line."""

    line: int
    hits: int

    @model_validator(mode="after")
    def _validate_line_record(self) -> CoverageLineRecord:
        if self.line < 1:
            raise ValueError("line must be >= 1")
        if self.hits < 0:
            raise ValueError("hits must be non-negative")
        return self


class CoverageFileEvidence(BaseModel):
    """Revision-bound line-coverage evidence for one source file.

    Read from a previously generated Cobertura-format coverage report (the
    same shape ``coverage xml`` / ``pytest --cov-report=xml`` already
    produces). ``revision`` is the git commit the report was collected
    against, always equal to the caller's ``expected_revision`` by the time
    a record reaches this model -- a provider that cannot confirm the
    revision matches must return an explicit ``STALE`` receipt with no
    records instead of constructing one.
    """

    file_path: str
    lines: list[CoverageLineRecord] = []
    line_rate: float
    revision: str

    @model_validator(mode="after")
    def _validate_coverage_evidence(self) -> CoverageFileEvidence:
        if not self.file_path.strip():
            raise ValueError("file_path must not be empty")
        if not 0.0 <= self.line_rate <= 1.0:
            raise ValueError("line_rate must be between 0.0 and 1.0")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        return self


class RuntimeStackSample(BaseModel):
    """One folded call-stack sample.

    ``frames`` is root-first: each entry is ``<repo-relative file
    path>:<qualified symbol name>``, the same addressable shape used
    throughout archex's own evidence/receipt surfaces, so a sample can
    always be attributed to an indexed file and symbol without guessing at
    an external profiler's own naming convention.
    """

    frames: tuple[str, ...]
    sample_count: int

    @model_validator(mode="after")
    def _validate_stack_sample(self) -> RuntimeStackSample:
        if not self.frames:
            raise ValueError("frames must not be empty")
        for frame in self.frames:
            if ":" not in frame:
                raise ValueError(f"frame {frame!r} must be '<file_path>:<qualified_name>'")
        if self.sample_count < 1:
            raise ValueError("sample_count must be >= 1")
        return self


class RuntimeProfileEvidence(BaseModel):
    """Revision-bound folded-stack runtime evidence for one collection run.

    ``revision`` is the git commit the profile was collected against, always
    equal to the caller's ``expected_revision`` by the time a record reaches
    this model -- mirrors ``CoverageFileEvidence.revision``.
    """

    samples: list[RuntimeStackSample] = []
    total_samples: int
    revision: str

    @model_validator(mode="after")
    def _validate_profile_evidence(self) -> RuntimeProfileEvidence:
        if self.total_samples < 0:
            raise ValueError("total_samples must be non-negative")
        if not self.revision.strip():
            raise ValueError("revision must not be empty")
        return self


class RuntimeProviderReceipt(BaseModel):
    """Availability, revision-validity, and completeness receipt for one provider run.

    Always produced, whether or not the provider was usable. ``reason`` is
    required whenever ``availability`` is not ``AVAILABLE`` so an unusable or
    stale provider is always explained rather than silently absent.
    ``expected_revision``/``observed_revision`` record the exact revision
    comparison a ``STALE`` outcome was based on.
    """

    provider: RuntimeEvidenceProviderName
    availability: ProviderAvailability
    reason: str = ""
    tool_name: str | None = None
    tool_version: str | None = None
    expected_revision: str = ""
    observed_revision: str | None = None
    records_collected: int = 0
    collected_at: str = ""

    @model_validator(mode="after")
    def _validate_receipt(self) -> RuntimeProviderReceipt:
        if self.availability != ProviderAvailability.AVAILABLE and not self.reason.strip():
            raise ValueError(f"reason is required when availability is {self.availability!r}")
        if self.records_collected < 0:
            raise ValueError("records_collected must be non-negative")
        return self
