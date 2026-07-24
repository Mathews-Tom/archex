"""Runtime/coverage evidence provider contracts (M7).

A provider turns previously collected, revision-stamped local evidence (a
Cobertura coverage report, a folded-stack profiling run) into typed evidence
records plus a ``RuntimeProviderReceipt`` describing whether the run actually
produced usable evidence. Providers must never raise for ordinary
unavailability (missing evidence directory, missing manifest, mismatched
revision) -- that is an expected outcome represented by an
``UNAVAILABLE``/``PARTIAL``/``STALE`` receipt, never a fallback record and
never an exception that would abort indexing. Providers never collect data by
running a profiler or a coverage-instrumented test suite themselves; they
only read evidence an operator collected out of band.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.runtime.models import (
        CoverageFileEvidence,
        RuntimeProfileEvidence,
        RuntimeProviderReceipt,
    )


class CoverageEvidenceProvider(Protocol):
    """A conditional line-coverage evidence source (a Cobertura coverage report)."""

    @property
    def name(self) -> str:
        """The provider identity recorded on every receipt it produces."""
        ...

    def probe(self, repo_root: Path, *, expected_revision: str) -> RuntimeProviderReceipt:
        """Cheaply check availability and revision validity without collecting evidence.

        Must not raise for ordinary unavailability; returns a receipt whose
        ``availability`` reflects what was found (missing evidence, an
        unreadable manifest, a revision mismatch, or ``AVAILABLE`` when a
        full collect is expected to produce evidence).
        """
        ...

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[CoverageFileEvidence], RuntimeProviderReceipt]:
        """Collect coverage evidence bound to *expected_revision*.

        Returns ``([], receipt)`` with an explanatory non-``AVAILABLE``
        receipt when the provider cannot run or the collected evidence was
        collected against a different revision, rather than raising or
        applying stale evidence.
        """
        ...


class RuntimeProfileEvidenceProvider(Protocol):
    """A conditional folded-stack runtime-profile evidence source."""

    @property
    def name(self) -> str:
        """The provider identity recorded on every receipt it produces."""
        ...

    def probe(self, repo_root: Path, *, expected_revision: str) -> RuntimeProviderReceipt:
        """Cheaply check availability and revision validity without collecting evidence.

        Must not raise for ordinary unavailability; see
        ``CoverageEvidenceProvider.probe``.
        """
        ...

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[RuntimeProfileEvidence], RuntimeProviderReceipt]:
        """Collect runtime-profile evidence bound to *expected_revision*.

        Returns ``([], receipt)`` with an explanatory non-``AVAILABLE``
        receipt when the provider cannot run or the collected evidence was
        collected against a different revision, rather than raising or
        applying stale evidence.
        """
        ...
