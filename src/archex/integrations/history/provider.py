"""Repository-memory (conditional history) evidence provider contracts (M8).

A provider turns local repository state (the git commit log already on
disk, or previously supplied operator rationale) into typed evidence
records plus a ``HistoryProviderReceipt`` describing whether the run
actually produced usable evidence. Providers must never raise for ordinary
unavailability (not a git repository, no commits in the requested window, a
missing rationale directory, a stale rationale revision) -- that is an
expected outcome represented by an ``UNAVAILABLE``/``PARTIAL``/``STALE``
receipt, never a fallback record and never an exception that would abort
indexing. Neither provider ever contacts a remote service: ``GitLogEvidenceProvider``
reads only the local ``.git`` history, and ``OperatorRationaleEvidenceProvider``
reads only a previously written local file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.history.models import (
        ChangeCard,
        HistoryProviderReceipt,
        OperatorRationale,
        TemporalCouplingObservation,
    )


class GitLogEvidenceProvider(Protocol):
    """A conditional local-git-history evidence source."""

    @property
    def name(self) -> str:
        """The provider identity recorded on every receipt it produces."""
        ...

    def probe(self, repo_root: Path, *, expected_revision: str) -> HistoryProviderReceipt:
        """Cheaply check availability without collecting evidence.

        Must not raise for ordinary unavailability; returns a receipt whose
        ``availability`` reflects what was found (not a git repository, an
        unresolvable revision, or ``AVAILABLE`` when a full collect is
        expected to produce evidence).
        """
        ...

    def collect(
        self, repo_root: Path, *, expected_revision: str, max_commits: int
    ) -> tuple[list[ChangeCard], list[TemporalCouplingObservation], HistoryProviderReceipt]:
        """Collect change cards and temporal-coupling observations bound to
        the ``max_commits``-commit window ending at ``expected_revision``.

        Returns ``([], [], receipt)`` with an explanatory non-``AVAILABLE``
        receipt when the provider cannot run, rather than raising.
        """
        ...


class OperatorRationaleEvidenceProvider(Protocol):
    """A conditional operator-authored-rationale evidence source."""

    @property
    def name(self) -> str:
        """The provider identity recorded on every receipt it produces."""
        ...

    def probe(self, repo_root: Path, *, expected_revision: str) -> HistoryProviderReceipt:
        """Cheaply check availability and revision validity without collecting evidence.

        Must not raise for ordinary unavailability; see
        ``GitLogEvidenceProvider.probe``.
        """
        ...

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[OperatorRationale], HistoryProviderReceipt]:
        """Collect operator rationale bound to *expected_revision*.

        Returns ``([], receipt)`` with an explanatory non-``AVAILABLE``
        receipt when the provider cannot run or the collected evidence was
        authored against a different revision, rather than raising or
        applying stale evidence.
        """
        ...
