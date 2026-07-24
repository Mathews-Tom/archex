"""Documentation-graph (conditional doc/ADR/ownership) evidence provider contracts (M9).

A provider turns local repository state (markdown documentation already on
disk, a conventional ADR directory, or a CODEOWNERS-style ownership
manifest) into typed evidence records plus a ``DocProviderReceipt``
describing whether the run actually produced usable evidence. Providers
must never raise for ordinary unavailability (no markdown found, no ADR
directory present, no ownership manifest present) -- that is an expected
outcome represented by an ``UNAVAILABLE``/``PARTIAL``/``STALE`` receipt,
never a fallback record and never an exception that would abort indexing.
No provider ever contacts a remote service: every one reads only files
already present on local disk under the repository root.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.docs.models import (
        AdrRecord,
        DocProviderReceipt,
        DocumentationLink,
        OwnershipRecord,
    )


class DocLinkEvidenceProvider(Protocol):
    """A conditional local-markdown-documentation-link evidence source."""

    @property
    def name(self) -> str:
        """The provider identity recorded on every receipt it produces."""
        ...

    def probe(self, repo_root: Path, *, expected_revision: str) -> DocProviderReceipt:
        """Cheaply check availability without collecting evidence.

        Must not raise for ordinary unavailability; returns a receipt whose
        ``availability`` reflects what was found (no markdown documentation,
        or ``AVAILABLE`` when a full collect is expected to produce
        evidence).
        """
        ...

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[DocumentationLink], DocProviderReceipt]:
        """Collect documentation links bound to *expected_revision*.

        Returns ``([], receipt)`` with an explanatory non-``AVAILABLE``
        receipt when the provider cannot run, rather than raising.
        """
        ...


class AdrEvidenceProvider(Protocol):
    """A conditional local-architecture-decision-record evidence source."""

    @property
    def name(self) -> str:
        """The provider identity recorded on every receipt it produces."""
        ...

    def probe(self, repo_root: Path, *, expected_revision: str) -> DocProviderReceipt:
        """Cheaply check availability without collecting evidence.

        Must not raise for ordinary unavailability; see
        ``DocLinkEvidenceProvider.probe``.
        """
        ...

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[AdrRecord], DocProviderReceipt]:
        """Collect ADR records bound to *expected_revision*.

        Returns ``([], receipt)`` with an explanatory non-``AVAILABLE``
        receipt when no ADR directory is present, rather than raising.
        """
        ...


class OwnershipEvidenceProvider(Protocol):
    """A conditional local CODEOWNERS-style ownership evidence source."""

    @property
    def name(self) -> str:
        """The provider identity recorded on every receipt it produces."""
        ...

    def probe(self, repo_root: Path, *, expected_revision: str) -> DocProviderReceipt:
        """Cheaply check availability without collecting evidence.

        Must not raise for ordinary unavailability; see
        ``DocLinkEvidenceProvider.probe``.
        """
        ...

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[OwnershipRecord], DocProviderReceipt]:
        """Collect ownership records bound to *expected_revision*.

        Returns ``([], receipt)`` with an explanatory non-``AVAILABLE``
        receipt when no ownership manifest is present, rather than raising.
        """
        ...
