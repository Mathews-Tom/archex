"""Semantic evidence provider contract.

A provider turns an external, versioned tool (a SCIP compiler index, an LSP
server) into ``SemanticEdgeEvidence`` records plus a ``SemanticProviderReceipt``
describing whether the run actually produced evidence. Providers must never
raise for ordinary unavailability (missing index file, missing optional
dependency, unreachable server) — that is an expected outcome represented by
an ``UNAVAILABLE``/``PARTIAL``/``STALE`` receipt, never a fallback edge and
never an exception that would abort indexing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.semantic.models import (
        SemanticEdgeEvidence,
        SemanticProviderName,
        SemanticProviderReceipt,
    )
    from archex.models import ParsedFile


class SemanticEvidenceProvider(Protocol):
    """A conditional semantic evidence source (SCIP index, LSP server, ...)."""

    @property
    def name(self) -> SemanticProviderName:
        """The provider identity recorded on every edge and receipt it produces."""
        ...

    def probe(self, repo_root: Path) -> SemanticProviderReceipt:
        """Cheaply check availability without collecting evidence.

        Must not raise for ordinary unavailability; returns a receipt whose
        ``availability`` reflects what was found (missing index/tooling,
        unreachable server, or ``AVAILABLE`` when a full run is expected to
        produce evidence).
        """
        ...

    def collect(
        self, parsed_files: list[ParsedFile], repo_root: Path
    ) -> tuple[list[SemanticEdgeEvidence], SemanticProviderReceipt]:
        """Collect semantic evidence for the given parsed files.

        Returns ``([], receipt)`` with an explanatory non-``AVAILABLE``
        receipt when the provider cannot run, rather than raising or
        inventing edges. Must not mutate ``parsed_files``.
        """
        ...
