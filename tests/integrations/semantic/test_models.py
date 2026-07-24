"""Tests for semantic evidence models."""

from __future__ import annotations

import pytest

from archex.integrations.semantic.models import (
    ProviderAvailability,
    SemanticEdgeEvidence,
    SemanticEdgeKind,
    SemanticEvidenceLocation,
    SemanticProviderName,
    SemanticProviderReceipt,
)


def _location(**overrides: object) -> SemanticEvidenceLocation:
    defaults: dict[str, object] = {"file_path": "a.py", "line": 1, "character": 0}
    defaults.update(overrides)
    return SemanticEvidenceLocation(**defaults)  # type: ignore[arg-type]


class TestSemanticEvidenceLocation:
    def test_rejects_negative_line(self) -> None:
        with pytest.raises(ValueError, match="line"):
            _location(line=-1)

    def test_rejects_negative_character(self) -> None:
        with pytest.raises(ValueError, match="character"):
            _location(character=-1)


class TestSemanticEdgeEvidence:
    def test_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            SemanticEdgeEvidence(
                provider=SemanticProviderName.SCIP,
                provider_version="1.0",
                kind=SemanticEdgeKind.DEFINITION,
                source=_location(),
                target=_location(file_path="b.py"),
                confidence=1.5,
            )

    def test_accepts_boundary_confidence(self) -> None:
        edge = SemanticEdgeEvidence(
            provider=SemanticProviderName.LSP,
            provider_version="1.0",
            kind=SemanticEdgeKind.REFERENCE,
            source=_location(),
            target=_location(file_path="b.py"),
            confidence=0.0,
        )
        assert edge.confidence == 0.0


class TestSemanticProviderReceipt:
    def test_reason_required_when_unavailable(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            SemanticProviderReceipt(
                provider=SemanticProviderName.SCIP,
                availability=ProviderAvailability.UNAVAILABLE,
            )

    def test_reason_optional_when_available(self) -> None:
        receipt = SemanticProviderReceipt(
            provider=SemanticProviderName.SCIP,
            availability=ProviderAvailability.AVAILABLE,
        )
        assert receipt.reason == ""

    def test_rejects_succeeded_exceeding_attempted(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            SemanticProviderReceipt(
                provider=SemanticProviderName.SCIP,
                availability=ProviderAvailability.AVAILABLE,
                files_attempted=1,
                files_succeeded=2,
            )

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SemanticProviderReceipt(
                provider=SemanticProviderName.SCIP,
                availability=ProviderAvailability.AVAILABLE,
                evidence_count=-1,
            )
