"""Tests for documentation-graph (doc/ADR/ownership) evidence models (M9)."""

from __future__ import annotations

import pytest

from archex.integrations.docs.models import (
    AdrRecord,
    DocEvidenceProviderName,
    DocProviderReceipt,
    DocumentationLink,
    OwnershipRecord,
    ProviderAvailability,
)


class TestDocumentationLink:
    def test_rejects_empty_doc_path(self) -> None:
        with pytest.raises(ValueError, match="doc_path"):
            DocumentationLink(doc_path=" ", target_path="src/a.py", link_text="a", revision="r")

    def test_rejects_empty_target_path(self) -> None:
        with pytest.raises(ValueError, match="target_path"):
            DocumentationLink(doc_path="README.md", target_path=" ", link_text="a", revision="r")

    def test_rejects_empty_revision(self) -> None:
        with pytest.raises(ValueError, match="revision"):
            DocumentationLink(
                doc_path="README.md", target_path="src/a.py", link_text="a", revision=" "
            )

    def test_accepts_valid_link(self) -> None:
        link = DocumentationLink(
            doc_path="README.md", target_path="src/a.py", link_text="module a", revision="rev"
        )
        assert link.target_path == "src/a.py"


class TestAdrRecord:
    def test_rejects_empty_adr_id(self) -> None:
        with pytest.raises(ValueError, match="adr_id"):
            AdrRecord(
                adr_id=" ",
                title="Use X",
                status="Accepted",
                doc_path="docs/adr/0001.md",
                revision="r",
            )

    def test_rejects_empty_title(self) -> None:
        with pytest.raises(ValueError, match="title"):
            AdrRecord(
                adr_id="0001",
                title=" ",
                status="Accepted",
                doc_path="docs/adr/0001.md",
                revision="r",
            )

    def test_accepts_valid_record(self) -> None:
        record = AdrRecord(
            adr_id="0001",
            title="Use X",
            status="Accepted",
            doc_path="docs/adr/0001.md",
            referenced_paths=["src/a.py"],
            revision="rev",
        )
        assert record.status == "Accepted"
        assert record.referenced_paths == ["src/a.py"]


class TestOwnershipRecord:
    def test_rejects_empty_pattern(self) -> None:
        with pytest.raises(ValueError, match="path_pattern"):
            OwnershipRecord(
                path_pattern=" ", owners=["@team"], source_path="CODEOWNERS", revision="r"
            )

    def test_rejects_empty_owners(self) -> None:
        with pytest.raises(ValueError, match="owners"):
            OwnershipRecord(path_pattern="/src/", owners=[], source_path="CODEOWNERS", revision="r")

    def test_accepts_valid_record(self) -> None:
        record = OwnershipRecord(
            path_pattern="/src/", owners=["@team"], source_path="CODEOWNERS", revision="rev"
        )
        assert record.owners == ["@team"]


class TestDocProviderReceipt:
    def test_reason_required_when_unavailable(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            DocProviderReceipt(
                provider=DocEvidenceProviderName.DOC_LINK,
                availability=ProviderAvailability.UNAVAILABLE,
            )

    def test_reason_not_required_when_available(self) -> None:
        receipt = DocProviderReceipt(
            provider=DocEvidenceProviderName.DOC_LINK,
            availability=ProviderAvailability.AVAILABLE,
        )
        assert receipt.reason == ""

    def test_rejects_negative_sources_scanned(self) -> None:
        with pytest.raises(ValueError, match="sources_scanned"):
            DocProviderReceipt(
                provider=DocEvidenceProviderName.ADR,
                availability=ProviderAvailability.AVAILABLE,
                sources_scanned=-1,
            )

    def test_rejects_negative_records_collected(self) -> None:
        with pytest.raises(ValueError, match="records_collected"):
            DocProviderReceipt(
                provider=DocEvidenceProviderName.OWNERSHIP,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=-1,
            )
