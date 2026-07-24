"""Tests for the ADR documentation evidence provider (M9)."""

from __future__ import annotations

from pathlib import Path

from archex.integrations.docs.adr_provider import AdrProvider
from archex.integrations.docs.models import ProviderAvailability

_REVISION = "a" * 40


class TestAdrProviderProbe:
    def test_unavailable_when_no_adr_directory(self, tmp_path: Path) -> None:
        provider = AdrProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "no ADR directory" in receipt.reason

    def test_available_when_adr_directory_present(self, tmp_path: Path) -> None:
        adr_dir = tmp_path / "docs" / "adr"
        adr_dir.mkdir(parents=True)
        (adr_dir / "0001-use-x.md").write_text("# Use X\n\nStatus: Accepted\n")
        provider = AdrProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.sources_scanned == 1


class TestAdrProviderCollect:
    def test_reads_title_status_and_id_from_filename(self, tmp_path: Path) -> None:
        adr_dir = tmp_path / "docs" / "adr"
        adr_dir.mkdir(parents=True)
        (adr_dir / "0001-use-x.md").write_text(
            "# Use X for caching\n\nStatus: Accepted\n\nSee [a](../../src/a.py).\n"
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("x = 1\n")

        provider = AdrProvider()
        records, receipt = provider.collect(tmp_path, expected_revision=_REVISION)

        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert len(records) == 1
        record = records[0]
        assert record.adr_id == "0001"
        assert record.title == "Use X for caching"
        assert record.status == "Accepted"
        assert record.referenced_paths == ["src/a.py"]
        assert record.revision == _REVISION

    def test_status_unknown_when_not_declared(self, tmp_path: Path) -> None:
        adr_dir = tmp_path / "docs" / "adr"
        adr_dir.mkdir(parents=True)
        (adr_dir / "0002-no-status.md").write_text("# Untitled decision\n")

        provider = AdrProvider()
        records, _receipt = provider.collect(tmp_path, expected_revision=_REVISION)

        assert records[0].status == "unknown"

    def test_returns_probe_receipt_unchanged_when_unavailable(self, tmp_path: Path) -> None:
        provider = AdrProvider()
        records, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert records == []
        assert receipt.availability == ProviderAvailability.UNAVAILABLE

    def test_checks_dot_docs_adr_fallback(self, tmp_path: Path) -> None:
        adr_dir = tmp_path / ".docs" / "adr"
        adr_dir.mkdir(parents=True)
        (adr_dir / "0003-fallback.md").write_text("# Fallback location\n")

        provider = AdrProvider()
        records, receipt = provider.collect(tmp_path, expected_revision=_REVISION)

        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert records[0].doc_path == ".docs/adr/0003-fallback.md"
