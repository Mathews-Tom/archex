"""Tests for the doc-link documentation evidence provider (M9)."""

from __future__ import annotations

from pathlib import Path

from archex.integrations.docs.doc_link_provider import DocLinkProvider
from archex.integrations.docs.models import ProviderAvailability

_REVISION = "a" * 40


class TestDocLinkProviderProbe:
    def test_unavailable_when_no_markdown_present(self, tmp_path: Path) -> None:
        provider = DocLinkProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "no markdown documentation" in receipt.reason

    def test_available_when_readme_present(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# Repo\n")
        provider = DocLinkProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.sources_scanned == 1


class TestDocLinkProviderCollect:
    def test_collects_links_resolving_to_real_files(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("x = 1\n")
        (tmp_path / "README.md").write_text(
            "See [module a](src/a.py) and [ghost](src/missing.py) and "
            "[remote](https://example.com/a.py).\n"
        )
        provider = DocLinkProvider()
        links, receipt = provider.collect(tmp_path, expected_revision=_REVISION)

        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert [link.target_path for link in links] == ["src/a.py"]
        assert links[0].doc_path == "README.md"
        assert links[0].link_text == "module a"
        assert links[0].revision == _REVISION
        assert receipt.records_collected == 1

    def test_scans_docs_and_dot_docs_trees(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# Repo\n")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("x = 1\n")
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("See [a](../src/a.py).\n")
        (tmp_path / ".docs").mkdir()
        (tmp_path / ".docs" / "notes.md").write_text("See [a again](../src/a.py).\n")

        provider = DocLinkProvider()
        links, receipt = provider.collect(tmp_path, expected_revision=_REVISION)

        doc_paths = {link.doc_path for link in links}
        assert doc_paths == {"docs/guide.md", ".docs/notes.md"}
        assert receipt.sources_scanned == 3

    def test_never_records_link_to_nonexistent_target(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("See [ghost](does/not/exist.py).\n")
        provider = DocLinkProvider()
        links, _receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert links == []

    def test_returns_probe_receipt_unchanged_when_unavailable(self, tmp_path: Path) -> None:
        provider = DocLinkProvider()
        links, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert links == []
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
