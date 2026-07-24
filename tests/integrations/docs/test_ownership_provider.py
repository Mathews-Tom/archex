"""Tests for the ownership (CODEOWNERS-style) documentation evidence provider (M9)."""

from __future__ import annotations

from pathlib import Path

from archex.integrations.docs.models import ProviderAvailability
from archex.integrations.docs.ownership_provider import OwnershipProvider

_REVISION = "a" * 40


class TestOwnershipProviderProbe:
    def test_unavailable_when_no_codeowners_file(self, tmp_path: Path) -> None:
        provider = OwnershipProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "no CODEOWNERS file" in receipt.reason

    def test_available_when_github_codeowners_present(self, tmp_path: Path) -> None:
        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "CODEOWNERS").write_text("/src/ @team-core\n")
        provider = OwnershipProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.sources_scanned == 1


class TestOwnershipProviderCollect:
    def test_parses_pattern_and_owners(self, tmp_path: Path) -> None:
        (tmp_path / "CODEOWNERS").write_text(
            "# comment\n\n/src/ @team-core @alice\n/docs/ @team-docs\n"
        )
        provider = OwnershipProvider()
        records, receipt = provider.collect(tmp_path, expected_revision=_REVISION)

        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert len(records) == 2
        assert records[0].path_pattern == "/src/"
        assert records[0].owners == ["@team-core", "@alice"]
        assert records[0].source_path == "CODEOWNERS"
        assert records[0].revision == _REVISION

    def test_skips_no_owner_override_lines(self, tmp_path: Path) -> None:
        (tmp_path / "CODEOWNERS").write_text("/generated/\n/src/ @team-core\n")
        provider = OwnershipProvider()
        records, _receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert [r.path_pattern for r in records] == ["/src/"]

    def test_prefers_github_codeowners_over_root(self, tmp_path: Path) -> None:
        (tmp_path / "CODEOWNERS").write_text("/root/ @root-team\n")
        github_dir = tmp_path / ".github"
        github_dir.mkdir()
        (github_dir / "CODEOWNERS").write_text("/gh/ @gh-team\n")

        provider = OwnershipProvider()
        records, receipt = provider.collect(tmp_path, expected_revision=_REVISION)

        assert receipt.sources_scanned == 1
        assert records[0].source_path == ".github/CODEOWNERS"
        assert records[0].path_pattern == "/gh/"

    def test_returns_probe_receipt_unchanged_when_unavailable(self, tmp_path: Path) -> None:
        provider = OwnershipProvider()
        records, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert records == []
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
