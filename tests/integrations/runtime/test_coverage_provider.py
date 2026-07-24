"""Tests for the coverage evidence provider."""

from __future__ import annotations

import json
from pathlib import Path

from archex.integrations.runtime.coverage_provider import (
    CoverageXmlProvider,
    current_repo_revision,
)
from archex.integrations.runtime.models import ProviderAvailability

_REVISION = "a" * 40
_OTHER_REVISION = "b" * 40

_COBERTURA_XML = """<?xml version="1.0" ?>
<coverage line-rate="0.75">
  <packages>
    <package name="pkg">
      <classes>
        <class filename="src/pkg/module.py" line-rate="0.75">
          <lines>
            <line number="1" hits="3"/>
            <line number="2" hits="0"/>
          </lines>
        </class>
        <class filename="../outside/escape.py" line-rate="1.0">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""


def _write_evidence(
    tmp_path: Path,
    *,
    revision: str = _REVISION,
    tool: str = "coverage.py",
    tool_version: str | None = "7.6.0",
) -> Path:
    evidence_dir = tmp_path / ".archex" / "runtime-evidence" / "coverage"
    evidence_dir.mkdir(parents=True)
    manifest = {"revision": revision, "tool": tool}
    if tool_version is not None:
        manifest["tool_version"] = tool_version
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
    (evidence_dir / "coverage.xml").write_text(_COBERTURA_XML)
    return tmp_path


class TestCoverageXmlProviderProbe:
    def test_unavailable_when_no_evidence(self, tmp_path: Path) -> None:
        provider = CoverageXmlProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "no coverage evidence" in receipt.reason

    def test_unavailable_when_manifest_has_no_revision(self, tmp_path: Path) -> None:
        evidence_dir = tmp_path / ".archex" / "runtime-evidence" / "coverage"
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "manifest.json").write_text(json.dumps({"tool": "coverage.py"}))
        (evidence_dir / "coverage.xml").write_text(_COBERTURA_XML)
        provider = CoverageXmlProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "no revision" in receipt.reason

    def test_unavailable_when_manifest_is_malformed_json(self, tmp_path: Path) -> None:
        evidence_dir = tmp_path / ".archex" / "runtime-evidence" / "coverage"
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "manifest.json").write_text("{not json")
        (evidence_dir / "coverage.xml").write_text(_COBERTURA_XML)
        provider = CoverageXmlProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE
        assert "could not read coverage manifest" in receipt.reason

    def test_stale_on_revision_mismatch(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, revision=_OTHER_REVISION)
        provider = CoverageXmlProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.STALE
        assert receipt.observed_revision == _OTHER_REVISION
        assert receipt.expected_revision == _REVISION

    def test_available_on_revision_match(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path)
        provider = CoverageXmlProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.tool_name == "coverage.py"
        assert receipt.tool_version == "7.6.0"

    def test_uses_injected_evidence_dir(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "custom-evidence"
        custom_dir.mkdir()
        manifest = {"revision": _REVISION, "tool": "coverage.py"}
        (custom_dir / "manifest.json").write_text(json.dumps(manifest))
        (custom_dir / "coverage.xml").write_text(_COBERTURA_XML)
        provider = CoverageXmlProvider(evidence_dir=custom_dir)
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE


class TestCoverageXmlProviderCollect:
    def test_collect_returns_unavailable_receipt_and_empty_when_missing(
        self, tmp_path: Path
    ) -> None:
        provider = CoverageXmlProvider()
        evidence, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.UNAVAILABLE

    def test_collect_returns_stale_receipt_and_empty_on_mismatch(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, revision=_OTHER_REVISION)
        provider = CoverageXmlProvider()
        evidence, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.STALE

    def test_collect_parses_lines_and_line_rate(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path)
        provider = CoverageXmlProvider()
        evidence, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert receipt.records_collected == 1
        assert len(evidence) == 1
        record = evidence[0]
        assert record.file_path == "src/pkg/module.py"
        assert record.revision == _REVISION
        assert record.line_rate == 0.75
        assert {(line.line, line.hits) for line in record.lines} == {(1, 3), (2, 0)}

    def test_collect_drops_files_outside_repo_root(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path)
        provider = CoverageXmlProvider()
        evidence, _receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert all("outside" not in record.file_path for record in evidence)

    def test_collect_reports_stale_on_unparseable_xml(self, tmp_path: Path) -> None:
        evidence_dir = tmp_path / ".archex" / "runtime-evidence" / "coverage"
        evidence_dir.mkdir(parents=True)
        manifest = {"revision": _REVISION, "tool": "coverage.py"}
        (evidence_dir / "manifest.json").write_text(json.dumps(manifest))
        (evidence_dir / "coverage.xml").write_text("<not-xml")
        provider = CoverageXmlProvider()
        evidence, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert evidence == []
        assert receipt.availability == ProviderAvailability.STALE
        assert "could not parse" in receipt.reason


class TestCurrentRepoRevision:
    def test_returns_none_for_non_git_dir(self, tmp_path: Path) -> None:
        assert current_repo_revision(tmp_path) is None

    def test_resolves_head_for_this_repo(self) -> None:
        revision = current_repo_revision(Path(__file__).resolve().parents[3])
        assert revision is not None
        assert len(revision) == 40
