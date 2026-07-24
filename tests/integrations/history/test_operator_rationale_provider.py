"""Tests for the operator-rationale history evidence provider."""

from __future__ import annotations

import json
from pathlib import Path

from archex.integrations.history.models import ProviderAvailability
from archex.integrations.history.operator_rationale_provider import OperatorRationaleProvider

_REVISION = "a" * 40
_OTHER_REVISION = "b" * 40


def _write_evidence(
    tmp_path: Path,
    *,
    revision: str = _REVISION,
    entries: list[dict[str, object]] | None = None,
) -> Path:
    evidence_dir = tmp_path / ".archex" / "history-evidence" / "rationale"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "manifest.json").write_text(json.dumps({"revision": revision}))
    default_entries = (
        entries
        if entries is not None
        else [
            {
                "target_path": "src/a.py",
                "rationale": "Chosen for compatibility with legacy callers.",
                "author": "op",
                "recorded_at": "2026-01-01T00:00:00Z",
            }
        ]
    )
    (evidence_dir / "rationale.json").write_text(json.dumps(default_entries))
    return tmp_path


class TestOperatorRationaleProviderProbe:
    def test_unavailable_when_no_evidence(self, tmp_path: Path) -> None:
        provider = OperatorRationaleProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.UNAVAILABLE

    def test_stale_on_revision_mismatch(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, revision=_OTHER_REVISION)
        provider = OperatorRationaleProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.STALE
        assert receipt.observed_revision == _OTHER_REVISION

    def test_available_on_revision_match(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path)
        provider = OperatorRationaleProvider()
        receipt = provider.probe(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE


class TestOperatorRationaleProviderCollect:
    def test_collect_returns_unavailable_and_empty_when_missing(self, tmp_path: Path) -> None:
        provider = OperatorRationaleProvider()
        entries, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert entries == []
        assert receipt.availability == ProviderAvailability.UNAVAILABLE

    def test_collect_returns_stale_and_empty_on_mismatch(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path, revision=_OTHER_REVISION)
        provider = OperatorRationaleProvider()
        entries, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert entries == []
        assert receipt.availability == ProviderAvailability.STALE

    def test_collect_parses_valid_entries(self, tmp_path: Path) -> None:
        _write_evidence(tmp_path)
        provider = OperatorRationaleProvider()
        entries, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert receipt.availability == ProviderAvailability.AVAILABLE
        assert len(entries) == 1
        assert entries[0].target_path == "src/a.py"
        assert entries[0].author == "op"
        assert entries[0].revision == _REVISION

    def test_collect_skips_entries_missing_required_fields(self, tmp_path: Path) -> None:
        _write_evidence(
            tmp_path,
            entries=[
                {"target_path": "a.py", "rationale": "ok", "recorded_at": "t"},
                {"target_path": "b.py"},
                {"rationale": "no path", "recorded_at": "t"},
            ],
        )
        provider = OperatorRationaleProvider()
        entries, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert len(entries) == 1
        assert entries[0].target_path == "a.py"
        assert receipt.records_collected == 1

    def test_collect_reports_stale_on_non_list_rationale_file(self, tmp_path: Path) -> None:
        evidence_dir = tmp_path / ".archex" / "history-evidence" / "rationale"
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "manifest.json").write_text(json.dumps({"revision": _REVISION}))
        (evidence_dir / "rationale.json").write_text(json.dumps({"not": "a list"}))
        provider = OperatorRationaleProvider()
        entries, receipt = provider.collect(tmp_path, expected_revision=_REVISION)
        assert entries == []
        assert receipt.availability == ProviderAvailability.STALE
