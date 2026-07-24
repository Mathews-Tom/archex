"""Tests for the runtime/coverage evidence ingestion dispatcher (M7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from archex.index.runtime_evidence import collect_runtime_evidence
from archex.integrations.runtime.models import (
    CoverageFileEvidence,
    ProviderAvailability,
    RuntimeEvidenceProviderName,
    RuntimeProfileEvidence,
    RuntimeProviderReceipt,
)

_REVISION = "a" * 40


class _StubCoverageProvider:
    def __init__(self, *, raise_error: bool = False) -> None:
        self._raise_error = raise_error

    @property
    def name(self) -> str:
        return "coverage"

    def probe(self, repo_root: Path, *, expected_revision: str) -> RuntimeProviderReceipt:
        del repo_root, expected_revision
        return RuntimeProviderReceipt(
            provider=RuntimeEvidenceProviderName.COVERAGE,
            availability=ProviderAvailability.AVAILABLE,
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[CoverageFileEvidence], RuntimeProviderReceipt]:
        del repo_root
        if self._raise_error:
            raise RuntimeError("boom")
        return (
            [CoverageFileEvidence(file_path="a.py", line_rate=1.0, revision=expected_revision)],
            RuntimeProviderReceipt(
                provider=RuntimeEvidenceProviderName.COVERAGE,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=1,
            ),
        )


class _StubProfileProvider:
    @property
    def name(self) -> str:
        return "runtime_profile"

    def probe(self, repo_root: Path, *, expected_revision: str) -> RuntimeProviderReceipt:
        del repo_root, expected_revision
        return RuntimeProviderReceipt(
            provider=RuntimeEvidenceProviderName.RUNTIME_PROFILE,
            availability=ProviderAvailability.AVAILABLE,
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[RuntimeProfileEvidence], RuntimeProviderReceipt]:
        del repo_root
        return (
            [RuntimeProfileEvidence(total_samples=0, revision=expected_revision)],
            RuntimeProviderReceipt(
                provider=RuntimeEvidenceProviderName.RUNTIME_PROFILE,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=1,
            ),
        )


class TestCollectRuntimeEvidence:
    def test_returns_empty_when_no_providers_requested(self, tmp_path: Path) -> None:
        coverage, profile, receipts = collect_runtime_evidence(
            tmp_path, [], expected_revision=_REVISION
        )
        assert coverage == []
        assert profile == []
        assert receipts == []

    def test_rejects_unknown_provider_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown runtime evidence providers"):
            collect_runtime_evidence(tmp_path, ["bogus"], expected_revision=_REVISION)

    def test_uses_injected_coverage_provider(self, tmp_path: Path) -> None:
        coverage, profile, receipts = collect_runtime_evidence(
            tmp_path,
            ["coverage"],
            expected_revision=_REVISION,
            coverage_provider=_StubCoverageProvider(),
        )
        assert len(coverage) == 1
        assert profile == []
        assert len(receipts) == 1
        assert receipts[0].provider == RuntimeEvidenceProviderName.COVERAGE

    def test_uses_injected_profile_provider(self, tmp_path: Path) -> None:
        coverage, profile, receipts = collect_runtime_evidence(
            tmp_path,
            ["runtime_profile"],
            expected_revision=_REVISION,
            profile_provider=_StubProfileProvider(),
        )
        assert coverage == []
        assert len(profile) == 1
        assert len(receipts) == 1
        assert receipts[0].provider == RuntimeEvidenceProviderName.RUNTIME_PROFILE

    def test_runs_both_providers_when_both_requested(self, tmp_path: Path) -> None:
        coverage, profile, receipts = collect_runtime_evidence(
            tmp_path,
            ["coverage", "runtime_profile"],
            expected_revision=_REVISION,
            coverage_provider=_StubCoverageProvider(),
            profile_provider=_StubProfileProvider(),
        )
        assert len(coverage) == 1
        assert len(profile) == 1
        assert len(receipts) == 2

    def test_provider_exception_degrades_to_unavailable_receipt(self, tmp_path: Path) -> None:
        coverage, _profile, receipts = collect_runtime_evidence(
            tmp_path,
            ["coverage"],
            expected_revision=_REVISION,
            coverage_provider=_StubCoverageProvider(raise_error=True),
        )
        assert coverage == []
        assert len(receipts) == 1
        assert receipts[0].availability == ProviderAvailability.UNAVAILABLE
        assert "boom" in receipts[0].reason

    def test_default_providers_report_unavailable_without_evidence(self, tmp_path: Path) -> None:
        coverage, profile, receipts = collect_runtime_evidence(
            tmp_path, ["coverage", "runtime_profile"], expected_revision=_REVISION
        )
        assert coverage == []
        assert profile == []
        assert {r.provider for r in receipts} == {
            RuntimeEvidenceProviderName.COVERAGE,
            RuntimeEvidenceProviderName.RUNTIME_PROFILE,
        }
        assert all(r.availability == ProviderAvailability.UNAVAILABLE for r in receipts)
