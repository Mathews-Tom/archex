"""Tests for runtime/coverage evidence models (M7)."""

from __future__ import annotations

import pytest

from archex.integrations.runtime.models import (
    CoverageFileEvidence,
    CoverageLineRecord,
    ProviderAvailability,
    RuntimeEvidenceProviderName,
    RuntimeProfileEvidence,
    RuntimeProviderReceipt,
    RuntimeStackSample,
)


class TestCoverageLineRecord:
    def test_rejects_line_below_one(self) -> None:
        with pytest.raises(ValueError, match="line"):
            CoverageLineRecord(line=0, hits=1)

    def test_rejects_negative_hits(self) -> None:
        with pytest.raises(ValueError, match="hits"):
            CoverageLineRecord(line=1, hits=-1)

    def test_accepts_zero_hits(self) -> None:
        record = CoverageLineRecord(line=1, hits=0)
        assert record.hits == 0


class TestCoverageFileEvidence:
    def test_rejects_empty_file_path(self) -> None:
        with pytest.raises(ValueError, match="file_path"):
            CoverageFileEvidence(file_path=" ", line_rate=0.5, revision="abc123")

    def test_rejects_out_of_range_line_rate(self) -> None:
        with pytest.raises(ValueError, match="line_rate"):
            CoverageFileEvidence(file_path="a.py", line_rate=1.5, revision="abc123")

    def test_rejects_empty_revision(self) -> None:
        with pytest.raises(ValueError, match="revision"):
            CoverageFileEvidence(file_path="a.py", line_rate=0.5, revision=" ")

    def test_accepts_valid_evidence(self) -> None:
        evidence = CoverageFileEvidence(
            file_path="a.py",
            lines=[CoverageLineRecord(line=1, hits=3)],
            line_rate=1.0,
            revision="abc123",
        )
        assert evidence.line_rate == 1.0
        assert evidence.lines[0].hits == 3


class TestRuntimeStackSample:
    def test_rejects_empty_frames(self) -> None:
        with pytest.raises(ValueError, match="frames"):
            RuntimeStackSample(frames=(), sample_count=1)

    def test_rejects_frame_without_colon(self) -> None:
        with pytest.raises(ValueError, match="file_path.*qualified_name"):
            RuntimeStackSample(frames=("no_colon_here",), sample_count=1)

    def test_rejects_zero_sample_count(self) -> None:
        with pytest.raises(ValueError, match="sample_count"):
            RuntimeStackSample(frames=("a.py:func",), sample_count=0)

    def test_accepts_valid_sample(self) -> None:
        sample = RuntimeStackSample(frames=("a.py:outer", "b.py:inner"), sample_count=5)
        assert sample.sample_count == 5
        assert sample.frames == ("a.py:outer", "b.py:inner")


class TestRuntimeProfileEvidence:
    def test_rejects_negative_total_samples(self) -> None:
        with pytest.raises(ValueError, match="total_samples"):
            RuntimeProfileEvidence(total_samples=-1, revision="abc123")

    def test_rejects_empty_revision(self) -> None:
        with pytest.raises(ValueError, match="revision"):
            RuntimeProfileEvidence(total_samples=0, revision=" ")

    def test_accepts_empty_samples(self) -> None:
        evidence = RuntimeProfileEvidence(total_samples=0, revision="abc123")
        assert evidence.samples == []


class TestRuntimeProviderReceipt:
    def test_reason_required_when_unavailable(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            RuntimeProviderReceipt(
                provider=RuntimeEvidenceProviderName.COVERAGE,
                availability=ProviderAvailability.UNAVAILABLE,
            )

    def test_reason_required_when_stale(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            RuntimeProviderReceipt(
                provider=RuntimeEvidenceProviderName.COVERAGE,
                availability=ProviderAvailability.STALE,
            )

    def test_reason_optional_when_available(self) -> None:
        receipt = RuntimeProviderReceipt(
            provider=RuntimeEvidenceProviderName.COVERAGE,
            availability=ProviderAvailability.AVAILABLE,
        )
        assert receipt.reason == ""

    def test_rejects_negative_records_collected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RuntimeProviderReceipt(
                provider=RuntimeEvidenceProviderName.RUNTIME_PROFILE,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=-1,
            )
