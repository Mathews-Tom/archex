"""Tests for repository-memory (history) evidence models (M8)."""

from __future__ import annotations

import pytest

from archex.integrations.history.models import (
    ChangeCard,
    HistoryEvidenceProviderName,
    HistoryProviderReceipt,
    LinkedReference,
    OperatorRationale,
    ProviderAvailability,
    TemporalCouplingObservation,
)


class TestLinkedReference:
    def test_rejects_empty_raw_text(self) -> None:
        with pytest.raises(ValueError, match="raw_text"):
            LinkedReference(raw_text=" ", identifier="123")

    def test_rejects_empty_identifier(self) -> None:
        with pytest.raises(ValueError, match="identifier"):
            LinkedReference(raw_text="#123", identifier=" ")

    def test_accepts_valid_reference(self) -> None:
        ref = LinkedReference(raw_text="#123", identifier="123")
        assert ref.identifier == "123"


class TestChangeCard:
    def test_rejects_empty_commit_sha(self) -> None:
        with pytest.raises(ValueError, match="commit_sha"):
            ChangeCard(commit_sha=" ", commit_subject="fix", committed_at="t", revision="rev")

    def test_rejects_empty_revision(self) -> None:
        with pytest.raises(ValueError, match="revision"):
            ChangeCard(commit_sha="abc", commit_subject="fix", committed_at="t", revision=" ")

    def test_rejects_touched_test_files_not_subset_of_changed_files(self) -> None:
        with pytest.raises(ValueError, match="touched_test_files"):
            ChangeCard(
                commit_sha="abc",
                commit_subject="fix",
                committed_at="t",
                changed_files=["a.py"],
                touched_test_files=["test_a.py"],
                revision="rev",
            )

    def test_accepts_valid_change_card(self) -> None:
        card = ChangeCard(
            commit_sha="abc",
            commit_subject="fix bug",
            committed_at="2026-01-01T00:00:00Z",
            changed_files=["a.py", "test_a.py"],
            touched_test_files=["test_a.py"],
            revision="rev",
        )
        assert card.touched_test_files == ["test_a.py"]


class TestTemporalCouplingObservation:
    def test_rejects_same_file(self) -> None:
        with pytest.raises(ValueError, match="different files"):
            TemporalCouplingObservation(
                file_a="a.py",
                file_b="a.py",
                co_change_count=2,
                window_commit_count=5,
                revision="rev",
            )

    def test_rejects_zero_co_change_count(self) -> None:
        with pytest.raises(ValueError, match="co_change_count"):
            TemporalCouplingObservation(
                file_a="a.py",
                file_b="b.py",
                co_change_count=0,
                window_commit_count=5,
                revision="rev",
            )

    def test_rejects_window_smaller_than_co_change_count(self) -> None:
        with pytest.raises(ValueError, match="window_commit_count"):
            TemporalCouplingObservation(
                file_a="a.py",
                file_b="b.py",
                co_change_count=5,
                window_commit_count=2,
                revision="rev",
            )

    def test_accepts_valid_observation(self) -> None:
        obs = TemporalCouplingObservation(
            file_a="a.py", file_b="b.py", co_change_count=3, window_commit_count=10, revision="rev"
        )
        assert obs.co_change_count == 3


class TestOperatorRationale:
    def test_rejects_empty_target_path(self) -> None:
        with pytest.raises(ValueError, match="target_path"):
            OperatorRationale(target_path=" ", rationale="why", recorded_at="t", revision="rev")

    def test_rejects_empty_rationale(self) -> None:
        with pytest.raises(ValueError, match="rationale"):
            OperatorRationale(target_path="a.py", rationale=" ", recorded_at="t", revision="rev")

    def test_accepts_valid_rationale(self) -> None:
        rationale = OperatorRationale(
            target_path="a.py", rationale="why this exists", recorded_at="t", revision="rev"
        )
        assert rationale.author is None


class TestHistoryProviderReceipt:
    def test_reason_required_when_unavailable(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            HistoryProviderReceipt(
                provider=HistoryEvidenceProviderName.GIT_LOG,
                availability=ProviderAvailability.UNAVAILABLE,
            )

    def test_reason_optional_when_available(self) -> None:
        receipt = HistoryProviderReceipt(
            provider=HistoryEvidenceProviderName.GIT_LOG,
            availability=ProviderAvailability.AVAILABLE,
        )
        assert receipt.reason == ""

    def test_rejects_negative_window_commit_count(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            HistoryProviderReceipt(
                provider=HistoryEvidenceProviderName.GIT_LOG,
                availability=ProviderAvailability.AVAILABLE,
                window_commit_count=-1,
            )

    def test_rejects_negative_records_collected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            HistoryProviderReceipt(
                provider=HistoryEvidenceProviderName.OPERATOR_RATIONALE,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=-1,
            )
