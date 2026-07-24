"""Tests for the M8 repository-memory eligibility policy."""

from __future__ import annotations

import pytest

from archex.integrations.history.eligibility import (
    MIN_DENSITY,
    MIN_LINKAGE,
    MIN_RELEVANCE,
    HistoryEligibilityDecision,
    evaluate_history_eligibility,
)
from archex.integrations.history.models import (
    ChangeCard,
    HistoryEvidenceProviderName,
    HistoryProviderReceipt,
    ProviderAvailability,
    TemporalCouplingObservation,
)


def _card(commit_sha: str, changed_files: list[str]) -> ChangeCard:
    return ChangeCard(
        commit_sha=commit_sha,
        commit_subject="s",
        committed_at="t",
        changed_files=changed_files,
        revision="rev",
    )


def _available_receipt(window_commit_count: int) -> HistoryProviderReceipt:
    return HistoryProviderReceipt(
        provider=HistoryEvidenceProviderName.GIT_LOG,
        availability=ProviderAvailability.AVAILABLE,
        window_commit_count=window_commit_count,
    )


class TestHistoryEligibilityDecision:
    def test_rejects_out_of_range_score(self) -> None:
        with pytest.raises(ValueError, match="density_score"):
            HistoryEligibilityDecision(
                enabled=True, density_score=1.5, linkage_score=0.5, relevance_score=0.5
            )

    def test_reason_required_when_disabled(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            HistoryEligibilityDecision(
                enabled=False, density_score=0.1, linkage_score=0.1, relevance_score=0.1
            )


class TestEvaluateHistoryEligibility:
    def test_disabled_when_git_log_receipt_missing(self) -> None:
        decision = evaluate_history_eligibility(
            [], [], set(), git_log_receipt=None, window_commit_count=0
        )
        assert decision.enabled is False
        assert "unavailable" in decision.reason

    def test_disabled_when_git_log_receipt_unavailable(self) -> None:
        receipt = HistoryProviderReceipt(
            provider=HistoryEvidenceProviderName.GIT_LOG,
            availability=ProviderAvailability.UNAVAILABLE,
            reason="not a git repo",
        )
        decision = evaluate_history_eligibility(
            [], [], set(), git_log_receipt=receipt, window_commit_count=0
        )
        assert decision.enabled is False
        assert "not a git repo" in decision.reason

    def test_disabled_below_density_threshold(self) -> None:
        # 1 change card in a 100-commit window -> density far below threshold.
        cards = [_card("c1", ["a.py"])]
        decision = evaluate_history_eligibility(
            cards,
            [],
            {"a.py"},
            git_log_receipt=_available_receipt(100),
            window_commit_count=100,
        )
        assert decision.enabled is False
        assert "density" in decision.reason

    def test_disabled_below_linkage_threshold(self) -> None:
        # Dense window, but files never co-change -> zero linkage.
        cards = [_card(f"c{i}", [f"f{i}.py"]) for i in range(10)]
        decision = evaluate_history_eligibility(
            cards,
            [],
            {"f0.py"},
            git_log_receipt=_available_receipt(10),
            window_commit_count=10,
        )
        assert decision.enabled is False
        assert "linkage" in decision.reason

    def test_disabled_below_relevance_threshold(self) -> None:
        cards = [_card("c1", ["a.py", "b.py"]), _card("c2", ["a.py", "b.py"])]
        coupling = [
            TemporalCouplingObservation(
                file_a="a.py",
                file_b="b.py",
                co_change_count=2,
                window_commit_count=2,
                revision="rev",
            )
        ]
        # candidate files are entirely unrelated to the collected history.
        decision = evaluate_history_eligibility(
            cards,
            coupling,
            {"unrelated.py"},
            git_log_receipt=_available_receipt(2),
            window_commit_count=2,
        )
        assert decision.enabled is False
        assert "relevance" in decision.reason

    def test_enabled_when_all_thresholds_clear(self) -> None:
        cards = [_card("c1", ["a.py", "b.py"]), _card("c2", ["a.py", "b.py"])]
        coupling = [
            TemporalCouplingObservation(
                file_a="a.py",
                file_b="b.py",
                co_change_count=2,
                window_commit_count=2,
                revision="rev",
            )
        ]
        decision = evaluate_history_eligibility(
            cards,
            coupling,
            {"a.py"},
            git_log_receipt=_available_receipt(2),
            window_commit_count=2,
        )
        assert decision.enabled is True
        assert decision.density_score >= MIN_DENSITY
        assert decision.linkage_score >= MIN_LINKAGE
        assert decision.relevance_score >= MIN_RELEVANCE

    def test_empty_candidate_paths_yields_zero_relevance(self) -> None:
        cards = [_card("c1", ["a.py", "b.py"]), _card("c2", ["a.py", "b.py"])]
        coupling = [
            TemporalCouplingObservation(
                file_a="a.py",
                file_b="b.py",
                co_change_count=2,
                window_commit_count=2,
                revision="rev",
            )
        ]
        decision = evaluate_history_eligibility(
            cards,
            coupling,
            set(),
            git_log_receipt=_available_receipt(2),
            window_commit_count=2,
        )
        assert decision.enabled is False
        assert decision.relevance_score == 0.0
