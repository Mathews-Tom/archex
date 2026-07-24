"""Tests for the repository-memory evidence ingestion dispatcher (M8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from archex.index.history_evidence import collect_history_evidence
from archex.integrations.history.models import (
    ChangeCard,
    HistoryEvidenceProviderName,
    HistoryProviderReceipt,
    OperatorRationale,
    ProviderAvailability,
    TemporalCouplingObservation,
)

_REVISION = "a" * 40


class _StubGitLogProvider:
    def __init__(self, *, raise_error: bool = False) -> None:
        self._raise_error = raise_error

    @property
    def name(self) -> str:
        return "git_log"

    def probe(self, repo_root: Path, *, expected_revision: str) -> HistoryProviderReceipt:
        del repo_root, expected_revision
        return HistoryProviderReceipt(
            provider=HistoryEvidenceProviderName.GIT_LOG,
            availability=ProviderAvailability.AVAILABLE,
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str, max_commits: int
    ) -> tuple[list[ChangeCard], list[TemporalCouplingObservation], HistoryProviderReceipt]:
        del repo_root, max_commits
        if self._raise_error:
            raise RuntimeError("boom")
        return (
            [
                ChangeCard(
                    commit_sha="abc",
                    commit_subject="s",
                    committed_at="t",
                    revision=expected_revision,
                )
            ],
            [],
            HistoryProviderReceipt(
                provider=HistoryEvidenceProviderName.GIT_LOG,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=1,
            ),
        )


class _StubRationaleProvider:
    @property
    def name(self) -> str:
        return "operator_rationale"

    def probe(self, repo_root: Path, *, expected_revision: str) -> HistoryProviderReceipt:
        del repo_root, expected_revision
        return HistoryProviderReceipt(
            provider=HistoryEvidenceProviderName.OPERATOR_RATIONALE,
            availability=ProviderAvailability.AVAILABLE,
        )

    def collect(
        self, repo_root: Path, *, expected_revision: str
    ) -> tuple[list[OperatorRationale], HistoryProviderReceipt]:
        del repo_root
        return (
            [
                OperatorRationale(
                    target_path="a.py", rationale="why", recorded_at="t", revision=expected_revision
                )
            ],
            HistoryProviderReceipt(
                provider=HistoryEvidenceProviderName.OPERATOR_RATIONALE,
                availability=ProviderAvailability.AVAILABLE,
                records_collected=1,
            ),
        )


class TestCollectHistoryEvidence:
    def test_returns_empty_when_no_providers_requested(self, tmp_path: Path) -> None:
        cards, coupling, rationale, receipts = collect_history_evidence(
            tmp_path, [], expected_revision=_REVISION
        )
        assert cards == []
        assert coupling == []
        assert rationale == []
        assert receipts == []

    def test_rejects_unknown_provider_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown history evidence providers"):
            collect_history_evidence(tmp_path, ["bogus"], expected_revision=_REVISION)

    def test_uses_injected_git_log_provider(self, tmp_path: Path) -> None:
        cards, coupling, rationale, receipts = collect_history_evidence(
            tmp_path,
            ["git_log"],
            expected_revision=_REVISION,
            git_log_provider=_StubGitLogProvider(),
        )
        assert len(cards) == 1
        assert coupling == []
        assert rationale == []
        assert len(receipts) == 1
        assert receipts[0].provider == HistoryEvidenceProviderName.GIT_LOG

    def test_uses_injected_rationale_provider(self, tmp_path: Path) -> None:
        cards, coupling, rationale, receipts = collect_history_evidence(
            tmp_path,
            ["operator_rationale"],
            expected_revision=_REVISION,
            operator_rationale_provider=_StubRationaleProvider(),
        )
        assert cards == []
        assert coupling == []
        assert len(rationale) == 1
        assert len(receipts) == 1
        assert receipts[0].provider == HistoryEvidenceProviderName.OPERATOR_RATIONALE

    def test_runs_both_providers_when_both_requested(self, tmp_path: Path) -> None:
        cards, _coupling, rationale, receipts = collect_history_evidence(
            tmp_path,
            ["git_log", "operator_rationale"],
            expected_revision=_REVISION,
            git_log_provider=_StubGitLogProvider(),
            operator_rationale_provider=_StubRationaleProvider(),
        )
        assert len(cards) == 1
        assert len(rationale) == 1
        assert len(receipts) == 2

    def test_provider_exception_degrades_to_unavailable_receipt(self, tmp_path: Path) -> None:
        cards, _coupling, _rationale, receipts = collect_history_evidence(
            tmp_path,
            ["git_log"],
            expected_revision=_REVISION,
            git_log_provider=_StubGitLogProvider(raise_error=True),
        )
        assert cards == []
        assert len(receipts) == 1
        assert receipts[0].availability == ProviderAvailability.UNAVAILABLE
        assert "boom" in receipts[0].reason

    def test_default_providers_report_unavailable_without_a_git_repo(self, tmp_path: Path) -> None:
        cards, coupling, rationale, receipts = collect_history_evidence(
            tmp_path, ["git_log", "operator_rationale"], expected_revision=_REVISION
        )
        assert cards == []
        assert coupling == []
        assert rationale == []
        assert {r.provider for r in receipts} == {
            HistoryEvidenceProviderName.GIT_LOG,
            HistoryEvidenceProviderName.OPERATOR_RATIONALE,
        }
        assert all(r.availability == ProviderAvailability.UNAVAILABLE for r in receipts)
