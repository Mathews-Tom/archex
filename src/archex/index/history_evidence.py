"""Conditional repository-memory evidence collection (M8): dispatch configured providers.

Zero-cost when no provider names are requested (the default): no provider
module import happens beyond this module's own top-level imports, and no
provider runs. Enabling a provider by name runs it against the given
repository root and revision, and folds every non-``AVAILABLE`` outcome
(unavailable, partial, or revision-stale) into an explicit receipt rather
than a silent gap or a mismatched-revision record being silently applied.

Kept structurally separate from ``archex.index.semantic_evidence`` (M6) and
``archex.index.runtime_evidence`` (M7): every conditional evidence channel
is independently disableable and independently rollback-able, so none of
them import or depend on each other.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import TYPE_CHECKING

from archex.integrations.history.git_log_provider import GitLogHistoryProvider
from archex.integrations.history.models import (
    HistoryEvidenceProviderName,
    HistoryProviderReceipt,
    ProviderAvailability,
)
from archex.integrations.history.operator_rationale_provider import OperatorRationaleProvider

if TYPE_CHECKING:
    from pathlib import Path

    from archex.integrations.history.models import (
        ChangeCard,
        OperatorRationale,
        TemporalCouplingObservation,
    )
    from archex.integrations.history.provider import (
        GitLogEvidenceProvider,
        OperatorRationaleEvidenceProvider,
    )

logger = logging.getLogger(__name__)

#: Provider names accepted by ``IndexConfig.history_evidence_providers``.
KNOWN_HISTORY_EVIDENCE_PROVIDERS = frozenset({"git_log", "operator_rationale"})

#: Default commit-window size for ``git_log`` collection -- bounds cost on a
#: large repository and gives every collection run an explicit, recorded
#: revision range rather than an unbounded full-history walk.
DEFAULT_MAX_COMMITS = 200


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _collect_git_log(
    provider: GitLogEvidenceProvider, repo_root: Path, expected_revision: str, max_commits: int
) -> tuple[list[ChangeCard], list[TemporalCouplingObservation], HistoryProviderReceipt]:
    try:
        return provider.collect(
            repo_root, expected_revision=expected_revision, max_commits=max_commits
        )
    except Exception as exc:
        logger.warning("history evidence provider 'git_log' raised during collect()", exc_info=True)
        return (
            [],
            [],
            HistoryProviderReceipt(
                provider=HistoryEvidenceProviderName.GIT_LOG,
                availability=ProviderAvailability.UNAVAILABLE,
                reason=f"provider raised {type(exc).__name__}: {exc}",
                expected_revision=expected_revision,
                collected_at=_now_iso(),
            ),
        )


def _collect_operator_rationale(
    provider: OperatorRationaleEvidenceProvider, repo_root: Path, expected_revision: str
) -> tuple[list[OperatorRationale], HistoryProviderReceipt]:
    try:
        return provider.collect(repo_root, expected_revision=expected_revision)
    except Exception as exc:
        logger.warning(
            "history evidence provider 'operator_rationale' raised during collect()",
            exc_info=True,
        )
        return [], HistoryProviderReceipt(
            provider=HistoryEvidenceProviderName.OPERATOR_RATIONALE,
            availability=ProviderAvailability.UNAVAILABLE,
            reason=f"provider raised {type(exc).__name__}: {exc}",
            expected_revision=expected_revision,
            collected_at=_now_iso(),
        )


def collect_history_evidence(
    repo_root: Path,
    provider_names: list[str],
    *,
    expected_revision: str,
    max_commits: int = DEFAULT_MAX_COMMITS,
    git_log_provider: GitLogEvidenceProvider | None = None,
    operator_rationale_provider: OperatorRationaleEvidenceProvider | None = None,
) -> tuple[
    list[ChangeCard],
    list[TemporalCouplingObservation],
    list[OperatorRationale],
    list[HistoryProviderReceipt],
]:
    """Run every provider named in *provider_names* against *repo_root*.

    Returns ``([], [], [], [])`` immediately when *provider_names* is empty
    -- the default path adds no cost. *expected_revision* is the current
    index build's resolved git revision; ``git_log`` collects the
    *max_commits*-commit window ending at it, and ``operator_rationale``
    validates its evidence's declared revision against it, reporting
    ``STALE`` rather than applying mismatched evidence.
    *git_log_provider*/*operator_rationale_provider* let a caller inject an
    already-configured provider; the corresponding stock provider is used
    when omitted.

    A provider must never raise for ordinary unavailability, but this is an
    external-tool boundary (a built-in provider's own bug, or a caller-
    injected custom provider), so every ``collect()`` call is defended here:
    an unexpected exception degrades that one provider to an explicit
    ``UNAVAILABLE`` receipt rather than aborting the entire index build.
    """
    if not provider_names:
        return [], [], [], []
    unknown = set(provider_names) - KNOWN_HISTORY_EVIDENCE_PROVIDERS
    if unknown:
        raise ValueError(f"unknown history evidence providers: {sorted(unknown)}")

    change_cards: list[ChangeCard] = []
    coupling_observations: list[TemporalCouplingObservation] = []
    rationale_entries: list[OperatorRationale] = []
    receipts: list[HistoryProviderReceipt] = []

    if "git_log" in provider_names:
        resolved_git_log = git_log_provider or GitLogHistoryProvider()
        change_cards, coupling_observations, git_log_receipt = _collect_git_log(
            resolved_git_log, repo_root, expected_revision, max_commits
        )
        receipts.append(git_log_receipt)
    if "operator_rationale" in provider_names:
        resolved_rationale = operator_rationale_provider or OperatorRationaleProvider()
        rationale_entries, rationale_receipt = _collect_operator_rationale(
            resolved_rationale, repo_root, expected_revision
        )
        receipts.append(rationale_receipt)

    return change_cards, coupling_observations, rationale_entries, receipts
