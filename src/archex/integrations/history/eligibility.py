"""Repository-memory eligibility policy (M8): density, linkage, and relevance gates.

History evidence is collected unconditionally once ``git_log`` is enabled
(cheap: a single bounded ``git log`` walk), but it must never be *surfaced*
to a report or receipt unless it clears three predeclared thresholds:

- **density** -- is there enough real commit activity in the collected
  window to say anything meaningful (a nearly-empty window is not
  history-rich)?
- **linkage** -- do the collected commits actually relate to each other
  (temporal coupling), or is this a pile of isolated, unrelated changes?
- **relevance** -- does the collected history actually cover the files this
  specific operation (query or diff) cares about?

Any threshold miss disables the channel for that operation with an explicit
reason -- never a partial, silently-degraded surface. This mirrors the
milestone's acceptance: "History is disabled when density, linkage quality,
or query relevance is below its declared threshold."
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from archex.integrations.history.models import (
    ChangeCard,
    HistoryProviderReceipt,
    ProviderAvailability,
    TemporalCouplingObservation,
)

#: Predeclared thresholds (see module docstring). Not user-configurable:
#: DEVELOPMENT_PLAN.md's M8 row calls these "declared thresholds", matching
#: the plan's broader "no automatic provider promotion" posture for every
#: conditional evidence channel -- a caller cannot loosen the gate to force
#: history onto a sparse repository.
MIN_DENSITY = 0.30
MIN_LINKAGE = 0.10
MIN_RELEVANCE = 0.20


class HistoryEligibilityDecision(BaseModel):
    """Explains whether repository-memory evidence is surfaced for one operation.

    Always produced when the ``git_log`` provider was configured, whether or
    not history ends up enabled -- a disabled decision still reports its
    three scores and a human-readable reason.
    """

    enabled: bool
    density_score: float
    linkage_score: float
    relevance_score: float
    reason: str = ""

    @model_validator(mode="after")
    def _validate_decision(self) -> HistoryEligibilityDecision:
        for name, value in (
            ("density_score", self.density_score),
            ("linkage_score", self.linkage_score),
            ("relevance_score", self.relevance_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if not self.enabled and not self.reason.strip():
            raise ValueError("reason is required when history is disabled")
        return self


def _density_score(change_cards: list[ChangeCard], window_commit_count: int) -> float:
    if window_commit_count <= 0:
        return 0.0
    return min(1.0, len(change_cards) / window_commit_count)


def _linkage_score(
    change_cards: list[ChangeCard], coupling_observations: list[TemporalCouplingObservation]
) -> float:
    changed_files = {path for card in change_cards for path in card.changed_files}
    if not changed_files:
        return 0.0
    linked_files = {
        path
        for observation in coupling_observations
        for path in (observation.file_a, observation.file_b)
    }
    return len(linked_files & changed_files) / len(changed_files)


def _relevance_score(
    change_cards: list[ChangeCard],
    coupling_observations: list[TemporalCouplingObservation],
    candidate_file_paths: set[str],
) -> float:
    if not candidate_file_paths:
        return 0.0
    history_files = {path for card in change_cards for path in card.changed_files}
    history_files.update(
        path
        for observation in coupling_observations
        for path in (observation.file_a, observation.file_b)
    )
    return len(candidate_file_paths & history_files) / len(candidate_file_paths)


def evaluate_history_eligibility(
    change_cards: list[ChangeCard],
    coupling_observations: list[TemporalCouplingObservation],
    candidate_file_paths: set[str],
    *,
    git_log_receipt: HistoryProviderReceipt | None,
    window_commit_count: int,
    min_density: float = MIN_DENSITY,
    min_linkage: float = MIN_LINKAGE,
    min_relevance: float = MIN_RELEVANCE,
) -> HistoryEligibilityDecision:
    """Decide whether to surface history evidence for one query or diff.

    ``candidate_file_paths`` is the operation's own relevant file set (a
    query's returned/candidate files, or a diff's changed files) -- the
    basis for the relevance score. Disabled whenever the ``git_log``
    provider itself was not ``AVAILABLE``, or when any of the three scores
    falls below its threshold.
    """
    if git_log_receipt is None or git_log_receipt.availability != ProviderAvailability.AVAILABLE:
        reason_detail = git_log_receipt.reason if git_log_receipt is not None else "not collected"
        return HistoryEligibilityDecision(
            enabled=False,
            density_score=0.0,
            linkage_score=0.0,
            relevance_score=0.0,
            reason=f"git_log evidence unavailable: {reason_detail or 'unknown reason'}",
        )

    density = _density_score(change_cards, window_commit_count)
    linkage = _linkage_score(change_cards, coupling_observations)
    relevance = _relevance_score(change_cards, coupling_observations, candidate_file_paths)

    failures: list[str] = []
    if density < min_density:
        failures.append(f"density {density:.2f} < {min_density:.2f}")
    if linkage < min_linkage:
        failures.append(f"linkage {linkage:.2f} < {min_linkage:.2f}")
    if relevance < min_relevance:
        failures.append(f"relevance {relevance:.2f} < {min_relevance:.2f}")

    if failures:
        return HistoryEligibilityDecision(
            enabled=False,
            density_score=density,
            linkage_score=linkage,
            relevance_score=relevance,
            reason="below threshold: " + "; ".join(failures),
        )
    return HistoryEligibilityDecision(
        enabled=True,
        density_score=density,
        linkage_score=linkage,
        relevance_score=relevance,
    )
