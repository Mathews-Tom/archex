"""Efficiency-aware packing score model.

A deterministic per-candidate packing score that folds intrinsic retrieval
signals into a single priority value plus provenance explaining the include /
compress / elide / skip decision. The model uses ONLY signals already present in
a retrieved bundle and its receipt — retrieval score, direct path/symbol match
status, graph edge confidence and distance, token count, compression
eligibility and loss risk, scout/fetch handle priority, and the budget tier. It
never reads benchmark ground truth (expected files, expected regions, or
completion labels), so it is safe to import from product code; the
benchmark-only ``archex_query_efficiency_packed`` lane is the first consumer.

The score model decides *priority* and a *provisional decision* per candidate.
The packer (built on top of this model) enforces the token budget and may
downgrade a provisional ``INCLUDE`` toward ``COMPRESS``/``ELIDE``/``SKIP`` under
budget pressure, but a direct/high-confidence target is always preserved before
optional context.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from archex.models import CompressionLossRisk
from archex.serve.modality import BudgetTier


class PackDecision(StrEnum):
    """What the packer does with one candidate region."""

    INCLUDE = "include"  # keep the region verbatim
    COMPRESS = "compress"  # keep a deterministic, low-risk compressed representation
    ELIDE = "elide"  # keep only an anchor/marker pointing at the original
    SKIP = "skip"  # drop the region entirely


# A candidate is "large" past this many tokens; combined with a low retrieval
# score and graph distance it becomes a budget-wasting expansion.
_LARGE_TOKEN_COUNT = 400
# Normalized final retrieval score below which a candidate is "low score" for
# the large-low-score-graph-distant penalty.
_LOW_SCORE = 0.25
# A graph hop of at least this distance is "graph-distant".
_DISTANT_HOP = 1

# Score weights. ``value`` is a relevance-per-token efficiency score adjusted by
# distance/risk penalties; direct-match preservation is enforced structurally by
# ``order_candidates`` rather than by an unbounded score bonus.
_HANDLE_PRIORITY_WEIGHT = 50.0
_GRAPH_DISTANCE_PENALTY = 20.0  # per hop, always applied to graph-expanded context
_LOW_CONFIDENCE_PENALTY = 30.0  # per hop, scaled by (1 - edge confidence)
_HIGH_RISK_PENALTY = 40.0
_LARGE_LOW_SCORE_PENALTY = 30.0

# Compression is only ever *preferred* for genuinely low-risk context; medium and
# high loss risk are never auto-compressed by the packer.
_COMPRESSIBLE_RISKS = frozenset({CompressionLossRisk.NONE, CompressionLossRisk.LOW})


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


@dataclass(frozen=True)
class PackingSignals:
    """Intrinsic per-candidate signals for the packer. No benchmark ground truth.

    Every field is derivable from a retrieved bundle and its receipt. Expected
    file/region coverage is deliberately absent so the score model cannot leak
    benchmark ground truth into a retrieval decision.
    """

    candidate_id: str
    file_path: str
    # Normalized 0..1 final retrieval score for the region.
    retrieval_score: float
    # Direct path/symbol match or otherwise high-confidence edit target.
    direct_match: bool
    # 0 = seed/direct retrieval, >=1 = graph-expanded neighbour.
    graph_distance: int
    # 0..1 confidence of the edge that admitted the region (1.0 when direct).
    graph_edge_confidence: float
    # Tokens the verbatim region would consume.
    token_count: int
    # A deterministic compression mode would shrink the region.
    compression_eligible: bool
    compression_loss_risk: CompressionLossRisk
    # Scout/fetch handle priority, 0..1 (0 when the region has no handle).
    handle_priority: float
    # Region is whole-file/module level rather than a smaller symbol/block.
    whole_file: bool
    # Count of high-confidence (direct-match) regions in the same file.
    file_evidence_regions: int
    # Query intent forbids lossy in-place compression of code; an anchor/elide
    # that still points at untouched source is allowed (fix/debug/review intents).
    protect_code: bool


@dataclass(frozen=True)
class PackingScore:
    """Deterministic packing score and provenance for one candidate.

    ``value`` is the relevance-per-token priority among same-tier candidates
    (higher packs first). ``decision`` is the provisional pack decision derived
    from per-candidate signals and budget tier; the packer may downgrade it under
    budget pressure but never drops a direct match below an anchor.
    """

    candidate_id: str
    value: float
    relevance_per_1k_tokens: float
    decision: PackDecision
    reason: str
    direct_match: bool
    graph_distance: int
    compression_loss_risk: CompressionLossRisk

    def to_provenance(self) -> dict[str, str]:
        """Flatten the score into string-valued provenance entries."""
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "value": f"{self.value:.4f}",
            "relevance_per_1k_tokens": f"{self.relevance_per_1k_tokens:.4f}",
            "direct_match": str(self.direct_match).lower(),
            "graph_distance": str(self.graph_distance),
            "compression_loss_risk": self.compression_loss_risk.value,
        }


def relevance_per_1k_tokens(retrieval_score: float, token_count: int) -> float:
    """Retrieval score normalized per 1000 consumed tokens (0 for empty regions)."""
    if token_count <= 0:
        return 0.0
    return retrieval_score / token_count * 1000.0


def _is_low_value(signals: PackingSignals) -> bool:
    """Large, low-score, and graph-distant: a budget-wasting expansion."""
    return (
        signals.token_count >= _LARGE_TOKEN_COUNT
        and signals.retrieval_score < _LOW_SCORE
        and signals.graph_distance >= _DISTANT_HOP
    )


def _provisional_decision(
    signals: PackingSignals, *, budget_tier: BudgetTier
) -> tuple[PackDecision, str]:
    """Per-candidate pack decision before the packer applies the token budget."""
    if signals.direct_match:
        return PackDecision.INCLUDE, "direct/high-confidence target preserved"

    low_risk = (
        signals.compression_eligible
        and signals.compression_loss_risk in _COMPRESSIBLE_RISKS
        and not signals.protect_code
    )

    # Whole-file context only earns the budget at a large tier or when the file
    # carries multiple high-confidence regions; otherwise a smaller enclosing
    # symbol/block is preferred and the whole-file region is shrunk to evidence.
    if signals.whole_file and not (
        budget_tier is BudgetTier.LARGE or signals.file_evidence_regions >= 2
    ):
        if low_risk:
            return (
                PackDecision.COMPRESS,
                "whole-file region compressed (low risk); smaller enclosing "
                "evidence preferred under non-large budget",
            )
        return (
            PackDecision.ELIDE,
            "whole-file region elided to an anchor; smaller enclosing evidence "
            "preferred under non-large budget",
        )

    if _is_low_value(signals):
        if low_risk:
            return (
                PackDecision.COMPRESS,
                "large low-score graph-distant region compressed (low risk)",
            )
        high_risk = signals.compression_loss_risk is CompressionLossRisk.HIGH
        return (
            PackDecision.SKIP,
            "large, low-score, graph-distant"
            + (", high compression risk" if high_risk else "")
            + " -> skipped",
        )

    return PackDecision.INCLUDE, "relevant region included"


def score_candidate(signals: PackingSignals, *, budget_tier: BudgetTier) -> PackingScore:
    """Combine intrinsic signals into a deterministic packing score + decision."""
    rel_per_1k = relevance_per_1k_tokens(signals.retrieval_score, signals.token_count)

    value = rel_per_1k
    value += _clamp01(signals.handle_priority) * _HANDLE_PRIORITY_WEIGHT
    if signals.graph_distance > 0:
        confidence_gap = 1.0 - _clamp01(signals.graph_edge_confidence)
        value -= signals.graph_distance * (
            _GRAPH_DISTANCE_PENALTY + _LOW_CONFIDENCE_PENALTY * confidence_gap
        )
    if signals.compression_loss_risk is CompressionLossRisk.HIGH:
        value -= _HIGH_RISK_PENALTY
    if _is_low_value(signals):
        value -= _LARGE_LOW_SCORE_PENALTY

    decision, reason = _provisional_decision(signals, budget_tier=budget_tier)
    return PackingScore(
        candidate_id=signals.candidate_id,
        value=value,
        relevance_per_1k_tokens=rel_per_1k,
        decision=decision,
        reason=reason,
        direct_match=signals.direct_match,
        graph_distance=signals.graph_distance,
        compression_loss_risk=signals.compression_loss_risk,
    )


def order_candidates(scores: list[PackingScore]) -> list[PackingScore]:
    """Deterministic packing order with direct-match preservation.

    Direct/high-confidence targets sort ahead of all optional context regardless
    of raw efficiency, so they are always packed first. Within a tier, regions
    sort by descending ``value``; ties break toward nearer graph distance and
    then by ``candidate_id`` so the order is fully deterministic and independent
    of input ordering.
    """
    return sorted(
        scores,
        key=lambda score: (
            not score.direct_match,
            -score.value,
            score.graph_distance,
            score.candidate_id,
        ),
    )
