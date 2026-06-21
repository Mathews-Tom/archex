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


# Pack decisions ordered richest-first; the packer downgrades a region along this
# chain under budget pressure (INCLUDE -> COMPRESS -> ELIDE -> SKIP).
_CHEAPER: dict[PackDecision, PackDecision] = {
    PackDecision.INCLUDE: PackDecision.COMPRESS,
    PackDecision.COMPRESS: PackDecision.ELIDE,
    PackDecision.ELIDE: PackDecision.SKIP,
}


@dataclass(frozen=True)
class PackingCandidate:
    """A region offered to the packer: its intrinsic signals plus shaped costs.

    ``compressed_token_count`` is the token count of the deterministic compressed
    representation (equal to ``signals.token_count`` when the region is not
    compressible); the packer reads it only when a COMPRESS decision is viable.
    ``elided_token_count`` is the token count of the anchor that replaces the body
    when a region is elided; the packer charges it (capped at the verbatim cost)
    so budget admission reflects the real anchor size rather than a fixed guess.
    """

    signals: PackingSignals
    compressed_token_count: int
    elided_token_count: int


@dataclass(frozen=True)
class PackedRegion:
    """Final pack decision for one candidate, with the score that produced it."""

    candidate_id: str
    decision: PackDecision
    score: PackingScore
    tokens_charged: int
    retrieval_score: float


@dataclass(frozen=True)
class PackingPlan:
    """Result of packing a candidate set under a token budget.

    ``regions`` lists every candidate (including skipped ones) in packed order so
    provenance can explain each include/compress/elide/skip decision.
    """

    regions: list[PackedRegion]
    included_tokens: int
    budget_tier: BudgetTier
    token_budget: int

    def kept(self) -> list[PackedRegion]:
        """Regions that survived packing (everything except SKIP), in packed order."""
        return [region for region in self.regions if region.decision is not PackDecision.SKIP]

    def kept_ids(self) -> list[str]:
        return [region.candidate_id for region in self.kept()]

    def decision_for(self, candidate_id: str) -> PackDecision | None:
        for region in self.regions:
            if region.candidate_id == candidate_id:
                return region.decision
        return None

    def relevance_per_1k_tokens(self) -> float:
        """Delivered retrieval relevance per 1000 packed tokens (0 for an empty pack).

        Only INCLUDE and COMPRESS regions deliver content; an ELIDE region keeps
        an anchor/fetch handle but not the region body, so its retrieval mass is
        excluded from the numerator (counting it would reward eliding evidence).
        """
        if self.included_tokens <= 0:
            return 0.0
        retained = sum(
            region.retrieval_score
            for region in self.regions
            if region.decision in (PackDecision.INCLUDE, PackDecision.COMPRESS)
        )
        return retained / self.included_tokens * 1000.0

    def to_provenance(self) -> dict[str, str]:
        """Aggregate decision counts and efficiency for the strategy/report layer."""
        counts = dict.fromkeys(PackDecision, 0)
        direct = 0
        for region in self.regions:
            counts[region.decision] += 1
            if region.score.direct_match:
                direct += 1
        return {
            "budget_tier": self.budget_tier.value,
            "token_budget": str(self.token_budget),
            "packed_tokens": str(self.included_tokens),
            "candidate_count": str(len(self.regions)),
            "direct_match_count": str(direct),
            "include_count": str(counts[PackDecision.INCLUDE]),
            "compress_count": str(counts[PackDecision.COMPRESS]),
            "elide_count": str(counts[PackDecision.ELIDE]),
            "skip_count": str(counts[PackDecision.SKIP]),
            "relevance_per_1k_tokens": f"{self.relevance_per_1k_tokens():.4f}",
        }


def _decision_cost(decision: PackDecision, candidate: PackingCandidate) -> int:
    if decision is PackDecision.INCLUDE:
        return candidate.signals.token_count
    if decision is PackDecision.COMPRESS:
        return candidate.compressed_token_count
    if decision is PackDecision.ELIDE:
        return min(candidate.elided_token_count, candidate.signals.token_count)
    return 0  # SKIP


def _compress_viable(candidate: PackingCandidate) -> bool:
    """Whether the packer may represent a region as a deterministic compression.

    Direct/high-confidence targets are never compressed (their content is the
    evidence). Only genuinely low-risk, compressible, non-protected regions whose
    compressed form is strictly smaller are eligible.
    """
    signals = candidate.signals
    return (
        not signals.direct_match
        and signals.compression_eligible
        and signals.compression_loss_risk in _COMPRESSIBLE_RISKS
        and not signals.protect_code
        and candidate.compressed_token_count < signals.token_count
    )


def _representation_chain(candidate: PackingCandidate, start: PackDecision) -> list[PackDecision]:
    """Decisions from ``start`` down to SKIP, dropping non-viable representations."""
    chain: list[PackDecision] = []
    decision: PackDecision | None = start
    while decision is not None:
        if decision is not PackDecision.COMPRESS or _compress_viable(candidate):
            chain.append(decision)
        decision = _CHEAPER.get(decision)
    return chain


def _fit_decision(
    candidate: PackingCandidate,
    start: PackDecision,
    remaining: int,
    *,
    allow_skip: bool,
) -> PackDecision:
    """Pick the richest representation at or below ``start`` that fits ``remaining``.

    Optional context falls through to SKIP when nothing fits. A direct/high-
    confidence target is never skipped: if even an anchor exceeds the remaining
    budget it is still kept as an anchor so its fetch handle survives.
    """
    chain = _representation_chain(candidate, start)
    cheapest_kept: PackDecision | None = None
    for decision in chain:
        if decision is PackDecision.SKIP:
            if allow_skip:
                return PackDecision.SKIP
            continue
        cheapest_kept = decision
        if _decision_cost(decision, candidate) <= remaining:
            return decision
    if allow_skip:
        return PackDecision.SKIP
    return cheapest_kept if cheapest_kept is not None else PackDecision.ELIDE


def pack_efficiently(
    candidates: list[PackingCandidate],
    *,
    token_budget: int,
    budget_tier: BudgetTier,
) -> PackingPlan:
    """Select and shape candidates to maximize relevance per token within budget.

    Direct/high-confidence targets are packed before optional context and never
    dropped below an anchor. Optional regions are admitted in efficiency order and
    downgraded (compress -> elide -> skip) as the budget tightens. Whole-file and
    low-value graph-distant context is shrunk or skipped per the score model.
    """
    by_id = {candidate.signals.candidate_id: candidate for candidate in candidates}
    scores = [
        score_candidate(candidate.signals, budget_tier=budget_tier) for candidate in candidates
    ]
    ordered = order_candidates(scores)

    used = 0
    regions: list[PackedRegion] = []
    for score in ordered:
        candidate = by_id[score.candidate_id]
        allow_skip = not score.direct_match
        decision = _fit_decision(
            candidate, score.decision, token_budget - used, allow_skip=allow_skip
        )
        cost = _decision_cost(decision, candidate)
        charged = 0 if decision is PackDecision.SKIP else cost
        used += charged
        regions.append(
            PackedRegion(
                candidate_id=score.candidate_id,
                decision=decision,
                score=score,
                tokens_charged=charged,
                retrieval_score=candidate.signals.retrieval_score,
            )
        )
    return PackingPlan(
        regions=regions,
        included_tokens=used,
        budget_tier=budget_tier,
        token_budget=token_budget,
    )


# --------------------------------------------------------------------------- #
# Query-adaptive diversity packing (benchmark-only experimental lane)
# --------------------------------------------------------------------------- #
#
# Pointwise packing can still spend budget on cross-file/cross-chunk redundancy
# (re-imported symbols, near-duplicate code). Maximal-marginal-relevance (MMR)
# de-selection trims that redundant tail to lift relevance-per-token, but MMR is
# a diversity control, not a guaranteed precision win, so the lane is
# benchmark-only and conservative:
#
# - lambda is query-adaptive: narrow single-aspect lookups keep lambda at 1.0
#   (diversity off; identical to pack_efficiently); multi-aspect queries lower it.
# - required/direct/high-confidence regions are never de-selected.
# - a redundant optional region is dropped only when its file is already
#   represented by another kept region, so the kept-file set is a superset of
#   pack_efficiently's and file recall never regresses.
#
# Similarity is a deterministic, local token-signature Jaccard — no model, no
# network — consistent with the local-first no-inference posture.

# Query-adaptive MMR lambda schedule (DF-RAG style). A single-aspect query keeps
# lambda at 1.0 so diversity is off and the pack is pure relevance; multi-aspect
# queries lower lambda to apply diversity pressure on the redundant tail.
_DIVERSITY_LAMBDA_NARROW = 1.0
_DIVERSITY_LAMBDA_BROAD = 0.7

# Two optional regions are "redundant" when their token signatures overlap at or
# above this Jaccard similarity. Only a redundant region whose file is already
# represented may be de-selected, so file recall is preserved.
_REDUNDANCY_SIMILARITY_THRESHOLD = 0.8


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token signatures (0 when either is empty)."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def query_adaptive_lambda(query_aspects: int) -> float:
    """Resolve the MMR lambda from the query's aspect count.

    A single-aspect (narrow) query keeps lambda at 1.0 so diversity is off and the
    pack is pure relevance; a multi-aspect query lowers lambda to apply diversity.
    """
    return _DIVERSITY_LAMBDA_NARROW if query_aspects <= 1 else _DIVERSITY_LAMBDA_BROAD


@dataclass(frozen=True)
class DiversityPackingPlan(PackingPlan):
    """A :class:`PackingPlan` produced with query-adaptive diversity de-selection.

    Adds the diversity provenance: whether diversity ran, the query aspect count
    and resolved lambda, how many optional regions were de-selected for
    redundancy, and how many protected (direct/high-confidence) regions were
    retained. ``deselected_for_diversity`` counts only redundant-tail de-selection;
    a protected region is never counted because it is never de-selected.
    """

    diversity_applied: bool
    query_aspects: int
    diversity_lambda: float
    deselected_for_diversity: int
    protected_regions: int

    def to_provenance(self) -> dict[str, str]:
        provenance = super().to_provenance()
        provenance.update(
            {
                "diversity_applied": str(self.diversity_applied).lower(),
                "query_aspects": str(self.query_aspects),
                "diversity_lambda": f"{self.diversity_lambda:.2f}",
                "deselected_for_diversity": str(self.deselected_for_diversity),
                "protected_regions": str(self.protected_regions),
            }
        )
        return provenance


def pack_with_diversity(
    candidates: list[PackingCandidate],
    signatures: dict[str, frozenset[str]],
    *,
    token_budget: int,
    budget_tier: BudgetTier,
    query_aspects: int,
) -> DiversityPackingPlan:
    """Pack with query-adaptive MMR diversity over the low-confidence tail.

    Derived from :func:`pack_efficiently`: the baseline plan is computed first,
    then for a multi-aspect query a redundant non-protected optional region is
    flipped to SKIP. Every other region keeps its baseline decision verbatim — the
    diversity plan never upgrades a region to a richer (costlier) representation, so
    it can never re-spend freed budget and starve a later region. A region is
    de-selected only when it is a near-duplicate (token-signature Jaccard at or
    above the redundancy threshold) of an already-kept content region AND its file
    is already represented by another kept region. Because every region's decision
    is the baseline's or SKIP, the per-region charged cost never exceeds the
    baseline's and the first kept region of each file is never de-selected, so the
    diversity kept-file set is a superset of pack_efficiently's — file recall never
    regresses. A required/direct/high-confidence region is never de-selected, and a
    narrow single-aspect query (lambda 1.0) returns the baseline plan unchanged.

    ``signatures`` maps candidate id to a deterministic token signature used for
    redundancy scoring; a missing or empty signature is treated as non-redundant.
    """
    lam = query_adaptive_lambda(query_aspects)
    diversity_applied = lam < _DIVERSITY_LAMBDA_NARROW

    baseline = pack_efficiently(candidates, token_budget=token_budget, budget_tier=budget_tier)
    file_by_id = {
        candidate.signals.candidate_id: candidate.signals.file_path for candidate in candidates
    }

    used = 0
    regions: list[PackedRegion] = []
    kept_signatures: list[frozenset[str]] = []
    file_kept_counts: dict[str, int] = {}
    deselected = 0
    protected = 0
    for region in baseline.regions:
        # Baseline already dropped this region; carry the SKIP through untouched.
        if region.decision is PackDecision.SKIP:
            regions.append(region)
            continue
        is_protected = region.score.direct_match
        if is_protected:
            protected += 1
        signature = signatures.get(region.candidate_id, frozenset())
        file_path = file_by_id[region.candidate_id]
        if diversity_applied and not is_protected and signature:
            max_similarity = max(
                (jaccard(signature, kept) for kept in kept_signatures), default=0.0
            )
            file_represented = file_kept_counts.get(file_path, 0) > 0
            # MMR de-selection: drop a redundant optional region only when its file
            # stays represented, so file recall never regresses.
            if max_similarity >= _REDUNDANCY_SIMILARITY_THRESHOLD and file_represented:
                deselected += 1
                regions.append(
                    PackedRegion(
                        candidate_id=region.candidate_id,
                        decision=PackDecision.SKIP,
                        score=region.score,
                        tokens_charged=0,
                        retrieval_score=region.retrieval_score,
                    )
                )
                continue
        # Keep the baseline decision verbatim — never richer, so budget is never
        # re-spent and a downstream baseline-kept region is never starved.
        regions.append(region)
        used += region.tokens_charged
        file_kept_counts[file_path] = file_kept_counts.get(file_path, 0) + 1
        # Only content-bearing regions anchor redundancy comparisons; an ELIDE
        # keeps an anchor, not the body, so it does not represent content here.
        if region.decision in (PackDecision.INCLUDE, PackDecision.COMPRESS):
            kept_signatures.append(signature)
    return DiversityPackingPlan(
        regions=regions,
        included_tokens=used,
        budget_tier=budget_tier,
        token_budget=token_budget,
        diversity_applied=diversity_applied,
        query_aspects=query_aspects,
        diversity_lambda=lam,
        deselected_for_diversity=deselected,
        protected_regions=protected,
    )
