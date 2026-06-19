"""Task-aware retrieval policy: map query modality + budget tier to choices.

Benchmark-only. ``policy_for`` is a pure mapping from a
:class:`~archex.serve.modality.ModalityClassification` to a deterministic
retrieval plan: sparse-vs-dense routing, a conditional dense-expansion trigger,
hard candidate caps, and the expensive steps this lane never runs. The benchmark
``archex_query_task_aware`` strategy consumes this plan; nothing here touches the
product query path, and no hosted calls, telemetry, or network behaviour are
introduced.

The mapping follows the research-backed rules in Workstream 2 of
``.docs/2026-06-18-retrieval-quality-token-efficiency-enhancement-design.md``:
BM25-heavy for PL-to-PL, bounded hybrid/dense for NL-to-PL, sparse-first with
targeted dense expansion for mixed, scaled by the requested token budget.

Whole-file/enclosing-block preservation and graph-expansion strictness are
recorded as declared policy preferences, as are the candidate caps beyond the
sparse confidence window. Mechanical retrieval-aware packing is intentionally
deferred to Workstream 4. The only expensive step this lane unconditionally
skips is the cross-encoder reranker; confidence-aware fusion runs only on the
conditional hybrid/dense pass, and total work is bounded to at most two
retrieval passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from archex.models import RetrievalPolicy
from archex.serve.modality import BudgetTier, ModalityClassification, QueryModality


class DenseTrigger(StrEnum):
    """When the lane may run a dense pass beyond the sparse-first default."""

    NEVER = "never"
    ALWAYS = "always"
    LOW_SPARSE_CONFIDENCE = "low_sparse_confidence"
    DIFFUSE_TOP_SCORES = "diffuse_top_scores"


class GraphExpansion(StrEnum):
    """Declared dependency-frontier expansion regime, scaled by budget."""

    STRICT = "strict"
    NORMAL = "normal"


# The cross-encoder reranker is never run by this lane (recorded as skipped so
# provenance shows the bounded retrieval surface). RRF/confidence-aware fusion is
# NOT listed here: it is used on the conditional hybrid/dense pass and is recorded
# per run in the strategy's runtime provenance, not as an always-skipped step.
_SKIPPED_EXPENSIVE_STEPS = ("cross_encoder_rerank",)


@dataclass(frozen=True)
class TaskAwarePolicy:
    """Deterministic retrieval plan for one modality + budget-tier combination."""

    modality: QueryModality
    budget_tier: BudgetTier
    initial_retrieval_policy: RetrievalPolicy
    use_vector_initial: bool
    dense_trigger: DenseTrigger
    candidate_cap: int
    dense_candidate_cap: int
    rerank_candidate_limit: int
    allow_rerank: bool
    graph_expansion: GraphExpansion
    prefer_whole_file: bool
    diffuse_gap_threshold: float
    skipped_steps: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_provenance(self) -> dict[str, str]:
        """Flatten the policy into string-valued provenance entries."""
        return {
            "policy_initial_retrieval": self.initial_retrieval_policy.value,
            "policy_use_vector_initial": str(self.use_vector_initial).lower(),
            "policy_dense_trigger": self.dense_trigger.value,
            "policy_candidate_cap": str(self.candidate_cap),
            "policy_dense_candidate_cap": str(self.dense_candidate_cap),
            "policy_rerank_candidate_limit": str(self.rerank_candidate_limit),
            "policy_allow_rerank": str(self.allow_rerank).lower(),
            "policy_graph_expansion": self.graph_expansion.value,
            "policy_prefer_whole_file": str(self.prefer_whole_file).lower(),
            "policy_diffuse_gap_threshold": f"{self.diffuse_gap_threshold:.2f}",
            "policy_skipped_steps": ", ".join(self.skipped_steps),
            "policy_reasons": "; ".join(self.reasons),
        }


@dataclass(frozen=True)
class _ModalityRouting:
    initial_retrieval_policy: RetrievalPolicy
    use_vector_initial: bool
    dense_trigger: DenseTrigger
    reason: str


@dataclass(frozen=True)
class _BudgetRouting:
    candidate_cap: int
    dense_candidate_cap: int
    rerank_candidate_limit: int
    graph_expansion: GraphExpansion
    prefer_whole_file: bool
    diffuse_gap_threshold: float
    reason: str


_MODALITY_ROUTING: dict[QueryModality, _ModalityRouting] = {
    QueryModality.PL_TO_PL: _ModalityRouting(
        initial_retrieval_policy=RetrievalPolicy.BM25_ONLY,
        use_vector_initial=False,
        dense_trigger=DenseTrigger.LOW_SPARSE_CONFIDENCE,
        reason="pl_to_pl: BM25-heavy; dense/fusion avoided unless sparse confidence is low",
    ),
    QueryModality.NL_TO_PL: _ModalityRouting(
        initial_retrieval_policy=RetrievalPolicy.HYBRID,
        use_vector_initial=True,
        dense_trigger=DenseTrigger.ALWAYS,
        reason="nl_to_pl: hybrid/dense expansion within strict candidate and latency caps",
    ),
    QueryModality.MIXED: _ModalityRouting(
        initial_retrieval_policy=RetrievalPolicy.BM25_ONLY,
        use_vector_initial=False,
        dense_trigger=DenseTrigger.DIFFUSE_TOP_SCORES,
        reason=(
            "mixed: sparse first; targeted dense expansion only if top sparse scores are diffuse"
        ),
    ),
}


_BUDGET_ROUTING: dict[BudgetTier, _BudgetRouting] = {
    BudgetTier.TIGHT: _BudgetRouting(
        candidate_cap=20,
        dense_candidate_cap=10,
        rerank_candidate_limit=4,
        graph_expansion=GraphExpansion.STRICT,
        prefer_whole_file=False,
        diffuse_gap_threshold=0.10,
        reason=(
            "tight budget: smaller candidate caps, stricter graph expansion, "
            "reluctant dense expansion"
        ),
    ),
    BudgetTier.STANDARD: _BudgetRouting(
        candidate_cap=40,
        dense_candidate_cap=20,
        rerank_candidate_limit=6,
        graph_expansion=GraphExpansion.NORMAL,
        prefer_whole_file=False,
        diffuse_gap_threshold=0.15,
        reason="standard budget: default candidate caps and graph expansion",
    ),
    BudgetTier.LARGE: _BudgetRouting(
        candidate_cap=80,
        dense_candidate_cap=40,
        rerank_candidate_limit=8,
        graph_expansion=GraphExpansion.NORMAL,
        prefer_whole_file=True,
        diffuse_gap_threshold=0.25,
        reason=(
            "large budget: larger candidate caps and whole-file/enclosing-block "
            "preservation preference for top files"
        ),
    ),
}


def policy_for(classification: ModalityClassification) -> TaskAwarePolicy:
    """Map a modality + budget-tier classification to a deterministic policy."""
    routing = _MODALITY_ROUTING[classification.modality]
    budget = _BUDGET_ROUTING[classification.budget_tier]
    return TaskAwarePolicy(
        modality=classification.modality,
        budget_tier=classification.budget_tier,
        initial_retrieval_policy=routing.initial_retrieval_policy,
        use_vector_initial=routing.use_vector_initial,
        dense_trigger=routing.dense_trigger,
        candidate_cap=budget.candidate_cap,
        dense_candidate_cap=budget.dense_candidate_cap,
        rerank_candidate_limit=budget.rerank_candidate_limit,
        allow_rerank=False,
        graph_expansion=budget.graph_expansion,
        prefer_whole_file=budget.prefer_whole_file,
        diffuse_gap_threshold=budget.diffuse_gap_threshold,
        skipped_steps=_SKIPPED_EXPENSIVE_STEPS,
        reasons=(routing.reason, budget.reason),
    )
