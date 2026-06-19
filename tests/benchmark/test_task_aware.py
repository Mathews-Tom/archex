"""Tests for the benchmark-only task-aware retrieval policy mapping."""

from __future__ import annotations

import pytest

from archex.benchmark.task_aware import (
    DenseTrigger,
    GraphExpansion,
    TaskAwarePolicy,
    policy_for,
)
from archex.models import RetrievalPolicy
from archex.serve.intent import QueryIntent
from archex.serve.modality import (
    BudgetTier,
    ModalityClassification,
    ModalitySignals,
    QueryModality,
    classify_query,
)


def _classification(modality: QueryModality, tier: BudgetTier) -> ModalityClassification:
    signals = ModalitySignals(
        word_count=0,
        identifier_count=0,
        identifier_density=0.0,
        has_code_fence=False,
        has_stack_trace=False,
        path_mentions=0,
        symbol_mentions=0,
        natural_language_word_count=0,
        natural_language_ratio=0.0,
        query_intent=QueryIntent.GENERAL,
    )
    return ModalityClassification(
        modality=modality,
        budget_tier=tier,
        token_budget=4096,
        signals=signals,
        reasons=(),
    )


_MODALITY_EXPECTATIONS = {
    QueryModality.PL_TO_PL: (RetrievalPolicy.BM25_ONLY, False, DenseTrigger.LOW_SPARSE_CONFIDENCE),
    QueryModality.NL_TO_PL: (RetrievalPolicy.HYBRID, True, DenseTrigger.ALWAYS),
    QueryModality.MIXED: (RetrievalPolicy.BM25_ONLY, False, DenseTrigger.DIFFUSE_TOP_SCORES),
}

_BUDGET_EXPECTATIONS = {
    # tier: (candidate_cap, dense_candidate_cap, rerank_limit, graph, whole_file, gap)
    BudgetTier.TIGHT: (20, 10, 4, GraphExpansion.STRICT, False, 0.10),
    BudgetTier.STANDARD: (40, 20, 6, GraphExpansion.NORMAL, False, 0.15),
    BudgetTier.LARGE: (80, 40, 8, GraphExpansion.NORMAL, True, 0.25),
}


class TestPolicyMapping:
    @pytest.mark.parametrize("modality", list(QueryModality))
    @pytest.mark.parametrize("tier", list(BudgetTier))
    def test_policy_routes_each_combination(
        self, modality: QueryModality, tier: BudgetTier
    ) -> None:
        policy = policy_for(_classification(modality, tier))
        assert isinstance(policy, TaskAwarePolicy)
        assert policy.modality is modality
        assert policy.budget_tier is tier

        retrieval, use_vector, trigger = _MODALITY_EXPECTATIONS[modality]
        assert policy.initial_retrieval_policy is retrieval
        assert policy.use_vector_initial is use_vector
        assert policy.dense_trigger is trigger

        cap, dense_cap, rerank_limit, graph, whole_file, gap = _BUDGET_EXPECTATIONS[tier]
        assert policy.candidate_cap == cap
        assert policy.dense_candidate_cap == dense_cap
        assert policy.rerank_candidate_limit == rerank_limit
        assert policy.graph_expansion is graph
        assert policy.prefer_whole_file is whole_file
        assert abs(policy.diffuse_gap_threshold - gap) < 1e-9

    @pytest.mark.parametrize("modality", list(QueryModality))
    @pytest.mark.parametrize("tier", list(BudgetTier))
    def test_cross_encoder_rerank_always_skipped(
        self, modality: QueryModality, tier: BudgetTier
    ) -> None:
        policy = policy_for(_classification(modality, tier))
        # The cross-encoder reranker is never run. Fusion is conditional and is
        # therefore not listed as an always-skipped step.
        assert policy.allow_rerank is False
        assert "cross_encoder_rerank" in policy.skipped_steps
        assert "rrf_fusion" not in policy.skipped_steps

    @pytest.mark.parametrize("tier", list(BudgetTier))
    def test_cap_ordering_invariants(self, tier: BudgetTier) -> None:
        policy = policy_for(_classification(QueryModality.NL_TO_PL, tier))
        # The lane must never request more dense/rerank candidates than its
        # overall candidate cap, or it would defeat the bound.
        assert policy.dense_candidate_cap <= policy.candidate_cap
        assert policy.rerank_candidate_limit <= policy.dense_candidate_cap

    def test_caps_increase_with_budget(self) -> None:
        tight = policy_for(_classification(QueryModality.NL_TO_PL, BudgetTier.TIGHT))
        standard = policy_for(_classification(QueryModality.NL_TO_PL, BudgetTier.STANDARD))
        large = policy_for(_classification(QueryModality.NL_TO_PL, BudgetTier.LARGE))
        # Caps and dense willingness grow monotonically tight < standard < large.
        assert tight.candidate_cap < standard.candidate_cap < large.candidate_cap
        assert tight.dense_candidate_cap < standard.dense_candidate_cap < large.dense_candidate_cap
        assert (
            tight.rerank_candidate_limit
            < standard.rerank_candidate_limit
            < large.rerank_candidate_limit
        )
        assert (
            tight.diffuse_gap_threshold
            < standard.diffuse_gap_threshold
            < large.diffuse_gap_threshold
        )

    def test_pl_to_pl_is_sparse_first(self) -> None:
        policy = policy_for(_classification(QueryModality.PL_TO_PL, BudgetTier.STANDARD))
        assert policy.initial_retrieval_policy is RetrievalPolicy.BM25_ONLY
        assert policy.use_vector_initial is False

    def test_nl_to_pl_allows_initial_dense(self) -> None:
        policy = policy_for(_classification(QueryModality.NL_TO_PL, BudgetTier.STANDARD))
        assert policy.initial_retrieval_policy is RetrievalPolicy.HYBRID
        assert policy.use_vector_initial is True


class TestPolicyProvenance:
    def test_provenance_is_string_valued(self) -> None:
        policy = policy_for(_classification(QueryModality.MIXED, BudgetTier.LARGE))
        provenance = policy.to_provenance()
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in provenance.items())
        assert provenance["policy_initial_retrieval"] == RetrievalPolicy.BM25_ONLY.value
        assert provenance["policy_dense_trigger"] == DenseTrigger.DIFFUSE_TOP_SCORES.value
        assert provenance["policy_candidate_cap"] == "80"
        assert provenance["policy_dense_candidate_cap"] == "40"
        assert provenance["policy_allow_rerank"] == "false"
        assert provenance["policy_prefer_whole_file"] == "true"
        assert "cross_encoder_rerank" in provenance["policy_skipped_steps"]
        assert provenance["policy_diffuse_gap_threshold"] == "0.25"
        assert set(provenance) == {
            "policy_initial_retrieval",
            "policy_use_vector_initial",
            "policy_dense_trigger",
            "policy_candidate_cap",
            "policy_dense_candidate_cap",
            "policy_rerank_candidate_limit",
            "policy_allow_rerank",
            "policy_graph_expansion",
            "policy_prefer_whole_file",
            "policy_diffuse_gap_threshold",
            "policy_skipped_steps",
            "policy_reasons",
        }
        assert provenance["policy_reasons"]


class TestPolicyFromClassifier:
    def test_code_query_tight_budget_is_bm25_only(self) -> None:
        classification = classify_query("AuthManager.validate_token refresh_session", 2048)
        policy = policy_for(classification)
        assert policy.modality is QueryModality.PL_TO_PL
        assert policy.budget_tier is BudgetTier.TIGHT
        assert policy.initial_retrieval_policy is RetrievalPolicy.BM25_ONLY
        assert policy.dense_trigger is DenseTrigger.LOW_SPARSE_CONFIDENCE

    def test_natural_language_large_budget_allows_dense(self) -> None:
        classification = classify_query("How does the authentication system work overall?", 16384)
        policy = policy_for(classification)
        assert policy.modality is QueryModality.NL_TO_PL
        assert policy.budget_tier is BudgetTier.LARGE
        assert policy.use_vector_initial is True
        assert policy.prefer_whole_file is True

    def test_mixed_issue_standard_budget_uses_diffuse_trigger(self) -> None:
        classification = classify_query(
            "The session cache never clears even after logout, can you check why "
            "`cache.invalidate(session_id)` is not being called here",
            6144,
        )
        policy = policy_for(classification)
        assert policy.modality is QueryModality.MIXED
        assert policy.budget_tier is BudgetTier.STANDARD
        assert policy.use_vector_initial is False
        assert policy.dense_trigger is DenseTrigger.DIFFUSE_TOP_SCORES
