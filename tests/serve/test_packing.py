"""Tests for the efficiency-aware packing score model."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import archex.serve.packing as packing
from archex.models import CompressionLossRisk
from archex.serve.modality import BudgetTier
from archex.serve.packing import (
    PackDecision,
    PackingSignals,
    order_candidates,
    relevance_per_1k_tokens,
    score_candidate,
)


def _signals(
    candidate_id: str = "c0",
    *,
    file_path: str = "src/app.py",
    retrieval_score: float = 0.5,
    direct_match: bool = False,
    graph_distance: int = 0,
    graph_edge_confidence: float = 1.0,
    token_count: int = 100,
    compression_eligible: bool = False,
    compression_loss_risk: CompressionLossRisk = CompressionLossRisk.NONE,
    handle_priority: float = 0.0,
    whole_file: bool = False,
    file_evidence_regions: int = 1,
    protect_code: bool = False,
) -> PackingSignals:
    return PackingSignals(
        candidate_id=candidate_id,
        file_path=file_path,
        retrieval_score=retrieval_score,
        direct_match=direct_match,
        graph_distance=graph_distance,
        graph_edge_confidence=graph_edge_confidence,
        token_count=token_count,
        compression_eligible=compression_eligible,
        compression_loss_risk=compression_loss_risk,
        handle_priority=handle_priority,
        whole_file=whole_file,
        file_evidence_regions=file_evidence_regions,
        protect_code=protect_code,
    )


class TestRelevancePerToken:
    def test_score_normalized_per_1k_tokens(self) -> None:
        assert relevance_per_1k_tokens(0.5, 100) == 5.0
        assert relevance_per_1k_tokens(1.0, 1000) == 1.0

    def test_empty_region_is_zero(self) -> None:
        assert relevance_per_1k_tokens(0.9, 0) == 0.0


class TestScoreOrdering:
    def test_ordering_is_deterministic_and_input_independent(self) -> None:
        # Three non-direct candidates with strictly distinct efficiency.
        high = score_candidate(
            _signals("high", retrieval_score=0.9, token_count=100), budget_tier=BudgetTier.STANDARD
        )
        mid = score_candidate(
            _signals("mid", retrieval_score=0.5, token_count=100), budget_tier=BudgetTier.STANDARD
        )
        low = score_candidate(
            _signals("low", retrieval_score=0.1, token_count=100), budget_tier=BudgetTier.STANDARD
        )
        forward = [s.candidate_id for s in order_candidates([high, mid, low])]
        shuffled = [s.candidate_id for s in order_candidates([low, high, mid])]
        assert forward == ["high", "mid", "low"]
        assert forward == shuffled

    def test_equal_value_breaks_by_graph_distance_then_id(self) -> None:
        # Identical efficiency and value; only the tie-break keys differ. Both are
        # graph-expanded with full edge confidence so the distance penalty matches.
        near = score_candidate(
            _signals("z_near", retrieval_score=0.4, token_count=100, graph_distance=1),
            budget_tier=BudgetTier.STANDARD,
        )
        far = score_candidate(
            _signals("a_far", retrieval_score=0.4, token_count=100, graph_distance=2),
            budget_tier=BudgetTier.STANDARD,
        )
        same_a = score_candidate(
            _signals("a_near", retrieval_score=0.4, token_count=100, graph_distance=1),
            budget_tier=BudgetTier.STANDARD,
        )
        ordered = [s.candidate_id for s in order_candidates([far, near, same_a])]
        # Nearer distance first; within equal distance, candidate_id ascending.
        assert ordered == ["a_near", "z_near", "a_far"]


class TestDirectMatchPreservation:
    def test_direct_match_packs_before_more_efficient_optional(self) -> None:
        # A tiny-but-efficient optional region would outrank the direct match on
        # raw efficiency, yet the direct/high-confidence target must pack first.
        direct = score_candidate(
            _signals("direct", retrieval_score=0.2, token_count=500, direct_match=True),
            budget_tier=BudgetTier.STANDARD,
        )
        optional = score_candidate(
            _signals("optional", retrieval_score=1.0, token_count=10),
            budget_tier=BudgetTier.STANDARD,
        )
        assert optional.value > direct.value  # optional is more efficient
        ordered = [s.candidate_id for s in order_candidates([optional, direct])]
        assert ordered == ["direct", "optional"]

    def test_direct_match_decision_is_always_include(self) -> None:
        # Even a large, whole-file, high-risk region stays INCLUDE when direct.
        score = score_candidate(
            _signals(
                "d",
                retrieval_score=0.05,
                token_count=2000,
                direct_match=True,
                whole_file=True,
                graph_distance=0,
                compression_eligible=True,
                compression_loss_risk=CompressionLossRisk.HIGH,
            ),
            budget_tier=BudgetTier.TIGHT,
        )
        assert score.decision is PackDecision.INCLUDE


class TestWholeFilePreservation:
    def test_whole_file_compressed_under_standard_budget_when_low_risk(self) -> None:
        score = score_candidate(
            _signals(
                "wf",
                whole_file=True,
                file_evidence_regions=1,
                compression_eligible=True,
                compression_loss_risk=CompressionLossRisk.LOW,
            ),
            budget_tier=BudgetTier.STANDARD,
        )
        assert score.decision is PackDecision.COMPRESS

    def test_whole_file_elided_under_standard_budget_when_not_compressible(self) -> None:
        score = score_candidate(
            _signals("wf", whole_file=True, file_evidence_regions=1, compression_eligible=False),
            budget_tier=BudgetTier.STANDARD,
        )
        assert score.decision is PackDecision.ELIDE

    def test_whole_file_included_under_large_budget(self) -> None:
        score = score_candidate(
            _signals("wf", whole_file=True, file_evidence_regions=1),
            budget_tier=BudgetTier.LARGE,
        )
        assert score.decision is PackDecision.INCLUDE

    def test_whole_file_included_with_multiple_evidence_regions(self) -> None:
        score = score_candidate(
            _signals("wf", whole_file=True, file_evidence_regions=2),
            budget_tier=BudgetTier.STANDARD,
        )
        assert score.decision is PackDecision.INCLUDE


class TestPenalties:
    def test_graph_distance_lowers_value(self) -> None:
        seed = score_candidate(
            _signals("seed", retrieval_score=0.6, token_count=100, graph_distance=0),
            budget_tier=BudgetTier.STANDARD,
        )
        expanded = score_candidate(
            _signals("exp", retrieval_score=0.6, token_count=100, graph_distance=1),
            budget_tier=BudgetTier.STANDARD,
        )
        assert expanded.value < seed.value

    def test_low_edge_confidence_lowers_value_further(self) -> None:
        confident = score_candidate(
            _signals("hi", retrieval_score=0.6, graph_distance=1, graph_edge_confidence=0.9),
            budget_tier=BudgetTier.STANDARD,
        )
        diffuse = score_candidate(
            _signals("lo", retrieval_score=0.6, graph_distance=1, graph_edge_confidence=0.1),
            budget_tier=BudgetTier.STANDARD,
        )
        assert diffuse.value < confident.value

    def test_handle_priority_bonus_is_clamped(self) -> None:
        capped = score_candidate(
            _signals("a", handle_priority=1.0), budget_tier=BudgetTier.STANDARD
        )
        over = score_candidate(_signals("b", handle_priority=5.0), budget_tier=BudgetTier.STANDARD)
        assert over.value == capped.value

    def test_edge_confidence_gap_is_clamped(self) -> None:
        # Confidence above 1 behaves like 1 (no extra distance penalty); below 0
        # behaves like 0 (full low-confidence penalty).
        at_one = score_candidate(
            _signals("a", graph_distance=1, graph_edge_confidence=1.0),
            budget_tier=BudgetTier.STANDARD,
        )
        over_one = score_candidate(
            _signals("b", graph_distance=1, graph_edge_confidence=5.0),
            budget_tier=BudgetTier.STANDARD,
        )
        at_zero = score_candidate(
            _signals("c", graph_distance=1, graph_edge_confidence=0.0),
            budget_tier=BudgetTier.STANDARD,
        )
        below_zero = score_candidate(
            _signals("d", graph_distance=1, graph_edge_confidence=-2.0),
            budget_tier=BudgetTier.STANDARD,
        )
        assert over_one.value == at_one.value
        assert below_zero.value == at_zero.value


class TestCompressionRiskGating:
    def test_high_risk_low_value_region_is_skipped_not_compressed(self) -> None:
        score = score_candidate(
            _signals(
                "risky",
                retrieval_score=0.05,
                token_count=800,
                graph_distance=2,
                compression_eligible=True,
                compression_loss_risk=CompressionLossRisk.HIGH,
            ),
            budget_tier=BudgetTier.TIGHT,
        )
        assert score.decision is PackDecision.SKIP

    def test_medium_risk_region_is_not_auto_compressed(self) -> None:
        # Medium loss risk is not "low risk", so a low-value region is skipped
        # rather than compressed; only NONE/LOW risk earns COMPRESS.
        score = score_candidate(
            _signals(
                "med",
                retrieval_score=0.05,
                token_count=800,
                graph_distance=2,
                compression_eligible=True,
                compression_loss_risk=CompressionLossRisk.MEDIUM,
            ),
            budget_tier=BudgetTier.TIGHT,
        )
        assert score.decision is PackDecision.SKIP

    def test_protect_code_blocks_compression_preference(self) -> None:
        # Fix/debug/review intent: a compressible whole-file region is elided to an
        # anchor rather than compressed in place.
        score = score_candidate(
            _signals(
                "wf",
                whole_file=True,
                compression_eligible=True,
                compression_loss_risk=CompressionLossRisk.LOW,
                protect_code=True,
            ),
            budget_tier=BudgetTier.STANDARD,
        )
        assert score.decision is PackDecision.ELIDE


class TestProvenance:
    def test_to_provenance_formats_fields(self) -> None:
        score = score_candidate(
            _signals("c0", retrieval_score=0.5, token_count=100, direct_match=True),
            budget_tier=BudgetTier.STANDARD,
        )
        prov = score.to_provenance()
        assert prov["candidate_id"] == "c0"
        assert prov["decision"] == PackDecision.INCLUDE.value
        assert prov["direct_match"] == "true"  # lowercased boolean, not "True"
        assert prov["relevance_per_1k_tokens"] == "5.0000"  # 0.5 / 100 * 1000, 4dp
        assert prov["value"] == f"{score.value:.4f}"
        assert prov["compression_loss_risk"] == CompressionLossRisk.NONE.value


class TestNoBenchmarkGroundTruthDependency:
    def test_score_signature_takes_only_intrinsic_signals(self) -> None:
        params = set(inspect.signature(score_candidate).parameters)
        assert params == {"signals", "budget_tier"}

    def test_module_does_not_import_benchmark_package(self) -> None:
        tree = ast.parse(Path(packing.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("archex.benchmark") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith("archex.benchmark")
                if module == "archex":  # guard `from archex import benchmark`
                    assert all(alias.name != "benchmark" for alias in node.names)

    def test_module_does_not_reference_ground_truth_fields(self) -> None:
        source = Path(packing.__file__).read_text(encoding="utf-8")
        for forbidden in ("expected_region", "expected_file", "ground_truth"):
            assert forbidden not in source

    def test_provisional_decision_ignores_unknown_relevance(self) -> None:
        # The signals carry no region-coverage field; identical signals always
        # produce the identical decision regardless of any external label.
        a = score_candidate(_signals("a"), budget_tier=BudgetTier.STANDARD)
        b = score_candidate(_signals("a"), budget_tier=BudgetTier.STANDARD)
        assert a == b
