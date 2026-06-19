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
    PackingCandidate,
    PackingSignals,
    order_candidates,
    pack_efficiently,
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


def _candidate(
    signals: PackingSignals,
    compressed_token_count: int | None = None,
    elided_token_count: int | None = None,
) -> PackingCandidate:
    return PackingCandidate(
        signals=signals,
        compressed_token_count=(
            signals.token_count if compressed_token_count is None else compressed_token_count
        ),
        elided_token_count=(
            min(12, signals.token_count) if elided_token_count is None else elided_token_count
        ),
    )


class TestPackerDirectMatchWins:
    def test_direct_match_packed_before_more_efficient_optional(self) -> None:
        # The optional region is more efficient per token, but the budget only
        # fits one region and the direct/high-confidence target must win.
        direct = _candidate(
            _signals("direct", direct_match=True, token_count=200, retrieval_score=0.2)
        )
        optional = _candidate(_signals("optional", token_count=60, retrieval_score=1.0))
        plan = pack_efficiently([optional, direct], token_budget=200, budget_tier=BudgetTier.TIGHT)
        assert plan.decision_for("direct") is PackDecision.INCLUDE
        assert plan.decision_for("optional") is PackDecision.SKIP
        assert plan.kept_ids() == ["direct"]
        assert plan.included_tokens == 200


class TestPackerGraphExpansion:
    def _candidates(self) -> list[PackingCandidate]:
        seed = _candidate(
            _signals(
                "seed", direct_match=True, graph_distance=0, token_count=100, retrieval_score=0.9
            )
        )
        expanded = _candidate(
            _signals(
                "expanded",
                graph_distance=1,
                graph_edge_confidence=0.5,
                token_count=100,
                retrieval_score=0.4,
            )
        )
        return [seed, expanded]

    def test_graph_expansion_loses_under_tight_budget(self) -> None:
        plan = pack_efficiently(self._candidates(), token_budget=100, budget_tier=BudgetTier.TIGHT)
        assert plan.decision_for("seed") is PackDecision.INCLUDE
        assert plan.decision_for("expanded") is PackDecision.SKIP
        assert plan.kept_ids() == ["seed"]

    def test_graph_expansion_included_when_budget_allows(self) -> None:
        plan = pack_efficiently(self._candidates(), token_budget=400, budget_tier=BudgetTier.LARGE)
        assert plan.decision_for("seed") is PackDecision.INCLUDE
        assert plan.decision_for("expanded") is PackDecision.INCLUDE
        assert set(plan.kept_ids()) == {"seed", "expanded"}


class TestPackerWholeFilePreservation:
    def test_whole_file_included_only_under_large_budget(self) -> None:
        wf = _signals(
            "wf", whole_file=True, file_evidence_regions=1, token_count=300, retrieval_score=0.5
        )
        large = pack_efficiently([_candidate(wf)], token_budget=2000, budget_tier=BudgetTier.LARGE)
        standard = pack_efficiently(
            [_candidate(wf)], token_budget=2000, budget_tier=BudgetTier.STANDARD
        )
        assert large.decision_for("wf") is PackDecision.INCLUDE
        assert standard.decision_for("wf") is PackDecision.ELIDE


class TestPackerCompressionRisk:
    def test_high_risk_optional_is_never_compressed_under_pressure(self) -> None:
        risky = _candidate(
            _signals(
                "risky",
                token_count=300,
                retrieval_score=0.6,
                compression_eligible=True,
                compression_loss_risk=CompressionLossRisk.HIGH,
            ),
            compressed_token_count=80,
        )
        plan = pack_efficiently([risky], token_budget=100, budget_tier=BudgetTier.TIGHT)
        decision = plan.decision_for("risky")
        assert decision is not PackDecision.COMPRESS
        assert decision in (PackDecision.ELIDE, PackDecision.SKIP)

    def test_low_risk_optional_is_compressed_under_pressure(self) -> None:
        low = _candidate(
            _signals(
                "low",
                token_count=300,
                retrieval_score=0.6,
                compression_eligible=True,
                compression_loss_risk=CompressionLossRisk.LOW,
            ),
            compressed_token_count=80,
        )
        plan = pack_efficiently([low], token_budget=100, budget_tier=BudgetTier.TIGHT)
        assert plan.decision_for("low") is PackDecision.COMPRESS
        assert plan.included_tokens == 80


class TestPackerInvariants:
    def test_direct_match_kept_as_anchor_when_budget_exhausted(self) -> None:
        first = _candidate(
            _signals("first", direct_match=True, token_count=100, retrieval_score=0.9)
        )
        second = _candidate(
            _signals("second", direct_match=True, token_count=100, retrieval_score=0.8)
        )
        plan = pack_efficiently([first, second], token_budget=100, budget_tier=BudgetTier.TIGHT)
        assert plan.decision_for("first") is PackDecision.INCLUDE
        # The second direct match no longer fits but is preserved as an anchor.
        assert plan.decision_for("second") is PackDecision.ELIDE
        assert "second" in plan.kept_ids()

    def test_optional_regions_respect_budget(self) -> None:
        candidates = [
            _candidate(_signals(f"c{i}", token_count=100, retrieval_score=0.9 - i * 0.1))
            for i in range(5)
        ]
        plan = pack_efficiently(candidates, token_budget=250, budget_tier=BudgetTier.STANDARD)
        # Optional regions never overflow: the full packed total (include + anchors)
        # stays within budget; only forced direct-match anchors may exceed it.
        assert plan.included_tokens <= 250

    def test_provenance_and_relevance_per_token(self) -> None:
        seed = _candidate(_signals("seed", direct_match=True, token_count=100, retrieval_score=1.0))
        opt = _candidate(_signals("opt", token_count=100, retrieval_score=0.5))
        plan = pack_efficiently([seed, opt], token_budget=1000, budget_tier=BudgetTier.STANDARD)
        prov = plan.to_provenance()
        assert prov["include_count"] == "2"
        assert prov["direct_match_count"] == "1"
        assert prov["skip_count"] == "0"
        assert prov["budget_tier"] == BudgetTier.STANDARD.value
        # (1.0 + 0.5) retrieval mass over 200 packed tokens -> 7.5 per 1k tokens.
        assert plan.relevance_per_1k_tokens() == 7.5
        assert prov["relevance_per_1k_tokens"] == "7.5000"

    def test_relevance_excludes_anchor_only_elided_regions(self) -> None:
        # The elided region keeps only an anchor, so its retrieval mass must not
        # count toward delivered relevance-per-token.
        kept = _candidate(_signals("kept", direct_match=True, token_count=100, retrieval_score=1.0))
        elided = _signals(
            "wf", whole_file=True, file_evidence_regions=1, token_count=300, retrieval_score=0.9
        )
        plan = pack_efficiently(
            [kept, _candidate(elided)], token_budget=2000, budget_tier=BudgetTier.STANDARD
        )
        assert plan.decision_for("wf") is PackDecision.ELIDE
        # Only the INCLUDE region's mass (1.0) counts, over packed tokens (100 + 12).
        assert plan.relevance_per_1k_tokens() == 1.0 / plan.included_tokens * 1000.0


class TestPackerEdgeCases:
    def test_empty_candidate_set(self) -> None:
        plan = pack_efficiently([], token_budget=1000, budget_tier=BudgetTier.STANDARD)
        assert plan.regions == []
        assert plan.kept_ids() == []
        assert plan.included_tokens == 0
        assert plan.relevance_per_1k_tokens() == 0.0

    def test_single_direct_match_larger_than_budget_is_anchored(self) -> None:
        big = _candidate(_signals("big", direct_match=True, token_count=5000, retrieval_score=0.9))
        plan = pack_efficiently([big], token_budget=100, budget_tier=BudgetTier.TIGHT)
        # Cannot be INCLUDEd within budget, but a direct match is never dropped:
        # it is preserved as an anchor even though that exceeds the budget.
        assert plan.decision_for("big") is PackDecision.ELIDE
        assert "big" in plan.kept_ids()

    def test_zero_budget_skips_all_optional(self) -> None:
        candidates = [_candidate(_signals(f"c{i}", token_count=100)) for i in range(3)]
        plan = pack_efficiently(candidates, token_budget=0, budget_tier=BudgetTier.TIGHT)
        assert plan.kept_ids() == []
        assert plan.included_tokens == 0
        assert plan.to_provenance()["skip_count"] == "3"

    def test_packing_is_input_order_independent(self) -> None:
        seed = _candidate(_signals("seed", direct_match=True, token_count=100, retrieval_score=0.9))
        mid = _candidate(_signals("mid", token_count=100, retrieval_score=0.6))
        low = _candidate(_signals("low", token_count=100, retrieval_score=0.2))
        forward = pack_efficiently(
            [seed, mid, low], token_budget=250, budget_tier=BudgetTier.STANDARD
        )
        shuffled = pack_efficiently(
            [low, seed, mid], token_budget=250, budget_tier=BudgetTier.STANDARD
        )
        assert forward.kept_ids() == shuffled.kept_ids()
        assert [r.decision for r in forward.regions] == [r.decision for r in shuffled.regions]

    def test_elide_cost_uses_real_anchor_token_count(self) -> None:
        # A whole-file region forced to an anchor is charged its real anchor cost
        # (elided_token_count), not a fixed marker guess.
        wf = _candidate(
            _signals(
                "wf", whole_file=True, file_evidence_regions=1, token_count=600, retrieval_score=0.4
            ),
            elided_token_count=30,
        )
        plan = pack_efficiently([wf], token_budget=2000, budget_tier=BudgetTier.STANDARD)
        assert plan.decision_for("wf") is PackDecision.ELIDE
        assert plan.included_tokens == 30
