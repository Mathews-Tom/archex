"""Tests for the M3 candidate lane matrix."""

from __future__ import annotations

from archex.benchmark.external_frontier import (
    CAST_LANE,
    DEFAULT_LANE,
    PROFILE_BALANCED_LANE,
    PROFILE_FAST_LANE,
    SYMBOLIC_RERANK_LANE,
    build_external_frontier_lanes,
    lane_strategies,
    profile_purity_violations,
)
from archex.benchmark.models import BenchmarkRetrievalOptions, Strategy


class TestBuildExternalFrontierLanes:
    def test_default_excludes_symbolic_rerank(self) -> None:
        lanes = build_external_frontier_lanes()
        assert lanes == [DEFAULT_LANE, CAST_LANE, PROFILE_FAST_LANE, PROFILE_BALANCED_LANE]

    def test_opt_in_includes_symbolic_rerank(self) -> None:
        lanes = build_external_frontier_lanes(include_symbolic_rerank=True)
        assert lanes[-1] == SYMBOLIC_RERANK_LANE
        assert len(lanes) == 5


class TestLaneStrategies:
    def test_deduplicates_default_and_cast_same_strategy(self) -> None:
        lanes = build_external_frontier_lanes()
        strategies = lane_strategies(lanes)
        assert strategies == [
            Strategy.ARCHEX_QUERY,
            Strategy.ARCHEX_QUERY_PROFILE_FAST,
            Strategy.ARCHEX_QUERY_PROFILE_BALANCED,
        ]


class TestCliRunArgs:
    def test_default_lane_pins_default_chunker(self) -> None:
        args = DEFAULT_LANE.cli_run_args()
        assert args == ["--strategy", "archex_query", "--chunker", "default"]

    def test_cast_lane_pins_cast_chunker(self) -> None:
        args = CAST_LANE.cli_run_args()
        assert args == ["--strategy", "archex_query", "--chunker", "cast"]

    def test_profile_lane_has_no_chunker_flag(self) -> None:
        args = PROFILE_FAST_LANE.cli_run_args()
        assert args == ["--strategy", "archex_query_profile_fast"]


class TestProfilePurityViolations:
    def test_empty_without_any_profile_lane(self) -> None:
        options = BenchmarkRetrievalOptions(splade=True)
        violations = profile_purity_violations(options, [DEFAULT_LANE, CAST_LANE])
        assert violations == []

    def test_empty_when_options_are_clean(self) -> None:
        options = BenchmarkRetrievalOptions()
        violations = profile_purity_violations(options, [PROFILE_FAST_LANE])
        assert violations == []

    def test_detects_splade(self) -> None:
        options = BenchmarkRetrievalOptions(splade=True)
        violations = profile_purity_violations(options, [PROFILE_FAST_LANE])
        assert violations == ["splade"]

    def test_detects_module_prefilter(self) -> None:
        options = BenchmarkRetrievalOptions(module_prefilter=True)
        violations = profile_purity_violations(options, [PROFILE_BALANCED_LANE])
        assert violations == ["module_prefilter"]

    def test_detects_rerank_model(self) -> None:
        options = BenchmarkRetrievalOptions(rerank_model="some/model")
        violations = profile_purity_violations(options, [PROFILE_FAST_LANE])
        assert violations == ["rerank_model"]

    def test_detects_all_three_at_once(self) -> None:
        options = BenchmarkRetrievalOptions(
            splade=True, module_prefilter=True, rerank_model="some/model"
        )
        violations = profile_purity_violations(options, [PROFILE_BALANCED_LANE])
        assert violations == ["splade", "module_prefilter", "rerank_model"]
