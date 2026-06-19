"""Tests for the benchmark-only archex_query_task_aware strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import archex.benchmark.strategies as strategies
from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, DEFAULT_STRATEGIES
from archex.benchmark.strategies import (
    _should_expand_dense,  # pyright: ignore[reportPrivateUsage]
    _sparse_confidence,  # pyright: ignore[reportPrivateUsage]
    run_archex_query_task_aware,
)
from archex.benchmark.task_aware import policy_for
from archex.models import CodeChunk, ContextBundle, IndexConfig, PipelineTiming, RankedChunk
from archex.serve.modality import BudgetTier, QueryModality, classify_query

if TYPE_CHECKING:
    from pathlib import Path


def _ranked(chunk_id: str, score: float) -> RankedChunk:
    chunk = CodeChunk(
        id=chunk_id,
        content=f"content {chunk_id}",
        file_path=f"{chunk_id}.py",
        start_line=1,
        end_line=1,
        language="python",
        token_count=4,
    )
    return RankedChunk(chunk=chunk, final_score=score)


def _bundle(*scores: float) -> ContextBundle:
    chunks = [_ranked(f"c{i}", score) for i, score in enumerate(scores)]
    return ContextBundle(query="q", chunks=chunks, token_count=0)


class TestStrategyRegistry:
    def test_runner_registered(self) -> None:
        runner = strategies.default_strategy_registry.get(Strategy.ARCHEX_QUERY_TASK_AWARE)
        assert runner is run_archex_query_task_aware

    def test_name_in_registry(self) -> None:
        names = strategies.default_strategy_registry.strategy_names
        assert Strategy.ARCHEX_QUERY_TASK_AWARE.value in names

    def test_not_a_product_default(self) -> None:
        assert Strategy.ARCHEX_QUERY_TASK_AWARE not in DEFAULT_STRATEGIES

    def test_is_available_strategy(self) -> None:
        assert Strategy.ARCHEX_QUERY_TASK_AWARE in AVAILABLE_STRATEGIES

    def test_default_strategy_list_unchanged(self) -> None:
        assert DEFAULT_STRATEGIES == [
            Strategy.RAW_FILES,
            Strategy.RAW_RIPGREP,
            Strategy.ARCHEX_QUERY,
        ]


class TestSparseConfidence:
    def test_empty_bundle(self) -> None:
        top, gap, count = _sparse_confidence(_bundle(), window=40)
        assert top == 0.0
        assert gap == 0.0
        assert count == 0

    def test_single_chunk_is_confident(self) -> None:
        top, gap, count = _sparse_confidence(_bundle(0.8), window=40)
        assert top == 0.8
        assert gap == 1.0
        assert count == 1

    def test_diffuse_top_scores_small_gap(self) -> None:
        _top, gap, count = _sparse_confidence(_bundle(1.0, 0.95, 0.9), window=40)
        assert count == 3
        assert abs(gap - 0.05) < 1e-9

    def test_clear_top_score_large_gap(self) -> None:
        _top, gap, _count = _sparse_confidence(_bundle(1.0, 0.2), window=40)
        assert abs(gap - 0.8) < 1e-9

    def test_window_caps_candidate_count(self) -> None:
        _top, _gap, count = _sparse_confidence(_bundle(1.0, 0.9, 0.8, 0.7), window=2)
        assert count == 2


class TestShouldExpandDense:
    def test_no_candidates_expands(self) -> None:
        policy = policy_for(classify_query("AuthManager.validate_token refresh", 4096))
        assert _should_expand_dense(policy, relative_gap=1.0, candidate_count=0) is True

    def test_diffuse_gap_expands(self) -> None:
        policy = policy_for(classify_query("AuthManager.validate_token refresh", 4096))
        # standard tier threshold is 0.15
        assert _should_expand_dense(policy, relative_gap=0.05, candidate_count=5) is True

    def test_confident_gap_does_not_expand(self) -> None:
        policy = policy_for(classify_query("AuthManager.validate_token refresh", 4096))
        assert _should_expand_dense(policy, relative_gap=0.5, candidate_count=5) is False


class TestRunTaskAwareFixture:
    @pytest.fixture(autouse=True)
    def _no_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Keep the lane hermetic: never reach for embedding backends.
        monkeypatch.setattr(strategies, "_vector_dependencies_available", lambda: False)

    def _task(self, question: str, token_budget: int = 4096) -> BenchmarkTask:
        return BenchmarkTask(
            task_id="task_aware_test",
            repo="test/repo",
            commit="abc",
            question=question,
            expected_files=["main.py"],
            token_budget=token_budget,
        )

    def test_pl_to_pl_runs_bm25_only(self, python_simple_repo: Path) -> None:
        task = self._task("main.py main function module")
        result = run_archex_query_task_aware(task, python_simple_repo)

        assert result.strategy == Strategy.ARCHEX_QUERY_TASK_AWARE
        assert result.tool_calls == 1
        assert 0.0 <= result.required_file_recall <= 1.0
        prov = result.provenance
        assert prov["modality"] == QueryModality.PL_TO_PL.value
        assert prov["budget_tier"] == BudgetTier.STANDARD.value
        assert prov["initial_pass"] == "bm25_only"
        assert prov["routing_decision"] == "bm25_only"
        # vector unavailable: a diffuse sparse result cannot escalate to dense.
        assert prov["dense_expansion"] in {"skipped:confident_sparse", "skipped:vector_unavailable"}
        assert prov["fusion_used"] == "false"

    def test_provenance_records_policy_and_routing(self, python_simple_repo: Path) -> None:
        task = self._task("main.py main function module")
        prov = run_archex_query_task_aware(task, python_simple_repo).provenance
        for key in (
            "modality",
            "budget_tier",
            "routing_decision",
            "dense_expansion",
            "initial_pass",
            "policy_candidate_cap",
            "policy_dense_candidate_cap",
            "policy_skipped_steps",
            "sparse_relative_gap",
            "vector_dependencies_available",
            "fusion_used",
            "initial_cache_state",
        ):
            assert key in prov, key
        assert "cross_encoder_rerank" in prov["policy_skipped_steps"]
        assert prov["vector_dependencies_available"] == "false"

    def test_nl_to_pl_falls_back_without_vector(self, python_simple_repo: Path) -> None:
        task = self._task("How does the application start up and run overall?")
        prov = run_archex_query_task_aware(task, python_simple_repo).provenance
        assert prov["modality"] == QueryModality.NL_TO_PL.value
        assert prov["initial_pass"] == "bm25_only(vector_unavailable_fallback)"
        assert prov["dense_expansion"] == "skipped:vector_unavailable"


def _install_query_stub(
    monkeypatch: pytest.MonkeyPatch,
    bundle: ContextBundle,
    vector_flags: list[bool],
) -> None:
    def fake_query_bundle(
        task: BenchmarkTask,
        repo_path: Path,
        *,
        strategy: Strategy,
        index_config: IndexConfig,
        cache: bool,
    ) -> tuple[ContextBundle, IndexConfig, PipelineTiming]:
        # Core safety claim: a task-aware pass never enables the cross-encoder reranker.
        assert index_config.rerank is False
        vector_flags.append(index_config.vector)
        return bundle, index_config, PipelineTiming()

    monkeypatch.setattr(strategies, "_query_bundle", fake_query_bundle)


def _stub_task(question: str, token_budget: int = 4096) -> BenchmarkTask:
    return BenchmarkTask(
        task_id="t",
        repo="r",
        commit="abc",
        question=question,
        expected_files=["main.py"],
        token_budget=token_budget,
    )


class TestRunTaskAwareDenseRouting:
    """Stub the retrieval pass to exercise dense routing deterministically."""

    def test_dense_expansion_runs_when_diffuse(
        self, monkeypatch: pytest.MonkeyPatch, python_simple_repo: Path
    ) -> None:
        monkeypatch.setattr(strategies, "_vector_dependencies_available", lambda: True)
        vector_flags: list[bool] = []
        _install_query_stub(monkeypatch, _bundle(1.0, 0.99, 0.98), vector_flags)

        prov = run_archex_query_task_aware(
            _stub_task("AuthManager.validate_token refresh_session handler"),
            python_simple_repo,
        ).provenance
        assert prov["modality"] == QueryModality.PL_TO_PL.value
        assert prov["dense_expansion"] == "ran"
        assert prov["fusion_used"] == "true"
        assert "+dense_expansion" in prov["routing_decision"]
        # Initial sparse pass then a dense pass.
        assert vector_flags == [False, True]

    def test_dense_skipped_when_confident(
        self, monkeypatch: pytest.MonkeyPatch, python_simple_repo: Path
    ) -> None:
        monkeypatch.setattr(strategies, "_vector_dependencies_available", lambda: True)
        vector_flags: list[bool] = []
        _install_query_stub(monkeypatch, _bundle(1.0, 0.1), vector_flags)

        prov = run_archex_query_task_aware(
            _stub_task("AuthManager.validate_token refresh_session handler"),
            python_simple_repo,
        ).provenance
        assert prov["dense_expansion"] == "skipped:confident_sparse"
        assert prov["fusion_used"] == "false"
        # Only the initial sparse pass ran.
        assert vector_flags == [False]

    def test_nl_to_pl_initial_hybrid_single_pass(
        self, monkeypatch: pytest.MonkeyPatch, python_simple_repo: Path
    ) -> None:
        monkeypatch.setattr(strategies, "_vector_dependencies_available", lambda: True)
        vector_flags: list[bool] = []
        _install_query_stub(monkeypatch, _bundle(1.0, 0.9), vector_flags)

        prov = run_archex_query_task_aware(
            _stub_task("How does the application start up and run overall?"),
            python_simple_repo,
        ).provenance
        assert prov["modality"] == QueryModality.NL_TO_PL.value
        assert prov["initial_pass"] == "hybrid"
        assert prov["dense_expansion"] == "initial_hybrid"
        assert prov["fusion_used"] == "true"
        # A single hybrid pass; no separate dense pass.
        assert vector_flags == [True]

    def test_mixed_dense_expansion_when_diffuse(
        self, monkeypatch: pytest.MonkeyPatch, python_simple_repo: Path
    ) -> None:
        monkeypatch.setattr(strategies, "_vector_dependencies_available", lambda: True)
        vector_flags: list[bool] = []
        _install_query_stub(monkeypatch, _bundle(1.0, 0.99, 0.98), vector_flags)

        prov = run_archex_query_task_aware(
            _stub_task(
                "The session cache never clears even after logout, can you check why "
                "`cache.invalidate(session_id)` is not being called here",
                6144,
            ),
            python_simple_repo,
        ).provenance
        assert prov["modality"] == QueryModality.MIXED.value
        assert prov["dense_expansion"] == "ran"
        assert vector_flags == [False, True]
