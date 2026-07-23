"""Tests for the M3 fast/balanced retrieval-profile benchmark lanes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkTask, RepoSizeClass, Strategy
from archex.benchmark.strategies import (
    run_archex_query_profile_balanced,
    run_archex_query_profile_fast,
)

if TYPE_CHECKING:
    from pathlib import Path


def _task(**overrides: object) -> BenchmarkTask:
    defaults: dict[str, object] = {
        "task_id": "test",
        "repo": "test/repo",
        "commit": "abc",
        "question": "How does the main module work?",
        "expected_files": ["main.py"],
        "token_budget": 4096,
    }
    defaults.update(overrides)
    return BenchmarkTask.model_validate(defaults)


class TestRunArchexQueryProfileFast:
    def test_strategy_and_metrics_shape(self, python_simple_repo: Path) -> None:
        result = run_archex_query_profile_fast(_task(), python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_PROFILE_FAST
        assert result.tool_calls == 1
        assert result.timing is not None
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0

    def test_never_performs_vector_or_rerank_work(self, python_simple_repo: Path) -> None:
        # The fast profile forces vector=False and rerank=False regardless of
        # any global CLI override; index_chunk_count/mean_chunk_tokens still
        # come from the BM25 chunk store, so this asserts no vector-specific
        # storage-provenance field ever populates for this lane.
        result = run_archex_query_profile_fast(_task(), python_simple_repo)
        assert result.rerank_model_storage_bytes is None

    def test_classifies_repo_size(self, python_simple_repo: Path) -> None:
        result = run_archex_query_profile_fast(_task(), python_simple_repo)
        assert result.repo_size_class == RepoSizeClass.SMALL

    def test_duplicate_rate_is_bounded(self, python_simple_repo: Path) -> None:
        result = run_archex_query_profile_fast(_task(), python_simple_repo)
        assert 0.0 <= result.duplicate_rate <= 1.0


class TestRunArchexQueryProfileBalanced:
    def test_strategy_and_metrics_shape(self, python_simple_repo: Path) -> None:
        result = run_archex_query_profile_balanced(_task(), python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_PROFILE_BALANCED
        assert result.tool_calls == 1
        assert result.timing is not None
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0

    def test_classifies_repo_size(self, python_simple_repo: Path) -> None:
        result = run_archex_query_profile_balanced(_task(), python_simple_repo)
        assert result.repo_size_class == RepoSizeClass.SMALL
