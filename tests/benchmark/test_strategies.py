"""Tests for benchmark strategy implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.strategies import (
    compute_precision,
    compute_recall,
    count_file_tokens,
    extract_keywords,
    run_archex_query,
    run_archex_query_hybrid,
    run_archex_symbol_lookup,
    run_raw_files,
    run_raw_grepped,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="test",
        repo="test/repo",
        commit="abc",
        question="How does auth work?",
        expected_files=["main.py", "services/auth.py"],
        keywords=["auth", "login"],
    )


class TestComputeRecall:
    def test_full_recall(self) -> None:
        assert compute_recall({"a.py", "b.py"}, ["a.py", "b.py"]) == 1.0

    def test_partial_recall(self) -> None:
        assert compute_recall({"a.py"}, ["a.py", "b.py"]) == 0.5

    def test_zero_recall(self) -> None:
        assert compute_recall({"c.py"}, ["a.py", "b.py"]) == 0.0

    def test_empty_expected(self) -> None:
        assert compute_recall({"a.py"}, []) == 0.0

    def test_empty_results(self) -> None:
        assert compute_recall(set(), ["a.py"]) == 0.0


class TestComputePrecision:
    def test_full_precision(self) -> None:
        assert compute_precision({"a.py", "b.py"}, ["a.py", "b.py"]) == 1.0

    def test_partial_precision(self) -> None:
        assert compute_precision({"a.py", "c.py"}, ["a.py", "b.py"]) == 0.5

    def test_zero_precision(self) -> None:
        assert compute_precision({"c.py", "d.py"}, ["a.py", "b.py"]) == 0.0

    def test_empty_results(self) -> None:
        assert compute_precision(set(), ["a.py"]) == 0.0


class TestExtractKeywords:
    def test_filters_stopwords(self) -> None:
        kws = extract_keywords("How does the auth module work?", [])
        assert "how" not in kws
        assert "does" not in kws
        assert "the" not in kws
        assert "auth" in kws
        assert "module" in kws

    def test_includes_extra_keywords(self) -> None:
        kws = extract_keywords("test query", ["special"])
        assert "special" in kws

    def test_deduplicates_extras(self) -> None:
        kws = extract_keywords("auth query", ["auth"])
        assert kws.count("auth") == 1

    def test_filters_short_words(self) -> None:
        kws = extract_keywords("a is on go", [])
        # "go" has len 2, should be filtered
        assert "go" not in kws


class TestCountFileTokens:
    def test_counts_real_files(self, python_simple_repo: Path) -> None:
        tokens = count_file_tokens(python_simple_repo, ["main.py"])
        assert tokens > 0

    def test_missing_file_skipped(self, python_simple_repo: Path) -> None:
        tokens = count_file_tokens(python_simple_repo, ["nonexistent.py"])
        assert tokens == 0

    def test_empty_file_list(self, python_simple_repo: Path) -> None:
        tokens = count_file_tokens(python_simple_repo, [])
        assert tokens == 0


class TestRunRawFiles:
    def test_raw_files_strategy(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How?",
            expected_files=["main.py", "utils.py"],
        )
        result = run_raw_files(task, python_simple_repo)
        assert result.strategy == Strategy.RAW_FILES
        assert result.tokens_total > 0
        assert result.recall == 1.0
        assert result.precision == 1.0
        assert result.savings_vs_raw == 0.0
        assert result.files_accessed == 2


class TestRunRawGrepped:
    def test_grep_finds_files(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How does authentication work?",
            expected_files=["services/auth.py"],
            keywords=["authenticate"],
        )
        result = run_raw_grepped(task, python_simple_repo)
        assert result.strategy == Strategy.RAW_GREPPED
        assert result.files_accessed >= 0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0

    def test_grep_no_matches(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="zzz_unique_nonexistent_term_xyz",
            expected_files=["main.py"],
            keywords=["zzz_unique_nonexistent_term_xyz"],
        )
        result = run_raw_grepped(task, python_simple_repo)
        assert result.strategy == Strategy.RAW_GREPPED
        assert result.tokens_total == 0
        assert result.files_accessed == 0
        assert result.recall == 0.0

    def test_grep_result_fields(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="import models",
            expected_files=["main.py", "utils.py"],
            keywords=["import"],
        )
        result = run_raw_grepped(task, python_simple_repo)
        assert result.wall_time_ms >= 0
        assert result.cached is False
        assert result.savings_vs_raw == 0.0  # Not yet backfilled
        assert result.tool_calls > 0  # At least one keyword searched


class TestRunArchexQuery:
    def test_archex_query_strategy(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How does the main module work?",
            expected_files=["main.py"],
            token_budget=4096,
        )
        result = run_archex_query(task, python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY
        assert result.tokens_total >= 0
        assert result.tool_calls == 1
        assert result.timing is not None


class TestRunArchexQueryHybrid:
    def test_hybrid_strategy(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How does the main module work?",
            expected_files=["main.py"],
            token_budget=4096,
        )
        result = run_archex_query_hybrid(task, python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_HYBRID
        assert result.tokens_total >= 0
        assert result.tool_calls == 1
        assert result.timing is not None
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0

    def test_hybrid_recall_precision(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="authentication login service",
            expected_files=["services/auth.py", "main.py"],
            token_budget=8192,
        )
        result = run_archex_query_hybrid(task, python_simple_repo)
        # Should return files and compute metrics
        assert result.files_accessed >= 0
        assert isinstance(result.recall, float)
        assert isinstance(result.precision, float)


class TestRunArchexSymbolLookup:
    def test_raises_not_implemented(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How?",
            expected_files=["main.py"],
        )
        with pytest.raises(NotImplementedError, match="Enhancement 1"):
            run_archex_symbol_lookup(task, python_simple_repo)
