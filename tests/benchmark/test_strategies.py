"""Tests for benchmark strategy implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.strategies import (
    _deduplicate_ranked,  # pyright: ignore[reportPrivateUsage]
    compute_map,
    compute_mrr,
    compute_ndcg,
    compute_precision,
    compute_recall,
    compute_symbol_recall,
    count_file_tokens,
    extract_keywords,
    run_archex_query,
    run_archex_query_fusion,
    run_archex_query_vector,
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


class TestComputeNdcg:
    def test_perfect_ranking(self) -> None:
        ranked = ["a.py", "b.py", "c.py"]
        expected = ["a.py", "b.py"]
        assert compute_ndcg(ranked, expected) == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]

    def test_worst_ranking(self) -> None:
        ranked = ["x.py", "y.py", "z.py"]
        expected = ["a.py", "b.py"]
        assert compute_ndcg(ranked, expected) == 0.0

    def test_partial_ranking(self) -> None:
        ranked = ["x.py", "a.py", "b.py"]
        expected = ["a.py", "b.py"]
        result = compute_ndcg(ranked, expected)
        assert 0.0 < result < 1.0

    def test_empty_expected(self) -> None:
        assert compute_ndcg(["a.py"], []) == 0.0

    def test_empty_ranked(self) -> None:
        assert compute_ndcg([], ["a.py"]) == 0.0

    def test_k_parameter(self) -> None:
        ranked = [f"filler_{i}.py" for i in range(20)] + ["a.py"]
        expected = ["a.py"]
        # With k=10, "a.py" is beyond cutoff
        assert compute_ndcg(ranked, expected, k=10) == 0.0
        # With k=25, "a.py" is included
        assert compute_ndcg(ranked, expected, k=25) > 0.0


class TestComputeMap:
    def test_perfect_ranking(self) -> None:
        ranked = ["a.py", "b.py", "c.py"]
        expected = ["a.py", "b.py"]
        assert compute_map(ranked, expected) == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]

    def test_worst_ranking(self) -> None:
        ranked = ["x.py", "y.py", "z.py"]
        expected = ["a.py", "b.py"]
        assert compute_map(ranked, expected) == 0.0

    def test_partial_ranking(self) -> None:
        # a.py at position 2: precision@2 = 1/2 = 0.5
        # b.py at position 3: precision@3 = 2/3
        # MAP = (0.5 + 2/3) / 2 = 7/12
        ranked = ["x.py", "a.py", "b.py"]
        expected = ["a.py", "b.py"]
        assert compute_map(ranked, expected) == pytest.approx(7.0 / 12.0)  # pyright: ignore[reportUnknownMemberType]

    def test_empty_expected(self) -> None:
        assert compute_map(["a.py"], []) == 0.0

    def test_empty_ranked(self) -> None:
        assert compute_map([], ["a.py"]) == 0.0


class TestDeduplicateRanked:
    def test_removes_duplicates_preserves_order(self) -> None:
        assert _deduplicate_ranked(["a.py", "b.py", "a.py", "c.py"]) == [
            "a.py",
            "b.py",
            "c.py",
        ]

    def test_empty_list(self) -> None:
        assert _deduplicate_ranked([]) == []

    def test_no_duplicates(self) -> None:
        assert _deduplicate_ranked(["a.py", "b.py"]) == ["a.py", "b.py"]

    def test_all_same(self) -> None:
        assert _deduplicate_ranked(["a.py", "a.py", "a.py"]) == ["a.py"]


class TestRankingMetricsDedup:
    """Verify that ranking metrics deduplicate before scoring."""

    def test_mrr_with_duplicates(self) -> None:
        # Without dedup: "x.py" at pos 1, "a.py" at pos 2 → MRR = 0.5
        # Same after dedup since no relevant dup before first hit
        assert compute_mrr(["x.py", "a.py", "a.py"], ["a.py"]) == 0.5

    def test_ndcg_not_inflated_by_duplicates(self) -> None:
        # ["a.py", "a.py"] with expected=["a.py"] should score same as ["a.py"]
        perfect = compute_ndcg(["a.py"], ["a.py"])
        with_dup = compute_ndcg(["a.py", "a.py"], ["a.py"])
        assert with_dup == perfect

    def test_map_not_inflated_by_duplicates(self) -> None:
        # ["a.py", "a.py", "b.py"] should score same as ["a.py", "b.py"]
        clean = compute_map(["a.py", "b.py"], ["a.py", "b.py"])
        with_dup = compute_map(["a.py", "a.py", "b.py"], ["a.py", "b.py"])
        assert with_dup == clean


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


class TestComputeSymbolRecall:
    def test_full_recall(self) -> None:
        assert compute_symbol_recall({"foo", "bar"}, ["foo", "bar"]) == 1.0

    def test_partial_recall(self) -> None:
        assert compute_symbol_recall({"foo"}, ["foo", "bar"]) == 0.5

    def test_zero_recall(self) -> None:
        assert compute_symbol_recall({"baz"}, ["foo", "bar"]) == 0.0

    def test_empty_expected(self) -> None:
        assert compute_symbol_recall({"foo"}, []) == 0.0

    def test_empty_results(self) -> None:
        assert compute_symbol_recall(set(), ["foo"]) == 0.0


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
        # Token efficiency fields
        assert result.tokens_input == result.tokens_total
        assert result.tokens_output == result.tokens_total
        assert result.token_efficiency == 1.0
        assert result.tokens_raw_baseline == result.tokens_total


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
        # Token efficiency + MRR fields
        assert result.tokens_input >= 0
        assert result.tokens_output >= 0
        assert result.tokens_raw_baseline >= 0
        assert isinstance(result.mrr, float)


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
        # Token efficiency fields
        assert result.tokens_input >= 0
        assert result.tokens_output >= 0
        assert result.tokens_raw_baseline >= 0


class _StubEmbedder:
    """Deterministic stub embedder for vector/fusion tests without onnxruntime."""

    @property
    def dimension(self) -> int:
        return 64

    def encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        result: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [float(b) / 255.0 for b in h[: self.dimension]]
            result.append(vec)
        return result


def _stub_get_embedder(_index_config: object) -> _StubEmbedder:
    return _StubEmbedder()


class TestRunArchexQueryVector:
    def test_vector_strategy(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How does the main module work?",
            expected_files=["main.py"],
            token_budget=4096,
        )
        with patch("archex.api._get_embedder", _stub_get_embedder):
            result = run_archex_query_vector(task, python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_VECTOR
        assert result.tokens_total >= 0
        assert result.tool_calls == 1
        assert result.timing is not None
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0

    def test_vector_recall_precision(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="authentication login service",
            expected_files=["services/auth.py", "main.py"],
            token_budget=8192,
        )
        with patch("archex.api._get_embedder", _stub_get_embedder):
            result = run_archex_query_vector(task, python_simple_repo)
        assert result.files_accessed >= 0
        assert isinstance(result.recall, float)
        assert isinstance(result.precision, float)


class TestRunArchexQueryFusion:
    def test_fusion_strategy(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How does the main module work?",
            expected_files=["main.py"],
            token_budget=4096,
        )
        with patch("archex.api._get_embedder", _stub_get_embedder):
            result = run_archex_query_fusion(task, python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_FUSION
        assert result.tokens_total >= 0
        assert result.tool_calls == 1
        assert result.timing is not None
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0

    def test_fusion_recall_precision(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="authentication login service",
            expected_files=["services/auth.py", "main.py"],
            token_budget=8192,
        )
        with patch("archex.api._get_embedder", _stub_get_embedder):
            result = run_archex_query_fusion(task, python_simple_repo)
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
