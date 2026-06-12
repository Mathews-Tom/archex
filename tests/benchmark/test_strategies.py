"""Tests for benchmark strategy implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from archex.benchmark.models import BenchmarkRetrievalOptions, BenchmarkTask, Strategy
from archex.benchmark.strategies import (
    _archex_fields,  # pyright: ignore[reportPrivateUsage]
    _deduplicate_ranked,  # pyright: ignore[reportPrivateUsage]
    benchmark_repo_source,
    compute_bundle_completion_penalty,
    compute_map,
    compute_mrr,
    compute_ndcg,
    compute_precision,
    compute_recall,
    compute_symbol_recall,
    compute_token_efficiency,
    count_file_tokens,
    extract_keywords,
    measure_archex_freshness,
    reset_benchmark_retrieval_options,
    run_archex_query,
    run_archex_query_fusion,
    run_archex_query_fusion_rerank,
    run_archex_query_vector,
    run_archex_symbol_lookup,
    run_cross_layer_fusion,
    run_raw_files,
    run_raw_grepped,
    run_surrogate_vector,
    set_benchmark_retrieval_options,
)
from archex.cache import CacheManager
from archex.exceptions import ConfigError
from archex.index.embeddings import (
    JINA_BERT_CODE_REVISION,
    JINA_V2_MAX_SEQ_LENGTH,
    JINA_V2_MODEL_REVISION,
)
from archex.models import CodeChunk, ContextBundle, IndexConfig, RankedChunk, RetrievalMetadata

JINA_V2_CACHE_IDENTITY = (
    f"jina-v2@{JINA_V2_MODEL_REVISION}"
    f"+code={JINA_BERT_CODE_REVISION}"
    f"+max_seq={JINA_V2_MAX_SEQ_LENGTH}"
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


def test_measure_archex_freshness_returns_correct_probe(
    sample_task: BenchmarkTask,
    python_simple_repo: Path,
) -> None:
    latency_ms, correct = measure_archex_freshness(sample_task, python_simple_repo)

    assert latency_ms > 0
    assert correct is True


def _ranked_chunk(chunk_id: str, file_path: str, *, score: float) -> RankedChunk:
    chunk = CodeChunk(
        id=chunk_id,
        content=f"content for {file_path}",
        file_path=file_path,
        start_line=1,
        end_line=1,
        language="python",
        token_count=4,
    )
    return RankedChunk(chunk=chunk, final_score=score)


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


class TestBundleCompletionPenalty:
    def test_missing_expected_files_count_as_completion_tokens(self, tmp_path: Path) -> None:
        (tmp_path / "found.py").write_text("print('found')\n", encoding="utf-8")
        (tmp_path / "missing.py").write_text("print('missing')\n", encoding="utf-8")

        tokens, files = compute_bundle_completion_penalty(
            tmp_path, {"found.py"}, ["found.py", "missing.py"]
        )

        assert tokens == count_file_tokens(tmp_path, ["missing.py"])
        assert files == ["missing.py"]


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


class TestComputeTokenEfficiency:
    def test_full_raw_read_has_no_savings(self) -> None:
        assert compute_token_efficiency(tokens_output=100, tokens_input=100) == 0.0

    def test_smaller_output_has_higher_efficiency(self) -> None:
        assert compute_token_efficiency(tokens_output=25, tokens_input=100) == 0.75

    def test_empty_input_has_no_efficiency(self) -> None:
        assert compute_token_efficiency(tokens_output=25, tokens_input=0) == 0.0

    def test_output_larger_than_input_clamps_to_zero(self) -> None:
        assert compute_token_efficiency(tokens_output=125, tokens_input=100) == 0.0


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
        assert result.token_efficiency == 0.0
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
    def test_benchmark_source_uses_stable_repo_commit_identity(
        self,
        sample_task: BenchmarkTask,
        tmp_path: Path,
    ) -> None:
        cache = CacheManager(cache_dir=str(tmp_path / "cache"))
        repo_a = tmp_path / "clone-a"
        repo_b = tmp_path / "clone-b"
        repo_a.mkdir()
        repo_b.mkdir()

        source_a = benchmark_repo_source(sample_task, repo_a)
        source_b = benchmark_repo_source(sample_task, repo_b)

        assert source_a.stable_identity == f"test/repo@abc#embedder={JINA_V2_CACHE_IDENTITY}"
        assert source_b.stable_identity == f"test/repo@abc#embedder={JINA_V2_CACHE_IDENTITY}"
        assert cache.cache_key(source_a) == cache.cache_key(source_b)

    def test_benchmark_source_resolves_missing_commit_from_git_head(
        self,
        tmp_path: Path,
    ) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="",
            question="How?",
            expected_files=["main.py"],
        )

        with patch.object(CacheManager, "git_head", return_value="resolved"):
            source = benchmark_repo_source(task, tmp_path)

        assert source.stable_identity == (f"test/repo@resolved#embedder={JINA_V2_CACHE_IDENTITY}")

    def test_benchmark_source_rejects_missing_commit_without_git_head(
        self,
        tmp_path: Path,
    ) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="",
            question="How?",
            expected_files=["main.py"],
        )

        with (
            patch.object(CacheManager, "git_head", return_value=None),
            pytest.raises(ConfigError, match="has no commit"),
        ):
            benchmark_repo_source(task, tmp_path)

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

    def test_expanded_files_split_uses_file_count_boundary(self, tmp_path: Path) -> None:
        for file_path in ("seed_a.py", "seed_b.py", "expanded_a.py", "expanded_b.py"):
            (tmp_path / file_path).write_text("print('x')\n", encoding="utf-8")
        bundle = ContextBundle(
            query="How does graph expansion work?",
            chunks=[
                _ranked_chunk("seed-a-1", "seed_a.py", score=1.0),
                _ranked_chunk("seed-a-2", "seed_a.py", score=0.9),
                _ranked_chunk("seed-b-1", "seed_b.py", score=0.8),
                _ranked_chunk("expanded-a-1", "expanded_a.py", score=0.7),
                _ranked_chunk("expanded-b-1", "expanded_b.py", score=0.6),
            ],
            token_count=20,
            token_budget=100,
            retrieval_metadata=RetrievalMetadata(
                candidates_found=3,
                candidates_after_expansion=5,
                seed_files_found=2,
                expansion_files_added=2,
                expansion_eligible_seeds=2,
                expansion_candidates_found=3,
                expansion_import_neighbor_edges=3,
                expansion_same_module_candidates=1,
                expansion_hub_candidates=1,
                expansion_test_candidates_skipped=1,
                expansion_zero_candidate_reason="",
            ),
        )
        task = BenchmarkTask(
            task_id="archex_graph_expansion",
            repo="Mathews-Tom/archex",
            commit="abc",
            question="How does graph expansion work?",
            expected_files=["expanded_a.py"],
        )

        fields = _archex_fields(bundle, task, tmp_path)

        assert fields.seed_files == ["seed_a.py", "seed_b.py"]
        assert fields.expanded_files == ["expanded_a.py", "expanded_b.py"]
        assert fields.expansion_ratio == 1.0
        assert fields.seed_recall == 0.0
        assert fields.expansion_eligible_seeds == 2
        assert fields.expansion_candidates_found == 3
        assert fields.expansion_import_neighbor_edges == 3
        assert fields.expansion_same_module_candidates == 1
        assert fields.expansion_hub_candidates == 1
        assert fields.expansion_test_candidates_skipped == 1
        assert fields.expansion_zero_candidate_reason == ""

    def test_expanded_files_uses_metadata_paths_when_expansion_is_not_included(
        self,
        tmp_path: Path,
    ) -> None:
        for file_path in ("seed_a.py", "seed_b.py", "expanded_a.py", "expanded_b.py"):
            (tmp_path / file_path).write_text("print('x')\n", encoding="utf-8")
        bundle = ContextBundle(
            query="How does graph expansion work?",
            chunks=[
                _ranked_chunk("seed-a-1", "seed_a.py", score=1.0),
                _ranked_chunk("seed-b-1", "seed_b.py", score=0.8),
            ],
            token_count=8,
            token_budget=100,
            retrieval_metadata=RetrievalMetadata(
                candidates_found=2,
                candidates_after_expansion=4,
                seed_files_found=2,
                seed_file_paths=["seed_a.py", "seed_b.py"],
                expanded_file_paths=["expanded_a.py", "expanded_b.py"],
                expansion_files_added=2,
                expansion_eligible_seeds=2,
                expansion_candidates_found=0,
                expansion_import_neighbor_edges=0,
                expansion_same_module_candidates=0,
                expansion_hub_candidates=0,
                expansion_test_candidates_skipped=0,
                expansion_zero_candidate_reason="no_import_neighbors",
            ),
        )
        task = BenchmarkTask(
            task_id="archex_graph_expansion",
            repo="Mathews-Tom/archex",
            commit="abc",
            question="How does graph expansion work?",
            expected_files=["expanded_a.py"],
        )

        fields = _archex_fields(bundle, task, tmp_path)

        assert fields.seed_files == ["seed_a.py", "seed_b.py"]
        assert fields.expanded_files == ["expanded_a.py", "expanded_b.py"]
        assert fields.expansion_ratio == 1.0

        assert fields.expansion_zero_candidate_reason == "no_import_neighbors"


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


def test_vector_strategies_read_configured_embedder(
    sample_task: BenchmarkTask,
    python_simple_repo: Path,
) -> None:
    captured: list[str | None] = []

    def fake_query(
        _source: object,
        question: str,
        *,
        token_budget: int,
        explicit_token_budget: bool,
        config: object,
        index_config: IndexConfig,
        timing: object | None = None,
    ) -> ContextBundle:
        del config, explicit_token_budget, timing
        captured.append(index_config.embedder)
        return ContextBundle(
            query=question,
            chunks=[],
            token_count=0,
            token_budget=token_budget,
        )

    token = set_benchmark_retrieval_options(BenchmarkRetrievalOptions(embedder="coderank"))
    try:
        with patch("archex.api.query", fake_query):
            for runner in (
                run_archex_query_vector,
                run_surrogate_vector,
                run_archex_query_fusion,
                run_cross_layer_fusion,
                run_archex_query_fusion_rerank,
            ):
                runner(sample_task, python_simple_repo)
    finally:
        reset_benchmark_retrieval_options(token)

    assert captured == ["coderank"] * 5


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


class TestRunSurrogateVector:
    def test_surrogate_vector_strategy(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How does authentication work?",
            expected_files=["services/auth.py"],
            token_budget=4096,
        )
        with patch("archex.api._get_embedder", _stub_get_embedder):
            result = run_surrogate_vector(task, python_simple_repo)
        assert result.strategy == Strategy.SURROGATE_VECTOR
        assert result.vector_mode == "surrogate"
        assert result.surrogate_version == "v1"


class TestRunCrossLayerFusion:
    def test_cross_layer_fusion_strategy(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How does authentication work?",
            expected_files=["services/auth.py", "main.py"],
            token_budget=4096,
        )
        with patch("archex.api._get_embedder", _stub_get_embedder):
            result = run_cross_layer_fusion(task, python_simple_repo)
        assert result.strategy == Strategy.CROSS_LAYER_FUSION
        assert result.vector_mode == "surrogate"
        assert result.surrogate_version == "v1"


class TestRunArchexQueryFusionRerank:
    def test_fusion_rerank_strategy(self, python_simple_repo: Path) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="test/repo",
            commit="abc",
            question="How does the main module work?",
            expected_files=["main.py"],
            token_budget=4096,
        )
        with patch("archex.api._get_embedder", _stub_get_embedder):
            result = run_archex_query_fusion_rerank(task, python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_FUSION_RERANK
        assert result.tokens_total >= 0
        assert result.tool_calls == 1
        assert result.timing is not None
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0


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
