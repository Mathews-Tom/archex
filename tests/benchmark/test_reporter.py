"""Tests for benchmark report generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkReport, BenchmarkResult, Strategy, TaskCategory
from archex.benchmark.reporter import (
    format_bucketed_summary,
    format_chunker_frontier_table,
    format_json,
    format_markdown,
    format_strategy_comparison,
    format_summary,
)

if TYPE_CHECKING:
    from archex.models import ChunkerName


def _make_result(
    strategy: Strategy,
    tokens: int = 1000,
    savings: float = 0.0,
    recall: float = 1.0,
    precision: float = 1.0,
    tokens_input: int = 2000,
    tokens_output: int = 1000,
    token_efficiency: float = 0.5,
    chunker: ChunkerName = "default",
    index_chunk_count: int = 10,
    mean_chunk_tokens: float = 50.0,
    category: TaskCategory | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="test",
        strategy=strategy,
        tokens_total=tokens,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        token_efficiency=token_efficiency,
        tool_calls=1,
        files_accessed=3,
        recall=recall,
        precision=precision,
        savings_vs_raw=savings,
        wall_time_ms=50.0,
        cached=False,
        timestamp="2025-01-01T00:00:00Z",
        chunker=chunker,
        index_chunk_count=index_chunk_count,
        mean_chunk_tokens=mean_chunk_tokens,
        category=category,
    )


def _make_report(results: list[BenchmarkResult] | None = None) -> BenchmarkReport:
    if results is None:
        results = [
            _make_result(Strategy.RAW_FILES, tokens=2000),
            _make_result(Strategy.ARCHEX_QUERY, tokens=500, savings=75.0, recall=0.8),
        ]
    return BenchmarkReport(
        task_id="test",
        repo="owner/repo",
        question="How does X work?",
        results=results,
        baseline_tokens=2000,
    )


class TestFormatMarkdown:
    def test_contains_header(self) -> None:
        md = format_markdown(_make_report())
        assert "## Benchmark: test" in md

    def test_contains_table_header(self) -> None:
        md = format_markdown(_make_report())
        assert "| Strategy |" in md
        assert "Required Recall" in md
        assert "Missed Task Rate" in md
        assert "Completion" in md
        assert "Receipt Accuracy" in md

    def test_contains_strategy_rows(self) -> None:
        md = format_markdown(_make_report())
        assert "raw_files" in md
        assert "archex_query" in md

    def test_contains_repo_and_question(self) -> None:
        md = format_markdown(_make_report())
        assert "owner/repo" in md
        assert "How does X work?" in md

    def test_contains_baseline(self) -> None:
        md = format_markdown(_make_report())
        assert "2,000" in md


class TestFormatJson:
    def test_valid_json(self) -> None:
        import json

        output = format_json(_make_report())
        data = json.loads(output)
        assert data["task_id"] == "test"
        assert len(data["results"]) == 2


class TestFormatSummary:
    def test_empty_reports(self) -> None:
        summary = format_summary([])
        assert "No benchmark results" in summary

    def test_summary_header(self) -> None:
        summary = format_summary([_make_report()])
        assert "# Benchmark Summary" in summary
        assert "**Tasks:** 1" in summary

    def test_summary_table(self) -> None:
        summary = format_summary([_make_report()])
        assert "| Strategy |" in summary
        assert "raw_files" in summary
        assert "archex_query" in summary
        assert "Avg Efficiency" in summary
        assert "Avg Required Recall" in summary
        assert "Missed Task Rate" in summary

    def test_multi_report_aggregation(self) -> None:
        r1 = _make_report(
            [
                _make_result(Strategy.RAW_FILES, tokens=2000),
                _make_result(Strategy.ARCHEX_QUERY, tokens=500, savings=75.0, recall=0.8),
            ]
        )
        r2 = _make_report(
            [
                _make_result(Strategy.RAW_FILES, tokens=3000),
                _make_result(Strategy.ARCHEX_QUERY, tokens=600, savings=80.0, recall=0.9),
            ]
        )
        summary = format_summary([r1, r2])
        assert "**Tasks:** 2" in summary


class TestFormatStrategyComparison:
    def test_empty_reports(self) -> None:
        result = format_strategy_comparison([])
        assert "No benchmark results" in result

    def test_contains_per_task_table(self) -> None:
        report = _make_report()
        result = format_strategy_comparison([report])
        assert "## test" in result
        assert "raw_files" in result
        assert "archex_query" in result
        assert "Tokens Total" in result
        assert "Required Recall" in result
        assert "Receipt Accuracy" in result

    def test_includes_missing_required_file_appendix(self) -> None:
        report = _make_report(
            [
                _make_result(Strategy.ARCHEX_QUERY),
                _make_result(Strategy.ARCHEX_SCOUT_FETCH),
            ]
        )
        report.results[0].required_files_missing = ["missing.py"]
        report.results[0].missed_required_file_rate = 1.0
        result = format_strategy_comparison([report])
        assert "Missing required files appendix" in result
        assert "missing.py" in result

    def test_contains_head_to_head(self) -> None:
        report = _make_report()
        result = format_strategy_comparison([report])
        assert "Head-to-Head Wins" in result

    def test_contains_best_strategy(self) -> None:
        report = _make_report()
        result = format_strategy_comparison([report])
        assert "Best Strategy per Metric" in result


class TestFormatBucketedSummary:
    def test_empty_reports(self) -> None:
        assert "No benchmark results" in format_bucketed_summary([])

    def test_uncategorized_fallback(self) -> None:
        result = format_bucketed_summary([_make_report()])
        assert "uncategorized" in result

    def test_categorized_reports(self) -> None:
        r_self = _make_result(Strategy.ARCHEX_QUERY, recall=0.9, category=TaskCategory.SELF)
        r_ext = _make_result(
            Strategy.ARCHEX_QUERY, recall=0.5, category=TaskCategory.EXTERNAL_FRAMEWORK
        )
        report_self = BenchmarkReport(
            task_id="self_task",
            repo=".",
            question="q",
            results=[r_self],
            baseline_tokens=1000,
        )
        report_ext = BenchmarkReport(
            task_id="ext_task",
            repo="owner/repo",
            question="q",
            results=[r_ext],
            baseline_tokens=1000,
        )
        output = format_bucketed_summary([report_self, report_ext])
        assert "self (1 tasks)" in output
        assert "external-framework (1 tasks)" in output
        assert "All Tasks (2 tasks)" in output
        assert "Seed Recall" in output


class TestFormatChunkerFrontierTable:
    def test_contains_quality_latency_and_granularity_axes(self) -> None:
        baseline = _make_report(
            [
                _make_result(
                    Strategy.ARCHEX_QUERY_FUSION_RERANK,
                    recall=0.80,
                    precision=0.40,
                    token_efficiency=0.60,
                    chunker="default",
                    index_chunk_count=100,
                    mean_chunk_tokens=80.0,
                )
            ]
        )
        candidate = _make_report(
            [
                _make_result(
                    Strategy.ARCHEX_QUERY_FUSION_RERANK,
                    recall=0.82,
                    precision=0.42,
                    token_efficiency=0.61,
                    chunker="cast",
                    index_chunk_count=120,
                    mean_chunk_tokens=70.0,
                )
            ]
        )

        table = format_chunker_frontier_table([candidate], [baseline])

        assert "Chunker Frontier Comparison" in table
        assert "archex_query_fusion_rerank | default" in table
        assert "archex_query_fusion_rerank | cast" in table
        assert "Chunk Count" in table
        assert "Mean Chunk Tokens" in table
