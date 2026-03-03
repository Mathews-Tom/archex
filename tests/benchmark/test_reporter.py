"""Tests for benchmark report generation."""

from __future__ import annotations

from archex.benchmark.models import BenchmarkReport, BenchmarkResult, Strategy
from archex.benchmark.reporter import format_json, format_markdown, format_summary


def _make_result(
    strategy: Strategy,
    tokens: int = 1000,
    savings: float = 0.0,
    recall: float = 1.0,
    precision: float = 1.0,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="test",
        strategy=strategy,
        tokens_total=tokens,
        tool_calls=1,
        files_accessed=3,
        recall=recall,
        precision=precision,
        savings_vs_raw=savings,
        wall_time_ms=50.0,
        cached=False,
        timestamp="2025-01-01T00:00:00Z",
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
