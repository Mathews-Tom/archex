"""End-to-end benchmark integration tests using fixture-based tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from archex.benchmark.baseline import Baseline, compare_baseline, load_baseline, save_baseline
from archex.benchmark.gate import GateViolation, check_gate
from archex.benchmark.models import BenchmarkReport, BenchmarkTask, Strategy
from archex.benchmark.reporter import format_markdown, format_summary
from archex.benchmark.runner import run_benchmark

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fixture_task(python_simple_repo: Path) -> tuple[BenchmarkTask, Path]:
    task = BenchmarkTask(
        task_id="e2e_test_task",
        repo="test/python_simple",
        commit="HEAD",
        question="How does authentication work?",
        expected_files=["services/auth.py"],
    )
    return task, python_simple_repo


class TestBenchmarkEndToEnd:
    """Run benchmark with fixture-based tasks end-to-end."""

    def test_run_benchmark_produces_report(self, fixture_task: tuple[BenchmarkTask, Path]) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.ARCHEX_QUERY],
            repo_path=repo_path,
        )
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) > 0

    def test_run_benchmark_raw_files(self, fixture_task: tuple[BenchmarkTask, Path]) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES],
            repo_path=repo_path,
        )
        assert isinstance(report, BenchmarkReport)
        assert report.baseline_tokens > 0

    def test_run_benchmark_multiple_strategies(
        self, fixture_task: tuple[BenchmarkTask, Path]
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.ARCHEX_QUERY],
            repo_path=repo_path,
        )
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 2

    def test_format_markdown_renders_all_columns(
        self, fixture_task: tuple[BenchmarkTask, Path]
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.ARCHEX_QUERY],
            repo_path=repo_path,
        )
        md = format_markdown(report)
        assert isinstance(md, str)
        assert len(md) > 0
        assert "e2e_test_task" in md
        assert "Strategy" in md

    def test_format_summary_multiple_reports(
        self, fixture_task: tuple[BenchmarkTask, Path]
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES],
            repo_path=repo_path,
        )
        summary = format_summary([report])
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_check_gate_with_real_results(self, fixture_task: tuple[BenchmarkTask, Path]) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.ARCHEX_QUERY],
            repo_path=repo_path,
        )
        violations = check_gate([report])
        assert isinstance(violations, list)
        for v in violations:
            assert isinstance(v, GateViolation)

    def test_baseline_roundtrip(self, fixture_task: tuple[BenchmarkTask, Path]) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES],
            repo_path=repo_path,
        )
        baseline = save_baseline([report])
        assert isinstance(baseline, Baseline)
        assert len(baseline.entries) > 0

        data = baseline.model_dump()
        loaded = load_baseline(data)
        assert isinstance(loaded, Baseline)
        assert len(loaded.entries) == len(baseline.entries)

    def test_compare_baseline_no_regression(self, fixture_task: tuple[BenchmarkTask, Path]) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES],
            repo_path=repo_path,
        )
        baseline = save_baseline([report])
        comparisons = compare_baseline([report], baseline)
        # Comparing against self: no regressions
        regressions = [c for c in comparisons if c.regression]
        assert regressions == []

    def test_task_id_preserved_in_report(self, fixture_task: tuple[BenchmarkTask, Path]) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES],
            repo_path=repo_path,
        )
        assert report.task_id == "e2e_test_task"
        assert report.repo == "test/python_simple"

    def test_savings_backfill_with_raw_baseline(
        self, fixture_task: tuple[BenchmarkTask, Path]
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.ARCHEX_QUERY],
            repo_path=repo_path,
        )
        raw = next(r for r in report.results if r.strategy == Strategy.RAW_FILES)
        assert raw.savings_vs_raw == 0.0
        for r in report.results:
            if r.strategy != Strategy.RAW_FILES:
                assert isinstance(r.savings_vs_raw, float)
