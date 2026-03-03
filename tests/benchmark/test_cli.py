"""Tests for benchmark CLI commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from archex.benchmark.models import BenchmarkReport, BenchmarkResult, Strategy
from archex.cli.benchmark_cmd import benchmark_cmd

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    """Create a results directory with a sample JSON result."""
    results = tmp_path / "results"
    results.mkdir()

    result = BenchmarkResult(
        task_id="test",
        strategy=Strategy.RAW_FILES,
        tokens_total=1000,
        tool_calls=1,
        files_accessed=3,
        recall=1.0,
        precision=1.0,
        savings_vs_raw=0.0,
        wall_time_ms=50.0,
        cached=False,
        timestamp="2025-01-01T00:00:00Z",
    )
    report = BenchmarkReport(
        task_id="test",
        repo="owner/repo",
        question="How?",
        results=[result],
        baseline_tokens=1000,
    )
    (results / "test.json").write_text(report.model_dump_json(indent=2))
    return results


@pytest.fixture
def tasks_dir(tmp_path: Path) -> Path:
    """Create a tasks directory with sample YAML files."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "test_task.yaml").write_text("""\
task_id: test_task
repo: owner/repo
commit: abc123
question: "How does X work?"
expected_files:
  - src/main.py
""")
    return tasks


class TestReportCommand:
    def test_markdown_output(self, runner: CliRunner, results_dir: Path) -> None:
        result = runner.invoke(benchmark_cmd, ["report", "--input", str(results_dir)])
        assert result.exit_code == 0
        assert "## Benchmark: test" in result.output
        assert "raw_files" in result.output

    def test_json_output(self, runner: CliRunner, results_dir: Path) -> None:
        result = runner.invoke(
            benchmark_cmd,
            ["report", "--format", "json", "--input", str(results_dir)],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["task_id"] == "test"

    def test_no_results_error(self, runner: CliRunner, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(benchmark_cmd, ["report", "--input", str(empty_dir)])
        assert result.exit_code != 0
        assert "No result files" in result.output


class TestValidateCommand:
    def test_valid_tasks(self, runner: CliRunner, tasks_dir: Path) -> None:
        result = runner.invoke(benchmark_cmd, ["validate", "--tasks-dir", str(tasks_dir)])
        assert result.exit_code == 0
        assert "All 1 task(s) valid" in result.output

    def test_invalid_task(self, runner: CliRunner, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks"
        tasks.mkdir()
        (tasks / "bad.yaml").write_text("""\
task_id: bad_task
repo: owner/repo
commit: ""
question: "  "
expected_files: []
""")
        result = runner.invoke(benchmark_cmd, ["validate", "--tasks-dir", str(tasks)])
        assert result.exit_code != 0


class TestRunCommand:
    def test_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(benchmark_cmd, ["run", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--task" in result.output
        assert "--strategy" in result.output
