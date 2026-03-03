"""Tests for delta benchmark strategies: models, quality gate, and integration."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from archex.benchmark.delta_strategies import run_delta_benchmark
from archex.benchmark.gate import (
    DeltaGateViolation,
    DeltaQualityThresholds,
    check_delta_gate,
)
from archex.benchmark.models import (
    DeltaBenchmarkResult,
    DeltaBenchmarkTask,
    DeltaStrategy,
)
from archex.benchmark.reporter import format_delta_summary

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _git(repo: Path, *args: str) -> str:
    """Run a git command in repo and return stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# Unit tests: model construction and serialization
# ---------------------------------------------------------------------------


class TestDeltaBenchmarkModels:
    def test_delta_strategy_values(self) -> None:
        assert DeltaStrategy.DELTA_INDEX == "delta_index"
        assert DeltaStrategy.FULL_REINDEX == "full_reindex"

    def test_delta_task_defaults(self) -> None:
        task = DeltaBenchmarkTask(
            task_id="test",
            repo=".",
            base_commit="abc",
            delta_commit="def",
        )
        assert task.expected_delta == []
        assert task.language == "python"

    def test_delta_task_with_expected(self) -> None:
        task = DeltaBenchmarkTask(
            task_id="test",
            repo="owner/repo",
            base_commit="abc",
            delta_commit="def",
            expected_delta=["src/main.py"],
            language="typescript",
        )
        assert task.expected_delta == ["src/main.py"]
        assert task.language == "typescript"

    def test_delta_result_construction(self) -> None:
        result = DeltaBenchmarkResult(
            task_id="test",
            strategy=DeltaStrategy.DELTA_INDEX,
            delta_files=5,
            total_files=20,
            delta_pct=25.0,
            delta_time_ms=150.0,
            full_reindex_time_ms=600.0,
            speedup_factor=4.0,
            correctness=True,
            chunks_updated=10,
            chunks_unchanged=50,
            edges_updated=3,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        assert result.speedup_factor == 4.0
        assert result.correctness is True
        assert result.delta_meta is None

    def test_delta_result_serialization_roundtrip(self) -> None:
        result = DeltaBenchmarkResult(
            task_id="test",
            strategy=DeltaStrategy.DELTA_INDEX,
            delta_files=3,
            total_files=10,
            delta_pct=30.0,
            delta_time_ms=100.0,
            full_reindex_time_ms=500.0,
            speedup_factor=5.0,
            correctness=True,
            chunks_updated=5,
            chunks_unchanged=15,
            edges_updated=2,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        json_str = result.model_dump_json()
        restored = DeltaBenchmarkResult.model_validate_json(json_str)
        assert restored.task_id == result.task_id
        assert restored.speedup_factor == result.speedup_factor


# ---------------------------------------------------------------------------
# Unit tests: delta quality gate
# ---------------------------------------------------------------------------


class TestDeltaQualityGate:
    def _make_result(
        self,
        *,
        correctness: bool = True,
        speedup: float = 3.0,
        task_id: str = "test",
    ) -> DeltaBenchmarkResult:
        return DeltaBenchmarkResult(
            task_id=task_id,
            strategy=DeltaStrategy.DELTA_INDEX,
            delta_files=5,
            total_files=20,
            delta_pct=25.0,
            delta_time_ms=100.0,
            full_reindex_time_ms=100.0 * speedup,
            speedup_factor=speedup,
            correctness=correctness,
            chunks_updated=5,
            chunks_unchanged=15,
            edges_updated=2,
            timestamp="2024-01-01T00:00:00+00:00",
        )

    def test_pass_on_correct_and_fast(self) -> None:
        results = [self._make_result(correctness=True, speedup=3.0)]
        violations = check_delta_gate(results)
        assert violations == []

    def test_fail_on_incorrect(self) -> None:
        results = [self._make_result(correctness=False, speedup=3.0)]
        violations = check_delta_gate(results)
        assert len(violations) == 1
        assert violations[0].metric == "correctness"
        assert violations[0].actual == 0.0

    def test_fail_on_slow(self) -> None:
        results = [self._make_result(correctness=True, speedup=1.0)]
        violations = check_delta_gate(results)
        assert len(violations) == 1
        assert violations[0].metric == "speedup_factor"

    def test_custom_thresholds(self) -> None:
        thresholds = DeltaQualityThresholds(min_speedup=5.0)
        results = [self._make_result(speedup=3.0)]
        violations = check_delta_gate(results, thresholds)
        assert len(violations) == 1
        assert violations[0].threshold == 5.0

    def test_no_correctness_requirement(self) -> None:
        thresholds = DeltaQualityThresholds(require_correctness=False, min_speedup=1.0)
        results = [self._make_result(correctness=False, speedup=2.0)]
        violations = check_delta_gate(results, thresholds)
        assert violations == []

    def test_multiple_violations(self) -> None:
        results = [self._make_result(correctness=False, speedup=1.0)]
        violations = check_delta_gate(results)
        assert len(violations) == 2
        metrics = {v.metric for v in violations}
        assert metrics == {"correctness", "speedup_factor"}

    def test_violation_model(self) -> None:
        v = DeltaGateViolation(
            task_id="test",
            metric="speedup_factor",
            threshold=1.5,
            actual=1.0,
        )
        assert v.threshold == 1.5
        assert v.actual == 1.0


# ---------------------------------------------------------------------------
# Unit tests: delta reporter
# ---------------------------------------------------------------------------


class TestDeltaReporter:
    def test_empty_results(self) -> None:
        output = format_delta_summary([])
        assert "No delta benchmark results" in output

    def test_format_with_results(self) -> None:
        result = DeltaBenchmarkResult(
            task_id="test_task",
            strategy=DeltaStrategy.DELTA_INDEX,
            delta_files=3,
            total_files=10,
            delta_pct=30.0,
            delta_time_ms=100.0,
            full_reindex_time_ms=400.0,
            speedup_factor=4.0,
            correctness=True,
            chunks_updated=5,
            chunks_unchanged=15,
            edges_updated=2,
            timestamp="2024-01-01T00:00:00+00:00",
        )
        output = format_delta_summary([result])
        assert "test_task" in output
        assert "4.0x" in output
        assert "yes" in output
        assert "Delta Benchmark Summary" in output


# ---------------------------------------------------------------------------
# Integration test: delta vs full correctness (requires git)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRunDeltaBenchmark:
    def test_delta_vs_full_correctness(self, tmp_path: Path) -> None:
        """Create a 2-commit repo, run delta benchmark, verify correctness."""
        src = FIXTURES_DIR / "python_simple"
        repo = tmp_path / "bench_repo"
        shutil.copytree(src, repo)

        _git(repo, "init")
        _git(repo, "config", "user.email", "test@archex.test")
        _git(repo, "config", "user.name", "archex-test")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "initial")
        base_commit = _git_head(repo)

        # Make a change: modify utils.py and add a new file
        (repo / "utils.py").write_text("def updated_util(): return 42\n")
        (repo / "extra.py").write_text("def extra_func(): pass\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "delta changes")
        delta_commit = _git_head(repo)

        task = DeltaBenchmarkTask(
            task_id="integration_test",
            repo=".",
            base_commit=base_commit,
            delta_commit=delta_commit,
        )

        result = run_delta_benchmark(task, repo)

        assert result.correctness is True
        assert result.speedup_factor > 0
        assert result.delta_files > 0
        assert result.total_files > 0
        assert result.delta_time_ms > 0
        assert result.full_reindex_time_ms > 0
        assert result.delta_meta is not None
