"""Tests for benchmark runner logic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.runner import AVAILABLE_STRATEGIES, run_benchmark

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fixture_task(python_simple_repo: Path) -> tuple[BenchmarkTask, Path]:
    task = BenchmarkTask(
        task_id="fixture_test",
        repo="test/python_simple",
        commit="HEAD",
        question="How does the main module work?",
        expected_files=["main.py", "utils.py"],
    )
    return task, python_simple_repo


class TestAvailableStrategies:
    def test_default_strategies(self) -> None:
        assert Strategy.RAW_FILES in AVAILABLE_STRATEGIES
        assert Strategy.RAW_GREPPED in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_QUERY in AVAILABLE_STRATEGIES
        # Fusion is in the default set (skipped at runtime if vector deps missing)
        assert Strategy.ARCHEX_QUERY_FUSION in AVAILABLE_STRATEGIES
        assert Strategy.CROSS_LAYER_FUSION in AVAILABLE_STRATEGIES
        assert Strategy.ARCHEX_SYMBOL_LOOKUP not in AVAILABLE_STRATEGIES


class TestRunBenchmark:
    def test_run_with_fixture_repo(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.RAW_GREPPED],
            repo_path=repo_path,
        )
        assert report.task_id == "fixture_test"
        assert report.repo == "test/python_simple"
        assert len(report.results) == 2

    def test_baseline_tokens_from_raw_files(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.RAW_GREPPED],
            repo_path=repo_path,
        )
        assert report.baseline_tokens > 0
        raw_result = next(r for r in report.results if r.strategy == Strategy.RAW_FILES)
        assert report.baseline_tokens == raw_result.tokens_total

    def test_savings_backfill(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.ARCHEX_QUERY],
            repo_path=repo_path,
        )
        raw = next(r for r in report.results if r.strategy == Strategy.RAW_FILES)
        assert raw.savings_vs_raw == 0.0
        # Other strategies should have savings backfilled
        for r in report.results:
            if r.strategy != Strategy.RAW_FILES:
                # savings_vs_raw is computed; could be negative if query returns more
                assert isinstance(r.savings_vs_raw, float)

    def test_strategy_filtering(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES],
            repo_path=repo_path,
        )
        assert len(report.results) == 1
        assert report.results[0].strategy == Strategy.RAW_FILES

    def test_symbol_lookup_skipped_gracefully(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_FILES, Strategy.ARCHEX_SYMBOL_LOOKUP],
            repo_path=repo_path,
        )
        # symbol_lookup should be skipped, only raw_files in results
        assert len(report.results) == 1
        assert report.results[0].strategy == Strategy.RAW_FILES

    def test_default_strategies_used_when_none(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        """strategies=None should use AVAILABLE_STRATEGIES."""
        task, repo_path = fixture_task
        report = run_benchmark(task, strategies=None, repo_path=repo_path)
        strategy_names = {r.strategy for r in report.results}
        # All three default strategies should have run
        assert Strategy.RAW_FILES in strategy_names
        assert Strategy.RAW_GREPPED in strategy_names
        assert Strategy.ARCHEX_QUERY in strategy_names

    def test_no_baseline_when_raw_files_omitted(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        """Without raw_files, baseline_tokens=0 and savings stay 0."""
        task, repo_path = fixture_task
        report = run_benchmark(
            task,
            strategies=[Strategy.RAW_GREPPED],
            repo_path=repo_path,
        )
        assert report.baseline_tokens == 0
        assert report.results[0].savings_vs_raw == 0.0

    def test_unknown_strategy_runner_skipped(
        self,
        fixture_task: tuple[BenchmarkTask, Path],
    ) -> None:
        """A strategy missing from the registry is logged and skipped."""
        from archex.benchmark.strategies import default_strategy_registry

        task, repo_path = fixture_task
        key = Strategy.RAW_GREPPED.value
        removed = default_strategy_registry._runners.pop(key)  # pyright: ignore[reportPrivateUsage]
        try:
            report = run_benchmark(
                task,
                strategies=[Strategy.RAW_FILES, Strategy.RAW_GREPPED],
                repo_path=repo_path,
            )
        finally:
            default_strategy_registry._runners[key] = removed  # pyright: ignore[reportPrivateUsage]

        # RAW_GREPPED was skipped; only RAW_FILES ran
        assert len(report.results) == 1
        assert report.results[0].strategy == Strategy.RAW_FILES


class TestCloneAtCommit:
    def testclone_at_commit(self, python_simple_repo: Path) -> None:
        """Exercise clone_at_commit with a local file:// URL substitute."""
        import subprocess

        import archex.benchmark.runner as runner_mod

        calls: list[list[str]] = []
        original_run = subprocess.run

        def mock_run(
            cmd: list[str],
            **kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        runner_mod.subprocess.run = mock_run  # type: ignore[assignment]
        try:
            path, needs_cleanup = runner_mod.clone_at_commit("owner/repo", "abc123")
        finally:
            runner_mod.subprocess.run = original_run  # type: ignore[assignment]

        assert needs_cleanup is True
        assert path.exists()
        # Shallow clone with --branch succeeds (returncode=0), so only 1 call.
        assert len(calls) == 1
        assert "clone" in calls[0]
        assert "https://github.com/owner/repo.git" in calls[0]
        assert "--depth" in calls[0]
        assert "abc123" in calls[0]

        # Cleanup
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    def test_cleanup_on_needs_cleanup(
        self,
        python_simple_repo: Path,
        tmp_path: Path,
    ) -> None:
        """When needs_cleanup=True, repo dir is removed after run_benchmark."""
        import archex.benchmark.runner as runner_mod

        # Create a temp dir that should get cleaned up
        clone_dir = tmp_path / "clone_target"
        clone_dir.mkdir()
        (clone_dir / "main.py").write_text("x = 1\n")

        def fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            return clone_dir, True

        original = runner_mod.clone_at_commit
        runner_mod.clone_at_commit = fake_clone  # type: ignore[assignment]
        try:
            task = BenchmarkTask(
                task_id="cleanup_test",
                repo="test/repo",
                commit="abc",
                question="test",
                expected_files=["main.py"],
            )
            # repo_path=None triggers the clone + cleanup path
            run_benchmark(task, strategies=[Strategy.RAW_FILES], repo_path=None)
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        # clone_dir should have been cleaned up
        assert not clone_dir.exists()


class TestRunAll:
    def _make_tasks_dir(self, tmp_path: Path) -> Path:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "test.yaml").write_text("""\
task_id: test_all
repo: test/repo
commit: HEAD
question: "How does main work?"
expected_files:
  - main.py
""")
        return tasks_dir

    def test_run_all_with_task_dir(
        self,
        python_simple_repo: Path,
        tmp_path: Path,
    ) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        output_dir = tmp_path / "results"

        import archex.benchmark.runner as runner_mod

        original = runner_mod.clone_at_commit

        def _fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            return python_simple_repo, False

        runner_mod.clone_at_commit = _fake_clone  # type: ignore[assignment]
        try:
            reports = run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                strategies=[Strategy.RAW_FILES],
            )
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        assert len(reports) == 1
        assert reports[0].task_id == "test_all"
        assert (output_dir / "test_all.json").exists()

    def test_task_filter_nonexistent_raises(self, tmp_path: Path) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        output_dir = tmp_path / "results"

        with pytest.raises(ValueError, match="No task found with id 'nonexistent'"):
            run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                task_filter="nonexistent",
            )

    def test_task_filter_selects_matching(
        self,
        python_simple_repo: Path,
        tmp_path: Path,
    ) -> None:
        from archex.benchmark.runner import run_all

        tasks_dir = self._make_tasks_dir(tmp_path)
        # Add a second task
        (tasks_dir / "other.yaml").write_text("""\
task_id: other_task
repo: test/repo
commit: HEAD
question: "Other?"
expected_files:
  - main.py
""")
        output_dir = tmp_path / "results"

        import archex.benchmark.runner as runner_mod

        original = runner_mod.clone_at_commit

        def _fake_clone(repo_slug: str, commit: str) -> tuple[Path, bool]:
            return python_simple_repo, False

        runner_mod.clone_at_commit = _fake_clone  # type: ignore[assignment]
        try:
            reports = run_all(
                tasks_dir=tasks_dir,
                output_dir=output_dir,
                strategies=[Strategy.RAW_FILES],
                task_filter="test_all",
            )
        finally:
            runner_mod.clone_at_commit = original  # type: ignore[assignment]

        assert len(reports) == 1
        assert reports[0].task_id == "test_all"
