"""Tests for benchmark CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from click.testing import CliRunner

from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    Strategy,
)
from archex.cli.benchmark_cmd import benchmark_cmd

if TYPE_CHECKING:
    from archex.benchmark.progress import BenchmarkProgress


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


def _empty_tasks(
    tasks_dir: Path,
    *,
    task_filter: str | None = None,
    self_only: bool = False,
) -> list[BenchmarkTask]:
    del tasks_dir, task_filter, self_only
    return []


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


class TestTriageCommand:
    def test_markdown_output(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        results = tmp_path / "results"
        tasks = tmp_path / "tasks"
        results.mkdir()
        tasks.mkdir()
        result = BenchmarkResult(
            task_id="task",
            strategy=Strategy.ARCHEX_QUERY,
            tokens_total=100,
            tool_calls=1,
            files_accessed=1,
            recall=0.0,
            precision=0.0,
            f1_score=0.0,
            savings_vs_raw=0.0,
            wall_time_ms=50.0,
            cached=False,
            timestamp="2026-01-01T00:00:00Z",
            seed_files=["src/noise.py"],
        )
        report = BenchmarkReport(
            task_id="task",
            repo="owner/repo",
            question="How does dispatch work?",
            results=[result],
            baseline_tokens=100,
        )
        (results / "task.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
        (tasks / "task.yaml").write_text(
            """\
task_id: task
repo: owner/repo
commit: abc123
category: external-large
question: "How does dispatch work?"
expected_files:
  - src/task.py
""",
            encoding="utf-8",
        )

        output = runner.invoke(
            benchmark_cmd,
            ["triage", "--input", str(results), "--tasks-dir", str(tasks)],
        )

        assert output.exit_code == 0
        assert "# Benchmark Failure Triage" in output.output
        assert "`task`" in output.output
        assert "zero_recall" in output.output

    def test_json_output(self, runner: CliRunner, results_dir: Path, tasks_dir: Path) -> None:
        output = runner.invoke(
            benchmark_cmd,
            [
                "triage",
                "--format",
                "json",
                "--input",
                str(results_dir),
                "--tasks-dir",
                str(tasks_dir),
            ],
        )

        assert output.exit_code == 0
        assert json.loads(output.output) == []


class TestReadinessCommand:
    def test_markdown_output(self, runner: CliRunner, results_dir: Path, tasks_dir: Path) -> None:
        output = runner.invoke(
            benchmark_cmd,
            ["readiness", "--input", str(results_dir), "--tasks-dir", str(tasks_dir)],
        )

        assert output.exit_code == 0
        assert "# Benchmark Readiness" in output.output
        assert "No `archex_query` results found" in output.output

    def test_json_output(self, runner: CliRunner, results_dir: Path, tasks_dir: Path) -> None:
        output = runner.invoke(
            benchmark_cmd,
            [
                "readiness",
                "--format",
                "json",
                "--input",
                str(results_dir),
                "--tasks-dir",
                str(tasks_dir),
            ],
        )

        assert output.exit_code == 0
        payload = json.loads(output.output)
        assert payload["strategy"] == "archex_query"
        assert payload["task_count"] == 0
        assert payload["ready"] is True


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
commit: abc
question: "How?"
expected_files: []
""")
        result = runner.invoke(benchmark_cmd, ["validate", "--tasks-dir", str(tasks)])
        assert result.exit_code != 0
        assert "bad.yaml" in result.output
        assert "expected_files" in result.output

    def test_validate_rejects_missing_local_expected_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        tasks = tmp_path / "tasks"
        tasks.mkdir()
        (tasks / "bad.yaml").write_text("""\
task_id: missing_local_file
repo: "."
commit: HEAD
question: "How?"
expected_files:
  - missing.py
""")

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(benchmark_cmd, ["validate", "--tasks-dir", str(tasks)])

        assert result.exit_code != 0
        assert "Expected file not found: missing.py" in result.output

    def test_validate_all_task_families(self, runner: CliRunner, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks"
        arch_tasks = tmp_path / "arch_tasks"
        delta_tasks = tmp_path / "delta_tasks"
        tasks.mkdir()
        arch_tasks.mkdir()
        delta_tasks.mkdir()
        (tasks / "task.yaml").write_text("""\
task_id: task
repo: owner/repo
commit: abc
question: "How?"
expected_files:
  - src/main.py
""")
        (arch_tasks / "arch.yaml").write_text("""\
task_id: arch
repo: "."
commit: HEAD
question: "Which architecture is present?"
include_paths:
  - src
arch_oracle:
  patterns:
    - name: pattern
""")
        (delta_tasks / "delta.yaml").write_text("""\
task_id: delta
repo: "."
base_commit: base
delta_commit: delta
expected_delta:
  - src/main.py
""")

        result = runner.invoke(
            benchmark_cmd,
            [
                "validate",
                "--kind",
                "all",
                "--tasks-dir",
                str(tasks),
                "--arch-tasks-dir",
                str(arch_tasks),
                "--delta-tasks-dir",
                str(delta_tasks),
            ],
        )

        assert result.exit_code == 0
        assert "1 task" in result.output
        assert "1 architecture task" in result.output
        assert "1 delta task" in result.output

    def test_validate_reports_malformed_yaml(self, runner: CliRunner, tmp_path: Path) -> None:
        tasks = tmp_path / "tasks"
        tasks.mkdir()
        (tasks / "bad.yaml").write_text("task_id: [unterminated")

        result = runner.invoke(benchmark_cmd, ["validate", "--tasks-dir", str(tasks)])
        assert result.exit_code != 0
        assert "Failed to parse YAML" in result.output


class TestRunCommand:
    @pytest.fixture(autouse=True)
    def _disable_model_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
        def no_preflight(
            strategies: list[Strategy],
            retrieval_options: BenchmarkRetrievalOptions,
        ) -> list[str]:
            del strategies, retrieval_options
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.warm_benchmark_models", no_preflight)

    def test_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(benchmark_cmd, ["run", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--task" in result.output
        assert "--strategy" in result.output
        assert "--scout" in result.output
        assert "--query-fusion" in result.output
        assert "--cross_layer_fusion" in result.output
        assert "--rerank" in result.output
        assert "--splade" in result.output
        assert "--module-prefilter" in result.output
        assert "--allow-remote-code" in result.output
        assert "--self-only" in result.output
        assert "--no-progress" in result.output
        assert "--embedder" in result.output
        assert "--rerank-model" in result.output

    def test_run_uses_default_strategies_without_flags(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, task_filter, self_only, progress, tasks
            captured["output_dir"] = output_dir
            captured["strategies"] = strategies
            captured["retrieval_options"] = retrieval_options
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(benchmark_cmd, ["run"])
        assert result.exit_code == 0
        assert captured["output_dir"] == Path(".archex/benchmark-results")
        assert captured["strategies"] == [
            Strategy.RAW_FILES,
            Strategy.RAW_GREPPED,
            Strategy.ARCHEX_QUERY,
        ]
        assert captured["retrieval_options"] == BenchmarkRetrievalOptions()

    def test_run_scout_flag_adds_scout_strategy(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, task_filter, self_only, progress, tasks, retrieval_options
            captured["strategies"] = strategies
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(benchmark_cmd, ["run", "--scout"])
        assert result.exit_code == 0
        assert captured["strategies"] == [
            Strategy.RAW_FILES,
            Strategy.RAW_GREPPED,
            Strategy.ARCHEX_QUERY,
            Strategy.ARCHEX_SCOUT_FETCH,
        ]

    def test_run_adds_experimental_flags(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, task_filter, self_only, progress, tasks, retrieval_options
            captured["strategies"] = strategies
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(
            benchmark_cmd,
            ["run", "--query-fusion", "--cross_layer_fusion"],
        )
        assert result.exit_code == 0
        assert captured["strategies"] == [
            Strategy.RAW_FILES,
            Strategy.RAW_GREPPED,
            Strategy.ARCHEX_QUERY,
            Strategy.ARCHEX_QUERY_FUSION,
            Strategy.CROSS_LAYER_FUSION,
        ]

    def test_run_rerank_flag_adds_fusion_and_rerank(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, task_filter, self_only, progress, tasks, retrieval_options
            captured["strategies"] = strategies
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(benchmark_cmd, ["run", "--rerank"])
        assert result.exit_code == 0
        assert captured["strategies"] == [
            Strategy.RAW_FILES,
            Strategy.RAW_GREPPED,
            Strategy.ARCHEX_QUERY,
            Strategy.ARCHEX_QUERY_FUSION,
            Strategy.ARCHEX_QUERY_FUSION_RERANK,
        ]

    def test_run_passes_retrieval_measurement_flags(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, strategies, task_filter, self_only, progress, tasks
            captured["retrieval_options"] = retrieval_options
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(
            benchmark_cmd,
            ["run", "--splade", "--module-prefilter", "--allow-remote-code"],
        )
        assert result.exit_code == 0
        assert captured["retrieval_options"] == BenchmarkRetrievalOptions(
            splade=True,
            module_prefilter=True,
            allow_remote_code=True,
        )

    def test_run_reports_remote_code_preflight_error(
        self,
        runner: CliRunner,
    ) -> None:
        result = runner.invoke(benchmark_cmd, ["run", "--query-fusion"])

        assert result.exit_code != 0
        assert "Remote code is disabled" in result.output
        assert "Traceback" not in result.output

    def test_run_passes_embedder_flag(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, strategies, task_filter, self_only, progress, tasks
            captured["retrieval_options"] = retrieval_options
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(benchmark_cmd, ["run", "--embedder", "coderank"])
        assert result.exit_code == 0
        assert captured["retrieval_options"] == BenchmarkRetrievalOptions(embedder="coderank")

    def test_run_passes_chunker_flag(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, strategies, task_filter, self_only, progress, tasks
            captured["retrieval_options"] = retrieval_options
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(benchmark_cmd, ["run", "--chunker", "cast"])
        assert result.exit_code == 0
        assert captured["retrieval_options"] == BenchmarkRetrievalOptions(chunker="cast")

    def test_run_passes_strategy_chunker_flags(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, strategies, task_filter, self_only, progress, tasks
            captured["retrieval_options"] = retrieval_options
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(
            benchmark_cmd,
            ["run", "--bm25-chunker", "default", "--vector-chunker", "cast"],
        )
        assert result.exit_code == 0
        assert captured["retrieval_options"] == BenchmarkRetrievalOptions(
            bm25_chunker="default",
            vector_chunker="cast",
        )

    def test_run_passes_rerank_candidate_limit_flag(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, strategies, task_filter, self_only, progress, tasks
            captured["retrieval_options"] = retrieval_options
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(benchmark_cmd, ["run", "--rerank-candidate-limit", "3"])
        assert result.exit_code == 0
        assert captured["retrieval_options"] == BenchmarkRetrievalOptions(rerank_candidate_limit=3)

    def test_run_passes_rerank_model_flag(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, strategies, task_filter, self_only, progress, tasks
            captured["retrieval_options"] = retrieval_options
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(
            benchmark_cmd,
            ["run", "--rerank-model", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
        )
        assert result.exit_code == 0
        assert captured["retrieval_options"] == BenchmarkRetrievalOptions(
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def test_run_preflights_models_before_loading_tasks(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        events: list[str] = []

        def fake_preflight(
            strategies: list[Strategy],
            retrieval_options: BenchmarkRetrievalOptions,
        ) -> list[str]:
            events.append("preflight")
            assert Strategy.ARCHEX_QUERY_FUSION_RERANK in strategies
            assert retrieval_options == BenchmarkRetrievalOptions(splade=True)
            return ["splade", "reranker"]

        def fake_load_selected_tasks(
            tasks_dir: Path,
            *,
            task_filter: str | None = None,
            self_only: bool = False,
        ) -> list[BenchmarkTask]:
            del tasks_dir, task_filter, self_only
            events.append("load_tasks")
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.warm_benchmark_models", fake_preflight)
        monkeypatch.setattr(
            "archex.cli.benchmark_cmd.load_selected_tasks", fake_load_selected_tasks
        )

        def fake_run_all(**kwargs: object) -> list[BenchmarkReport]:
            del kwargs
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)

        result = runner.invoke(benchmark_cmd, ["run", "--splade", "--rerank"])

        assert result.exit_code == 0
        assert events == ["preflight", "load_tasks"]
        assert "Benchmark model preflight loaded 2 model(s)." in result.output

    def test_run_self_only_flag_filters_tasks(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, strategies, task_filter, progress, tasks, retrieval_options
            captured["self_only"] = self_only
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(benchmark_cmd, ["run", "--self-only"])
        assert result.exit_code == 0
        assert captured["self_only"] is True

    def test_run_no_progress_forces_disabled_controller(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run_all(
            tasks_dir: Path,
            output_dir: Path,
            strategies: list[Strategy] | None = None,
            task_filter: str | None = None,
            self_only: bool = False,
            progress: object | None = None,
            tasks: object | None = None,
            retrieval_options: BenchmarkRetrievalOptions | None = None,
        ) -> list[BenchmarkReport]:
            del tasks_dir, output_dir, strategies, task_filter, self_only, tasks, retrieval_options
            assert progress is not None
            captured["progress_enabled"] = cast("BenchmarkProgress", progress).live_display_enabled
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_selected_tasks", _empty_tasks)
        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all", fake_run_all)
        result = runner.invoke(benchmark_cmd, ["run", "--no-progress"])
        assert result.exit_code == 0
        assert captured["progress_enabled"] is False


class TestDeltaCommand:
    def test_delta_run_defaults_to_generated_state_dir(
        self,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = tmp_path / "delta_tasks"
        tasks.mkdir()
        captured: dict[str, object] = {}

        def fake_run_all_delta(
            tasks_dir: Path,
            output_dir: Path,
            task_filter: str | None = None,
        ) -> list[object]:
            captured["tasks_dir"] = tasks_dir
            captured["output_dir"] = output_dir
            captured["task_filter"] = task_filter
            return []

        monkeypatch.setattr("archex.cli.benchmark_cmd.run_all_delta", fake_run_all_delta)

        result = runner.invoke(
            benchmark_cmd,
            ["delta", "run", "--tasks-dir", str(tasks)],
        )

        assert result.exit_code == 0
        assert captured == {
            "tasks_dir": tasks,
            "output_dir": Path(".archex/delta-results"),
            "task_filter": None,
        }
