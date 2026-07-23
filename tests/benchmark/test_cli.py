"""Tests for benchmark CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from click.testing import CliRunner

from archex.benchmark.baseline import RankingSnapshotEntry
from archex.benchmark.models import (
    BenchmarkEvidenceManifest,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    BundleOnlyEvaluatorCommand,
    Strategy,
    TaskCompletionResult,
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


def _patch_gate_evidence(
    monkeypatch: pytest.MonkeyPatch,
    reports: list[BenchmarkReport],
) -> None:
    def load_evidence(
        input_dir: Path,
        tasks_dir: Path,
    ) -> tuple[BenchmarkEvidenceManifest, list[BenchmarkReport]]:
        del input_dir, tasks_dir
        return cast("BenchmarkEvidenceManifest", object()), reports

    monkeypatch.setattr("archex.cli.benchmark_cmd.load_evidence_reports", load_evidence)


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

    def test_markdown_output_with_baseline_comparison(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        candidate_dir = tmp_path / "candidate"
        baseline_dir = tmp_path / "baseline"
        candidate_dir.mkdir()
        baseline_dir.mkdir()
        baseline = BenchmarkReport(
            task_id="test",
            repo="owner/repo",
            question="How?",
            results=[
                BenchmarkResult(
                    task_id="test",
                    strategy=Strategy.ARCHEX_QUERY_HYBRID,
                    tokens_total=100,
                    tool_calls=1,
                    files_accessed=1,
                    recall=0.8,
                    precision=1.0,
                    f1_score=0.89,
                    mrr=0.5,
                    savings_vs_raw=0.0,
                    wall_time_ms=10.0,
                    cached=True,
                    timestamp="2025-01-01T00:00:00Z",
                )
            ],
            baseline_tokens=100,
        )
        candidate = baseline.model_copy(
            update={
                "results": [
                    BenchmarkResult(
                        task_id="test",
                        strategy=Strategy.ARCHEX_QUERY_HYBRID_QUANTIZED_4BIT,
                        tokens_total=100,
                        tool_calls=1,
                        files_accessed=1,
                        recall=0.79,
                        precision=1.0,
                        f1_score=0.88,
                        mrr=0.5,
                        savings_vs_raw=0.0,
                        wall_time_ms=12.0,
                        cached=True,
                        timestamp="2025-01-01T00:00:00Z",
                        provenance={"vector_compression_ratio": "7.5"},
                    )
                ]
            }
        )
        (baseline_dir / "test.json").write_text(baseline.model_dump_json(indent=2))
        (candidate_dir / "test.json").write_text(candidate.model_dump_json(indent=2))

        result = runner.invoke(
            benchmark_cmd,
            ["report", "--input", str(candidate_dir), "--baseline", str(baseline_dir)],
        )

        assert result.exit_code == 0
        assert "# Baseline Comparison" in result.output
        assert "| test | -0.010 | +0.000 | -0.010" in result.output
        assert "- Compression: 7.50x" in result.output

    def test_no_results_error(self, runner: CliRunner, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(benchmark_cmd, ["report", "--input", str(empty_dir)])
        assert result.exit_code != 0
        assert "No result files" in result.output


class TestBaselineSaveCommand:
    def test_without_ranking_source_omits_ranking(
        self,
        runner: CliRunner,
        results_dir: Path,
        tmp_path: Path,
    ) -> None:
        output_path = tmp_path / "baseline.json"

        result = runner.invoke(
            benchmark_cmd,
            ["baseline", "save", "--input", str(results_dir), "--output", str(output_path)],
        )

        assert result.exit_code == 0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["ranking"] == []

    def test_with_ranking_source_attaches_snapshot(
        self,
        runner: CliRunner,
        results_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ranking_source = tmp_path / "ranking_source"
        ranking_source.mkdir()
        output_path = tmp_path / "baseline.json"

        stub_ranking = [
            RankingSnapshotEntry(file_path="src/a.py", centrality=0.4, symbol_count=12),
            RankingSnapshotEntry(file_path="src/b.py", centrality=0.1, symbol_count=3),
        ]
        captured: list[Path] = []

        def fake_build_ranking_snapshot(repo_root: Path) -> list[RankingSnapshotEntry]:
            captured.append(repo_root)
            return stub_ranking

        monkeypatch.setattr(
            "archex.cli.benchmark_cmd.build_ranking_snapshot",
            fake_build_ranking_snapshot,
        )

        result = runner.invoke(
            benchmark_cmd,
            [
                "baseline",
                "save",
                "--input",
                str(results_dir),
                "--output",
                str(output_path),
                "--ranking-source",
                str(ranking_source),
            ],
        )

        assert result.exit_code == 0
        assert captured == [ranking_source]
        assert f"Ranking snapshot:   {len(stub_ranking)} files" in result.output
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["ranking"] == [
            {"file_path": "src/a.py", "centrality": 0.4, "symbol_count": 12},
            {"file_path": "src/b.py", "centrality": 0.1, "symbol_count": 3},
        ]


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


class TestGateCommand:
    def test_passes_when_max_latency_ms_not_set(
        self,
        runner: CliRunner,
        results_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        report = BenchmarkReport.model_validate_json((results_dir / "test.json").read_bytes())
        _patch_gate_evidence(monkeypatch, [report])
        result = runner.invoke(benchmark_cmd, ["gate", "--input", str(results_dir)])
        assert result.exit_code == 0
        assert "Quality gate passed." in result.output

    def test_promotion_gate_hard_checks_only_candidate_and_protects_evidence(
        self,
        runner: CliRunner,
        results_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        control = BenchmarkResult(
            task_id="test",
            strategy=Strategy.ARCHEX_QUERY,
            tokens_total=1000,
            tool_calls=1,
            files_accessed=3,
            recall=1.0,
            precision=0.1,
            f1_score=0.18,
            mrr=0.1,
            savings_vs_raw=0.0,
            token_efficiency=0.01,
            token_efficiency_with_completion=0.01,
            required_file_recall=1.0,
            region_recall=1.0,
            line_recall=1.0,
            wall_time_ms=100.0,
            cached=True,
            cache_state="warm",
            timestamp="2025-01-01T00:00:00Z",
        )
        candidate = control.model_copy(
            update={
                "strategy": Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE,
                "precision": 1.0,
                "f1_score": 1.0,
                "mrr": 1.0,
                "token_efficiency": 0.5,
                "token_efficiency_with_completion": 0.5,
                "warm_latency_ms": 100.0,
            }
        )
        report = BenchmarkReport(
            task_id="test",
            repo="owner/repo",
            question="How?",
            results=[control, candidate],
            baseline_tokens=1000,
        )
        _patch_gate_evidence(monkeypatch, [report])

        result = runner.invoke(
            benchmark_cmd,
            [
                "gate",
                "--input",
                str(results_dir),
                "--promotion-strategy",
                Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE.value,
                "--control-strategy",
                Strategy.ARCHEX_QUERY.value,
                "--min-token-efficiency-with-completion",
                "0.08",
                "--max-p95-warm-latency-ms",
                "3000",
            ],
        )

        assert result.exit_code == 0
        assert "Quality gate passed." in result.output

    def test_promotion_gate_fails_protected_region_regression(
        self,
        runner: CliRunner,
        results_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        control = BenchmarkResult(
            task_id="test",
            strategy=Strategy.ARCHEX_QUERY,
            tokens_total=1000,
            tool_calls=1,
            files_accessed=1,
            recall=1.0,
            precision=1.0,
            f1_score=1.0,
            mrr=1.0,
            savings_vs_raw=0.0,
            token_efficiency=0.5,
            token_efficiency_with_completion=0.5,
            required_file_recall=1.0,
            region_recall=1.0,
            line_recall=1.0,
            wall_time_ms=100.0,
            cached=True,
            cache_state="warm",
            timestamp="2025-01-01T00:00:00Z",
        )
        candidate = control.model_copy(
            update={
                "strategy": Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE,
                "region_recall": 0.5,
                "warm_latency_ms": 100.0,
            }
        )
        report = BenchmarkReport(
            task_id="test",
            repo="owner/repo",
            question="How?",
            results=[control, candidate],
            baseline_tokens=1000,
        )
        _patch_gate_evidence(monkeypatch, [report])

        result = runner.invoke(
            benchmark_cmd,
            [
                "gate",
                "--input",
                str(results_dir),
                "--promotion-strategy",
                Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE.value,
                "--control-strategy",
                Strategy.ARCHEX_QUERY.value,
                "--min-token-efficiency-with-completion",
                "0.08",
                "--max-p95-warm-latency-ms",
                "3000",
            ],
        )

        assert result.exit_code != 0
        assert "region_recall" in result.output

    def test_promotion_gate_fails_zero_recall_regression(
        self,
        runner: CliRunner,
        results_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        control = BenchmarkResult(
            task_id="test",
            strategy=Strategy.ARCHEX_QUERY,
            tokens_total=1000,
            tool_calls=1,
            files_accessed=1,
            recall=1.0,
            precision=1.0,
            f1_score=1.0,
            mrr=1.0,
            savings_vs_raw=0.0,
            token_efficiency=0.5,
            token_efficiency_with_completion=0.5,
            required_file_recall=1.0,
            wall_time_ms=100.0,
            cached=True,
            cache_state="warm",
            timestamp="2025-01-01T00:00:00Z",
        )
        candidate = control.model_copy(
            update={
                "strategy": Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE,
                "recall": 0.0,
                "required_file_recall": 0.0,
                "warm_latency_ms": 100.0,
            }
        )
        report = BenchmarkReport(
            task_id="test",
            repo="owner/repo",
            question="How?",
            results=[control, candidate],
            baseline_tokens=1000,
        )
        _patch_gate_evidence(monkeypatch, [report])

        result = runner.invoke(
            benchmark_cmd,
            [
                "gate",
                "--input",
                str(results_dir),
                "--promotion-strategy",
                Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE.value,
                "--control-strategy",
                Strategy.ARCHEX_QUERY.value,
                "--min-token-efficiency-with-completion",
                "0.08",
                "--max-p95-warm-latency-ms",
                "3000",
            ],
        )

        assert result.exit_code != 0
        assert "zero_recall_regression" in result.output

    def test_promotion_gate_fails_fixed_agent_success_regression(
        self,
        runner: CliRunner,
        results_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        control = BenchmarkResult(
            task_id="test",
            strategy=Strategy.ARCHEX_QUERY,
            tokens_total=1000,
            tool_calls=1,
            files_accessed=1,
            recall=1.0,
            precision=1.0,
            f1_score=1.0,
            mrr=1.0,
            savings_vs_raw=0.0,
            token_efficiency=0.5,
            token_efficiency_with_completion=0.5,
            required_file_recall=1.0,
            task_completion_result=TaskCompletionResult.PASS,
            wall_time_ms=100.0,
            cached=True,
            cache_state="warm",
            timestamp="2025-01-01T00:00:00Z",
        )
        candidate = control.model_copy(
            update={
                "strategy": Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE,
                "task_completion_result": TaskCompletionResult.FAIL,
                "warm_latency_ms": 100.0,
            }
        )
        report = BenchmarkReport(
            task_id="test",
            repo="owner/repo",
            question="How?",
            results=[control, candidate],
            baseline_tokens=1000,
        )
        _patch_gate_evidence(monkeypatch, [report])

        result = runner.invoke(
            benchmark_cmd,
            [
                "gate",
                "--input",
                str(results_dir),
                "--promotion-strategy",
                Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE.value,
                "--control-strategy",
                Strategy.ARCHEX_QUERY.value,
                "--min-token-efficiency-with-completion",
                "0.08",
                "--max-p95-warm-latency-ms",
                "3000",
            ],
        )

        assert result.exit_code != 0
        assert "fixed_agent_success_regression" in result.output

    def test_promotion_gate_fails_language_family_regression(
        self,
        runner: CliRunner,
        results_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "test.yaml").write_text(
            "task_id: test\nrepo: owner/repo\ncommit: v1.0.0\nquestion: How?\n"
            "expected_files: [a.py]\nlanguages: [python]\n"
        )
        control = BenchmarkResult(
            task_id="test",
            strategy=Strategy.ARCHEX_QUERY,
            tokens_total=1000,
            tool_calls=1,
            files_accessed=1,
            recall=0.9,
            precision=1.0,
            f1_score=1.0,
            mrr=1.0,
            savings_vs_raw=0.0,
            token_efficiency=0.5,
            token_efficiency_with_completion=0.5,
            required_file_recall=1.0,
            wall_time_ms=100.0,
            cached=True,
            cache_state="warm",
            timestamp="2025-01-01T00:00:00Z",
        )
        candidate = control.model_copy(
            update={
                "strategy": Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE,
                "recall": 0.4,
                "warm_latency_ms": 100.0,
            }
        )
        report = BenchmarkReport(
            task_id="test",
            repo="owner/repo",
            question="How?",
            results=[control, candidate],
            baseline_tokens=1000,
        )
        _patch_gate_evidence(monkeypatch, [report])

        result = runner.invoke(
            benchmark_cmd,
            [
                "gate",
                "--input",
                str(results_dir),
                "--tasks-dir",
                str(tasks_dir),
                "--promotion-strategy",
                Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE.value,
                "--control-strategy",
                Strategy.ARCHEX_QUERY.value,
                "--min-token-efficiency-with-completion",
                "0.08",
                "--max-p95-warm-latency-ms",
                "3000",
            ],
        )

        assert result.exit_code != 0
        assert "language_family_recall" in result.output
        assert "language:python" in result.output

    def test_promotion_gate_rejects_gate_exempt_candidate(
        self,
        runner: CliRunner,
        results_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        report = BenchmarkReport.model_validate_json((results_dir / "test.json").read_bytes())
        _patch_gate_evidence(monkeypatch, [report])

        result = runner.invoke(
            benchmark_cmd,
            [
                "gate",
                "--input",
                str(results_dir),
                "--promotion-strategy",
                Strategy.RAW_FILES.value,
                "--control-strategy",
                Strategy.ARCHEX_QUERY.value,
                "--min-token-efficiency-with-completion",
                "0.08",
                "--max-p95-warm-latency-ms",
                "3000",
            ],
        )

        assert result.exit_code != 0
        assert "gate-exempt" in result.output

    def test_max_latency_ms_hard_fails_on_breach(
        self,
        runner: CliRunner,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        results = tmp_path / "results"
        results.mkdir()
        result_obj = BenchmarkResult(
            task_id="slow_task",
            strategy=Strategy.RAW_FILES,
            tokens_total=1000,
            tool_calls=1,
            files_accessed=3,
            recall=1.0,
            precision=1.0,
            savings_vs_raw=0.0,
            wall_time_ms=9000.0,
            cached=False,
            timestamp="2025-01-01T00:00:00Z",
        )
        report = BenchmarkReport(
            task_id="slow_task",
            repo="owner/repo",
            question="How?",
            results=[result_obj],
            baseline_tokens=1000,
        )
        _patch_gate_evidence(monkeypatch, [report])

        result = runner.invoke(
            benchmark_cmd,
            ["gate", "--input", str(results), "--max-latency-ms", "5000"],
        )

        assert result.exit_code != 0
        assert "LATENCY GATE FAILED" in result.output
        assert "slow_task/raw_files" in result.output

    def test_max_latency_ms_passes_under_threshold(
        self,
        runner: CliRunner,
        results_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        report = BenchmarkReport.model_validate_json((results_dir / "test.json").read_bytes())
        _patch_gate_evidence(monkeypatch, [report])
        # results_dir fixture's sample result has wall_time_ms=50.0
        result = runner.invoke(
            benchmark_cmd,
            ["gate", "--input", str(results_dir), "--max-latency-ms", "5000"],
        )
        assert result.exit_code == 0
        assert "LATENCY GATE FAILED" not in result.output


class TestBundleEvalCommand:
    def test_bundle_eval_help(self, runner: CliRunner) -> None:
        result = runner.invoke(benchmark_cmd, ["bundle-eval", "--help"])

        assert result.exit_code == 0
        assert "--evaluator-command" in result.output
        assert "--bundle-format" in result.output

    def test_bundle_eval_requires_evaluator_command(
        self,
        runner: CliRunner,
        tasks_dir: Path,
    ) -> None:
        result = runner.invoke(benchmark_cmd, ["bundle-eval", "--tasks-dir", str(tasks_dir)])

        assert result.exit_code != 0
        assert "Missing option '--evaluator-command'" in result.output

    def test_bundle_eval_invokes_opt_in_runner(
        self,
        runner: CliRunner,
        tasks_dir: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def capture_run_bundle_only_eval_all(
            tasks: list[BenchmarkTask],
            output_dir: Path,
            *,
            command: object,
            bundle_format: str,
        ) -> list[BenchmarkReport]:
            captured["task_count"] = len(tasks)
            captured["output_dir"] = output_dir
            captured["command"] = command
            captured["bundle_format"] = bundle_format
            return []

        monkeypatch.setattr(
            "archex.cli.benchmark_cmd.run_bundle_only_eval_all",
            capture_run_bundle_only_eval_all,
        )

        result = runner.invoke(
            benchmark_cmd,
            [
                "bundle-eval",
                "--tasks-dir",
                str(tasks_dir),
                "--output",
                str(tmp_path / "bundle-eval"),
                "--evaluator-command",
                "python",
                "--evaluator-arg",
                "tests/fixtures/bundle_eval_command.py",
                "--evaluator-arg",
                "pass",
                "--timeout-seconds",
                "12",
                "--bundle-format",
                "json",
            ],
        )

        assert result.exit_code == 0
        assert captured["task_count"] == 1
        assert captured["output_dir"] == tmp_path / "bundle-eval"
        command = cast("BundleOnlyEvaluatorCommand", captured["command"])
        assert command.command == "python"
        assert command.args == ["tests/fixtures/bundle_eval_command.py", "pass"]
        assert command.timeout_seconds == 12
        assert captured["bundle_format"] == "json"


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
            Strategy.RAW_RIPGREP,
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
            Strategy.RAW_RIPGREP,
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
            Strategy.RAW_RIPGREP,
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
            Strategy.RAW_RIPGREP,
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
