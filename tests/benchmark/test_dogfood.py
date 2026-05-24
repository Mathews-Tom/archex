"""Tests for dogfood benchmark orchestration."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.benchmark.baseline import Baseline, BaselineEntry
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, BenchmarkTask, Strategy
from archex.cli.main import cli
from archex.dogfood import run_dogfood

if TYPE_CHECKING:
    import pytest


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)


def _write_tasks(path: Path) -> Path:
    tasks_dir = path / "benchmarks" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "archex_query_pipeline.yaml").write_text(
        """\
task_id: archex_query_pipeline
repo: "."
commit: "HEAD"
category: self
question: "How does archex implement the query pipeline?"
expected_files:
  - src/archex/api.py
expected_symbols: []
""",
        encoding="utf-8",
    )
    (tasks_dir / "external_task.yaml").write_text(
        """\
task_id: external_task
repo: owner/repo
commit: abc123
question: "How does an external project work?"
expected_files:
  - src/main.py
expected_symbols: []
""",
        encoding="utf-8",
    )
    return tasks_dir


def _report(task: BenchmarkTask, recall: float = 1.0) -> BenchmarkReport:
    result = BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.ARCHEX_QUERY,
        tokens_total=100,
        tool_calls=1,
        files_accessed=1,
        recall=recall,
        precision=1.0,
        f1_score=recall,
        mrr=recall,
        ndcg=recall,
        map_score=recall,
        token_efficiency=recall,
        savings_vs_raw=0.0,
        wall_time_ms=10.0,
        cached=False,
        timestamp="2026-05-24T00:00:00Z",
        seed_files=["src/archex/api.py"],
    )
    return BenchmarkReport(
        task_id=task.task_id,
        repo=task.repo,
        question=task.question,
        results=[result],
        baseline_tokens=100,
    )


def _regressing_report(
    task: BenchmarkTask,
    strategies: list[Strategy] | None = None,
    repo_path: Path | None = None,
) -> BenchmarkReport:
    del strategies, repo_path
    return _report(task, recall=0.5)


def _passing_report(
    task: BenchmarkTask,
    strategies: list[Strategy] | None = None,
    repo_path: Path | None = None,
) -> BenchmarkReport:
    del strategies, repo_path
    return _report(task)


def test_dogfood_runs_self_tasks_and_writes_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)
    captured: list[str] = []

    def fake_run_benchmark(
        task: BenchmarkTask,
        strategies: list[Strategy] | None = None,
        repo_path: Path | None = None,
    ) -> BenchmarkReport:
        captured.append(task.task_id)
        assert repo_path == tmp_path
        assert strategies == [Strategy.RAW_FILES, Strategy.RAW_GREPPED, Strategy.ARCHEX_QUERY]
        return _report(task)

    monkeypatch.setattr("archex.dogfood.run_benchmark", fake_run_benchmark)

    result = run_dogfood(tmp_path)

    assert captured == ["archex_query_pipeline"]
    assert result.latest_json_path == tmp_path / ".archex" / "dogfood" / "latest.json"
    assert result.latest_json_path.is_file()
    assert result.latest_markdown_path.is_file()
    assert result.history_json_path.is_file()
    payload = json.loads(result.latest_json_path.read_text(encoding="utf-8"))
    assert payload["tasks"] == ["archex_query_pipeline"]
    assert payload["regressions"] == []
    assert payload["retrieval_gaps"][0]["missing_expected_files"] == []


def test_dogfood_compares_explicit_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    baseline = Baseline(
        entries=[
            BaselineEntry(
                task_id="archex_query_pipeline",
                strategy=Strategy.ARCHEX_QUERY.value,
                recall=1.0,
                precision=1.0,
                f1_score=1.0,
                mrr=1.0,
                ndcg=1.0,
                map_score=1.0,
                token_efficiency=1.0,
            )
        ]
    )
    baseline_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr("archex.dogfood.run_benchmark", _regressing_report)

    result = run_dogfood(tmp_path, task_id="archex_query_pipeline", baseline_path=baseline_path)

    assert result.baseline_path == baseline_path
    assert result.regressions
    assert {regression.metric for regression in result.regressions} >= {"recall", "f1_score", "mrr"}


def test_dogfood_command_exits_nonzero_on_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        Baseline(
            entries=[
                BaselineEntry(
                    task_id="archex_query_pipeline",
                    strategy=Strategy.ARCHEX_QUERY.value,
                    recall=1.0,
                    precision=1.0,
                    f1_score=1.0,
                    mrr=1.0,
                )
            ]
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    monkeypatch.setattr("archex.dogfood.run_benchmark", _regressing_report)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dogfood",
            str(tmp_path),
            "--task",
            "archex_query_pipeline",
            "--baseline",
            str(baseline_path),
        ],
    )

    assert result.exit_code == 1
    assert "Regressions:" in result.output
    assert "archex_query_pipeline/archex_query recall" in result.output


def test_dogfood_command_json_outputs_latest_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)
    monkeypatch.setattr("archex.dogfood.run_benchmark", _passing_report)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["dogfood", str(tmp_path), "--task", "archex_query_pipeline", "--format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["tasks"] == ["archex_query_pipeline"]
    assert payload["regressions"] == []
