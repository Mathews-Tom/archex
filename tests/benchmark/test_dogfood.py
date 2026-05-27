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
  - src/archex/serve/context.py
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


def _write_baseline(
    path: Path,
    *,
    recall: float = 1.0,
    include_diagnostics: bool = False,
) -> Path:
    baseline_path = path / "baseline.json"
    entries = [
        BaselineEntry(
            task_id="archex_query_pipeline",
            strategy=Strategy.ARCHEX_QUERY.value,
            recall=recall,
            precision=1.0,
            f1_score=recall,
            mrr=recall,
            ndcg=recall,
            map_score=recall,
            token_efficiency=recall,
        )
    ]
    if include_diagnostics:
        entries.extend(
            [
                BaselineEntry(
                    task_id="archex_query_pipeline",
                    strategy=Strategy.RAW_FILES.value,
                    recall=1.0,
                    precision=1.0,
                    f1_score=1.0,
                    mrr=1.0,
                ),
                BaselineEntry(
                    task_id="archex_query_pipeline",
                    strategy=Strategy.RAW_GREPPED.value,
                    recall=1.0,
                    precision=1.0,
                    f1_score=1.0,
                    mrr=1.0,
                ),
            ]
        )
    baseline = Baseline(entries=entries)
    baseline_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    return baseline_path


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
        seed_files=["src/archex/api.py", "src/archex/serve/context.py"],
        seed_recall=recall,
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


def _diagnostic_regressing_report(
    task: BenchmarkTask,
    strategies: list[Strategy] | None = None,
    repo_path: Path | None = None,
) -> BenchmarkReport:
    del strategies, repo_path
    report = _report(task)
    diagnostic_result = report.results[0].model_copy(
        update={
            "strategy": Strategy.RAW_GREPPED,
            "recall": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "map_score": 0.0,
            "token_efficiency": 0.0,
        }
    )
    return report.model_copy(update={"results": [diagnostic_result, *report.results]})


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
    baseline_path = _write_baseline(tmp_path)

    result = run_dogfood(tmp_path, baseline_path=baseline_path)

    assert captured == ["archex_query_pipeline"]
    assert result.latest_json_path == tmp_path / ".archex" / "dogfood" / "latest.json"
    assert result.latest_json_path.is_file()
    assert result.latest_markdown_path.is_file()
    assert result.history_json_path.is_file()
    payload = json.loads(result.latest_json_path.read_text(encoding="utf-8"))
    assert payload["tasks"] == ["archex_query_pipeline"]
    assert payload["regressions"] == []
    assert payload["retrieval_diagnostics"][0]["missing_expected_files"] == []
    assert payload["retrieval_diagnostics"][0]["failure_classes"] == []


def test_dogfood_reports_failure_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)

    def partial_report(
        task: BenchmarkTask,
        strategies: list[Strategy] | None = None,
        repo_path: Path | None = None,
    ) -> BenchmarkReport:
        del strategies, repo_path
        result = BenchmarkResult(
            task_id=task.task_id,
            strategy=Strategy.ARCHEX_QUERY,
            tokens_total=100,
            tool_calls=1,
            files_accessed=2,
            recall=0.5,
            precision=0.5,
            f1_score=0.5,
            mrr=0.5,
            ndcg=0.5,
            map_score=0.5,
            token_efficiency=0.5,
            savings_vs_raw=0.0,
            wall_time_ms=10.0,
            cached=False,
            timestamp="2026-05-24T00:00:00Z",
            seed_files=["src/archex/api.py", "src/archex/models.py"],
            seed_recall=0.5,
        )
        return BenchmarkReport(
            task_id=task.task_id,
            repo=task.repo,
            question=task.question,
            results=[result],
            baseline_tokens=100,
        )

    monkeypatch.setattr("archex.dogfood.run_benchmark", partial_report)
    baseline_path = _write_baseline(tmp_path, recall=0.5)

    result = run_dogfood(tmp_path, task_id="archex_query_pipeline", baseline_path=baseline_path)
    payload = json.loads(result.latest_json_path.read_text(encoding="utf-8"))
    diagnostic = payload["retrieval_diagnostics"][0]

    assert diagnostic["missing_expected_files"] == ["src/archex/serve/context.py"]
    assert diagnostic["top_unexpected_files"] == ["src/archex/models.py"]
    assert diagnostic["failure_classes"] == ["partial_recall", "ranking_miss", "seed_miss"]
    markdown = result.latest_markdown_path.read_text(encoding="utf-8")
    assert "## Retrieval Diagnostics" in markdown
    assert "`src/archex/serve/context.py`" in markdown


def test_dogfood_compares_explicit_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)
    baseline_path = _write_baseline(tmp_path)

    monkeypatch.setattr("archex.dogfood.run_benchmark", _regressing_report)

    result = run_dogfood(tmp_path, task_id="archex_query_pipeline", baseline_path=baseline_path)

    assert result.baseline_path == baseline_path
    assert result.regressions
    assert {regression.metric for regression in result.regressions} >= {"recall", "f1_score", "mrr"}


def test_dogfood_resolves_relative_baseline_from_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)
    baseline_path = _write_baseline(tmp_path)
    monkeypatch.setattr("archex.dogfood.run_benchmark", _passing_report)

    result = run_dogfood(
        tmp_path,
        task_id="archex_query_pipeline",
        baseline_path=baseline_path.name,
    )

    assert result.baseline_path == baseline_path


def test_dogfood_filters_diagnostic_strategy_regressions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)
    baseline_path = _write_baseline(tmp_path, include_diagnostics=True)
    monkeypatch.setattr("archex.dogfood.run_benchmark", _diagnostic_regressing_report)

    result = run_dogfood(tmp_path, task_id="archex_query_pipeline", baseline_path=baseline_path)

    assert result.regressions == []
    assert {comparison.strategy for comparison in result.comparisons} == {
        Strategy.ARCHEX_QUERY.value
    }
    payload = json.loads(result.latest_json_path.read_text(encoding="utf-8"))
    assert payload["regressions"] == []
    assert {comparison["strategy"] for comparison in payload["comparisons"]} == {
        Strategy.ARCHEX_QUERY.value
    }


def test_dogfood_command_exits_nonzero_on_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)
    baseline_path = _write_baseline(tmp_path)
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
    baseline_path = _write_baseline(tmp_path)
    monkeypatch.setattr("archex.dogfood.run_benchmark", _passing_report)

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
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["tasks"] == ["archex_query_pipeline"]
    assert payload["regressions"] == []


def test_dogfood_requires_explicit_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_git_repo(tmp_path)
    _write_tasks(tmp_path)
    monkeypatch.setattr("archex.dogfood.run_benchmark", _passing_report)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["dogfood", str(tmp_path), "--task", "archex_query_pipeline"],
    )

    assert result.exit_code == 1
    assert "Dogfood requires an explicit --baseline path" in result.output
