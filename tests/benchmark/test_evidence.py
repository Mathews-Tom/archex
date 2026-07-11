"""Tests for immutable benchmark evidence directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from archex.benchmark.evidence import (
    BenchmarkEvidenceError,
    build_evidence_manifest,
    copy_evidence_as_baseline,
    prepare_evidence_directory,
    task_manifest_digest,
    validate_baseline_coverage,
    validate_evidence_directory,
    write_evidence_manifest,
)
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    Strategy,
)


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="evidence_task",
        repo="owner/repo",
        commit="0123456789abcdef",
        question="Where is the evidence contract?",
        expected_files=["src/evidence.py"],
    )


def _report() -> BenchmarkReport:
    result = BenchmarkResult(
        task_id="evidence_task",
        strategy=Strategy.ARCHEX_QUERY,
        tokens_total=100,
        tool_calls=1,
        files_accessed=1,
        recall=1.0,
        precision=1.0,
        f1_score=1.0,
        mrr=1.0,
        ndcg=1.0,
        map_score=1.0,
        savings_vs_raw=0.0,
        token_efficiency=0.2,
        wall_time_ms=1.0,
        cached=False,
        timestamp="2026-07-11T00:00:00+00:00",
    )
    return BenchmarkReport(
        task_id="evidence_task",
        repo="owner/repo",
        question="Where is the evidence contract?",
        results=[result],
        baseline_tokens=100,
    )


def _write_task(tasks_dir: Path) -> None:
    tasks_dir.mkdir()
    (tasks_dir / "evidence_task.yaml").write_text(
        "\n".join(
            [
                "task_id: evidence_task",
                "repo: owner/repo",
                "commit: 0123456789abcdef",
                "question: Where is the evidence contract?",
                "expected_files:",
                "  - src/evidence.py",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_evidence(tmp_path: Path) -> tuple[Path, Path]:
    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir)
    output_dir = tmp_path / "evidence"
    prepare_evidence_directory(output_dir)
    report = _report()
    (output_dir / "evidence_task.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    manifest = build_evidence_manifest(
        [report],
        [_task()],
        [Strategy.ARCHEX_QUERY],
        BenchmarkRetrievalOptions(),
        source_sha="a" * 40,
        tasks_dir=tasks_dir,
        hardware_advisory="test hardware",
    )
    write_evidence_manifest(output_dir, manifest)
    return output_dir, tasks_dir


def test_evidence_manifest_binds_reports_tasks_and_configuration(tmp_path: Path) -> None:
    output_dir, tasks_dir = _write_evidence(tmp_path)

    manifest = validate_evidence_directory(
        output_dir,
        tasks_dir,
        expected_source_sha="a" * 40,
    )

    assert manifest.task_ids == ["evidence_task"]
    assert manifest.strategies == [Strategy.ARCHEX_QUERY]
    assert manifest.task_manifest_digest == task_manifest_digest(tasks_dir)
    assert set(manifest.report_hashes) == {"evidence_task"}
    assert manifest.retrieval_options == BenchmarkRetrievalOptions()


def test_evidence_validation_rejects_tampered_report(tmp_path: Path) -> None:
    output_dir, tasks_dir = _write_evidence(tmp_path)
    (output_dir / "evidence_task.json").write_text("{}", encoding="utf-8")

    with pytest.raises(BenchmarkEvidenceError, match="report hash mismatch"):
        validate_evidence_directory(output_dir, tasks_dir)


def test_evidence_validation_rejects_incomplete_report_coverage(tmp_path: Path) -> None:
    output_dir, tasks_dir = _write_evidence(tmp_path)
    (output_dir / "evidence_task.json").unlink()

    with pytest.raises(BenchmarkEvidenceError, match="file coverage mismatch"):
        validate_evidence_directory(output_dir, tasks_dir)


def test_evidence_validation_rejects_stale_task_manifest(tmp_path: Path) -> None:
    output_dir, tasks_dir = _write_evidence(tmp_path)
    task_path = tasks_dir / "evidence_task.yaml"
    task_path.write_text(
        task_path.read_text(encoding="utf-8") + "keywords:\n  - stale\n", encoding="utf-8"
    )

    with pytest.raises(BenchmarkEvidenceError, match="task-manifest digest mismatch"):
        validate_evidence_directory(output_dir, tasks_dir)


def test_evidence_validation_rejects_stale_source_revision(tmp_path: Path) -> None:
    output_dir, tasks_dir = _write_evidence(tmp_path)

    with pytest.raises(BenchmarkEvidenceError, match="source revision mismatch"):
        validate_evidence_directory(
            output_dir,
            tasks_dir,
            expected_source_sha="b" * 40,
        )


def test_prepare_evidence_directory_rejects_stale_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "evidence"
    output_dir.mkdir()
    (output_dir / "old.json").write_text("{}", encoding="utf-8")

    with pytest.raises(BenchmarkEvidenceError, match="not empty"):
        prepare_evidence_directory(output_dir)


def test_validate_command_accepts_complete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from click.testing import CliRunner

    from archex.cli import benchmark_cmd as benchmark_module

    def fixed_source_revision(_repo_root: Path) -> str:
        return "a" * 40

    output_dir, tasks_dir = _write_evidence(tmp_path)
    monkeypatch.setattr(benchmark_module, "source_revision", fixed_source_revision)

    result = CliRunner().invoke(
        benchmark_module.benchmark_cmd,
        [
            "validate",
            "--kind",
            "evidence",
            "--tasks-dir",
            str(tasks_dir),
            "--input",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Valid benchmark evidence: 1 task(s), 1 strategy/strategies." in result.output


def test_gate_command_requires_complete_manifest_backed_evidence(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from archex.cli.benchmark_cmd import benchmark_cmd

    output_dir, tasks_dir = _write_evidence(tmp_path)
    result = CliRunner().invoke(
        benchmark_cmd,
        [
            "gate",
            "--input",
            str(output_dir),
            "--tasks-dir",
            str(tasks_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Quality gate passed." in result.output


def test_gate_command_rejects_reports_without_manifest(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from archex.cli.benchmark_cmd import benchmark_cmd

    output_dir, tasks_dir = _write_evidence(tmp_path)
    (output_dir / "manifest.json").unlink()
    result = CliRunner().invoke(
        benchmark_cmd,
        [
            "gate",
            "--input",
            str(output_dir),
            "--tasks-dir",
            str(tasks_dir),
        ],
    )

    assert result.exit_code != 0
    assert "manifest not found" in result.output


def test_baseline_coverage_rejects_configuration_drift(tmp_path: Path) -> None:
    output_dir, tasks_dir = _write_evidence(tmp_path)
    current = validate_evidence_directory(output_dir, tasks_dir)
    baseline = current.model_copy(
        update={"retrieval_options": BenchmarkRetrievalOptions(chunker="cast")}
    )

    with pytest.raises(BenchmarkEvidenceError, match="retrieval configuration"):
        validate_baseline_coverage(current, baseline)


def test_copy_evidence_as_baseline_preserves_manifest_and_reports(tmp_path: Path) -> None:
    output_dir, tasks_dir = _write_evidence(tmp_path)
    baseline_dir = tmp_path / "baseline"

    copied = copy_evidence_as_baseline(output_dir, baseline_dir, tasks_dir)
    loaded = validate_evidence_directory(baseline_dir, tasks_dir)

    assert copied == loaded


def test_run_command_records_evidence_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from click.testing import CliRunner

    from archex.cli import benchmark_cmd as benchmark_module

    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir)
    output_dir = tmp_path / "evidence"
    task = _task()
    base_report = _report()
    report = base_report.model_copy(
        update={
            "results": [
                base_report.results[0].model_copy(update={"strategy": strategy})
                for strategy in (
                    Strategy.RAW_FILES,
                    Strategy.RAW_RIPGREP,
                    Strategy.ARCHEX_QUERY,
                )
            ]
        }
    )

    def no_preflight(
        _strategies: list[Strategy],
        _retrieval_options: BenchmarkRetrievalOptions,
    ) -> list[str]:
        return []

    def selected_tasks(
        _tasks_dir: Path,
        *,
        task_filter: str | None = None,
        self_only: bool = False,
    ) -> list[BenchmarkTask]:
        del task_filter, self_only
        return [task]

    def fixed_source_revision(_repo_root: Path) -> str:
        return "a" * 40

    def fake_run_all(
        tasks_dir: Path,
        output_dir: Path,
        strategies: list[Strategy] | None = None,
        task_filter: str | None = None,
        self_only: bool = False,
        progress: object | None = None,
        tasks: list[BenchmarkTask] | None = None,
        retrieval_options: BenchmarkRetrievalOptions | None = None,
    ) -> list[BenchmarkReport]:
        del tasks_dir, strategies, task_filter, self_only, progress, tasks, retrieval_options
        (output_dir / "evidence_task.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return [report]

    monkeypatch.setattr(benchmark_module, "warm_benchmark_models", no_preflight)
    monkeypatch.setattr(benchmark_module, "load_selected_tasks", selected_tasks)
    monkeypatch.setattr(benchmark_module, "run_all", fake_run_all)
    monkeypatch.setattr(benchmark_module, "source_revision", fixed_source_revision)

    result = CliRunner().invoke(
        benchmark_module.benchmark_cmd,
        ["run", "--tasks-dir", str(tasks_dir), "--output", str(output_dir), "--no-progress"],
    )

    assert result.exit_code == 0
    assert "Recorded benchmark evidence manifest" in result.output
    validate_evidence_directory(output_dir, tasks_dir, expected_source_sha="a" * 40)
