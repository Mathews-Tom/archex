"""Tests for the M3 language/repo-size/intent/family scorecard and raw artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from archex.benchmark.evidence import BenchmarkEvidenceManifest
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRetrievalOptions,
    BenchmarkTask,
    RepoSizeClass,
    Strategy,
    TaskCategory,
    TaskCompletionResult,
    TaskFamily,
)
from archex.benchmark.scorecard import (
    build_family_scorecard,
    build_intent_scorecard,
    build_language_scorecard,
    build_m3_scorecard_artifact,
    build_repo_size_scorecard,
    classify_repo_size,
    format_m3_scorecard_markdown,
    load_m3_scorecard_artifact,
    save_m3_scorecard_artifact,
)
from archex.cli.benchmark_cmd import benchmark_cmd


def _task(**overrides: object) -> BenchmarkTask:
    defaults: dict[str, object] = {
        "task_id": "t",
        "repo": "owner/repo",
        "commit": "v1.0.0",
        "question": "q",
        "expected_files": ["a.py"],
    }
    defaults.update(overrides)
    return BenchmarkTask.model_validate(defaults)


def _result(task_id: str, **overrides: object) -> BenchmarkResult:
    defaults: dict[str, object] = {
        "task_id": task_id,
        "strategy": Strategy.ARCHEX_QUERY,
        "tokens_total": 100,
        "tool_calls": 1,
        "files_accessed": 1,
        "recall": 1.0,
        "precision": 1.0,
        "savings_vs_raw": 0.0,
        "timestamp": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return BenchmarkResult.model_validate(defaults)


def _report(task_id: str, results: list[BenchmarkResult], **overrides: object) -> BenchmarkReport:
    defaults: dict[str, object] = {
        "task_id": task_id,
        "repo": "owner/repo",
        "question": "q",
        "results": results,
        "baseline_tokens": 1000,
    }
    defaults.update(overrides)
    return BenchmarkReport.model_validate(defaults)


class TestClassifyRepoSize:
    def test_small_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "small"
        repo.mkdir()
        (repo / "a.py").write_text("\n".join(f"line {i}" for i in range(50)))
        assert classify_repo_size(repo) == RepoSizeClass.SMALL

    def test_medium_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "medium"
        repo.mkdir()
        (repo / "a.py").write_text("\n".join(f"line {i}" for i in range(15_000)))
        assert classify_repo_size(repo) == RepoSizeClass.MEDIUM

    def test_large_repo(self, tmp_path: Path) -> None:
        repo = tmp_path / "large"
        repo.mkdir()
        (repo / "a.py").write_text("\n".join(f"line {i}" for i in range(150_000)))
        assert classify_repo_size(repo) == RepoSizeClass.LARGE

    def test_excludes_vendored_directories(self, tmp_path: Path) -> None:
        repo = tmp_path / "vendored"
        repo.mkdir()
        (repo / "a.py").write_text("x = 1\n")
        vendored = repo / "node_modules"
        vendored.mkdir()
        (vendored / "big.js").write_text("\n".join(f"line {i}" for i in range(200_000)))
        assert classify_repo_size(repo) == RepoSizeClass.SMALL

    def test_memoizes_by_resolved_path(self, tmp_path: Path) -> None:
        repo = tmp_path / "memo"
        repo.mkdir()
        (repo / "a.py").write_text("x = 1\n")
        first = classify_repo_size(repo)
        (repo / "b.py").write_text("\n".join(f"line {i}" for i in range(200_000)))
        second = classify_repo_size(repo)
        assert first == second == RepoSizeClass.SMALL


class TestLanguageScorecard:
    def test_multi_language_task_contributes_to_every_language(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1", languages=["python", "go"])}
        reports = [_report("t1", [_result("t1")])]
        rows = build_language_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert {row.value for row in rows} == {"python", "go"}
        assert all(row.task_count == 1 for row in rows)

    def test_task_without_languages_buckets_as_unspecified(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1", languages=None)}
        reports = [_report("t1", [_result("t1")])]
        rows = build_language_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert [row.value for row in rows] == ["unspecified"]


class TestIntentAndFamilyScorecard:
    def test_intent_defaults_to_unspecified(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1", category=None)}
        reports = [_report("t1", [_result("t1")])]
        rows = build_intent_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert [row.value for row in rows] == ["unspecified"]

    def test_intent_uses_task_category(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1", category=TaskCategory.EXTERNAL_LARGE)}
        reports = [_report("t1", [_result("t1")])]
        rows = build_intent_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert [row.value for row in rows] == ["external-large"]

    def test_family_uses_task_family(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1", family=TaskFamily.LOCALIZATION)}
        reports = [_report("t1", [_result("t1")])]
        rows = build_family_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert [row.value for row in rows] == ["localization"]


class TestRepoSizeScorecard:
    def test_unmeasured_when_result_lacks_size_class(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1")}
        reports = [_report("t1", [_result("t1", repo_size_class=None)])]
        rows = build_repo_size_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert [row.value for row in rows] == ["unmeasured"]

    def test_buckets_by_measured_size_class(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1")}
        reports = [_report("t1", [_result("t1", repo_size_class=RepoSizeClass.LARGE)])]
        rows = build_repo_size_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert [row.value for row in rows] == ["large"]


class TestAggregation:
    def test_zero_recall_count_and_rate(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1"), "t2": _task(task_id="t2")}
        reports = [
            _report("t1", [_result("t1", recall=0.0)]),
            _report("t2", [_result("t2", recall=1.0)]),
        ]
        rows = build_family_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        row = rows[0]
        assert row.task_count == 2
        assert row.zero_recall_count == 1
        assert row.zero_recall_rate == pytest.approx(0.5)  # pyright: ignore[reportUnknownMemberType]

    def test_duplicate_rate_mean(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1"), "t2": _task(task_id="t2")}
        reports = [
            _report("t1", [_result("t1", duplicate_rate=0.2)]),
            _report("t2", [_result("t2", duplicate_rate=0.6)]),
        ]
        rows = build_family_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert rows[0].mean_duplicate_rate == pytest.approx(0.4)  # pyright: ignore[reportUnknownMemberType]

    def test_warm_latency_percentiles_only_use_measured_warm_samples(self) -> None:
        tasks_by_id = {f"t{i}": _task(task_id=f"t{i}") for i in range(1, 4)}
        reports = [
            _report("t1", [_result("t1", cache_state="warm", warm_latency_ms=10.0)]),
            _report("t2", [_result("t2", cache_state="warm", warm_latency_ms=20.0)]),
            _report("t3", [_result("t3", cache_state="cold", warm_latency_ms=0.0)]),
        ]
        rows = build_family_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        row = rows[0]
        assert row.warm_p50_latency_ms == pytest.approx(15.0)  # pyright: ignore[reportUnknownMemberType]

    def test_cold_latency_uses_wall_time_for_cold_results(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1")}
        reports = [_report("t1", [_result("t1", cache_state="cold", wall_time_ms=42.0)])]
        rows = build_family_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert rows[0].cold_p50_latency_ms == pytest.approx(42.0)  # pyright: ignore[reportUnknownMemberType]

    def test_required_file_completeness_rate_from_task_completion(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1"), "t2": _task(task_id="t2")}
        reports = [
            _report("t1", [_result("t1", task_completion_result=TaskCompletionResult.PASS)]),
            _report("t2", [_result("t2", task_completion_result=TaskCompletionResult.FAIL)]),
        ]
        rows = build_family_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert rows[0].required_file_completeness_rate == pytest.approx(0.5)  # pyright: ignore[reportUnknownMemberType]

    def test_required_file_completeness_rate_prefers_bundle_only_success(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1")}
        reports = [
            _report(
                "t1",
                [
                    _result(
                        "t1",
                        task_completion_result=TaskCompletionResult.FAIL,
                        bundle_only_success=TaskCompletionResult.PASS,
                    )
                ],
            )
        ]
        rows = build_family_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert rows[0].required_file_completeness_rate == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]

    def test_required_file_completeness_rate_none_when_all_unknown(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1")}
        reports = [
            _report("t1", [_result("t1", task_completion_result=TaskCompletionResult.UNKNOWN)])
        ]
        rows = build_family_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert rows[0].required_file_completeness_rate is None

    def test_ignores_reports_missing_the_requested_strategy(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1")}
        reports = [_report("t1", [_result("t1", strategy=Strategy.RAW_FILES)])]
        rows = build_family_scorecard(reports, tasks_by_id, strategy=Strategy.ARCHEX_QUERY)
        assert rows == []

    def test_ignores_reports_missing_from_tasks_by_id(self) -> None:
        reports = [_report("unknown", [_result("unknown")])]
        rows = build_family_scorecard(reports, {}, strategy=Strategy.ARCHEX_QUERY)
        assert rows == []


def _manifest() -> BenchmarkEvidenceManifest:
    return BenchmarkEvidenceManifest(
        source_revision="a" * 40,
        archex_version="0.0.0",
        task_manifest_digest="b" * 64,
        task_ids=["t1"],
        strategies=[Strategy.ARCHEX_QUERY],
        retrieval_options=BenchmarkRetrievalOptions(),
        generated_at="2026-01-01T00:00:00Z",
        hardware_advisory="test",
        report_hashes={"t1": "c" * 64},
    )


class TestM3ScorecardArtifact:
    def test_build_and_round_trip(self, tmp_path: Path) -> None:
        tasks_by_id = {"t1": _task(task_id="t1", languages=["python"])}
        reports = [_report("t1", [_result("t1")])]
        artifact = build_m3_scorecard_artifact(
            reports, tasks_by_id, _manifest(), strategy=Strategy.ARCHEX_QUERY
        )
        assert artifact.artifact_version == 2
        assert len(artifact.language_scorecard) == 1
        assert len(artifact.family_scorecard) == 1

        path = tmp_path / "artifact.json"
        save_m3_scorecard_artifact(path, artifact)
        loaded = load_m3_scorecard_artifact(path)
        assert loaded == artifact
        assert json.loads(path.read_text())["artifact_version"] == 2

    def test_markdown_renders_every_dimension_and_handles_missing_data(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1")}
        reports = [_report("t1", [_result("t1")])]
        artifact = build_m3_scorecard_artifact(
            reports, tasks_by_id, _manifest(), strategy=Strategy.ARCHEX_QUERY
        )
        markdown = format_m3_scorecard_markdown(artifact)
        assert "## By Language" in markdown
        assert "## By Repository Size" in markdown
        assert "## By Query Intent" in markdown
        assert "## By Task Family" in markdown
        assert "n/a" in markdown  # warm latency unmeasured for this fixture

    def test_markdown_column_names_the_required_file_completeness_metric(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1")}
        reports = [_report("t1", [_result("t1")])]
        artifact = build_m3_scorecard_artifact(
            reports, tasks_by_id, _manifest(), strategy=Strategy.ARCHEX_QUERY
        )
        markdown = format_m3_scorecard_markdown(artifact)
        assert "| Required-File Completeness |" in markdown
        assert "Downstream Success" not in markdown

    def test_markdown_carries_the_completeness_definition_with_the_table(self) -> None:
        tasks_by_id = {"t1": _task(task_id="t1")}
        reports = [_report("t1", [_result("t1")])]
        artifact = build_m3_scorecard_artifact(
            reports, tasks_by_id, _manifest(), strategy=Strategy.ARCHEX_QUERY
        )
        markdown = format_m3_scorecard_markdown(artifact)
        # The column's meaning must travel with the rendered table, not live only in
        # the source docstring, or a published scorecard reasserts a task-outcome claim
        # the metric cannot support.
        assert "a function of required-file recall, with no model in" in markdown
        assert "`bundle_only_success`" in markdown


class TestScorecardCliCommand:
    def _patch_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        manifest: BenchmarkEvidenceManifest,
        reports: list[BenchmarkReport],
    ) -> None:
        def load_evidence(
            input_dir: Path,
            tasks_dir: Path,
        ) -> tuple[BenchmarkEvidenceManifest, list[BenchmarkReport]]:
            del input_dir, tasks_dir
            return manifest, reports

        monkeypatch.setattr("archex.cli.benchmark_cmd.load_evidence_reports", load_evidence)

    def test_markdown_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "t1.yaml").write_text(
            "task_id: t1\nrepo: owner/repo\ncommit: v1.0.0\nquestion: q\nexpected_files: [a.py]\n"
        )
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        self._patch_evidence(monkeypatch, _manifest(), [_report("t1", [_result("t1")])])

        result = CliRunner().invoke(
            benchmark_cmd,
            ["scorecard", "--input", str(results_dir), "--tasks-dir", str(tasks_dir)],
        )
        assert result.exit_code == 0
        assert "M3 External Quality Frontier Scorecard" in result.output
        assert "## By Task Family" in result.output

    def test_json_output_and_artifact_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "t1.yaml").write_text(
            "task_id: t1\nrepo: owner/repo\ncommit: v1.0.0\nquestion: q\nexpected_files: [a.py]\n"
        )
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        artifact_path = tmp_path / "artifact.json"
        self._patch_evidence(monkeypatch, _manifest(), [_report("t1", [_result("t1")])])

        result = CliRunner().invoke(
            benchmark_cmd,
            [
                "scorecard",
                "--input",
                str(results_dir),
                "--tasks-dir",
                str(tasks_dir),
                "--format",
                "json",
                "--output",
                str(artifact_path),
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["artifact_version"] == 2
        assert artifact_path.exists()
        assert json.loads(artifact_path.read_text())["strategy"] == "archex_query"

    def test_rejects_sealed_tasks_dir_without_flag(self, tmp_path: Path) -> None:
        sealed_dir = tmp_path / "sealed_tasks"
        sealed_dir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        result = CliRunner().invoke(
            benchmark_cmd,
            ["scorecard", "--input", str(results_dir), "--tasks-dir", str(sealed_dir)],
        )
        assert result.exit_code != 0
        assert "--allow-sealed-corpus" in str(result.output)
