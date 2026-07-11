"""Tests for baseline save/load/compare functionality."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from archex.benchmark.baseline import (
    Baseline,
    BaselineContractError,
    BaselineCoverageError,
    BaselineEntry,
    BaselineManifest,
    RankingSnapshotEntry,
    build_ranking_snapshot,
    compare_baseline,
    load_baseline,
    save_baseline,
    task_manifest_digest,
    validate_baseline_directory,
)
from archex.benchmark.models import BenchmarkReport, BenchmarkResult, Strategy


def _make_report(
    task_id: str = "test_task",
    strategy: Strategy = Strategy.ARCHEX_QUERY,
    recall: float = 0.8,
    precision: float = 0.6,
    f1_score: float = 0.685,
    mrr: float = 0.5,
    ndcg: float = 0.7,
    map_score: float = 0.6,
) -> BenchmarkReport:
    result = BenchmarkResult(
        task_id=task_id,
        strategy=strategy,
        tokens_total=1000,
        tool_calls=1,
        files_accessed=3,
        recall=recall,
        precision=precision,
        f1_score=f1_score,
        mrr=mrr,
        ndcg=ndcg,
        map_score=map_score,
        savings_vs_raw=50.0,
        wall_time_ms=100.0,
        cached=False,
        timestamp="2026-01-01T00:00:00Z",
    )
    return BenchmarkReport(
        task_id=task_id,
        repo="test/repo",
        question="test question",
        results=[result],
        baseline_tokens=2000,
    )


def _write_task_manifest(tasks_dir: Path, task_id: str = "test_task") -> None:
    tasks_dir.mkdir()
    (tasks_dir / f"{task_id}.yaml").write_text(
        f"""\
task_id: {task_id}
repo: owner/repo
commit: abc123
question: "How does {task_id} work?"
expected_files:
  - src/{task_id}.py
""",
        encoding="utf-8",
    )


def _make_manifest(tasks_dir: Path, *, strategies: list[str]) -> BaselineManifest:
    return BaselineManifest(
        manifest_version=1,
        archex_version="0.19.2",
        source_revision="a" * 40,
        task_manifest_digest=task_manifest_digest(tasks_dir),
        task_ids=["test_task"],
        strategies=strategies,
        retrieval_options={"vector": "false"},
        generated_at="2026-07-11T00:00:00Z",
    )


def test_save_load_baseline_roundtrip() -> None:
    reports = [_make_report()]
    baseline = save_baseline(reports)
    assert len(baseline.entries) == 1
    assert baseline.entries[0].task_id == "test_task"
    assert baseline.entries[0].ndcg == 0.7
    assert baseline.entries[0].map_score == 0.6
    assert baseline.entries[0].token_efficiency == 0.0

    # Roundtrip through JSON
    data = baseline.model_dump()
    loaded = load_baseline(data)
    assert len(loaded.entries) == 1
    assert loaded.entries[0].recall == 0.8
    assert loaded.entries[0].ndcg == 0.7
    assert loaded.entries[0].map_score == 0.6
    assert loaded.entries[0].token_efficiency == 0.0


def test_baseline_directory_accepts_exact_task_strategy_coverage(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    _write_task_manifest(tasks_dir)

    manifest = _make_manifest(tasks_dir, strategies=[Strategy.ARCHEX_QUERY.value])
    validate_baseline_directory(manifest, [_make_report()], tasks_dir=tasks_dir)


def test_baseline_directory_rejects_missing_task_strategy_coverage(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    _write_task_manifest(tasks_dir)
    manifest = _make_manifest(
        tasks_dir,
        strategies=[Strategy.ARCHEX_QUERY.value, Strategy.RAW_FILES.value],
    )

    with pytest.raises(BaselineCoverageError, match="missing=test_task/raw_files"):
        validate_baseline_directory(manifest, [_make_report()], tasks_dir=tasks_dir)


def test_baseline_directory_rejects_unexpected_task_strategy_coverage(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    _write_task_manifest(tasks_dir)
    manifest = _make_manifest(tasks_dir, strategies=[Strategy.ARCHEX_QUERY.value])
    report = _make_report()
    report.results.append(_make_report(strategy=Strategy.RAW_FILES).results[0])

    with pytest.raises(BaselineCoverageError, match="unexpected=test_task/raw_files"):
        validate_baseline_directory(manifest, [report], tasks_dir=tasks_dir)


def test_baseline_directory_rejects_wrong_task_manifest_digest(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    _write_task_manifest(tasks_dir)
    manifest = _make_manifest(
        tasks_dir,
        strategies=[Strategy.ARCHEX_QUERY.value],
    ).model_copy(update={"task_manifest_digest": "b" * 64})

    with pytest.raises(BaselineContractError, match="digest does not match"):
        validate_baseline_directory(manifest, [_make_report()], tasks_dir=tasks_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("archex_version", ""),
        ("source_revision", ""),
    ],
)
def test_baseline_directory_rejects_empty_identity_fields(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    tasks_dir = tmp_path / "tasks"
    _write_task_manifest(tasks_dir)
    manifest = _make_manifest(
        tasks_dir,
        strategies=[Strategy.ARCHEX_QUERY.value],
    ).model_copy(update={field: value})

    with pytest.raises(BaselineContractError):
        validate_baseline_directory(manifest, [_make_report()], tasks_dir=tasks_dir)


def test_compare_baseline_detects_regression() -> None:
    baseline = Baseline(
        entries=[
            BaselineEntry(
                task_id="test_task",
                strategy="archex_query",
                recall=0.9,
                precision=0.8,
                f1_score=0.85,
                mrr=0.7,
                ndcg=0.8,
                map_score=0.7,
            )
        ]
    )
    # Current results are worse
    reports = [
        _make_report(
            recall=0.5,
            precision=0.3,
            f1_score=0.37,
            mrr=0.2,
            ndcg=0.3,
            map_score=0.2,
        )
    ]
    comparisons = compare_baseline(reports, baseline)
    regressions = [c for c in comparisons if c.regression]
    assert len(regressions) > 0
    regressed_metrics = {c.metric for c in regressions}
    assert "recall" in regressed_metrics
    assert "ndcg" in regressed_metrics
    assert "map_score" in regressed_metrics


def test_compare_baseline_no_regression() -> None:
    baseline = Baseline(
        entries=[
            BaselineEntry(
                task_id="test_task",
                strategy="archex_query",
                recall=0.8,
                precision=0.6,
                f1_score=0.685,
                mrr=0.5,
                ndcg=0.7,
                map_score=0.6,
            )
        ]
    )
    # Current results are the same or better
    reports = [
        _make_report(
            recall=0.85,
            precision=0.65,
            f1_score=0.72,
            mrr=0.55,
            ndcg=0.75,
            map_score=0.65,
        )
    ]
    comparisons = compare_baseline(reports, baseline)
    regressions = [c for c in comparisons if c.regression]
    assert len(regressions) == 0


def test_compare_baseline_rejects_missing_or_unexpected_coverage() -> None:
    baseline = Baseline(entries=[])

    with pytest.raises(BaselineCoverageError, match="unexpected=test_task/archex_query"):
        compare_baseline([_make_report()], baseline)


def test_compare_baseline_excludes_diagnostic_strategies() -> None:
    baseline = Baseline(
        entries=[
            BaselineEntry(
                task_id="test_task",
                strategy="raw_ripgrep",
                recall=1.0,
                precision=1.0,
                f1_score=1.0,
                mrr=1.0,
            ),
            BaselineEntry(
                task_id="test_task",
                strategy="archex_query",
                recall=0.8,
                precision=0.6,
                f1_score=0.685,
                mrr=0.5,
            ),
        ]
    )
    reports = [
        _make_report(strategy=Strategy.RAW_RIPGREP, recall=0.0),
        _make_report(strategy=Strategy.ARCHEX_QUERY, recall=0.85),
    ]

    comparisons = compare_baseline(
        reports,
        baseline,
        excluded_strategies={Strategy.RAW_RIPGREP.value},
    )

    assert {comparison.strategy for comparison in comparisons} == {Strategy.ARCHEX_QUERY.value}
    assert [comparison for comparison in comparisons if comparison.regression] == []


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)


def test_baseline_ranking_field_defaults_empty() -> None:
    assert Baseline().ranking == []
    baseline_with_entries = Baseline(
        entries=[
            BaselineEntry(
                task_id="test_task",
                strategy="archex_query",
                recall=1.0,
                precision=1.0,
                f1_score=1.0,
                mrr=1.0,
            )
        ]
    )
    assert baseline_with_entries.ranking == []


def test_baseline_ranking_roundtrips_through_json() -> None:
    baseline = Baseline(
        ranking=[
            RankingSnapshotEntry(file_path="src/a.py", centrality=0.4, symbol_count=12),
            RankingSnapshotEntry(file_path="src/b.py", centrality=0.1, symbol_count=3),
        ]
    )
    loaded = load_baseline(baseline.model_dump())
    assert [(e.file_path, e.centrality, e.symbol_count) for e in loaded.ranking] == [
        ("src/a.py", 0.4, 12),
        ("src/b.py", 0.1, 3),
    ]


def test_build_ranking_snapshot_reflects_indexed_files(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "target_pkg").mkdir()
    (tmp_path / "target_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "target_pkg" / "helper.py").write_text(
        "def one():\n    return 1\n\n\ndef two():\n    return 2\n\n\ndef three():\n    return 3\n",
        encoding="utf-8",
    )
    (tmp_path / "importer.py").write_text(
        "import target_pkg.helper\n\n\ndef use_helper():\n    return target_pkg.helper.one()\n",
        encoding="utf-8",
    )
    (tmp_path / "lonely.py").write_text(
        "def solo_a():\n    return 1\n\n\ndef solo_b():\n    return 2\n",
        encoding="utf-8",
    )

    snapshot = build_ranking_snapshot(tmp_path)
    by_path = {entry.file_path: entry for entry in snapshot}

    assert set(by_path) == {"target_pkg/helper.py", "importer.py", "lonely.py"}
    # helper.py has 3 top-level defs and no import statements, so its symbol
    # count matches the def count exactly (imports add their own chunk).
    assert by_path["target_pkg/helper.py"].symbol_count == 3
    assert by_path["lonely.py"].symbol_count == 2
    # helper.py is imported by importer.py, so PageRank credits it with
    # non-zero centrality; a file with no incoming or outgoing edges (lonely.py)
    # gets centrality 0.0.
    assert by_path["target_pkg/helper.py"].centrality > 0.0
    assert by_path["lonely.py"].centrality == 0.0
