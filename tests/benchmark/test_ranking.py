"""Tests for benchmark file-ranking and result-set-noise evidence reading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archex.benchmark.models import BenchmarkReport, BenchmarkResult, Strategy, TaskFamily
from archex.benchmark.ranking import (
    BELOW_F1_FLOOR,
    BELOW_MRR_FLOOR,
    BELOW_PRECISION_FLOOR,
    BELOW_RECALL_FLOOR,
    BROAD_RESULT_SET,
    RANK_BELOW_FIRST,
    read_rank_noise_observations,
)

_M03_TASK_MANIFEST_DIGEST = "0ae42fb18f96678b5e79591be8372d83fbeed646c4c08e7cc0f118eba4c2bd09"


_GATE = {"min_recall": 0.60, "min_precision": 0.20, "min_f1": 0.30, "min_mrr": 0.55}


def _result(
    task_id: str,
    strategy: Strategy,
    *,
    recall: float,
    precision: float,
    f1_score: float,
    mrr: float,
    required_file_recall: float,
    result_files: list[str],
    required_files_present: list[str],
    required_files_missing: list[str],
    family: TaskFamily = TaskFamily.COMPREHENSION,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        strategy=strategy,
        tokens_total=100,
        tool_calls=1,
        files_accessed=len(result_files),
        recall=recall,
        precision=precision,
        f1_score=f1_score,
        mrr=mrr,
        savings_vs_raw=0.0,
        wall_time_ms=10.0,
        cached=True,
        timestamp="2026-07-12T00:00:00+00:00",
        result_files=result_files,
        required_file_recall=required_file_recall,
        required_files_present=required_files_present,
        required_files_missing=required_files_missing,
        family=family,
    )


def _report(task_id: str, result: BenchmarkResult) -> BenchmarkReport:
    return BenchmarkReport(
        task_id=task_id,
        repo="owner/repo",
        question="Where is the required file?",
        results=[result],
        baseline_tokens=100,
    )


def test_classifies_rank_below_first_when_required_file_present_but_not_first() -> None:
    result = _result(
        "rank_task",
        Strategy.ARCHEX_QUERY,
        recall=1.0,
        precision=0.6,
        f1_score=0.75,
        mrr=0.5,
        required_file_recall=1.0,
        result_files=["a.py", "required.py", "c.py"],
        required_files_present=["required.py"],
        required_files_missing=[],
    )

    rows = read_rank_noise_observations(
        [_report("rank_task", result)], Strategy.ARCHEX_QUERY, **_GATE
    )

    assert len(rows) == 1
    row = rows[0]
    assert RANK_BELOW_FIRST in row.failure_classes
    assert BROAD_RESULT_SET not in row.failure_classes
    assert BELOW_MRR_FLOOR in row.failure_classes
    assert BELOW_RECALL_FLOOR not in row.failure_classes
    assert row.result_file_count == 3
    assert row.required_file_count == 1


def test_classifies_broad_result_set_when_precision_fails_despite_full_recall() -> None:
    result = _result(
        "noise_task",
        Strategy.ARCHEX_QUERY,
        recall=1.0,
        precision=0.05,
        f1_score=0.10,
        mrr=1.0,
        required_file_recall=1.0,
        result_files=[f"n{i}.py" for i in range(20)] + ["required.py"],
        required_files_present=["required.py"],
        required_files_missing=[],
    )

    rows = read_rank_noise_observations(
        [_report("noise_task", result)], Strategy.ARCHEX_QUERY, **_GATE
    )

    row = rows[0]
    assert BROAD_RESULT_SET in row.failure_classes
    assert RANK_BELOW_FIRST not in row.failure_classes
    assert BELOW_PRECISION_FLOOR in row.failure_classes
    assert BELOW_F1_FLOOR in row.failure_classes
    assert row.result_file_count == 21
    assert row.required_file_count == 1


def test_classifies_neither_class_for_a_clean_pass() -> None:
    result = _result(
        "clean_task",
        Strategy.ARCHEX_QUERY,
        recall=1.0,
        precision=1.0,
        f1_score=1.0,
        mrr=1.0,
        required_file_recall=1.0,
        result_files=["required.py"],
        required_files_present=["required.py"],
        required_files_missing=[],
    )

    rows = read_rank_noise_observations(
        [_report("clean_task", result)], Strategy.ARCHEX_QUERY, **_GATE
    )

    assert rows[0].failure_classes == ()


def test_classifies_neither_rank_class_for_a_pure_coverage_miss() -> None:
    # required_file_recall below 1.0 -- this is a coverage problem (M0.2's
    # scope), not a ranking/noise problem (M0.3's scope), and must not be
    # mislabeled either way.
    result = _result(
        "coverage_miss_task",
        Strategy.ARCHEX_QUERY,
        recall=0.0,
        precision=0.0,
        f1_score=0.0,
        mrr=0.0,
        required_file_recall=0.0,
        result_files=["unrelated.py"],
        required_files_present=[],
        required_files_missing=["required.py"],
    )

    rows = read_rank_noise_observations(
        [_report("coverage_miss_task", result)], Strategy.ARCHEX_QUERY, **_GATE
    )

    row = rows[0]
    assert RANK_BELOW_FIRST not in row.failure_classes
    assert BROAD_RESULT_SET not in row.failure_classes
    assert BELOW_RECALL_FLOOR in row.failure_classes


def test_rejects_report_without_requested_strategy() -> None:
    result = _result(
        "task",
        Strategy.RAW_FILES,
        recall=1.0,
        precision=1.0,
        f1_score=1.0,
        mrr=1.0,
        required_file_recall=1.0,
        result_files=["a.py"],
        required_files_present=["a.py"],
        required_files_missing=[],
    )
    with pytest.raises(ValueError, match="Expected exactly one archex_query result"):
        read_rank_noise_observations([_report("task", result)], Strategy.ARCHEX_QUERY, **_GATE)


def test_rejects_duplicate_report_task_ids() -> None:
    result = _result(
        "dup_task",
        Strategy.ARCHEX_QUERY,
        recall=1.0,
        precision=1.0,
        f1_score=1.0,
        mrr=1.0,
        required_file_recall=1.0,
        result_files=["a.py"],
        required_files_present=["a.py"],
        required_files_missing=[],
    )
    reports = [_report("dup_task", result), _report("dup_task", result)]

    with pytest.raises(ValueError, match="Duplicate benchmark report task ID"):
        read_rank_noise_observations(reports, Strategy.ARCHEX_QUERY, **_GATE)


def test_control_ranking_artifact_characterizes_seventeen_rank_noise_tasks() -> None:
    root = Path(__file__).parents[2]
    artifact = root / "benchmarks" / "evidence" / "m0.3-control-ranking.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["task_manifest_digest"] == _M03_TASK_MANIFEST_DIGEST
    assert payload["gate_thresholds"] == _GATE

    by_strategy = {block["strategy"]: block for block in payload["strategies"]}
    assert set(by_strategy) == {
        Strategy.ARCHEX_QUERY.value,
        Strategy.ARCHEX_QUERY_COVERAGE_CANDIDATE.value,
    }

    control = by_strategy[Strategy.ARCHEX_QUERY.value]
    assert len(control["tasks"]) == 17
    assert control["rank_below_first_count"] + control["broad_result_set_count"] >= 17
    for observation in control["tasks"]:
        classes = set(observation["failure_classes"])
        assert classes & {RANK_BELOW_FIRST, BROAD_RESULT_SET}
        if RANK_BELOW_FIRST in classes:
            assert observation["required_file_recall"] >= 1.0
        if BROAD_RESULT_SET in classes:
            assert observation["required_file_recall"] >= _GATE["min_recall"]
            assert observation["precision"] < _GATE["min_precision"]

    # The M0.2 coverage candidate is M0.3's declared input: it recovers
    # required-file coverage but, unmodified, makes rank/noise worse (more
    # flagged tasks) by always admitting a flat 32-seed/24-neighbor budget
    # regardless of whether the base query already found the file. This is
    # exactly the defect M0.3 must repair without discarding M0.2's recall.
    candidate = by_strategy[Strategy.ARCHEX_QUERY_COVERAGE_CANDIDATE.value]
    assert len(candidate["tasks"]) > len(control["tasks"])
