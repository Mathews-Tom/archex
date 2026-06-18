"""Tests for benchmark failure triage."""

from __future__ import annotations

import json

from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTask,
    Strategy,
    TaskCategory,
)
from archex.benchmark.triage import (
    format_triage_json,
    format_triage_markdown,
    triage_failures,
)


def _result(
    strategy: Strategy,
    *,
    recall: float,
    precision: float,
    f1_score: float,
    category: TaskCategory | None = None,
    seed_files: list[str] | None = None,
    expanded_files: list[str] | None = None,
    tokens_total: int = 100,
    token_efficiency: float = 0.5,
    savings_vs_raw: float = 25.0,
    expansion_eligible_seeds: int = 0,
    expansion_candidates_found: int = 0,
    expansion_zero_candidate_reason: str = "",
    expansion_reason_counts: dict[str, int] | None = None,
    expanded_file_reasons: dict[str, list[str]] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="task",
        strategy=strategy,
        tokens_total=tokens_total,
        token_efficiency=token_efficiency,
        tool_calls=1,
        files_accessed=1,
        recall=recall,
        precision=precision,
        f1_score=f1_score,
        mrr=recall,
        savings_vs_raw=savings_vs_raw,
        wall_time_ms=10.0,
        cached=False,
        timestamp="2026-01-01T00:00:00Z",
        seed_files=seed_files or [],
        expanded_files=expanded_files or [],
        expansion_ratio=0.5,
        expansion_eligible_seeds=expansion_eligible_seeds,
        expansion_candidates_found=expansion_candidates_found,
        expansion_zero_candidate_reason=expansion_zero_candidate_reason,
        expansion_reason_counts=expansion_reason_counts or {},
        expanded_file_reasons=expanded_file_reasons or {},
        category=category,
    )


def _report(
    task_id: str,
    result: BenchmarkResult,
    raw_read: BenchmarkResult | None = None,
) -> BenchmarkReport:
    result.task_id = task_id
    results = [result]
    if raw_read is not None:
        raw_read.task_id = task_id
        results.insert(0, raw_read)
    return BenchmarkReport(
        task_id=task_id,
        repo="owner/repo",
        question="How does task dispatch work?",
        results=results,
        baseline_tokens=100,
    )


def _task(task_id: str, category: TaskCategory = TaskCategory.EXTERNAL_LARGE) -> BenchmarkTask:
    return BenchmarkTask(
        task_id=task_id,
        repo="owner/repo",
        commit="abc123",
        question="How does task dispatch work?",
        expected_files=["src/task.py", "src/worker.py"],
        category=category,
    )


def test_triage_ranks_zero_recall_before_low_precision() -> None:
    zero = _result(
        Strategy.ARCHEX_QUERY,
        recall=0.0,
        precision=0.0,
        f1_score=0.0,
        category=TaskCategory.EXTERNAL_LARGE,
        seed_files=["src/noise.py"],
    )
    low_precision = _result(
        Strategy.ARCHEX_QUERY,
        recall=1.0,
        precision=0.2,
        f1_score=0.33,
        category=TaskCategory.SELF,
        seed_files=["src/task.py", "src/worker.py", "src/noise.py"],
    )
    reports = [_report("zero", zero), _report("low_precision", low_precision)]
    tasks = {"zero": _task("zero"), "low_precision": _task("low_precision", TaskCategory.SELF)}

    findings = triage_failures(reports, tasks)

    assert [finding.task_id for finding in findings] == ["zero", "low_precision"]
    assert findings[0].failure_bucket == "zero_recall"
    assert findings[0].missing_files == ["src/task.py", "src/worker.py"]
    assert findings[0].extra_files == ["src/noise.py"]


def test_triage_detects_raw_ripgrep_gap() -> None:
    archex = _result(
        Strategy.ARCHEX_QUERY,
        recall=0.4,
        precision=0.4,
        f1_score=0.4,
        category=TaskCategory.ARCHITECTURE_BROAD,
        seed_files=["src/task.py"],
    )
    raw_read = _result(
        Strategy.RAW_RIPGREP,
        recall=0.9,
        precision=0.1,
        f1_score=0.18,
    )
    findings = triage_failures([_report("gap", archex, raw_read)], {"gap": _task("gap")})

    assert len(findings) == 1
    assert findings[0].failure_bucket == "raw_ripgrep_gap"
    assert "raw_ripgrep_gap" in findings[0].failure_reasons
    assert findings[0].raw_read_recall == 0.9


def test_triage_uses_task_category_when_result_category_missing() -> None:
    result = _result(
        Strategy.ARCHEX_QUERY,
        recall=0.5,
        precision=0.2,
        f1_score=0.28,
        category=None,
        seed_files=["src/noise.py"],
    )
    findings = triage_failures(
        [_report("semantic", result)],
        {"semantic": _task("semantic", TaskCategory.FRAMEWORK_SEMANTIC)},
    )

    assert findings[0].category == "framework-semantic"
    assert findings[0].failure_bucket == "semantic_gap"


def test_triage_skips_passing_result() -> None:
    result = _result(
        Strategy.ARCHEX_QUERY,
        recall=1.0,
        precision=0.8,
        f1_score=0.88,
        seed_files=["src/task.py", "src/worker.py"],
    )

    assert triage_failures([_report("pass", result)], {"pass": _task("pass")}) == []


def test_format_triage_outputs_are_stable() -> None:
    result = _result(
        Strategy.ARCHEX_QUERY,
        recall=0.0,
        precision=0.0,
        f1_score=0.0,
        seed_files=[],
    )
    findings = triage_failures([_report("zero", result)], {"zero": _task("zero")})

    markdown = format_triage_markdown(findings)
    assert "# Benchmark Failure Triage" in markdown
    assert "`zero_recall`" in markdown
    assert "`src/task.py`" in markdown
    assert "Tokens Total" in markdown
    assert "Token Efficiency" in markdown
    assert "Savings vs Raw" in markdown
    assert "Expansion:" in markdown
    assert "raw_grepped" not in markdown

    payload = json.loads(format_triage_json(findings))
    assert payload[0]["task_id"] == "zero"
    assert payload[0]["failure_bucket"] == "zero_recall"
    assert payload[0]["metrics"]["tokens_total"] == 100
    assert payload[0]["metrics"]["token_efficiency"] == 0.5
    assert payload[0]["metrics"]["savings_vs_raw"] == 25.0
    assert payload[0]["expansion_diagnostics"]["eligible_seeds"] == 0
    assert "raw_grepped_metrics" not in payload[0]


def test_format_triage_json_serializes_expansion_diagnostics() -> None:
    result = _result(
        Strategy.ARCHEX_QUERY,
        recall=1.0,
        precision=0.2,
        f1_score=0.33,
        seed_files=["src/task.py"],
        expansion_reason_counts={"import_target": 2, "test_file": 1},
        expanded_file_reasons={"src/worker.py": ["import_target"]},
        expanded_files=["src/worker.py", "src/noise.py"],
        expansion_eligible_seeds=1,
        expansion_candidates_found=2,
        expansion_zero_candidate_reason="",
    )
    findings = triage_failures([_report("expanded", result)], {"expanded": _task("expanded")})

    payload = json.loads(format_triage_json(findings))

    assert payload[0]["task_id"] == "expanded"
    assert payload[0]["returned_files"] == ["src/task.py", "src/worker.py", "src/noise.py"]
    assert payload[0]["expansion_diagnostics"]["reason_counts"] == {
        "import_target": 2,
        "test_file": 1,
    }
    assert payload[0]["expansion_diagnostics"]["file_reasons"] == {
        "src/worker.py": ["import_target"]
    }
    assert payload[0]["extra_files"] == ["src/noise.py"]
    assert payload[0]["seed_files"] == ["src/task.py"]
    assert payload[0]["expanded_files"] == ["src/worker.py", "src/noise.py"]
    assert payload[0]["expansion_ratio"] == 0.5
    assert payload[0]["expansion_diagnostics"]["eligible_seeds"] == 1
    assert payload[0]["expansion_diagnostics"]["candidates_found"] == 2
