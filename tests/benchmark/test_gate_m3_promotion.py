"""Tests for the M3 promotion-gate dimensions: zero-recall, language-family, fixed-agent."""

from __future__ import annotations

from archex.benchmark.gate import (
    check_fixed_agent_non_regression,
    check_language_family_non_regression,
    check_zero_recall_non_regression,
)
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTask,
    Strategy,
    TaskCompletionResult,
)

_CANDIDATE = Strategy.ARCHEX_QUERY_CONTEXT_CANDIDATE
_CONTROL = Strategy.ARCHEX_QUERY


def _base_result(**overrides: object) -> BenchmarkResult:
    defaults: dict[str, object] = {
        "task_id": "t",
        "strategy": _CONTROL,
        "tokens_total": 1000,
        "tool_calls": 1,
        "files_accessed": 3,
        "recall": 0.8,
        "precision": 0.5,
        "savings_vs_raw": 50.0,
        "timestamp": "2026-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return BenchmarkResult.model_validate(defaults)


def _paired_report(
    task_id: str,
    control_overrides: dict[str, object],
    candidate_overrides: dict[str, object],
) -> BenchmarkReport:
    control = _base_result(task_id=task_id, strategy=_CONTROL, **control_overrides)
    candidate = _base_result(task_id=task_id, strategy=_CANDIDATE, **candidate_overrides)
    return BenchmarkReport(
        task_id=task_id,
        repo="test/repo",
        question="q",
        results=[control, candidate],
        baseline_tokens=2000,
    )


class TestCheckZeroRecallNonRegression:
    def test_flags_candidate_going_zero_when_control_was_not(self) -> None:
        report = _paired_report("t1", {"recall": 0.8}, {"recall": 0.0})
        violations = check_zero_recall_non_regression(
            [report], candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert [(v.task_id, v.metric) for v in violations] == [("t1", "zero_recall_regression")]

    def test_allows_both_zero_recall(self) -> None:
        report = _paired_report("t1", {"recall": 0.0}, {"recall": 0.0})
        violations = check_zero_recall_non_regression(
            [report], candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert violations == []

    def test_allows_candidate_recovering_from_control_zero_recall(self) -> None:
        report = _paired_report("t1", {"recall": 0.0}, {"recall": 0.6})
        violations = check_zero_recall_non_regression(
            [report], candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert violations == []

    def test_ignores_reports_missing_either_strategy(self) -> None:
        result = _base_result(task_id="t1", strategy=_CONTROL, recall=0.8)
        report = BenchmarkReport(
            task_id="t1", repo="test/repo", question="q", results=[result], baseline_tokens=2000
        )
        violations = check_zero_recall_non_regression(
            [report], candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert violations == []


def _task(task_id: str, languages: list[str] | None) -> BenchmarkTask:
    return BenchmarkTask.model_validate(
        {
            "task_id": task_id,
            "repo": "owner/repo",
            "commit": "v1.0.0",
            "question": "q",
            "expected_files": ["a.py"],
            "languages": languages,
        }
    )


class TestCheckLanguageFamilyNonRegression:
    def test_flags_a_single_regressed_language_behind_a_flat_aggregate(self) -> None:
        # python regresses (0.9 -> 0.4) while go improves (0.5 -> 1.0), so the
        # cross-family mean stays flat at 0.7 -- a naive aggregate check
        # would see no regression, but the python-only loss must still fire.
        tasks_by_id = {
            "py_task": _task("py_task", ["python"]),
            "go_task": _task("go_task", ["go"]),
        }
        reports = [
            _paired_report("py_task", {"recall": 0.9}, {"recall": 0.4}),
            _paired_report("go_task", {"recall": 0.5}, {"recall": 1.0}),
        ]
        violations = check_language_family_non_regression(
            reports, tasks_by_id, candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert [(v.task_id, v.metric) for v in violations] == [
            ("language:python", "language_family_recall")
        ]

    def test_no_violation_when_every_language_holds_or_improves(self) -> None:
        tasks_by_id = {
            "py_task": _task("py_task", ["python"]),
            "go_task": _task("go_task", ["go"]),
        }
        reports = [
            _paired_report("py_task", {"recall": 0.5}, {"recall": 0.6}),
            _paired_report("go_task", {"recall": 0.5}, {"recall": 1.0}),
        ]
        violations = check_language_family_non_regression(
            reports, tasks_by_id, candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert violations == []

    def test_multi_language_task_contributes_to_every_language(self) -> None:
        tasks_by_id = {"t1": _task("t1", ["python", "go"])}
        reports = [_paired_report("t1", {"recall": 0.8}, {"recall": 0.2})]
        violations = check_language_family_non_regression(
            reports, tasks_by_id, candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert {v.task_id for v in violations} == {"language:python", "language:go"}

    def test_tasks_without_languages_are_excluded(self) -> None:
        tasks_by_id = {"t1": _task("t1", None)}
        reports = [_paired_report("t1", {"recall": 0.8}, {"recall": 0.0})]
        violations = check_language_family_non_regression(
            reports, tasks_by_id, candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert violations == []

    def test_tolerance_permits_small_regressions(self) -> None:
        tasks_by_id = {"t1": _task("t1", ["python"])}
        reports = [_paired_report("t1", {"recall": 0.80}, {"recall": 0.78})]
        violations = check_language_family_non_regression(
            reports,
            tasks_by_id,
            candidate_strategy=_CANDIDATE,
            control_strategy=_CONTROL,
            tolerance=0.05,
        )
        assert violations == []


class TestCheckFixedAgentNonRegression:
    def test_flags_control_pass_becoming_candidate_fail(self) -> None:
        report = _paired_report(
            "t1",
            {"task_completion_result": TaskCompletionResult.PASS},
            {"task_completion_result": TaskCompletionResult.FAIL},
        )
        violations = check_fixed_agent_non_regression(
            [report], candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert [(v.task_id, v.metric) for v in violations] == [
            ("t1", "fixed_agent_success_regression")
        ]

    def test_allows_candidate_improving_on_control_fail(self) -> None:
        report = _paired_report(
            "t1",
            {"task_completion_result": TaskCompletionResult.FAIL},
            {"task_completion_result": TaskCompletionResult.PASS},
        )
        violations = check_fixed_agent_non_regression(
            [report], candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert violations == []

    def test_prefers_bundle_only_success_over_task_completion_result(self) -> None:
        report = _paired_report(
            "t1",
            {"task_completion_result": TaskCompletionResult.PASS},
            {
                "task_completion_result": TaskCompletionResult.PASS,
                "bundle_only_success": TaskCompletionResult.FAIL,
            },
        )
        violations = check_fixed_agent_non_regression(
            [report], candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert [(v.task_id, v.metric) for v in violations] == [
            ("t1", "fixed_agent_success_regression")
        ]

    def test_skips_tasks_with_unknown_outcome(self) -> None:
        report = _paired_report(
            "t1",
            {"task_completion_result": TaskCompletionResult.UNKNOWN},
            {"task_completion_result": TaskCompletionResult.FAIL},
        )
        violations = check_fixed_agent_non_regression(
            [report], candidate_strategy=_CANDIDATE, control_strategy=_CONTROL
        )
        assert violations == []
