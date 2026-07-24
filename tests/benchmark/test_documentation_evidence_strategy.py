"""Tests for the M9 archex_query_documentation_evidence benchmark lane."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.strategies import (
    run_archex_query,
    run_archex_query_documentation_evidence,
)

if TYPE_CHECKING:
    from pathlib import Path


def _task(**overrides: object) -> BenchmarkTask:
    defaults: dict[str, object] = {
        "task_id": "test",
        "repo": "test/repo",
        "commit": "abc",
        "question": "How does the main module work?",
        "expected_files": ["main.py"],
        "token_budget": 4096,
    }
    defaults.update(overrides)
    return BenchmarkTask.model_validate(defaults)


class TestRunArchexQueryDocumentationEvidence:
    def test_strategy_and_metrics_shape(self, python_simple_repo: Path) -> None:
        result = run_archex_query_documentation_evidence(_task(), python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_DOCUMENTATION_EVIDENCE
        assert result.tool_calls == 1
        assert result.timing is not None
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0

    def test_matches_baseline_when_no_markdown_present(self, python_simple_repo: Path) -> None:
        # Documentation evidence is a pure side channel (M9): it populates
        # the store and receipt surfaces but is never read by search,
        # ranking, or candidate expansion, so its presence must not change
        # retrieval output -- regardless of whether the doc_link provider
        # actually finds any markdown to scan in the fixture repo.
        task = _task()
        baseline = run_archex_query(task, python_simple_repo)
        candidate = run_archex_query_documentation_evidence(task, python_simple_repo)

        assert candidate.recall == baseline.recall
        assert candidate.precision == baseline.precision
        assert candidate.mrr == baseline.mrr
        assert candidate.tokens_output == baseline.tokens_output
        assert candidate.required_file_recall == baseline.required_file_recall

    def test_matches_baseline_when_markdown_is_present(self, python_simple_repo: Path) -> None:
        (python_simple_repo / "README.md").write_text("See [main](main.py) for the entry point.\n")
        task = _task()
        baseline = run_archex_query(task, python_simple_repo)
        candidate = run_archex_query_documentation_evidence(task, python_simple_repo)

        assert candidate.recall == baseline.recall
        assert candidate.precision == baseline.precision
        assert candidate.mrr == baseline.mrr
        assert candidate.tokens_output == baseline.tokens_output
        assert candidate.required_file_recall == baseline.required_file_recall

    def test_default_archex_query_never_enables_documentation_providers(
        self, python_simple_repo: Path
    ) -> None:
        result = run_archex_query(_task(), python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY
