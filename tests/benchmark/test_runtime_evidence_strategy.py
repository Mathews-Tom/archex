"""Tests for the M7 archex_query_runtime_evidence benchmark lane."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.strategies import run_archex_query, run_archex_query_runtime_evidence

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


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


class TestRunArchexQueryRuntimeEvidence:
    def test_strategy_and_metrics_shape(self, python_simple_repo: Path) -> None:
        result = run_archex_query_runtime_evidence(_task(), python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_RUNTIME_EVIDENCE
        assert result.tool_calls == 1
        assert result.timing is not None
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0

    def test_matches_baseline_when_no_evidence_present(self, python_simple_repo: Path) -> None:
        # No .archex/runtime-evidence/ directory -> both providers report
        # UNAVAILABLE and no evidence is collected; retrieval must be
        # identical to the unmodified archex_query baseline.
        task = _task()
        baseline = run_archex_query(task, python_simple_repo)
        candidate = run_archex_query_runtime_evidence(task, python_simple_repo)

        assert candidate.recall == baseline.recall
        assert candidate.precision == baseline.precision
        assert candidate.mrr == baseline.mrr
        assert candidate.tokens_output == baseline.tokens_output
        assert candidate.required_file_recall == baseline.required_file_recall

    def test_matches_baseline_with_real_evidence_present(self, python_simple_repo: Path) -> None:
        # Runtime/coverage evidence is read-only provenance (M7): it is
        # collected and persisted but never consulted by search/ranking, so
        # its presence must not change retrieval output either.
        head = _git_head(python_simple_repo)
        coverage_dir = python_simple_repo / ".archex" / "runtime-evidence" / "coverage"
        coverage_dir.mkdir(parents=True)
        (coverage_dir / "manifest.json").write_text(
            json.dumps({"revision": head, "tool": "coverage.py"})
        )
        (coverage_dir / "coverage.xml").write_text(
            '<?xml version="1.0" ?>\n'
            '<coverage line-rate="1.0"><packages><package><classes>\n'
            '<class filename="main.py" line-rate="1.0">'
            '<lines><line number="1" hits="1"/></lines></class>\n'
            "</classes></package></packages></coverage>\n"
        )

        task = _task()
        baseline = run_archex_query(task, python_simple_repo)
        candidate = run_archex_query_runtime_evidence(task, python_simple_repo)

        assert candidate.recall == baseline.recall
        assert candidate.precision == baseline.precision
        assert candidate.mrr == baseline.mrr
        assert candidate.tokens_output == baseline.tokens_output

    def test_default_archex_query_never_enables_runtime_providers(
        self, python_simple_repo: Path
    ) -> None:
        result = run_archex_query(_task(), python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY
