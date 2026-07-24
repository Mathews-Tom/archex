"""Tests for the M6 archex_query_semantic benchmark lane."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.benchmark.models import BenchmarkTask, Strategy
from archex.benchmark.strategies import run_archex_query, run_archex_query_semantic

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


class TestRunArchexQuerySemantic:
    def test_strategy_and_metrics_shape(self, python_simple_repo: Path) -> None:
        result = run_archex_query_semantic(_task(), python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_SEMANTIC
        assert result.tool_calls == 1
        assert result.timing is not None
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.precision <= 1.0

    def test_matches_baseline_when_no_scip_index_present(self, python_simple_repo: Path) -> None:
        # No index.scip at the repo root -> ScipEvidenceProvider reports
        # UNAVAILABLE and zero edges are added; retrieval must be identical
        # to the unmodified archex_query baseline.
        task = _task()
        baseline = run_archex_query(task, python_simple_repo)
        candidate = run_archex_query_semantic(task, python_simple_repo)

        assert candidate.recall == baseline.recall
        assert candidate.precision == baseline.precision
        assert candidate.mrr == baseline.mrr
        assert candidate.tokens_output == baseline.tokens_output
        assert candidate.required_file_recall == baseline.required_file_recall

    def test_runs_successfully_with_a_real_scip_index_present(
        self, python_simple_repo: Path
    ) -> None:
        from archex.integrations.semantic import scip_pb2

        index = scip_pb2.Index()
        index.metadata.tool_info.name = "scip-python"
        index.metadata.tool_info.version = "0.5.0"

        main_doc = index.documents.add()
        main_doc.relative_path = "main.py"
        main_doc.language = "python"
        definition = main_doc.occurrences.add()
        definition.symbol = "scip-python python . . main/entry()."
        definition.symbol_roles = scip_pb2.SymbolRole.Definition
        definition.single_line_range.line = 0

        models_doc = index.documents.add()
        models_doc.relative_path = "models.py"
        models_doc.language = "python"
        usage = models_doc.occurrences.add()
        usage.symbol = "scip-python python . . main/entry()."
        usage.range.extend([3, 0, 4])

        (python_simple_repo / "index.scip").write_bytes(index.SerializeToString())

        result = run_archex_query_semantic(_task(), python_simple_repo)
        assert result.strategy == Strategy.ARCHEX_QUERY_SEMANTIC
        assert 0.0 <= result.recall <= 1.0
