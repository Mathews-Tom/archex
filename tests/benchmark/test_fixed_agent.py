"""Tests for deterministic fixed-agent trajectory accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from archex.benchmark.fixed_agent import (
    FIXED_AGENT_MAX_SEARCH_TURNS,
    compute_fixed_agent_search_turns,
)
from archex.benchmark.models import BenchmarkTask
from archex.benchmark.strategies import run_archex_query

if TYPE_CHECKING:
    from pathlib import Path


class TestComputeFixedAgentSearchTurns:
    def test_zero_missing_files_costs_zero_turns(self) -> None:
        assert compute_fixed_agent_search_turns([]) == 0

    def test_under_budget_counts_one_turn_per_file(self) -> None:
        assert compute_fixed_agent_search_turns(["a.py"]) == 1
        assert compute_fixed_agent_search_turns(["a.py", "b.py"]) == 2

    def test_caps_at_default_budget(self) -> None:
        missing = [f"f{i}.py" for i in range(10)]
        assert compute_fixed_agent_search_turns(missing) == FIXED_AGENT_MAX_SEARCH_TURNS

    def test_respects_custom_budget(self) -> None:
        missing = ["a.py", "b.py", "c.py"]
        assert compute_fixed_agent_search_turns(missing, max_search_turns=1) == 1
        assert compute_fixed_agent_search_turns(missing, max_search_turns=10) == 3


class TestFixedAgentTrajectoryWiring:
    """End-to-end proof that every archex_query-family result carries the trajectory signal."""

    def test_default_strategy_populates_search_turns_alongside_read_turns(
        self, python_simple_repo: Path
    ) -> None:
        task = BenchmarkTask.model_validate(
            {
                "task_id": "test",
                "repo": "test/repo",
                "commit": "abc",
                "question": "How does the main module work?",
                "expected_files": [
                    "main.py",
                    "missing_1.py",
                    "missing_2.py",
                    "missing_3.py",
                    "missing_4.py",
                ],
                "token_budget": 4096,
            }
        )
        result = run_archex_query(task, python_simple_repo)
        assert result.post_bundle_read_turns == 4
        assert result.post_bundle_search_turns == 3
