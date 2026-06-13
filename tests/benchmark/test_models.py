"""Tests for benchmark data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from archex.benchmark.models import (
    ArchitectureBenchmarkTask,
    ArchitectureExpectedInterface,
    ArchitectureExpectedModule,
    ArchitectureExpectedPattern,
    ArchitectureOracle,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkTask,
    Strategy,
)


class TestStrategy:
    def test_enum_values(self) -> None:
        assert Strategy.RAW_FILES == "raw_files"
        assert Strategy.RAW_GREPPED == "raw_grepped"
        assert Strategy.ARCHEX_QUERY == "archex_query"
        assert Strategy.ARCHEX_SCOUT_FETCH == "archex_scout_fetch"
        assert Strategy.ARCHEX_QUERY_VECTOR == "archex_query_vector"
        assert Strategy.SURROGATE_VECTOR == "surrogate_vector"
        assert Strategy.ARCHEX_QUERY_FUSION == "archex_query_fusion"
        assert Strategy.CROSS_LAYER_FUSION == "cross_layer_fusion"
        assert Strategy.ARCHEX_QUERY_FUSION_RERANK == "archex_query_fusion_rerank"
        assert Strategy.ARCHEX_SYMBOL_LOOKUP == "archex_symbol_lookup"
        assert Strategy.EXTERNAL_MCP == "external_mcp"

    def test_enum_from_value(self) -> None:
        assert Strategy("raw_files") is Strategy.RAW_FILES

    def test_enum_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            Strategy("nonexistent")


class TestBenchmarkTask:
    def test_valid_task(self) -> None:
        task = BenchmarkTask(
            task_id="test_task",
            repo="owner/repo",
            commit="abc123",
            question="How does X work?",
            expected_files=["src/main.py"],
        )
        assert task.task_id == "test_task"
        assert task.token_budget == 8192
        assert task.keywords == []
        assert task.expected_symbols == []
        assert task.include_paths == []

    def test_include_paths_must_be_relative(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkTask(
                task_id="test",
                repo="owner/repo",
                commit="abc",
                question="test",
                expected_files=["src/main.py"],
                include_paths=["../secret"],
            )

        with pytest.raises(ValidationError):
            BenchmarkTask(
                task_id="test",
                repo="owner/repo",
                commit="abc",
                question="test",
                expected_files=["src/main.py"],
                include_paths=["/tmp/repo"],
            )

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkTask(  # type: ignore[call-arg]
                task_id="test",
                repo="owner/repo",
                # missing commit, question, expected_files
            )

    def test_empty_expected_files(self) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="owner/repo",
            commit="abc",
            question="test",
            expected_files=[],
        )
        assert task.expected_files == []

    def test_custom_token_budget(self) -> None:
        task = BenchmarkTask(
            task_id="test",
            repo="owner/repo",
            commit="abc",
            question="test",
            expected_files=["a.py"],
            token_budget=4096,
        )
        assert task.token_budget == 4096


class TestArchitectureBenchmarkTask:
    def test_valid_architecture_task(self) -> None:
        task = ArchitectureBenchmarkTask(
            task_id="arch_test",
            repo=".",
            commit="HEAD",
            question="What architecture does this fixture use?",
            include_paths=["tests/fixtures/python_patterns"],
            arch_oracle=ArchitectureOracle(
                modules=[
                    ArchitectureExpectedModule(
                        name="python_patterns",
                        root_path="tests/fixtures/python_patterns",
                        files=["tests/fixtures/python_patterns/strategies.py"],
                    )
                ],
                patterns=[ArchitectureExpectedPattern(name="strategy")],
                interfaces=[
                    ArchitectureExpectedInterface(
                        name="SortStrategy",
                        file_path="tests/fixtures/python_patterns/strategies.py",
                    )
                ],
            ),
        )

        assert task.repo == "."
        assert task.include_paths == ["tests/fixtures/python_patterns"]
        assert task.arch_oracle.patterns[0].name == "strategy"

    def test_architecture_task_requires_include_paths(self) -> None:
        with pytest.raises(ValidationError):
            ArchitectureBenchmarkTask(
                task_id="arch_test",
                repo=".",
                commit="HEAD",
                question="What architecture does this fixture use?",
                include_paths=[],
                arch_oracle=ArchitectureOracle(),
            )

    def test_architecture_include_paths_must_be_relative(self) -> None:
        with pytest.raises(ValidationError):
            ArchitectureBenchmarkTask(
                task_id="arch_test",
                repo=".",
                commit="HEAD",
                question="What architecture does this fixture use?",
                include_paths=["/tmp/repo"],
                arch_oracle=ArchitectureOracle(),
            )


class TestBenchmarkResult:
    def test_valid_result(self) -> None:
        result = BenchmarkResult(
            task_id="test",
            strategy=Strategy.RAW_FILES,
            tokens_total=1000,
            tool_calls=3,
            files_accessed=3,
            recall=1.0,
            precision=1.0,
            savings_vs_raw=0.0,
            wall_time_ms=50.0,
            cached=False,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert result.tokens_total == 1000
        assert result.timing is None
        assert result.symbol_recall == 0.0
        assert result.vector_mode == "raw"
        assert result.cache_state == "cold"

    def test_with_timing(self) -> None:
        from archex.models import PipelineTiming

        timing = PipelineTiming(total_ms=100.0)
        result = BenchmarkResult(
            task_id="test",
            strategy=Strategy.ARCHEX_QUERY,
            tokens_total=500,
            tool_calls=1,
            files_accessed=2,
            recall=0.8,
            precision=0.5,
            savings_vs_raw=50.0,
            wall_time_ms=100.0,
            cached=False,
            timing=timing,
            timestamp="2025-01-01T00:00:00Z",
        )
        assert result.timing is not None
        assert result.timing.total_ms == 100.0


class TestBenchmarkReport:
    def test_valid_report(self) -> None:
        report = BenchmarkReport(
            task_id="test",
            repo="owner/repo",
            question="How?",
            results=[],
            baseline_tokens=5000,
        )
        assert report.baseline_tokens == 5000
        assert report.results == []
        assert report.median_latency_ms == 0.0
        assert report.p95_latency_ms == 0.0

    def test_serialization_roundtrip(self) -> None:
        result = BenchmarkResult(
            task_id="test",
            strategy=Strategy.RAW_FILES,
            tokens_total=1000,
            tool_calls=1,
            files_accessed=1,
            recall=1.0,
            precision=1.0,
            savings_vs_raw=0.0,
            wall_time_ms=10.0,
            cached=False,
            timestamp="2025-01-01T00:00:00Z",
        )
        report = BenchmarkReport(
            task_id="test",
            repo="owner/repo",
            question="How?",
            results=[result],
            baseline_tokens=1000,
        )
        json_str = report.model_dump_json()
        restored = BenchmarkReport.model_validate_json(json_str)
        assert restored.task_id == report.task_id
        assert len(restored.results) == 1
        assert restored.results[0].strategy == Strategy.RAW_FILES
        assert restored.median_latency_ms == 0.0
        assert restored.p95_latency_ms == 0.0
