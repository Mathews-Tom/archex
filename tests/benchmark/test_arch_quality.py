"""Tests for architecture-quality benchmark scoring."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from archex.benchmark.arch_quality import (
    architecture_gate_warnings,
    format_architecture_summary,
    load_architecture_results,
    run_architecture_benchmark,
    score_architecture_profile,
)
from archex.benchmark.loader import load_arch_task
from archex.benchmark.models import (
    ArchitectureBenchmarkResult,
    ArchitectureBenchmarkTask,
    ArchitectureDimensionScores,
    ArchitectureExpectedDecision,
    ArchitectureExpectedInterface,
    ArchitectureExpectedModule,
    ArchitectureExpectedPattern,
    ArchitectureOracle,
)
from archex.cli.benchmark_cmd import benchmark_cmd
from archex.models import (
    ArchDecision,
    ArchProfile,
    DetectedPattern,
    Interface,
    Module,
    PatternCategory,
    PatternEvidence,
    RepoMetadata,
    SymbolKind,
    SymbolRef,
)


def _architecture_task() -> ArchitectureBenchmarkTask:
    return ArchitectureBenchmarkTask(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        question="Which architecture is present?",
        include_paths=["tests/fixtures/python_patterns"],
        arch_oracle=ArchitectureOracle(
            modules=[
                ArchitectureExpectedModule(
                    name="sorters",
                    root_path="sorters",
                    files=["sorters/base.py", "sorters/context.py"],
                    responsibility_terms=["sort", "strategy"],
                )
            ],
            patterns=[ArchitectureExpectedPattern(name="strategy")],
            interfaces=[
                ArchitectureExpectedInterface(
                    name="SortStrategy",
                    file_path="sorters/base.py",
                    kind=SymbolKind.CLASS,
                )
            ],
            decisions=[ArchitectureExpectedDecision(decision_terms=["strategy", "algorithm"])],
        ),
    )


def _profile() -> ArchProfile:
    symbol = SymbolRef(
        name="SortStrategy",
        qualified_name="SortStrategy",
        file_path="sorters/base.py",
        kind=SymbolKind.CLASS,
    )
    return ArchProfile(
        repo=RepoMetadata(local_path="."),
        module_map=[
            Module(
                name="sorters",
                root_path="sorters",
                files=["sorters/base.py", "sorters/context.py"],
                responsibility="sort strategy selection",
            )
        ],
        pattern_catalog=[
            DetectedPattern(
                name="strategy",
                display_name="Strategy Pattern",
                confidence=0.9,
                evidence=[
                    PatternEvidence(
                        file_path="sorters/base.py",
                        start_line=1,
                        end_line=10,
                        symbol="SortStrategy",
                        explanation="interface",
                    )
                ],
                description="Interchangeable algorithms",
                category=PatternCategory.BEHAVIORAL,
            )
        ],
        interface_surface=[Interface(symbol=symbol, signature="class SortStrategy")],
        decision_log=[
            ArchDecision(
                decision="Uses Strategy Pattern for interchangeable sorting algorithms",
                alternatives=["Conditional branching"],
                evidence=["sorters/base.py:1-10"],
                implications=["Algorithms are independently testable"],
            )
        ],
    )


def test_score_architecture_profile_perfect_match() -> None:
    scores = score_architecture_profile(_architecture_task(), _profile())

    assert scores.boundary_f1 == 1.0
    assert scores.pattern_precision == 1.0
    assert scores.pattern_recall == 1.0
    assert scores.interface_completeness == 1.0
    assert scores.decision_recall == 1.0
    assert scores.overall == 1.0


def test_score_architecture_profile_penalizes_false_positive_patterns() -> None:
    task = _architecture_task().model_copy(update={"arch_oracle": ArchitectureOracle(patterns=[])})
    scores = score_architecture_profile(task, _profile())

    assert scores.pattern_precision == 0.0
    assert scores.pattern_recall == 1.0


def test_architecture_gate_warnings_are_advisory() -> None:
    result = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(boundary_f1=0.5, pattern_precision=0.5),
    )

    warnings = architecture_gate_warnings([result])

    assert warnings == [
        "arch_fixture boundary_f1: 0.500 < 0.800",
        "arch_fixture pattern_precision: 0.500 < 0.800",
    ]


def test_format_architecture_summary_includes_gate_mode() -> None:
    result = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(),
    )

    summary = format_architecture_summary([result])

    assert "| arch_fixture | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |" in summary
    assert "Architecture-quality gate mode: ADVISORY" in summary


def test_load_architecture_results(tmp_path: Path) -> None:
    result = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(),
    )
    (tmp_path / "result.json").write_text(result.model_dump_json())

    loaded = load_architecture_results(tmp_path)

    assert [item.task_id for item in loaded] == ["arch_fixture"]


def test_arch_gate_cli_exits_zero_for_advisory_warning(tmp_path: Path) -> None:
    result = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(boundary_f1=0.5),
    )
    (tmp_path / "result.json").write_text(result.model_dump_json())

    output = CliRunner().invoke(benchmark_cmd, ["arch", "gate", "--input", str(tmp_path)])

    assert output.exit_code == 0
    assert "ARCHITECTURE QUALITY ADVISORY" in output.output


def test_run_architecture_benchmark_scores_fixture() -> None:
    task = load_arch_task(Path("benchmarks/arch_tasks/python_false_positives.yaml"))

    result = run_architecture_benchmark(task)

    assert result.task_id.startswith("python_false_positives")
    assert result.advisory is True
    assert 0.0 <= result.scores.pattern_precision <= 1.0
