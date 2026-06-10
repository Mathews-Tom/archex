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


def test_score_architecture_profile_does_not_match_unrelated_module_responsibility() -> None:
    task = _architecture_task().model_copy(
        update={
            "arch_oracle": ArchitectureOracle(
                modules=[
                    ArchitectureExpectedModule(
                        name="missing",
                        root_path="missing",
                        files=["missing/module.py"],
                        responsibility_terms=["sort"],
                    )
                ]
            )
        }
    )

    scores = score_architecture_profile(task, _profile())

    assert scores.responsibility_recall == 0.0


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


def test_architecture_gate_warns_on_baseline_regression() -> None:
    current = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(pattern_recall=0.5),
    )
    baseline = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(pattern_recall=1.0),
    )

    warnings = architecture_gate_warnings([current], baseline_results=[baseline])

    assert "arch_fixture pattern_recall regressed: 0.500 < baseline 1.000" in warnings


def test_architecture_gate_does_not_warn_for_equal_or_improved_baseline_scores() -> None:
    current = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(pattern_recall=1.0, overall=1.0),
    )
    baseline = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(pattern_recall=0.5, overall=1.0),
    )

    warnings = architecture_gate_warnings([current], baseline_results=[baseline])

    assert not any("regressed" in warning for warning in warnings)


def test_architecture_gate_warns_on_missing_baseline_task() -> None:
    current = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(),
    )

    warnings = architecture_gate_warnings([current], baseline_results=[])

    assert warnings == ["arch_fixture: missing architecture baseline result"]


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
    assert "Architecture baseline mode: FIRST RUN / seed candidate" in summary
    assert ".archex/arch-quality-baseline" in summary


def test_format_architecture_summary_reports_baseline_comparison(tmp_path: Path) -> None:
    result = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(),
    )

    summary = format_architecture_summary(
        [result],
        baseline_dir=tmp_path,
        baseline_results=[result],
    )

    assert f"Architecture baseline mode: REGRESSION COMPARISON ({tmp_path})" in summary


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
    assert "Architecture baseline mode: FIRST RUN / seed candidate" in output.output
    assert ".archex/arch-quality-baseline" in output.output
    assert "ARCHITECTURE QUALITY ADVISORY" in output.output


def test_arch_gate_cli_accepts_baseline_results(tmp_path: Path) -> None:
    current_dir = tmp_path / "current"
    baseline_dir = tmp_path / "baseline"
    current_dir.mkdir()
    baseline_dir.mkdir()
    current = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(pattern_recall=0.5),
    )
    baseline = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(pattern_recall=1.0),
    )
    (current_dir / "result.json").write_text(current.model_dump_json())
    (baseline_dir / "result.json").write_text(baseline.model_dump_json())

    output = CliRunner().invoke(
        benchmark_cmd,
        ["arch", "gate", "--input", str(current_dir), "--baseline", str(baseline_dir)],
    )

    assert output.exit_code == 0
    assert "pattern_recall regressed" in output.output


def test_arch_report_cli_defaults_to_seed_mode(tmp_path: Path) -> None:
    result = ArchitectureBenchmarkResult(
        task_id="arch_fixture",
        repo=".",
        commit="HEAD",
        scores=ArchitectureDimensionScores(),
    )
    (tmp_path / "result.json").write_text(result.model_dump_json())

    output = CliRunner().invoke(benchmark_cmd, ["arch", "report", "--input", str(tmp_path)])

    assert output.exit_code == 0
    assert "Architecture baseline mode: FIRST RUN / seed candidate" in output.output


def test_run_architecture_benchmark_scores_fixture() -> None:
    task = load_arch_task(Path("benchmarks/arch_tasks/python_false_positives.yaml"))

    result = run_architecture_benchmark(task)

    assert result.task_id.startswith("python_false_positives")
    assert result.advisory is True
    assert 0.0 <= result.scores.pattern_precision <= 1.0
