from __future__ import annotations

import pytest
from click.testing import CliRunner

from archex.cli.compare_cmd import compare_cmd
from archex.models import (
    ArchDecision,
    ArchProfile,
    CodebaseStats,
    ComparisonResult,
    DependencyGraphSummary,
    DetectedPattern,
    Interface,
    LanguageStats,
    Module,
    Parameter,
    PatternCategory,
    PatternEvidence,
    RepoMetadata,
    SymbolKind,
    SymbolRef,
)
from archex.serve.compare import SUPPORTED_DIMENSIONS, compare_repos

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ref(name: str, path: str, kind: SymbolKind = SymbolKind.FUNCTION) -> SymbolRef:
    return SymbolRef(name=name, qualified_name=f"{path}:{name}", file_path=path, kind=kind)


def _make_profile(
    *,
    name: str = "/tmp/repo",
    patterns: list[DetectedPattern] | None = None,
    interfaces: list[Interface] | None = None,
    modules: list[Module] | None = None,
    decisions: list[ArchDecision] | None = None,
    total_files: int = 10,
    total_lines: int = 500,
) -> ArchProfile:
    return ArchProfile(
        repo=RepoMetadata(
            local_path=name,
            languages={"python": total_files},
            total_files=total_files,
            total_lines=total_lines,
        ),
        stats=CodebaseStats(
            total_files=total_files,
            total_lines=total_lines,
            languages={"python": LanguageStats(files=total_files, lines=total_lines)},
        ),
        pattern_catalog=patterns or [],
        interface_surface=interfaces or [],
        module_map=modules or [],
        decision_log=decisions or [],
        dependency_graph=DependencyGraphSummary(),
    )


def _error_pattern() -> DetectedPattern:
    return DetectedPattern(
        name="custom_exception",
        display_name="Custom Exception Hierarchy",
        confidence=0.9,
        description="Custom exception classes for error handling",
        category=PatternCategory.STRUCTURAL,
        evidence=[
            PatternEvidence(
                file_path="errors.py",
                start_line=1,
                end_line=10,
                symbol="AppError",
                explanation="base exception",
            )
        ],
    )


def _api_interface() -> Interface:
    return Interface(
        symbol=_make_ref("get_users", "api/routes.py"),
        signature="def get_users(request: Request) -> Response",
        parameters=[Parameter(name="request", type_annotation="Request")],
        return_type="Response",
    )


def _config_module() -> Module:
    return Module(
        name="config",
        root_path="config/",
        files=["config/settings.py"],
        file_count=1,
        line_count=50,
        external_deps=["pydantic-settings"],
    )


def _concurrency_decision() -> ArchDecision:
    return ArchDecision(
        decision="Uses async I/O with asyncio for concurrency",
        evidence=["Found async def in 5 files"],
        source="structural",
    )


# ---------------------------------------------------------------------------
# Tests — compare_repos core
# ---------------------------------------------------------------------------


class TestCompareReposBasic:
    def test_returns_comparison_result(self) -> None:
        a = _make_profile(name="/tmp/a")
        b = _make_profile(name="/tmp/b")
        result = compare_repos(a, b)
        assert isinstance(result, ComparisonResult)

    def test_all_default_dimensions_included(self) -> None:
        a = _make_profile()
        b = _make_profile()
        result = compare_repos(a, b)
        dims = {d.dimension for d in result.dimensions}
        assert dims == SUPPORTED_DIMENSIONS

    def test_specific_dimensions_only(self) -> None:
        a = _make_profile()
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["error_handling", "concurrency"])
        dims = [d.dimension for d in result.dimensions]
        assert dims == ["error_handling", "concurrency"]  # user-provided order

    def test_unsupported_dimension_raises(self) -> None:
        a = _make_profile()
        b = _make_profile()
        with pytest.raises(ValueError, match="Unsupported dimensions"):
            compare_repos(a, b, dimensions=["nonexistent"])

    def test_repo_metadata_preserved(self) -> None:
        a = _make_profile(name="/tmp/a")
        b = _make_profile(name="/tmp/b")
        result = compare_repos(a, b)
        assert result.repo_a.local_path == "/tmp/a"
        assert result.repo_b.local_path == "/tmp/b"

    def test_summary_contains_repo_info(self) -> None:
        a = _make_profile(name="/tmp/a", total_files=5, total_lines=100)
        b = _make_profile(name="/tmp/b", total_files=20, total_lines=2000)
        result = compare_repos(a, b)
        assert "/tmp/a" in result.summary
        assert "/tmp/b" in result.summary
        assert "5 files" in result.summary
        assert "20 files" in result.summary


# ---------------------------------------------------------------------------
# Tests — dimension evidence extraction
# ---------------------------------------------------------------------------


class TestErrorHandlingDimension:
    def test_detects_error_pattern(self) -> None:
        a = _make_profile(patterns=[_error_pattern()])
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert len(dim.evidence_a) > 0
        assert len(dim.evidence_b) == 0
        assert any("Custom Exception" in e for e in dim.evidence_a)

    def test_both_have_evidence(self) -> None:
        a = _make_profile(patterns=[_error_pattern()])
        b = _make_profile(patterns=[_error_pattern()])
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert len(dim.evidence_a) > 0
        assert len(dim.evidence_b) > 0


class TestApiSurfaceDimension:
    def test_detects_api_interface(self) -> None:
        a = _make_profile(interfaces=[_api_interface()])
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["api_surface"])
        dim = result.dimensions[0]
        assert any("get_users" in e for e in dim.evidence_a)

    def test_counts_total_public_interfaces(self) -> None:
        a = _make_profile(interfaces=[_api_interface()])
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["api_surface"])
        dim = result.dimensions[0]
        assert any("Total public interfaces: 1" in e for e in dim.evidence_a)


class TestConfigurationDimension:
    def test_detects_config_module(self) -> None:
        a = _make_profile(modules=[_config_module()])
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["configuration"])
        dim = result.dimensions[0]
        assert any("config" in e.lower() for e in dim.evidence_a)

    def test_detects_config_external_dep(self) -> None:
        mod = Module(
            name="app",
            root_path="app/",
            files=["app/main.py"],
            file_count=1,
            line_count=100,
            external_deps=["python-dotenv", "pydantic-settings"],
        )
        a = _make_profile(modules=[mod])
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["configuration"])
        dim = result.dimensions[0]
        assert any("settings" in e.lower() for e in dim.evidence_a)


class TestConcurrencyDimension:
    def test_detects_concurrency_decision(self) -> None:
        a = _make_profile(decisions=[_concurrency_decision()])
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["concurrency"])
        dim = result.dimensions[0]
        assert any("async" in e.lower() for e in dim.evidence_a)


# ---------------------------------------------------------------------------
# Tests — empty profiles and edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_both_empty_profiles(self) -> None:
        a = _make_profile()
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert dim.repo_a_approach == "No error handling evidence found"
        assert dim.repo_b_approach == "No error handling evidence found"
        assert len(dim.trade_offs) > 0
        assert "Neither" in dim.trade_offs[0]

    def test_empty_dimensions_list(self) -> None:
        a = _make_profile()
        b = _make_profile()
        result = compare_repos(a, b, dimensions=[])
        assert result.dimensions == []

    def test_trade_off_asymmetric(self) -> None:
        a = _make_profile(patterns=[_error_pattern()])
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("Repo A" in t and "Repo B lacks" in t for t in dim.trade_offs)

    def test_trade_off_comparable(self) -> None:
        a = _make_profile(patterns=[_error_pattern()])
        b = _make_profile(patterns=[_error_pattern()])
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("comparable" in t for t in dim.trade_offs)


# ---------------------------------------------------------------------------
# Tests — CLI
# ---------------------------------------------------------------------------


class TestCompareCLI:
    def test_help_shows_dimensions(self) -> None:
        runner = CliRunner()
        result = runner.invoke(compare_cmd, ["--help"])
        assert result.exit_code == 0
        assert "dimensions" in result.output.lower()

    def test_help_shows_format(self) -> None:
        runner = CliRunner()
        result = runner.invoke(compare_cmd, ["--help"])
        assert result.exit_code == 0
        assert "json" in result.output
        assert "markdown" in result.output


# ---------------------------------------------------------------------------
# Tests — render_comparison_markdown
# ---------------------------------------------------------------------------

from unittest.mock import patch  # noqa: E402

from archex.cli.compare_cmd import render_comparison_markdown  # noqa: E402
from archex.models import DimensionComparison  # noqa: E402


def _make_comparison_result(
    *,
    url_a: str | None = None,
    local_path_a: str | None = "/tmp/a",
    url_b: str | None = None,
    local_path_b: str | None = "/tmp/b",
    dimensions: list[DimensionComparison] | None = None,
    summary: str = "Test summary.",
) -> ComparisonResult:
    return ComparisonResult(
        repo_a=RepoMetadata(url=url_a, local_path=local_path_a, total_files=5, total_lines=200),
        repo_b=RepoMetadata(url=url_b, local_path=local_path_b, total_files=10, total_lines=800),
        dimensions=dimensions if dimensions is not None else [],
        summary=summary,
    )


def _make_dimension(
    *,
    dimension: str = "error_handling",
    repo_a_approach: str = "Uses exceptions",
    repo_b_approach: str = "Returns error codes",
    evidence_a: list[str] | None = None,
    evidence_b: list[str] | None = None,
    trade_offs: list[str] | None = None,
) -> DimensionComparison:
    return DimensionComparison(
        dimension=dimension,
        repo_a_approach=repo_a_approach,
        repo_b_approach=repo_b_approach,
        evidence_a=evidence_a if evidence_a is not None else [],
        evidence_b=evidence_b if evidence_b is not None else [],
        trade_offs=trade_offs if trade_offs is not None else [],
    )


class TestRenderComparisonMarkdown:
    def test_basic_output_contains_repo_names_and_dimension_headers(self) -> None:
        dim = _make_dimension(dimension="error_handling")
        result = _make_comparison_result(
            local_path_a="/tmp/repo_a",
            local_path_b="/tmp/repo_b",
            dimensions=[dim],
        )
        md = render_comparison_markdown(result)
        assert "/tmp/repo_a" in md
        assert "/tmp/repo_b" in md
        assert "## Dimensions" in md
        assert "### Error Handling" in md

    def test_summary_section_included(self) -> None:
        result = _make_comparison_result(summary="Architecture differs significantly.")
        md = render_comparison_markdown(result)
        assert "## Summary" in md
        assert "Architecture differs significantly." in md

    def test_empty_dimensions_list_produces_no_dimensions_section(self) -> None:
        result = _make_comparison_result(dimensions=[])
        md = render_comparison_markdown(result)
        assert "## Dimensions" not in md

    def test_no_evidence_produces_no_evidence_sections(self) -> None:
        dim = _make_dimension(evidence_a=[], evidence_b=[])
        result = _make_comparison_result(dimensions=[dim])
        md = render_comparison_markdown(result)
        assert "**Repo A evidence:**" not in md
        assert "**Repo B evidence:**" not in md

    def test_no_trade_offs_produces_no_trade_offs_section(self) -> None:
        dim = _make_dimension(trade_offs=[])
        result = _make_comparison_result(dimensions=[dim])
        md = render_comparison_markdown(result)
        assert "**Trade-offs:**" not in md

    def test_repo_names_derived_from_url_when_local_path_is_none(self) -> None:
        result = _make_comparison_result(
            url_a="https://github.com/org/repo-a",
            local_path_a=None,
            url_b="https://github.com/org/repo-b",
            local_path_b=None,
        )
        md = render_comparison_markdown(result)
        assert "https://github.com/org/repo-a" in md
        assert "https://github.com/org/repo-b" in md


# ---------------------------------------------------------------------------
# Tests — trade-off text edge cases (via compare_repos)
# ---------------------------------------------------------------------------


class TestTradeOffEdgeCases:
    def test_repo_b_empty_trade_off_mentions_repo_b_lacks(self) -> None:
        a = _make_profile(patterns=[_error_pattern()])
        b = _make_profile()
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("Repo B lacks" in t for t in dim.trade_offs)

    def test_repo_a_significantly_more_evidence_mentions_significantly_more(self) -> None:
        extra_patterns: list[DetectedPattern] = []
        for i in range(6):
            extra_patterns.append(
                DetectedPattern(
                    name=f"error_pattern_{i}",
                    display_name=f"Error Pattern {i}",
                    confidence=0.8,
                    description="error handling",
                    category=PatternCategory.STRUCTURAL,
                    evidence=[
                        PatternEvidence(
                            file_path=f"errors_{i}.py",
                            start_line=1,
                            end_line=5,
                            symbol=f"Error{i}",
                            explanation="exception",
                        )
                    ],
                )
            )
        a = _make_profile(patterns=extra_patterns)
        b = _make_profile(patterns=[_error_pattern()])
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("significantly more" in t for t in dim.trade_offs)

    def test_repo_b_significantly_more_evidence_mentions_significantly_more(self) -> None:
        extra_patterns: list[DetectedPattern] = []
        for i in range(6):
            extra_patterns.append(
                DetectedPattern(
                    name=f"error_pattern_{i}",
                    display_name=f"Error Pattern {i}",
                    confidence=0.8,
                    description="error handling",
                    category=PatternCategory.STRUCTURAL,
                    evidence=[
                        PatternEvidence(
                            file_path=f"errors_{i}.py",
                            start_line=1,
                            end_line=5,
                            symbol=f"Error{i}",
                            explanation="exception",
                        )
                    ],
                )
            )
        a = _make_profile(patterns=[_error_pattern()])
        b = _make_profile(patterns=extra_patterns)
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("significantly more" in t for t in dim.trade_offs)

    def test_comparable_evidence_trade_off_mentions_comparable(self) -> None:
        a = _make_profile(patterns=[_error_pattern()])
        b = _make_profile(patterns=[_error_pattern()])
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("comparable" in t for t in dim.trade_offs)


# ---------------------------------------------------------------------------
# Tests — CLI extended (JSON, markdown, dimension parsing)
# ---------------------------------------------------------------------------


def _make_cli_comparison_result() -> ComparisonResult:
    return ComparisonResult(
        repo_a=RepoMetadata(local_path="/tmp/a", total_files=3, total_lines=100),
        repo_b=RepoMetadata(local_path="/tmp/b", total_files=7, total_lines=300),
        dimensions=[
            DimensionComparison(
                dimension="error_handling",
                repo_a_approach="Uses exceptions",
                repo_b_approach="Returns error codes",
                evidence_a=["Custom exception hierarchy"],
                evidence_b=[],
                trade_offs=["Repo A has structured error types; Repo B lacks them"],
            )
        ],
        summary="Repo A uses exceptions; Repo B uses error codes.",
    )


class TestCompareCLIExtended:
    def test_cli_json_output(self) -> None:
        import json

        runner = CliRunner()
        fake_result = _make_cli_comparison_result()
        with patch("archex.cli.compare_cmd.compare", return_value=fake_result):
            result = runner.invoke(compare_cmd, ["/tmp/a", "/tmp/b"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "dimensions" in parsed
        assert parsed["dimensions"][0]["dimension"] == "error_handling"

    def test_cli_markdown_output(self) -> None:
        runner = CliRunner()
        fake_result = _make_cli_comparison_result()
        with patch("archex.cli.compare_cmd.compare", return_value=fake_result):
            result = runner.invoke(compare_cmd, ["/tmp/a", "/tmp/b", "--format", "markdown"])
        assert result.exit_code == 0, result.output
        assert "# Comparison:" in result.output
        assert "## Dimensions" in result.output

    def test_cli_dimensions_parsing(self) -> None:
        runner = CliRunner()
        fake_result = _make_cli_comparison_result()
        captured: dict[str, list[str] | None] = {}

        def fake_compare(
            src_a: object,
            src_b: object,
            *,
            dimensions: list[str] | None = None,
            config: object = None,
        ) -> ComparisonResult:
            captured["dimensions"] = dimensions
            return fake_result

        with patch("archex.cli.compare_cmd.compare", side_effect=fake_compare):
            result = runner.invoke(
                compare_cmd,
                ["/tmp/a", "/tmp/b", "--dimensions", "error_handling,concurrency"],
            )
        assert result.exit_code == 0, result.output
        assert captured["dimensions"] == ["error_handling", "concurrency"]
