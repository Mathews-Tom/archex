"""Tests for the per-dimension compare package and CLI renderer."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from archex.cli.compare_cmd import compare_cmd, render_comparison_markdown
from archex.models import (
    ArchDecision,
    ArchProfile,
    CodebaseStats,
    ComparisonResult,
    DependencyGraphSummary,
    DetectedPattern,
    DimensionComparison,
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
from archex.serve.compare import (
    SUPPORTED_DIMENSIONS,
    compare_repos,
    validate_dimensions,
)

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
        docstring="List all users.",
    )


def _async_interface() -> Interface:
    return Interface(
        symbol=_make_ref("fetch_users", "api/routes.py"),
        signature="async def fetch_users(request: Request) -> Response",
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


def _test_module() -> Module:
    return Module(
        name="tests",
        root_path="tests/",
        files=["tests/test_main.py", "tests/test_utils.py"],
        file_count=2,
        line_count=200,
        external_deps=["pytest", "hypothesis"],
    )


# ---------------------------------------------------------------------------
# compare_repos core
# ---------------------------------------------------------------------------


class TestCompareReposBasic:
    def test_returns_comparison_result(self) -> None:
        a = _make_profile(name="/tmp/a")
        b = _make_profile(name="/tmp/b")
        result = compare_repos(a, b)
        assert isinstance(result, ComparisonResult)

    def test_all_default_dimensions_included(self) -> None:
        result = compare_repos(_make_profile(), _make_profile())
        dims = {d.dimension for d in result.dimensions}
        assert dims == SUPPORTED_DIMENSIONS

    def test_specific_dimensions_only_preserves_order(self) -> None:
        result = compare_repos(
            _make_profile(),
            _make_profile(),
            dimensions=["error_handling", "concurrency"],
        )
        assert [d.dimension for d in result.dimensions] == ["error_handling", "concurrency"]

    def test_unsupported_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dimensions"):
            compare_repos(_make_profile(), _make_profile(), dimensions=["nonexistent"])

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
# Per-dimension behavior — error_handling
# ---------------------------------------------------------------------------


class TestErrorHandlingDimension:
    def test_detects_error_pattern(self) -> None:
        a = _make_profile(patterns=[_error_pattern()])
        result = compare_repos(a, _make_profile(), dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("Exception-related patterns: 1" in e for e in dim.evidence_a)
        assert any("Custom Exception" in e for e in dim.evidence_a)
        assert any("Exception-related patterns: 0" in e for e in dim.evidence_b)

    def test_no_signals_reports_zero(self) -> None:
        result = compare_repos(_make_profile(), _make_profile(), dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert dim.repo_a_approach == "No explicit error-handling structure detected"
        assert dim.repo_b_approach == "No explicit error-handling structure detected"

    def test_one_sided_signals_emit_imbalance_trade_off(self) -> None:
        result = compare_repos(
            _make_profile(patterns=[_error_pattern()]),
            _make_profile(),
            dimensions=["error_handling"],
        )
        dim = result.dimensions[0]
        assert any("Repo B has none" in t for t in dim.trade_offs)


# ---------------------------------------------------------------------------
# Per-dimension behavior — api_surface
# ---------------------------------------------------------------------------


class TestApiSurfaceDimension:
    def test_counts_total_interfaces(self) -> None:
        a = _make_profile(interfaces=[_api_interface()])
        result = compare_repos(a, _make_profile(), dimensions=["api_surface"])
        dim = result.dimensions[0]
        assert any("Total public interfaces: 1" in e for e in dim.evidence_a)
        assert any("Total public interfaces: 0" in e for e in dim.evidence_b)

    def test_reports_return_type_and_docstring_coverage(self) -> None:
        a = _make_profile(interfaces=[_api_interface()])
        result = compare_repos(a, _make_profile(), dimensions=["api_surface"])
        dim = result.dimensions[0]
        assert any("Return-type annotation coverage: 100.0%" in e for e in dim.evidence_a)
        assert any("Docstring coverage: 100.0%" in e for e in dim.evidence_a)


# ---------------------------------------------------------------------------
# Per-dimension behavior — configuration
# ---------------------------------------------------------------------------


class TestConfigurationDimension:
    def test_detects_config_module(self) -> None:
        a = _make_profile(modules=[_config_module()])
        result = compare_repos(a, _make_profile(), dimensions=["configuration"])
        dim = result.dimensions[0]
        assert any("Config modules: 1" in e for e in dim.evidence_a)

    def test_detects_config_library_dep(self) -> None:
        mod = Module(
            name="app",
            root_path="app/",
            files=["app/main.py"],
            file_count=1,
            line_count=100,
            external_deps=["python-dotenv", "pydantic-settings"],
        )
        a = _make_profile(modules=[mod])
        result = compare_repos(a, _make_profile(), dimensions=["configuration"])
        dim = result.dimensions[0]
        assert any("dotenv" in e or "pydantic_settings" in e for e in dim.evidence_a)


# ---------------------------------------------------------------------------
# Per-dimension behavior — concurrency
# ---------------------------------------------------------------------------


class TestConcurrencyDimension:
    def test_detects_async_signature(self) -> None:
        a = _make_profile(interfaces=[_api_interface(), _async_interface()])
        result = compare_repos(a, _make_profile(), dimensions=["concurrency"])
        dim = result.dimensions[0]
        assert any("Async interfaces: 1 of 2" in e for e in dim.evidence_a)

    def test_zero_async_when_only_sync_interfaces(self) -> None:
        a = _make_profile(interfaces=[_api_interface()])
        result = compare_repos(a, _make_profile(), dimensions=["concurrency"])
        dim = result.dimensions[0]
        assert any("Async interfaces: 0 of 1" in e for e in dim.evidence_a)


# ---------------------------------------------------------------------------
# Per-dimension behavior — testing
# ---------------------------------------------------------------------------


class TestTestingDimension:
    def test_detects_test_module_and_dep(self) -> None:
        a = _make_profile(modules=[_test_module()], total_files=10)
        result = compare_repos(a, _make_profile(), dimensions=["testing"])
        dim = result.dimensions[0]
        assert any("Test modules: 1" in e for e in dim.evidence_a)
        assert any("pytest" in e for e in dim.evidence_a)

    def test_test_file_ratio_present(self) -> None:
        a = _make_profile(modules=[_test_module()], total_files=10)
        result = compare_repos(a, _make_profile(), dimensions=["testing"])
        dim = result.dimensions[0]
        assert any("Test files: 2 of 10" in e for e in dim.evidence_a)


# ---------------------------------------------------------------------------
# Per-dimension behavior — state_management
# ---------------------------------------------------------------------------


class TestStateManagementDimension:
    def test_detects_persistence_dep(self) -> None:
        mod = Module(
            name="app",
            root_path="app/",
            files=["app/db.py"],
            file_count=1,
            line_count=200,
            external_deps=["sqlalchemy", "redis"],
        )
        a = _make_profile(modules=[mod])
        result = compare_repos(a, _make_profile(), dimensions=["state_management"])
        dim = result.dimensions[0]
        assert any("Persistence backends:" in e for e in dim.evidence_a)
        assert any("sqlalchemy" in e for e in dim.evidence_a)


# ---------------------------------------------------------------------------
# Asymmetric trade-off branches — directional discipline / ratio / library diff
# ---------------------------------------------------------------------------


def _bare_interface(name: str = "foo", path: str = "x.py") -> Interface:
    """Interface with no return_type and no docstring — drops api_surface coverage."""
    return Interface(
        symbol=_make_ref(name, path),
        signature=f"def {name}():",
    )


class TestApiSurfaceAsymmetric:
    """api_surface emits a directional trade-off when coverage gaps exceed 15pp."""

    def test_a_has_stronger_return_type_and_docstring_discipline(self) -> None:
        a = _make_profile(name="/tmp/a", interfaces=[_api_interface()])
        b = _make_profile(name="/tmp/b", interfaces=[_bare_interface()])
        result = compare_repos(a, b, dimensions=["api_surface"])
        dim = result.dimensions[0]
        assert any("stronger return-type discipline" in t for t in dim.trade_offs)
        assert any("documents interfaces more thoroughly" in t for t in dim.trade_offs)
        assert any("/tmp/a" in t for t in dim.trade_offs)

    def test_b_has_stronger_return_type_and_docstring_discipline(self) -> None:
        a = _make_profile(name="/tmp/a", interfaces=[_bare_interface()])
        b = _make_profile(name="/tmp/b", interfaces=[_api_interface()])
        result = compare_repos(a, b, dimensions=["api_surface"])
        dim = result.dimensions[0]
        assert any("stronger return-type discipline" in t for t in dim.trade_offs)
        assert any("documents interfaces more thoroughly" in t for t in dim.trade_offs)
        assert any("/tmp/b" in t for t in dim.trade_offs)


class TestConcurrencyAsymmetric:
    """concurrency emits a directional trade-off when async ratios diverge by ≥15pp."""

    def test_a_more_async_heavy(self) -> None:
        a = _make_profile(name="/tmp/a", interfaces=[_async_interface()])
        b = _make_profile(name="/tmp/b", interfaces=[_api_interface()])
        result = compare_repos(a, b, dimensions=["concurrency"])
        dim = result.dimensions[0]
        assert any("markedly more async-heavy" in t and "/tmp/a" in t for t in dim.trade_offs)

    def test_b_more_async_heavy(self) -> None:
        a = _make_profile(name="/tmp/a", interfaces=[_api_interface()])
        b = _make_profile(name="/tmp/b", interfaces=[_async_interface()])
        result = compare_repos(a, b, dimensions=["concurrency"])
        dim = result.dimensions[0]
        assert any("markedly more async-heavy" in t and "/tmp/b" in t for t in dim.trade_offs)


class TestTestingAsymmetric:
    """testing emits a directional trade-off when test-file ratios diverge by ≥10pp."""

    def test_a_invests_more_in_test_surface(self) -> None:
        a = _make_profile(name="/tmp/a", modules=[_test_module()], total_files=10)
        b = _make_profile(name="/tmp/b", total_files=10)
        result = compare_repos(a, b, dimensions=["testing"])
        dim = result.dimensions[0]
        assert any("invests more in test surface" in t and "/tmp/a" in t for t in dim.trade_offs)

    def test_b_invests_more_in_test_surface(self) -> None:
        a = _make_profile(name="/tmp/a", total_files=10)
        b = _make_profile(name="/tmp/b", modules=[_test_module()], total_files=10)
        result = compare_repos(a, b, dimensions=["testing"])
        dim = result.dimensions[0]
        assert any("invests more in test surface" in t and "/tmp/b" in t for t in dim.trade_offs)


class TestLibraryDiffTradeOffs:
    """library_diff_trade_offs surfaces both A-only and B-only library callouts.

    Routed through state_management, which is the dimension whose tests already
    cover the persistence-deps codepath. The helper itself is shared across
    state_management, concurrency, testing, and configuration; one route is
    sufficient to exercise the helper's branches.
    """

    def test_both_sides_have_unique_libraries(self) -> None:
        mod_a = Module(
            name="store_a",
            root_path="a/",
            files=["a/db.py"],
            file_count=1,
            line_count=100,
            external_deps=["redis"],
        )
        mod_b = Module(
            name="store_b",
            root_path="b/",
            files=["b/db.py"],
            file_count=1,
            line_count=100,
            external_deps=["mongo"],
        )
        a = _make_profile(name="/tmp/a", modules=[mod_a])
        b = _make_profile(name="/tmp/b", modules=[mod_b])
        result = compare_repos(a, b, dimensions=["state_management"])
        dim = result.dimensions[0]
        assert any("redis" in t and "/tmp/a" in t and "not seen in" in t for t in dim.trade_offs)
        assert any("mongo" in t and "/tmp/b" in t and "not seen in" in t for t in dim.trade_offs)


# ---------------------------------------------------------------------------
# Trade-off classifier
# ---------------------------------------------------------------------------


class TestTradeOffs:
    def test_both_empty_reports_neither(self) -> None:
        result = compare_repos(_make_profile(), _make_profile(), dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("Neither repository" in t for t in dim.trade_offs)

    def test_substantially_more_when_one_side_dominates(self) -> None:
        many = [
            DetectedPattern(
                name=f"exception_{i}",
                display_name=f"Exception {i}",
                confidence=0.8,
                description="error pattern",
                category=PatternCategory.STRUCTURAL,
                evidence=[
                    PatternEvidence(
                        file_path=f"e_{i}.py",
                        start_line=1,
                        end_line=5,
                        symbol=f"E{i}",
                        explanation="exception",
                    )
                ],
            )
            for i in range(6)
        ]
        a = _make_profile(patterns=many)
        b = _make_profile(patterns=[_error_pattern()])
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("substantially more" in t for t in dim.trade_offs)

    def test_comparable_when_counts_match(self) -> None:
        a = _make_profile(patterns=[_error_pattern()])
        b = _make_profile(patterns=[_error_pattern()])
        result = compare_repos(a, b, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("comparable" in t for t in dim.trade_offs)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dimensions_list(self) -> None:
        result = compare_repos(_make_profile(), _make_profile(), dimensions=[])
        assert result.dimensions == []

    def test_identical_profiles_show_comparable(self) -> None:
        profile = _make_profile(patterns=[_error_pattern()])
        result = compare_repos(profile, profile, dimensions=["error_handling"])
        dim = result.dimensions[0]
        assert any("comparable" in t for t in dim.trade_offs)

    def test_validate_dimensions_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dimensions"):
            validate_dimensions(["nonexistent_dimension"])

    def test_validate_dimensions_mixed_valid_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dimensions"):
            validate_dimensions(["error_handling", "invalid_dim"])

    def test_validate_dimensions_empty_passes(self) -> None:
        validate_dimensions([])


# ---------------------------------------------------------------------------
# CLI smoke tests
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
# render_comparison_markdown
# ---------------------------------------------------------------------------


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
        result = _make_comparison_result(
            local_path_a="/tmp/repo_a",
            local_path_b="/tmp/repo_b",
            dimensions=[_make_dimension(dimension="error_handling")],
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
        md = render_comparison_markdown(_make_comparison_result(dimensions=[]))
        assert "## Dimensions" not in md

    def test_no_evidence_produces_no_evidence_sections(self) -> None:
        result = _make_comparison_result(dimensions=[_make_dimension(evidence_a=[], evidence_b=[])])
        md = render_comparison_markdown(result)
        assert "**Repo A evidence:**" not in md
        assert "**Repo B evidence:**" not in md

    def test_no_trade_offs_produces_no_trade_offs_section(self) -> None:
        md = render_comparison_markdown(
            _make_comparison_result(dimensions=[_make_dimension(trade_offs=[])])
        )
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
# CLI extended (JSON, markdown, dimension parsing)
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
                evidence_a=["Exception-related patterns: 1"],
                evidence_b=["Exception-related patterns: 0"],
                trade_offs=["Repo A has 1 exception-pattern signals; Repo B has none."],
            )
        ],
        summary="Repo A uses exceptions; Repo B uses error codes.",
    )


class TestCompareCLIExtended:
    def test_cli_json_output(self) -> None:
        import json

        runner = CliRunner()
        fake = _make_cli_comparison_result()
        with patch("archex.cli.compare_cmd.compare", return_value=fake):
            result = runner.invoke(compare_cmd, ["/tmp/a", "/tmp/b"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "dimensions" in parsed
        assert parsed["dimensions"][0]["dimension"] == "error_handling"

    def test_cli_markdown_output(self) -> None:
        runner = CliRunner()
        fake = _make_cli_comparison_result()
        with patch("archex.cli.compare_cmd.compare", return_value=fake):
            result = runner.invoke(compare_cmd, ["/tmp/a", "/tmp/b", "--format", "markdown"])
        assert result.exit_code == 0, result.output
        assert "# Comparison:" in result.output
        assert "## Dimensions" in result.output

    def test_cli_dimensions_parsing(self) -> None:
        runner = CliRunner()
        fake = _make_cli_comparison_result()
        captured: dict[str, list[str] | None] = {}

        def fake_compare(
            src_a: object,
            src_b: object,
            *,
            dimensions: list[str] | None = None,
            config: object = None,
        ) -> ComparisonResult:
            captured["dimensions"] = dimensions
            return fake

        with patch("archex.cli.compare_cmd.compare", side_effect=fake_compare):
            result = runner.invoke(
                compare_cmd,
                ["/tmp/a", "/tmp/b", "--dimensions", "error_handling,concurrency"],
            )
        assert result.exit_code == 0, result.output
        assert captured["dimensions"] == ["error_handling", "concurrency"]
