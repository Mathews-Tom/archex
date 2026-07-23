"""Tests for the MCP server integration."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("mcp", reason="mcp not installed")

from archex.context_facade import ContextResult, ContextRouteDecision
from archex.explain import ExplainError
from archex.graph_artifact import (
    ArchGraph,
    GraphEdge,
    GraphEdgeType,
    GraphExportMetadata,
    GraphNode,
    GraphNodeType,
    GraphProject,
    file_node_id,
)
from archex.integrations import mcp as mcp_integration
from archex.integrations.mcp import (
    build_server,
    clear_graph_query_cache,
    handle_analyze_repo,
    handle_compare_repos,
    handle_context,
    handle_explain_target,
    handle_generate_onboarding,
    handle_get_impact,
    handle_graph_hubs,
    handle_graph_neighbors,
    handle_graph_path,
    handle_graph_stats,
    handle_query_repo,
    handle_scout_repo,
)
from archex.models import (
    ArchProfile,
    CodeChunk,
    ComparisonResult,
    ContextBundle,
    ContextCompletenessReason,
    ContextCompletenessStatus,
    ContextFreshness,
    ContextReceipt,
    ContextReceiptTokenBudget,
    ContextRecommendedAction,
    EdgeConfidence,
    RankedChunk,
    RepoMetadata,
    RetrievalMetadata,
    TypeDefinition,
)
from archex.reporting import count_tokens
from archex.serve.intent import QueryIntent
from archex.serve.modality import BudgetTier, QueryModality
from archex.serve.renderers.xml import render_xml, render_xml_envelope

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_arch_profile(local_path: str = "/fake/repo") -> ArchProfile:
    return ArchProfile(repo=RepoMetadata(local_path=local_path))


def _make_context_bundle(question: str = "how does auth work?") -> ContextBundle:
    return ContextBundle(
        query=question,
        token_count=100,
        token_budget=8000,
        receipt=ContextReceipt(
            query=question,
            token_budget=ContextReceiptTokenBudget(requested=8000, consumed=100),
            index_revision="rev",
            freshness=ContextFreshness.CLEAN,
            context_complete=ContextCompletenessStatus.COMPLETE,
            context_complete_reason=ContextCompletenessReason.COMPLETE,
            recommended_next_action=ContextRecommendedAction.USE_BUNDLE,
        ),
    )


def _make_context_result(question: str = "how does auth work?") -> ContextResult:
    return ContextResult(
        bundle=_make_context_bundle(question),
        route=ContextRouteDecision(
            resolved_intent=QueryIntent.GENERAL,
            intent_source="auto",
            resolved_modality=QueryModality.NL_TO_PL,
            resolved_profile=None,
            profile_source="none",
            resolved_budget_tier=BudgetTier.STANDARD,
            token_budget_requested=8000,
            budget_source="intent_default",
            handles_mode=False,
            filters_active=False,
        ),
    )


def _make_comparison_result() -> ComparisonResult:
    return ComparisonResult(
        repo_a=RepoMetadata(local_path="/fake/repo_a"),
        repo_b=RepoMetadata(local_path="/fake/repo_b"),
    )


# ---------------------------------------------------------------------------
# Unit tests for handler functions
# ---------------------------------------------------------------------------


class TestHandleAnalyzeRepo:
    def test_returns_json_by_default(self) -> None:
        profile = _make_arch_profile()
        with (
            patch("archex.integrations.mcp.analyze", return_value=profile) as mock_analyze,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=10000),
        ):
            result = handle_analyze_repo("/fake/repo")
        mock_analyze.assert_called_once()
        assert isinstance(result, str)
        import json

        parsed = json.loads(result)
        assert "content" in parsed
        assert "_meta" in parsed
        assert parsed["_meta"]["tool_name"] == "analyze_repo"
        assert parsed["_meta"]["strategy"] == "full_analysis"
        assert parsed["_meta"]["tokens_raw_equivalent"] == 10000

    def test_returns_markdown_format(self) -> None:
        profile = _make_arch_profile()
        with (
            patch("archex.integrations.mcp.analyze", return_value=profile),
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=10000),
        ):
            result = handle_analyze_repo("/fake/repo", "markdown")
        import json

        parsed = json.loads(result)
        assert "# Architecture Profile" in parsed["content"]
        assert "_meta" in parsed

    def test_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="format must be one of"):
            handle_analyze_repo("/fake/repo", "xml")

    def test_resolves_local_path(self) -> None:
        profile = _make_arch_profile()
        with (
            patch("archex.integrations.mcp.analyze", return_value=profile) as mock_analyze,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=0),
        ):
            handle_analyze_repo("/some/local/path")
        call_args = mock_analyze.call_args[0]
        source = call_args[0]
        assert source.local_path == "/some/local/path"
        assert source.url is None

    def test_resolves_https_url(self) -> None:
        profile = _make_arch_profile()
        with (
            patch("archex.integrations.mcp.analyze", return_value=profile) as mock_analyze,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=0),
        ):
            handle_analyze_repo("https://github.com/example/repo")
        call_args = mock_analyze.call_args[0]
        source = call_args[0]
        assert source.url == "https://github.com/example/repo"
        assert source.local_path is None

    def test_resolves_http_url(self) -> None:
        profile = _make_arch_profile()
        with (
            patch("archex.integrations.mcp.analyze", return_value=profile) as mock_analyze,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=0),
        ):
            handle_analyze_repo("http://example.com/repo")
        call_args = mock_analyze.call_args[0]
        source = call_args[0]
        assert source.url == "http://example.com/repo"


class TestHandleQueryRepo:
    def test_returns_xml_prompt_with_meta(self) -> None:
        bundle = _make_context_bundle()
        with (
            patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            result = handle_query_repo("/fake/repo", "how does auth work?")
        mock_query.assert_called_once()
        assert isinstance(result, str)
        import json

        parsed = json.loads(result)
        assert "content" in parsed
        assert "_meta" in parsed
        assert parsed["_meta"]["tool_name"] == "query_repo"
        assert parsed["_meta"]["strategy"] == "bm25+graph"
        assert parsed["receipt"]["index_revision"] == "rev"
        assert parsed["receipt"]["context_complete"] == "complete"
        assert "<receipt" not in parsed["content"]

    def test_uses_adaptive_default_budget_when_omitted(self) -> None:
        bundle = _make_context_bundle()
        with (
            patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            handle_query_repo("/fake/repo", "what is the entry point?")
        assert mock_query.call_args.kwargs["token_budget"] == 8192
        assert mock_query.call_args.kwargs["explicit_token_budget"] is False

    def test_passes_explicit_token_budget_override(self) -> None:
        bundle = _make_context_bundle()
        with (
            patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            handle_query_repo("/fake/repo", "what is the entry point?", budget=4000)
        assert mock_query.call_args.kwargs["token_budget"] == 4000
        assert mock_query.call_args.kwargs["explicit_token_budget"] is True

    def test_query_savings_use_candidate_files_and_payload_tokens(self) -> None:
        chunk = CodeChunk(
            id="c1",
            content="def selected(): pass",
            file_path="selected.py",
            start_line=1,
            end_line=1,
            language="python",
            token_count=5,
        )
        bundle = ContextBundle(
            query="Where is selected defined?",
            chunks=[RankedChunk(chunk=chunk, final_score=1.0)],
            token_count=5,
            token_budget=2048,
            retrieval_metadata=RetrievalMetadata(
                seed_file_paths=["selected.py", "candidate.py"],
                expanded_file_paths=["neighbor.py"],
            ),
        )
        with (
            patch("archex.integrations.mcp.query", return_value=bundle),
            patch("archex.integrations.mcp.get_files_token_count", return_value=100) as raw_mock,
        ):
            output = handle_query_repo("/fake/repo", "Where is selected defined?")

        raw_mock.assert_called_once()
        assert raw_mock.call_args.args[1] == ["candidate.py", "neighbor.py", "selected.py"]
        parsed = json.loads(output)
        expected_tokens = count_tokens(render_xml(bundle)) - count_tokens(
            render_xml_envelope(bundle)
        )
        assert parsed["_meta"]["tokens_returned"] == expected_tokens
        assert parsed["_meta"]["savings_pct"] == round((1 - expected_tokens / 100) * 100, 1)

    def test_query_meta_reports_zero_savings_for_unknown_baseline(self) -> None:
        # A legacy index without per-file token totals yields a None baseline; the
        # meta must report 0% (not a bogus negative driven by a forced-1 denominator).
        bundle = _make_context_bundle()
        with (
            patch("archex.integrations.mcp.query", return_value=bundle),
            patch("archex.integrations.mcp.get_files_token_count", return_value=None),
        ):
            output = handle_query_repo("/fake/repo", "how does auth work?")
        parsed = json.loads(output)
        assert parsed["_meta"]["tokens_raw_equivalent"] == 0
        assert parsed["_meta"]["savings_pct"] == 0.0

    def test_query_savings_keep_type_definition_payload(self) -> None:
        content = "class Selected:\n    pass"
        chunk = CodeChunk(
            id="c1",
            content=content,
            file_path="selected.py",
            start_line=1,
            end_line=2,
            language="python",
            token_count=5,
            symbol_name="Selected",
        )
        bundle = ContextBundle(
            query="Where is Selected defined?",
            chunks=[RankedChunk(chunk=chunk, final_score=1.0)],
            type_definitions=[
                TypeDefinition(
                    symbol="Selected",
                    file_path="selected.py",
                    start_line=1,
                    end_line=2,
                    content=content,
                )
            ],
            token_count=5,
            token_budget=2048,
            retrieval_metadata=RetrievalMetadata(seed_file_paths=["selected.py"]),
        )
        with (
            patch("archex.integrations.mcp.query", return_value=bundle),
            patch("archex.integrations.mcp.get_files_token_count", return_value=100),
        ):
            output = handle_query_repo("/fake/repo", "Where is Selected defined?")

        import json

        parsed = json.loads(output)
        assert parsed["_meta"]["tokens_returned"] > bundle.token_count

    def test_rejects_empty_question(self) -> None:
        with pytest.raises(ValueError, match="question must not be empty"):
            handle_query_repo("/fake/repo", "   ")

    def test_rejects_nonpositive_budget(self) -> None:
        with pytest.raises(ValueError, match="budget must be positive"):
            handle_query_repo("/fake/repo", "question", budget=0)

    def test_rejects_negative_budget(self) -> None:
        with pytest.raises(ValueError, match="budget must be positive"):
            handle_query_repo("/fake/repo", "question", budget=-100)

    def test_resolves_source_from_url(self) -> None:
        bundle = _make_context_bundle()
        with (
            patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            handle_query_repo("https://github.com/example/repo", "question?")
        source = mock_query.call_args[0][0]
        assert source.url == "https://github.com/example/repo"

    def test_passes_runtime_through_to_query(self) -> None:
        from archex.serve.runtime import QueryRuntime

        bundle = _make_context_bundle()
        runtime = QueryRuntime()
        try:
            with (
                patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
                patch("archex.integrations.mcp.get_files_token_count", return_value=0),
            ):
                handle_query_repo("/fake/repo", "question?", runtime=runtime)
            assert mock_query.call_args.kwargs["runtime"] is runtime
        finally:
            runtime.close()

    def test_omits_runtime_by_default(self) -> None:
        bundle = _make_context_bundle()
        with (
            patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            handle_query_repo("/fake/repo", "question?")
        assert mock_query.call_args.kwargs["runtime"] is None

    def test_passes_valid_profile_string_through_to_query(self) -> None:
        from archex.models import RetrievalProfile

        bundle = _make_context_bundle()
        with (
            patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            handle_query_repo("/fake/repo", "question?", profile="fast")
        assert mock_query.call_args.kwargs["profile"] is RetrievalProfile.FAST

    def test_omits_profile_by_default(self) -> None:
        bundle = _make_context_bundle()
        with (
            patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            handle_query_repo("/fake/repo", "question?")
        assert mock_query.call_args.kwargs["profile"] is None

    def test_rejects_invalid_profile_string(self) -> None:
        with pytest.raises(ValueError, match="profile must be one of"):
            handle_query_repo("/fake/repo", "question?", profile="ultra")


class TestHandleContext:
    def test_returns_json_envelope_with_all_six_outputs(self) -> None:
        result = _make_context_result()
        with (
            patch("archex.integrations.mcp.context", return_value=result) as mock_context,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            output = handle_context("/fake/repo", "how does auth work?")
        mock_context.assert_called_once()
        parsed = json.loads(output)
        for key in (
            "content",
            "candidate_map",
            "fetch_handles",
            "relation_paths",
            "route",
            "receipt",
            "next_action",
            "_meta",
        ):
            assert key in parsed
        assert parsed["_meta"]["tool_name"] == "context"
        assert parsed["_meta"]["strategy"] == "context_facade"
        assert parsed["receipt"]["index_revision"] == "rev"
        assert parsed["route"]["resolved_intent"] == "general"
        assert parsed["next_action"] == "use_bundle"

    def test_markdown_format_returns_rendered_string_content(self) -> None:
        result = _make_context_result()
        with (
            patch("archex.integrations.mcp.context", return_value=result),
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            output = handle_context("/fake/repo", "how does auth work?", output_format="markdown")
        parsed = json.loads(output)
        assert isinstance(parsed["content"], str)
        assert "## Route" in parsed["content"]

    def test_rejects_blank_query(self) -> None:
        with pytest.raises(ValueError, match="query must not be empty"):
            handle_context("/fake/repo", "   ")

    def test_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="format must be one of"):
            handle_context("/fake/repo", "question?", output_format="xml")

    def test_rejects_invalid_intent(self) -> None:
        with pytest.raises(ValueError, match="invalid context request"):
            handle_context("/fake/repo", "question?", intent="not_a_real_intent")

    def test_rejects_invalid_profile(self) -> None:
        with pytest.raises(ValueError, match="invalid context request"):
            handle_context("/fake/repo", "question?", profile="ultra")

    def test_rejects_non_positive_budget(self) -> None:
        with pytest.raises(ValueError, match="invalid context request"):
            handle_context("/fake/repo", "question?", budgets={"token_budget": 0})

    def test_passes_request_fields_through_to_context(self) -> None:
        result = _make_context_result()
        with (
            patch("archex.integrations.mcp.context", return_value=result) as mock_context,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            handle_context(
                "/fake/repo",
                "question?",
                intent="debugging",
                profile="fast",
                filters={"include_paths": ["src/**"], "languages": ["python"]},
                budgets={"token_budget": 2048},
                handles=["chunk:a.py:1"],
            )
        request = mock_context.call_args.args[1]
        assert request.query == "question?"
        assert request.intent == QueryIntent.DEBUGGING
        assert request.filters.include_paths == ["src/**"]
        assert request.filters.languages == ["python"]
        assert request.budgets.token_budget == 2048
        assert request.handles == ["chunk:a.py:1"]

    def test_records_metrics_under_context_tool_name(self) -> None:
        result = _make_context_result()
        with (
            patch("archex.integrations.mcp.context", return_value=result),
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
            patch("archex.integrations.mcp.record_query_usage") as mock_record,
            patch("archex.integrations.mcp.resolve_metrics_policy") as mock_policy,
        ):
            mock_policy.return_value.metrics_enabled = True
            handle_context("/fake/repo", "how does auth work?")
        assert mock_record.call_args.kwargs["tool_name"] == "context"


class TestHandleScoutRepo:
    def test_returns_scout_markdown_with_meta(self) -> None:
        from archex.scout import ScoutBudget, ScoutFile, ScoutResult, file_handle

        result_model = ScoutResult(
            query="delta indexing",
            ranked_files=[
                ScoutFile(
                    path="src/archex/index/delta.py",
                    language="python",
                    lines=100,
                    symbol_count=4,
                    handle=file_handle("src/archex/index/delta.py"),
                )
            ],
            receipt=ContextReceipt(
                query="delta indexing",
                token_budget=ContextReceiptTokenBudget(requested=120, consumed=80),
                index_revision="rev",
                freshness=ContextFreshness.CLEAN,
                context_complete=ContextCompletenessStatus.COMPLETE,
                context_complete_reason=ContextCompletenessReason.COMPLETE,
                recommended_next_action=ContextRecommendedAction.USE_BUNDLE,
            ),
            budget=ScoutBudget(token_budget=120),
        )
        with (
            patch("archex.integrations.mcp.scout", return_value=result_model) as scout_mock,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=1000),
        ):
            output = handle_scout_repo(
                "/fake/repo",
                "delta indexing",
                budget=120,
                output_format="markdown",
            )

        parsed = json.loads(output)
        assert parsed["content"].startswith("# archex scout")
        assert file_handle("src/archex/index/delta.py") in parsed["content"]
        assert parsed["_meta"]["tool_name"] == "scout_repo"
        assert parsed["_meta"]["strategy"] == "scout"
        assert scout_mock.call_args.kwargs["token_budget"] == 120
        assert parsed["receipt"]["index_revision"] == "rev"

    def test_returns_scout_json_without_duplicate_receipt(self) -> None:
        from archex.scout import ScoutBudget, ScoutResult

        result_model = ScoutResult(
            query="delta indexing",
            budget=ScoutBudget(token_budget=120),
            receipt=ContextReceipt(
                query="delta indexing",
                token_budget=ContextReceiptTokenBudget(requested=120, consumed=80),
                index_revision="rev",
                freshness=ContextFreshness.CLEAN,
                context_complete=ContextCompletenessStatus.COMPLETE,
                context_complete_reason=ContextCompletenessReason.COMPLETE,
                recommended_next_action=ContextRecommendedAction.USE_BUNDLE,
            ),
        )
        with (
            patch("archex.integrations.mcp.scout", return_value=result_model),
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=1000),
        ):
            output = handle_scout_repo("/fake/repo", "delta indexing")

        parsed = json.loads(output)
        assert "receipt" not in parsed["content"]
        assert parsed["receipt"]["index_revision"] == "rev"

    def test_scout_json_always_uses_full_dump_regardless_of_m1_default(self) -> None:
        """M1 narrowed render_scout's default JSON dump; the MCP scout_repo tool
        must keep its pre-M1 contract (every field present, `None` included)."""
        from archex.models import SymbolKind
        from archex.scout import ScoutBudget, ScoutResult, ScoutSymbol, chunk_handle, file_handle

        result_model = ScoutResult(
            query="delta indexing",
            symbols=[
                ScoutSymbol(
                    name="run",
                    kind=SymbolKind.FUNCTION,
                    file_path="src/app.py",
                    start_line=1,
                    end_line=2,
                    chunk_id="c1",
                    file_handle=file_handle("src/app.py"),
                    chunk_handle=chunk_handle("c1"),
                )
            ],
            budget=ScoutBudget(token_budget=120),
        )
        with (
            patch("archex.integrations.mcp.scout", return_value=result_model),
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=1000),
        ):
            output = handle_scout_repo("/fake/repo", "delta indexing", output_format="json")

        parsed = json.loads(output)
        symbol = parsed["content"]["symbols"][0]
        assert symbol["signature"] is None
        assert symbol["visibility"] is None
        assert symbol["symbol_id"] is None


class TestHandleCompareRepos:
    def test_returns_json_with_meta(self) -> None:
        result = _make_comparison_result()
        with (
            patch("archex.integrations.mcp.compare", return_value=result) as mock_compare,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=5000),
        ):
            output = handle_compare_repos("/fake/a", "/fake/b")
        mock_compare.assert_called_once()
        import json

        parsed = json.loads(output)
        assert "content" in parsed
        assert "repo_a" in parsed["content"]
        assert "repo_b" in parsed["content"]
        assert "_meta" in parsed
        assert parsed["_meta"]["tool_name"] == "compare_repos"
        assert parsed["_meta"]["strategy"] == "full_comparison"
        assert parsed["_meta"]["tokens_raw_equivalent"] == 10000  # 5000 * 2 repos

    def test_passes_dimensions_list(self) -> None:
        result = _make_comparison_result()
        with (
            patch("archex.integrations.mcp.compare", return_value=result) as mock_compare,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=0),
        ):
            handle_compare_repos("/fake/a", "/fake/b", "api_surface,concurrency")
        call_kwargs = mock_compare.call_args[1]
        dims = call_kwargs.get("dimensions") or mock_compare.call_args[0][2]
        assert dims == ["api_surface", "concurrency"]

    def test_default_dimensions(self) -> None:
        result = _make_comparison_result()
        with (
            patch("archex.integrations.mcp.compare", return_value=result) as mock_compare,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=0),
        ):
            handle_compare_repos("/fake/a", "/fake/b")
        call_kwargs = mock_compare.call_args[1]
        dims = call_kwargs.get("dimensions") or mock_compare.call_args[0][2]
        assert "api_surface" in dims
        assert "error_handling" in dims

    def test_rejects_empty_dimensions(self) -> None:
        with pytest.raises(ValueError, match="dimensions must be a non-empty"):
            handle_compare_repos("/fake/a", "/fake/b", "  ,  ")

    def test_resolves_sources(self) -> None:
        result = _make_comparison_result()
        with (
            patch("archex.integrations.mcp.compare", return_value=result) as mock_compare,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=0),
        ):
            handle_compare_repos(
                "https://github.com/example/a",
                "/local/b",
                "api_surface",
            )
        source_a = mock_compare.call_args[0][0]
        source_b = mock_compare.call_args[0][1]
        assert source_a.url == "https://github.com/example/a"
        assert source_b.local_path == "/local/b"

    def test_validates_dimensions_valid(self) -> None:
        result = _make_comparison_result()
        with (
            patch("archex.integrations.mcp.compare", return_value=result) as mock_compare,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=5000),
        ):
            handle_compare_repos(
                "/fake/a",
                "/fake/b",
                "error_handling,api_surface,concurrency",
            )
        mock_compare.assert_called_once()

    def test_validates_dimensions_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dimensions"):
            handle_compare_repos(
                "/fake/a",
                "/fake/b",
                "invalid_dim,another_bad_dim",
            )

    def test_validates_dimensions_mixed_valid_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unsupported dimensions"):
            handle_compare_repos(
                "/fake/a",
                "/fake/b",
                "api_surface,nonexistent",
            )


# ---------------------------------------------------------------------------
# Server-level tests
# ---------------------------------------------------------------------------


class TestBuildServerImportError:
    def test_build_server_raises_import_error_when_mcp_missing(self) -> None:
        import builtins
        from typing import Any

        original_import: Any = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> object:
            if name.startswith("mcp"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(ImportError, match="mcp"),
        ):
            build_server()


class TestRunStdioServer:
    @pytest.mark.asyncio
    async def test_run_stdio_server_import_error(self) -> None:
        """run_stdio_server raises ImportError when mcp.server.stdio is missing."""
        import builtins
        from typing import Any

        original_import: Any = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> object:
            if name == "mcp.server.stdio":
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from archex.integrations.mcp import run_stdio_server

            with pytest.raises(ImportError, match="mcp"):
                await run_stdio_server()

    @pytest.mark.asyncio
    async def test_run_stdio_server_creates_and_closes_query_runtime(self) -> None:
        """One QueryRuntime lives for the server's lifetime and is closed on shutdown."""
        from collections.abc import AsyncIterator
        from contextlib import asynccontextmanager

        from archex.integrations.mcp import run_stdio_server
        from archex.serve.runtime import QueryRuntime

        captured: dict[str, object] = {}

        @asynccontextmanager
        async def fake_stdio_server() -> AsyncIterator[tuple[None, None]]:
            yield (None, None)

        class FakeServer:
            def create_initialization_options(self) -> object:
                return object()

            async def run(self, *args: object, **kwargs: object) -> None:
                return None

        def fake_build_server(runtime: QueryRuntime | None = None) -> object:
            captured["runtime"] = runtime
            return FakeServer()

        with (
            patch("archex.integrations.mcp.build_server", side_effect=fake_build_server),
            patch("mcp.server.stdio.stdio_server", fake_stdio_server),
            patch("archex.serve.runtime.QueryRuntime.close") as mock_close,
        ):
            await run_stdio_server()

        assert isinstance(captured["runtime"], QueryRuntime)
        mock_close.assert_called_once()


class TestBuildServer:
    def test_returns_server_instance(self) -> None:
        from mcp.server import Server

        server = build_server()
        assert isinstance(server, Server)
        assert server.name == "archex"

    def test_server_has_list_tools_handler(self) -> None:
        from mcp import types as mcp_types

        server = build_server()
        assert mcp_types.ListToolsRequest in server.request_handlers

    def test_server_has_call_tool_handler(self) -> None:
        from mcp import types as mcp_types

        server = build_server()
        assert mcp_types.CallToolRequest in server.request_handlers

    @pytest.mark.asyncio
    async def test_list_tools_returns_core_tools(self) -> None:
        server = build_server()
        # Call the registered list_tools handler directly
        from mcp import types as mcp_types

        handler = server.request_handlers[mcp_types.ListToolsRequest]
        req = mcp_types.ListToolsRequest(method="tools/list", params=None)
        server_result = await handler(req)
        result = server_result.root
        assert isinstance(result, mcp_types.ListToolsResult)
        tool_names = {t.name for t in result.tools}
        assert "analyze_repo" in tool_names
        assert "query_repo" in tool_names
        assert "compare_repos" in tool_names

        query_repo = next(tool for tool in result.tools if tool.name == "query_repo")
        budget_schema = query_repo.inputSchema["properties"]["budget"]
        assert "default" not in budget_schema
        assert "Omit to use adaptive intent routing" in budget_schema["description"]
        profile_schema = query_repo.inputSchema["properties"]["profile"]
        assert profile_schema["enum"] == ["fast", "balanced", "deep"]

    @pytest.mark.asyncio
    async def test_call_tool_analyze_repo(self) -> None:
        with patch("archex.integrations.mcp.handle_analyze_repo", return_value='{"repo": {}}'):
            server = build_server()
            from mcp import types as mcp_types

            handler = server.request_handlers[mcp_types.CallToolRequest]
            req = mcp_types.CallToolRequest(
                method="tools/call",
                params=mcp_types.CallToolRequestParams(
                    name="analyze_repo",
                    arguments={"repo_url": "/fake/repo"},
                ),
            )
            # Force list_tools to populate tool cache
            list_handler = server.request_handlers[mcp_types.ListToolsRequest]
            await list_handler(mcp_types.ListToolsRequest(method="tools/list", params=None))

            server_result = await handler(req)
            result = server_result.root
            assert isinstance(result, mcp_types.CallToolResult)
            assert len(result.content) == 1
            assert result.content[0].type == "text"

    @pytest.mark.asyncio
    async def test_call_tool_query_repo(self) -> None:
        with patch(
            "archex.integrations.mcp.handle_query_repo", return_value="<context>result</context>"
        ):
            server = build_server()
            from mcp import types as mcp_types

            handler = server.request_handlers[mcp_types.CallToolRequest]
            req = mcp_types.CallToolRequest(
                method="tools/call",
                params=mcp_types.CallToolRequestParams(
                    name="query_repo",
                    arguments={"repo_url": "/fake", "question": "what?", "budget": 4000},
                ),
            )
            list_handler = server.request_handlers[mcp_types.ListToolsRequest]
            await list_handler(mcp_types.ListToolsRequest(method="tools/list", params=None))

            server_result = await handler(req)
            result = server_result.root
            assert isinstance(result, mcp_types.CallToolResult)
            assert len(result.content) == 1
            assert result.content[0].type == "text"

    @pytest.mark.asyncio
    async def test_call_tool_compare_repos(self) -> None:
        mock_return = '{"repo_a": {}, "repo_b": {}}'
        with patch("archex.integrations.mcp.handle_compare_repos", return_value=mock_return):
            server = build_server()
            from mcp import types as mcp_types

            handler = server.request_handlers[mcp_types.CallToolRequest]
            req = mcp_types.CallToolRequest(
                method="tools/call",
                params=mcp_types.CallToolRequestParams(
                    name="compare_repos",
                    arguments={"repo_a": "/a", "repo_b": "/b", "dimensions": "api_surface"},
                ),
            )
            list_handler = server.request_handlers[mcp_types.ListToolsRequest]
            await list_handler(mcp_types.ListToolsRequest(method="tools/list", params=None))

            server_result = await handler(req)
            result = server_result.root
            assert isinstance(result, mcp_types.CallToolResult)
            assert len(result.content) == 1
            assert result.content[0].type == "text"

    @pytest.mark.asyncio
    async def test_call_tool_unknown_name_raises(self) -> None:
        server = build_server()
        from mcp import types as mcp_types

        handler = server.request_handlers[mcp_types.CallToolRequest]
        # Populate tool cache first
        list_handler = server.request_handlers[mcp_types.ListToolsRequest]
        await list_handler(mcp_types.ListToolsRequest(method="tools/list", params=None))

        req = mcp_types.CallToolRequest(
            method="tools/call",
            params=mcp_types.CallToolRequestParams(
                name="nonexistent_tool",
                arguments={},
            ),
        )
        # The MCP server converts unhandled exceptions to error results
        server_result = await handler(req)
        result = server_result.root
        # Should be an error result (isError=True) or ValidationError for bad tool name
        assert result.isError or isinstance(result, mcp_types.CallToolResult)

    @pytest.mark.asyncio
    async def test_list_tools_returns_graph_tools(self) -> None:
        server = build_server()
        from mcp import types as mcp_types

        handler = server.request_handlers[mcp_types.ListToolsRequest]
        req = mcp_types.ListToolsRequest(method="tools/list", params=None)
        server_result = await handler(req)
        result = server_result.root
        assert isinstance(result, mcp_types.ListToolsResult)
        assert len(result.tools) == 18
        tool_names = {t.name for t in result.tools}
        assert "scout_repo" in tool_names
        assert "get_file_tree" in tool_names
        assert "get_symbol" in tool_names
        assert "search_symbols" in tool_names
        assert "graph_neighbors" in tool_names
        assert "graph_path" in tool_names
        assert "graph_stats" in tool_names
        assert "get_impact" in tool_names
        assert "explain_target" in tool_names
        assert "generate_onboarding" in tool_names
        assert "context" in tool_names

    @pytest.mark.asyncio
    async def test_call_tool_query_repo_passes_explicit_runtime_to_handler(self) -> None:
        from archex.serve.runtime import QueryRuntime

        runtime = QueryRuntime()
        try:
            with patch(
                "archex.integrations.mcp.handle_query_repo", return_value="<context/>"
            ) as mock_handle:
                server = build_server(runtime=runtime)
                from mcp import types as mcp_types

                handler = server.request_handlers[mcp_types.CallToolRequest]
                list_handler = server.request_handlers[mcp_types.ListToolsRequest]
                await list_handler(mcp_types.ListToolsRequest(method="tools/list", params=None))
                req = mcp_types.CallToolRequest(
                    method="tools/call",
                    params=mcp_types.CallToolRequestParams(
                        name="query_repo",
                        arguments={"repo_url": "/fake", "question": "what?"},
                    ),
                )
                await handler(req)
            assert mock_handle.call_args[0][3] is runtime
        finally:
            runtime.close()

    @pytest.mark.asyncio
    async def test_call_tool_query_repo_default_runtime_is_query_runtime_instance(self) -> None:
        from archex.serve.runtime import QueryRuntime

        with patch(
            "archex.integrations.mcp.handle_query_repo", return_value="<context/>"
        ) as mock_handle:
            server = build_server()
            from mcp import types as mcp_types

            handler = server.request_handlers[mcp_types.CallToolRequest]
            list_handler = server.request_handlers[mcp_types.ListToolsRequest]
            await list_handler(mcp_types.ListToolsRequest(method="tools/list", params=None))
            req = mcp_types.CallToolRequest(
                method="tools/call",
                params=mcp_types.CallToolRequestParams(
                    name="query_repo",
                    arguments={"repo_url": "/fake", "question": "what?"},
                ),
            )
            await handler(req)
        passed_runtime = mock_handle.call_args[0][3]
        assert isinstance(passed_runtime, QueryRuntime)

    @pytest.mark.asyncio
    async def test_query_repo_reuses_warm_snapshot_across_calls(
        self, python_simple_repo: Path
    ) -> None:
        """End-to-end (unmocked): a second query_repo call against the same repo
        through one server-owned runtime reuses the exact same warm snapshot."""
        from archex.api import _cache_manager_for_source  # pyright: ignore[reportPrivateUsage]
        from archex.models import Config, RepoSource
        from archex.serve.runtime import QueryRuntime

        runtime = QueryRuntime()
        try:
            server = build_server(runtime=runtime)
            from mcp import types as mcp_types

            handler = server.request_handlers[mcp_types.CallToolRequest]
            list_handler = server.request_handlers[mcp_types.ListToolsRequest]
            await list_handler(mcp_types.ListToolsRequest(method="tools/list", params=None))

            def _call(question: str) -> mcp_types.CallToolRequest:
                return mcp_types.CallToolRequest(
                    method="tools/call",
                    params=mcp_types.CallToolRequestParams(
                        name="query_repo",
                        arguments={
                            "repo_url": str(python_simple_repo),
                            "question": question,
                            "budget": 100,
                        },
                    ),
                )

            source = RepoSource(local_path=str(python_simple_repo))
            cache_key = _cache_manager_for_source(source, Config()).cache_key(source)

            await handler(_call("calculate_sum"))
            first_snapshot = runtime._snapshots.get(cache_key)  # pyright: ignore[reportPrivateUsage]
            assert first_snapshot is not None

            await handler(_call("models"))
            second_snapshot = runtime._snapshots.get(cache_key)  # pyright: ignore[reportPrivateUsage]

            assert second_snapshot is first_snapshot
        finally:
            runtime.close()


# ---------------------------------------------------------------------------
# Precision Symbol Tool handler tests
# ---------------------------------------------------------------------------

from archex.integrations.mcp import (  # noqa: E402
    handle_get_file_outline,
    handle_get_file_tree,
    handle_get_symbol,
    handle_get_symbols_batch,
    handle_search_symbols,
)
from archex.models import (  # noqa: E402
    FileOutline,
    FileTree,
    SymbolKind,
    SymbolMatch,
    SymbolSource,
)


class TestHandleGetFileTree:
    def test_returns_json_with_meta(self) -> None:
        tree = FileTree(root="/repo", entries=[], total_files=0, languages={})
        with (
            patch("archex.integrations.mcp.file_tree", return_value=tree),
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=5000),
        ):
            result = handle_get_file_tree("/fake/repo")
        import json

        parsed = json.loads(result)
        assert parsed["content"]["root"] == "/repo"
        assert "_meta" in parsed
        assert parsed["_meta"]["tool_name"] == "get_file_tree"
        assert parsed["_meta"]["strategy"] == "file_tree"
        assert parsed["_meta"]["tokens_raw_equivalent"] == 5000
        # PipelineTiming integration: query_time_ms populated
        assert isinstance(parsed["_meta"]["query_time_ms"], float)

    def test_passes_params(self) -> None:
        tree = FileTree(root="/repo", entries=[], total_files=0, languages={})
        with (
            patch("archex.integrations.mcp.file_tree", return_value=tree) as mock,
            patch("archex.integrations.mcp.get_repo_total_tokens", return_value=0),
        ):
            handle_get_file_tree("/fake", max_depth=3, language="python")
        call_kwargs = mock.call_args[1]
        assert call_kwargs["max_depth"] == 3
        assert call_kwargs["language"] == "python"


class TestHandleGetFileOutline:
    def test_returns_json_with_meta(self) -> None:
        outline = FileOutline(
            file_path="f.py",
            language="python",
            lines=10,
            symbols=[],
            token_count_raw=100,
        )
        with patch("archex.integrations.mcp.file_outline", return_value=outline):
            result = handle_get_file_outline("/fake", "f.py")
        import json

        parsed = json.loads(result)
        assert parsed["content"]["file_path"] == "f.py"
        assert "_meta" in parsed
        assert parsed["_meta"]["tool_name"] == "get_file_outline"
        assert parsed["_meta"]["tokens_raw_equivalent"] == 100


class TestHandleSearchSymbols:
    def test_returns_json_with_meta(self) -> None:
        match = SymbolMatch(
            symbol_id="f.py::foo#function",
            name="foo",
            kind=SymbolKind.FUNCTION,
            file_path="f.py",
            start_line=1,
        )
        with (
            patch("archex.integrations.mcp.search_symbols", return_value=[match]),
            patch("archex.integrations.mcp.get_files_token_count", return_value=3000),
        ):
            result = handle_search_symbols("/fake", "foo")
        import json

        parsed = json.loads(result)
        assert isinstance(parsed["content"], list)
        assert len(parsed["content"]) == 1
        assert parsed["content"][0]["name"] == "foo"
        assert "_meta" in parsed
        assert parsed["_meta"]["tool_name"] == "search_symbols"


class TestHandleGetSymbol:
    def test_returns_symbol_json_with_meta(self) -> None:
        sym = SymbolSource(
            symbol_id="f.py::foo#function",
            name="foo",
            kind=SymbolKind.FUNCTION,
            file_path="f.py",
            start_line=1,
            end_line=3,
            source="def foo(): pass",
        )
        with (
            patch("archex.integrations.mcp.get_symbol", return_value=sym),
            patch("archex.integrations.mcp.get_file_token_count", return_value=2000),
        ):
            result = handle_get_symbol("/fake", "f.py::foo#function")
        import json

        parsed = json.loads(result)
        assert parsed["content"]["source"] == "def foo(): pass"
        assert "_meta" in parsed
        assert parsed["_meta"]["tool_name"] == "get_symbol"
        assert parsed["_meta"]["strategy"] == "symbol_lookup"
        assert parsed["_meta"]["tokens_raw_equivalent"] == 2000

    def test_returns_error_for_not_found(self) -> None:
        with patch("archex.integrations.mcp.get_symbol", return_value=None):
            result = handle_get_symbol("/fake", "nonexistent")
        import json

        parsed = json.loads(result)
        assert "error" in parsed
        # Not-found responses don't include _meta
        assert "_meta" not in parsed


class TestHandleGetSymbolsBatch:
    def test_returns_json_with_meta(self) -> None:
        sym = SymbolSource(
            symbol_id="f.py::foo#function",
            name="foo",
            kind=SymbolKind.FUNCTION,
            file_path="f.py",
            start_line=1,
            end_line=3,
            source="def foo(): pass",
        )
        with (
            patch("archex.integrations.mcp.get_symbols_batch", return_value=[sym, None]),
            patch("archex.integrations.mcp.get_files_token_count", return_value=4000),
        ):
            result = handle_get_symbols_batch("/fake", ["f.py::foo#function", "missing"])
        import json

        parsed = json.loads(result)
        assert len(parsed["content"]) == 2
        assert parsed["content"][0]["name"] == "foo"
        assert parsed["content"][1] is None
        assert "_meta" in parsed
        assert parsed["_meta"]["tool_name"] == "get_symbols_batch"

    def test_rejects_too_many_ids(self) -> None:
        with pytest.raises(ValueError, match="at most 50"):
            handle_get_symbols_batch("/fake", ["id"] * 51)


def _write_mcp_graph_artifact(tmp_path: Path) -> Path:
    app = file_node_id("pkg/app.py")
    models = file_node_id("pkg/models.py")
    db = file_node_id("pkg/db.py")
    hub = file_node_id("pkg/hub.py")
    graph = ArchGraph(
        project=GraphProject(name="mcp-graph", languages={"python": 4}, total_files=4),
        metadata=GraphExportMetadata(archex_version="0.8.0"),
        nodes=[
            GraphNode(id=app, type=GraphNodeType.FILE, label="app.py", path="pkg/app.py"),
            GraphNode(id=models, type=GraphNodeType.FILE, label="models.py", path="pkg/models.py"),
            GraphNode(id=db, type=GraphNodeType.FILE, label="db.py", path="pkg/db.py"),
            GraphNode(id=hub, type=GraphNodeType.FILE, label="hub.py", path="pkg/hub.py"),
        ],
        edges=[
            GraphEdge(
                source=app,
                target=models,
                type=GraphEdgeType.IMPORTS,
                location="pkg/app.py:1",
                confidence=EdgeConfidence.HEURISTIC,
                confidence_score=0.75,
                evidence=["fallback resolution"],
            ),
            GraphEdge(source=models, target=db, type=GraphEdgeType.IMPORTS),
            GraphEdge(source=hub, target=app, type=GraphEdgeType.IMPORTS),
            GraphEdge(source=hub, target=models, type=GraphEdgeType.IMPORTS),
            GraphEdge(source=hub, target=db, type=GraphEdgeType.IMPORTS),
        ],
    )
    artifact = tmp_path / "archgraph.json"
    artifact.write_text(graph.to_json(), encoding="utf-8")
    clear_graph_query_cache()
    return artifact


class TestGraphQueryHandlers:
    def test_neighbors_returns_confidence_and_evidence_without_reindexing(
        self, tmp_path: Path
    ) -> None:
        artifact = _write_mcp_graph_artifact(tmp_path)

        with patch("archex.integrations.mcp.index_repository", side_effect=AssertionError):
            result = handle_graph_neighbors(str(artifact), "pkg/app.py")

        parsed = json.loads(result)
        edge = parsed["content"]["edges"][0]
        assert edge["type"] == "imports"
        assert edge["confidence"] == "heuristic"
        assert edge["confidence_score"] == 0.75
        assert edge["evidence"] == ["fallback resolution"]
        assert edge["source"]["path"] == "pkg/app.py"
        assert parsed["_meta"]["tool_name"] == "graph_neighbors"

    def test_missing_node_returns_error(self, tmp_path: Path) -> None:
        artifact = _write_mcp_graph_artifact(tmp_path)

        result = handle_graph_neighbors(str(artifact), "missing.py")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "No graph node matches" in parsed["error"]

    def test_hubs_reports_high_degree_nodes(self, tmp_path: Path) -> None:
        artifact = _write_mcp_graph_artifact(tmp_path)

        result = handle_graph_hubs(str(artifact), threshold=3)

        parsed = json.loads(result)
        assert parsed["content"]["hubs"][0]["path"] == "pkg/hub.py"
        assert parsed["content"]["hubs"][0]["degree"] == 3

    def test_artifact_handle_is_reused_across_calls(self, tmp_path: Path) -> None:
        artifact = _write_mcp_graph_artifact(tmp_path)
        original = mcp_integration.GraphQuery.from_artifact

        with patch.object(mcp_integration.GraphQuery, "from_artifact", wraps=original) as loader:
            handle_graph_stats(str(artifact))
            handle_graph_stats(str(artifact))

        assert loader.call_count == 1

    def test_markdown_output_respects_token_budget(self, tmp_path: Path) -> None:
        artifact = _write_mcp_graph_artifact(tmp_path)

        result = handle_graph_neighbors(
            str(artifact),
            "pkg/hub.py",
            output_format="markdown",
            token_budget=20,
        )

        parsed = json.loads(result)
        assert "[truncated: token budget reached]" in parsed["content"]
        assert parsed["_meta"]["token_budget"] == 20
        assert parsed["_meta"]["token_budget_truncated"] is True

    def test_path_handler_returns_structural_path(self, tmp_path: Path) -> None:
        artifact = _write_mcp_graph_artifact(tmp_path)

        result = handle_graph_path(str(artifact), "pkg/app.py", "pkg/db.py")

        parsed = json.loads(result)
        assert parsed["content"]["found"] is True
        assert [node["path"] for node in parsed["content"]["nodes"]] == [
            "pkg/app.py",
            "pkg/models.py",
            "pkg/db.py",
        ]

    def test_reexported_artifact_busts_stale_cache(self, tmp_path: Path) -> None:
        """A long-running MCP daemon must not keep serving a pre-re-export snapshot.

        `archex graph export` writes to a fixed, repeatedly-overwritten path; the
        cached GraphQuery handle must be keyed on more than the path string or a
        daemon that already answered a query for this path never sees the update.
        """
        artifact = _write_mcp_graph_artifact(tmp_path)
        first = json.loads(handle_graph_stats(str(artifact)))
        assert first["content"]["nodes"] == 4

        solo = file_node_id("solo.py")
        smaller = ArchGraph(
            project=GraphProject(name="mcp-graph", languages={"python": 1}, total_files=1),
            metadata=GraphExportMetadata(archex_version="0.8.0"),
            nodes=[
                GraphNode(id=solo, type=GraphNodeType.FILE, label="solo.py", path="solo.py"),
            ],
            edges=[],
        )
        artifact.write_text(smaller.to_json(), encoding="utf-8")
        # Force the mtime strictly forward so the write is observable even on
        # filesystems/CI runners with coarse mtime resolution.
        newer = time.time() + 5
        os.utime(artifact, (newer, newer))

        second = json.loads(handle_graph_stats(str(artifact)))
        assert second["content"]["nodes"] == 1

    def test_reexport_replaces_stale_entry_without_accumulating(self, tmp_path: Path) -> None:
        """Repeated re-exports of one path must replace, not accumulate, cache slots.

        A composite (path, hub_degree, token) cache key would let every re-export
        of the same logical artifact occupy a fresh, never-superseded slot instead
        of replacing the stale one for that (path, hub_degree) pair.
        """
        artifact = _write_mcp_graph_artifact(tmp_path)
        handle_graph_stats(str(artifact))
        assert len(mcp_integration._graph_query_cache) == 1  # pyright: ignore[reportPrivateUsage]

        for i in range(4):
            solo = file_node_id(f"solo{i}.py")
            graph = ArchGraph(
                project=GraphProject(name="mcp-graph", languages={"python": 1}, total_files=1),
                metadata=GraphExportMetadata(archex_version="0.8.0"),
                nodes=[
                    GraphNode(
                        id=solo, type=GraphNodeType.FILE, label=f"solo{i}.py", path=f"solo{i}.py"
                    ),
                ],
                edges=[],
            )
            artifact.write_text(graph.to_json(), encoding="utf-8")
            newer = time.time() + 5 + i
            os.utime(artifact, (newer, newer))
            handle_graph_stats(str(artifact))

        assert len(mcp_integration._graph_query_cache) == 1  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# get_impact handler tests
# ---------------------------------------------------------------------------


class TestHandleGetImpact:
    def test_matches_cli_output_for_explicit_changed_file(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        cli_result = runner.invoke(
            cli,
            [
                "impact",
                str(python_simple_repo),
                "--changed-file",
                "utils.py",
                "--format",
                "json",
            ],
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_data = json.loads(cli_result.output)

        mcp_result = handle_get_impact(
            str(python_simple_repo), changed_files=["utils.py"], output_format="json"
        )
        envelope = json.loads(mcp_result)
        mcp_data = json.loads(envelope["content"])

        assert mcp_data == cli_data
        assert envelope["_meta"]["tool_name"] == "get_impact"
        assert envelope["_meta"]["strategy"] == "impact_analysis"

    def test_matches_cli_output_for_git_diff_markdown(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        cli_result = runner.invoke(cli, ["impact", str(python_simple_repo), "--base", "HEAD"])
        assert cli_result.exit_code == 0, cli_result.output

        mcp_result = handle_get_impact(
            str(python_simple_repo), base="HEAD", output_format="markdown"
        )
        envelope = json.loads(mcp_result)

        assert envelope["content"] == cli_result.output

    def test_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            handle_get_impact("/fake/repo", output_format="xml")

    def test_git_diff_mode_requires_local_repo_url(self) -> None:
        from archex.impact import ImpactError

        with pytest.raises(ImpactError, match="requires changed_files"):
            handle_get_impact("https://example.com/repo.git")

    def test_matches_cli_output_for_diff_json(self, impact_diff_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        hub = impact_diff_repo / "hub.py"
        hub.write_text(hub.read_text().replace("value * 2", "value * 3"))

        runner = CliRunner()
        cli_result = runner.invoke(
            cli,
            ["impact", str(impact_diff_repo), "--diff", "HEAD", "--format", "json"],
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_data = json.loads(cli_result.output)

        mcp_result = handle_get_impact(str(impact_diff_repo), diff_ref="HEAD", output_format="json")
        envelope = json.loads(mcp_result)
        mcp_data = json.loads(envelope["content"])

        assert mcp_data == cli_data
        assert mcp_data["diff_ref"] == "HEAD"
        assert mcp_data["affected_symbols"][0]["level"] == "high"

    def test_diff_ref_rejects_changed_files_combination(self) -> None:
        from archex.impact import ImpactError

        with pytest.raises(ImpactError, match="cannot be combined with changed_files"):
            handle_get_impact("/fake/repo", changed_files=["a.py"], diff_ref="HEAD")

    def test_without_diff_ref_output_has_no_diff_fields(self, python_simple_repo: Path) -> None:
        mcp_result = handle_get_impact(
            str(python_simple_repo), changed_files=["utils.py"], output_format="json"
        )
        envelope = json.loads(mcp_result)
        mcp_data = json.loads(envelope["content"])

        assert "diff_ref" not in mcp_data
        assert "affected_symbols" not in mcp_data


# ---------------------------------------------------------------------------
# explain_target handler tests
# ---------------------------------------------------------------------------


class TestHandleExplainTarget:
    def test_matches_cli_output_for_file_target(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        cli_result = runner.invoke(
            cli, ["explain", str(python_simple_repo), "main.py", "--format", "json"]
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_data = json.loads(cli_result.output)

        mcp_result = handle_explain_target(
            str(python_simple_repo), target="main.py", output_format="json"
        )
        envelope = json.loads(mcp_result)
        mcp_data = json.loads(envelope["content"])

        assert mcp_data == cli_data
        assert envelope["_meta"]["tool_name"] == "explain_target"
        assert envelope["_meta"]["strategy"] == "explain_context"

    def test_matches_cli_output_for_symbol_target(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        cli_result = runner.invoke(
            cli,
            [
                "explain",
                str(python_simple_repo),
                "main.py::run#function",
                "--format",
                "json",
            ],
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_data = json.loads(cli_result.output)

        mcp_result = handle_explain_target(
            str(python_simple_repo), target="main.py::run#function", output_format="json"
        )
        mcp_data = json.loads(json.loads(mcp_result)["content"])

        assert mcp_data == cli_data

    def test_matches_cli_output_for_module_target(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        cli_result = runner.invoke(
            cli,
            ["explain", str(python_simple_repo), "--module", "services", "--format", "json"],
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_data = json.loads(cli_result.output)

        mcp_result = handle_explain_target(
            str(python_simple_repo), module_name="services", output_format="json"
        )
        mcp_data = json.loads(json.loads(mcp_result)["content"])

        assert mcp_data == cli_data

    def test_matches_cli_output_for_markdown_format(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        cli_result = runner.invoke(cli, ["explain", str(python_simple_repo), "main.py"])
        assert cli_result.exit_code == 0, cli_result.output

        mcp_result = handle_explain_target(
            str(python_simple_repo), target="main.py", output_format="markdown"
        )
        envelope = json.loads(mcp_result)

        assert envelope["content"] == cli_result.output

    def test_matches_cli_output_for_graph_artifact_mode(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        graph_path = tmp_path / "graph.json"
        export_result = runner.invoke(
            cli, ["graph", "export", str(python_simple_repo), "--output", str(graph_path)]
        )
        assert export_result.exit_code == 0, export_result.output

        cli_result = runner.invoke(
            cli,
            [
                "explain",
                "ignored",
                "main.py",
                "--graph",
                str(graph_path),
                "--format",
                "json",
            ],
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_data = json.loads(cli_result.output)

        mcp_result = handle_explain_target(
            target="main.py", graph_path=str(graph_path), output_format="json"
        )
        envelope = json.loads(mcp_result)
        mcp_data = json.loads(envelope["content"])

        assert mcp_data == cli_data

    def test_rejects_both_target_and_module(self) -> None:
        with pytest.raises(ExplainError, match="not both"):
            handle_explain_target("/fake/repo", target="a.py", module_name="a")

    def test_rejects_neither_target_nor_module(self) -> None:
        with pytest.raises(ExplainError, match="requires target or module_name"):
            handle_explain_target("/fake/repo")

    def test_requires_repo_url_without_graph_path(self) -> None:
        with pytest.raises(ExplainError, match="requires repo_url"):
            handle_explain_target(target="a.py")

    def test_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            handle_explain_target("/fake/repo", target="a.py", output_format="xml")


# ---------------------------------------------------------------------------
# generate_onboarding handler tests
# ---------------------------------------------------------------------------


class TestHandleGenerateOnboarding:
    def test_matches_cli_output_for_indexed_repo(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        cli_result = runner.invoke(cli, ["onboard", str(python_simple_repo), "--max-files", "5"])
        assert cli_result.exit_code == 0, cli_result.output

        mcp_result = handle_generate_onboarding(str(python_simple_repo), max_files=5)
        envelope = json.loads(mcp_result)

        assert envelope["content"] == cli_result.output
        assert envelope["_meta"]["tool_name"] == "generate_onboarding"
        assert envelope["_meta"]["strategy"] == "onboarding_guide"

    def test_matches_cli_output_for_graph_artifact_mode(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        graph_path = tmp_path / "graph.json"
        export_result = runner.invoke(
            cli, ["graph", "export", str(python_simple_repo), "--output", str(graph_path)]
        )
        assert export_result.exit_code == 0, export_result.output

        cli_result = runner.invoke(
            cli, ["onboard", "ignored", "--graph", str(graph_path), "--max-files", "5"]
        )
        assert cli_result.exit_code == 0, cli_result.output

        mcp_result = handle_generate_onboarding(graph_path=str(graph_path), max_files=5)
        envelope = json.loads(mcp_result)

        assert envelope["content"] == cli_result.output

    def test_requires_repo_url_without_graph_path(self) -> None:
        from archex.onboarding import OnboardingError

        with pytest.raises(OnboardingError, match="requires repo_url"):
            handle_generate_onboarding()

    def test_rejects_non_positive_max_files(self, python_simple_repo: Path) -> None:
        from archex.onboarding import OnboardingError

        with pytest.raises(OnboardingError, match="max-files must be greater than zero"):
            handle_generate_onboarding(str(python_simple_repo), max_files=0)


class TestHandleContextEndToEnd:
    """Unmocked context() facade exercised through the real MCP dispatch path."""

    def test_matches_cli_output_for_json(self, python_simple_repo: Path) -> None:
        from click.testing import CliRunner

        from archex.cli.main import cli

        runner = CliRunner()
        cli_result = runner.invoke(
            cli, ["context", str(python_simple_repo), "how does authentication work?"]
        )
        assert cli_result.exit_code == 0, cli_result.output
        cli_data = json.loads(cli_result.output)

        mcp_result = handle_context(str(python_simple_repo), "how does authentication work?")
        envelope = json.loads(mcp_result)

        assert envelope["fetch_handles"] == cli_data["fetch_handles"]
        assert [item["handle"] for item in envelope["candidate_map"]] == [
            item["handle"] for item in cli_data["candidate_map"]
        ]
        assert envelope["route"]["resolved_intent"] == cli_data["route"]["resolved_intent"]

    @pytest.mark.asyncio
    async def test_stable_handle_round_trip_through_call_tool_dispatch(
        self, python_simple_repo: Path
    ) -> None:
        """A handle returned by one `context` MCP call fetches the exact same
        chunk on a second `context` call routed through the real dispatch
        path — no broad search re-run."""
        server = build_server()
        from mcp import types as mcp_types

        handler = server.request_handlers[mcp_types.CallToolRequest]
        list_handler = server.request_handlers[mcp_types.ListToolsRequest]
        await list_handler(mcp_types.ListToolsRequest(method="tools/list", params=None))

        def _call(query: str, handles: list[str] | None = None) -> mcp_types.CallToolRequest:
            arguments: dict[str, object] = {
                "repo_url": str(python_simple_repo),
                "query": query,
            }
            if handles is not None:
                arguments["handles"] = handles
            return mcp_types.CallToolRequest(
                method="tools/call",
                params=mcp_types.CallToolRequestParams(name="context", arguments=arguments),
            )

        first = (await handler(_call("how does authentication work?"))).root
        assert isinstance(first, mcp_types.CallToolResult)
        first_envelope = json.loads(first.content[0].text)  # type: ignore[union-attr]
        handle = first_envelope["fetch_handles"][0]
        expected_chunk = next(
            item for item in first_envelope["content"] if item["chunk"]["id"] in handle
        )

        second = (await handler(_call("ignored query text", handles=[handle]))).root
        assert isinstance(second, mcp_types.CallToolResult)
        second_envelope = json.loads(second.content[0].text)  # type: ignore[union-attr]

        assert second_envelope["route"]["handles_mode"] is True
        assert second_envelope["fetch_handles"] == [handle]
        assert len(second_envelope["content"]) == 1
        assert (
            second_envelope["content"][0]["chunk"]["content"] == expected_chunk["chunk"]["content"]
        )
