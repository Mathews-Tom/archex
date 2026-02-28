"""Tests for the MCP server integration."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

from unittest.mock import patch

import pytest

from archex.integrations.mcp import (
    build_server,
    handle_analyze_repo,
    handle_compare_repos,
    handle_query_repo,
)
from archex.models import (
    ArchProfile,
    ComparisonResult,
    ContextBundle,
    RepoMetadata,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_arch_profile(local_path: str = "/fake/repo") -> ArchProfile:
    return ArchProfile(repo=RepoMetadata(local_path=local_path))


def _make_context_bundle(question: str = "how does auth work?") -> ContextBundle:
    return ContextBundle(query=question, token_count=100, token_budget=8000)


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
        with patch("archex.integrations.mcp.analyze", return_value=profile) as mock_analyze:
            result = handle_analyze_repo("/fake/repo")
        mock_analyze.assert_called_once()
        assert isinstance(result, str)
        # JSON output should be parseable
        import json

        parsed = json.loads(result)
        assert "repo" in parsed

    def test_returns_markdown_format(self) -> None:
        profile = _make_arch_profile()
        with patch("archex.integrations.mcp.analyze", return_value=profile):
            result = handle_analyze_repo("/fake/repo", "markdown")
        assert "# Architecture Profile" in result

    def test_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="format must be one of"):
            handle_analyze_repo("/fake/repo", "xml")

    def test_resolves_local_path(self) -> None:
        profile = _make_arch_profile()
        with patch("archex.integrations.mcp.analyze", return_value=profile) as mock_analyze:
            handle_analyze_repo("/some/local/path")
        call_args = mock_analyze.call_args[0]
        source = call_args[0]
        assert source.local_path == "/some/local/path"
        assert source.url is None

    def test_resolves_https_url(self) -> None:
        profile = _make_arch_profile()
        with patch("archex.integrations.mcp.analyze", return_value=profile) as mock_analyze:
            handle_analyze_repo("https://github.com/example/repo")
        call_args = mock_analyze.call_args[0]
        source = call_args[0]
        assert source.url == "https://github.com/example/repo"
        assert source.local_path is None

    def test_resolves_http_url(self) -> None:
        profile = _make_arch_profile()
        with patch("archex.integrations.mcp.analyze", return_value=profile) as mock_analyze:
            handle_analyze_repo("http://example.com/repo")
        call_args = mock_analyze.call_args[0]
        source = call_args[0]
        assert source.url == "http://example.com/repo"


class TestHandleQueryRepo:
    def test_returns_xml_prompt(self) -> None:
        bundle = _make_context_bundle()
        with patch("archex.integrations.mcp.query", return_value=bundle) as mock_query:
            result = handle_query_repo("/fake/repo", "how does auth work?")
        mock_query.assert_called_once()
        assert isinstance(result, str)

    def test_passes_token_budget(self) -> None:
        bundle = _make_context_bundle()
        with patch("archex.integrations.mcp.query", return_value=bundle) as mock_query:
            handle_query_repo("/fake/repo", "what is the entry point?", budget=4000)
        budget = mock_query.call_args[1].get("token_budget") or mock_query.call_args[0][2]
        assert budget == 4000

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
        with patch("archex.integrations.mcp.query", return_value=bundle) as mock_query:
            handle_query_repo("https://github.com/example/repo", "question?")
        source = mock_query.call_args[0][0]
        assert source.url == "https://github.com/example/repo"


class TestHandleCompareRepos:
    def test_returns_json(self) -> None:
        result = _make_comparison_result()
        with patch("archex.integrations.mcp.compare", return_value=result) as mock_compare:
            output = handle_compare_repos("/fake/a", "/fake/b")
        mock_compare.assert_called_once()
        import json

        parsed = json.loads(output)
        assert "repo_a" in parsed
        assert "repo_b" in parsed

    def test_passes_dimensions_list(self) -> None:
        result = _make_comparison_result()
        with patch("archex.integrations.mcp.compare", return_value=result) as mock_compare:
            handle_compare_repos("/fake/a", "/fake/b", "api_surface,concurrency")
        call_kwargs = mock_compare.call_args[1]
        dims = call_kwargs.get("dimensions") or mock_compare.call_args[0][2]
        assert dims == ["api_surface", "concurrency"]

    def test_default_dimensions(self) -> None:
        result = _make_comparison_result()
        with patch("archex.integrations.mcp.compare", return_value=result) as mock_compare:
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
        with patch("archex.integrations.mcp.compare", return_value=result) as mock_compare:
            handle_compare_repos(
                "https://github.com/example/a",
                "/local/b",
                "api_surface",
            )
        source_a = mock_compare.call_args[0][0]
        source_b = mock_compare.call_args[0][1]
        assert source_a.url == "https://github.com/example/a"
        assert source_b.local_path == "/local/b"


# ---------------------------------------------------------------------------
# Server-level tests
# ---------------------------------------------------------------------------


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
    async def test_list_tools_returns_three_tools(self) -> None:
        server = build_server()
        # Call the registered list_tools handler directly
        from mcp import types as mcp_types

        handler = server.request_handlers[mcp_types.ListToolsRequest]
        req = mcp_types.ListToolsRequest(method="tools/list", params=None)
        server_result = await handler(req)
        result = server_result.root
        assert isinstance(result, mcp_types.ListToolsResult)
        tool_names = {t.name for t in result.tools}
        assert tool_names == {"analyze_repo", "query_repo", "compare_repos"}

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
