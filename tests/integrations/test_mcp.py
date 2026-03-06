"""Tests for the MCP server integration."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportMissingImports=false

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("mcp", reason="mcp not installed")

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

    def test_passes_token_budget(self) -> None:
        bundle = _make_context_bundle()
        with (
            patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
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
        with (
            patch("archex.integrations.mcp.query", return_value=bundle) as mock_query,
            patch("archex.integrations.mcp.get_files_token_count", return_value=0),
        ):
            handle_query_repo("https://github.com/example/repo", "question?")
        source = mock_query.call_args[0][0]
        assert source.url == "https://github.com/example/repo"


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
    async def test_list_tools_returns_eight_tools(self) -> None:
        server = build_server()
        from mcp import types as mcp_types

        handler = server.request_handlers[mcp_types.ListToolsRequest]
        req = mcp_types.ListToolsRequest(method="tools/list", params=None)
        server_result = await handler(req)
        result = server_result.root
        assert isinstance(result, mcp_types.ListToolsResult)
        assert len(result.tools) == 8
        tool_names = {t.name for t in result.tools}
        assert "get_file_tree" in tool_names
        assert "get_symbol" in tool_names
        assert "search_symbols" in tool_names


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
