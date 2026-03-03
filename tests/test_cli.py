from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from archex import __version__
from archex.cli.main import cli

if TYPE_CHECKING:
    from pathlib import Path


def test_version() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"archex, version {__version__}" in result.output


def test_help_contains_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    output = result.output
    assert "analyze" in output
    assert "query" in output
    assert "compare" in output
    assert "cache" in output


def test_analyze_local_json(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", str(python_simple_repo), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "repo" in data
    assert "stats" in data
    assert "interface_surface" in data


def test_analyze_local_markdown(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["analyze", str(python_simple_repo), "--format", "markdown"])
    assert result.exit_code == 0, result.output
    output = result.output
    assert "# Architecture Profile" in output
    assert "## Stats" in output


def test_analyze_error_handling() -> None:
    from unittest.mock import patch

    from archex.exceptions import ArchexError

    runner = CliRunner()
    with patch("archex.cli.analyze_cmd.analyze", side_effect=ArchexError("Test error")):
        result = runner.invoke(cli, ["analyze", "/fake/repo"])
    assert result.exit_code != 0
    assert "Test error" in result.output


def test_query_error_handling(python_simple_repo: Path) -> None:
    from unittest.mock import patch

    from archex.exceptions import ArchexError

    runner = CliRunner()
    with patch("archex.cli.query_cmd.query", side_effect=ArchexError("Query failed")):
        result = runner.invoke(cli, ["query", str(python_simple_repo), "test question"])
    assert result.exit_code != 0
    assert "Query failed" in result.output


def test_query_success_outputs_prompt(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["query", str(python_simple_repo), "what functions exist?"],
    )
    assert result.exit_code == 0, result.output
    assert len(result.output.strip()) > 0


def test_query_timing_flag(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["query", str(python_simple_repo), "what functions exist?", "--timing"],
    )
    assert result.exit_code == 0, result.output
    assert "[savings]" in result.output
    assert "[timing]" in result.output
    # Phase timing: should show acquire or cache hit
    output = result.output
    assert "Acquired repo" in output or "Cache hit" in output


def test_compare_error_handling() -> None:
    from unittest.mock import patch

    from archex.exceptions import ArchexError

    runner = CliRunner()
    with patch("archex.cli.compare_cmd.compare", side_effect=ArchexError("Analyze failed")):
        result = runner.invoke(cli, ["compare", "/fake/a", "/fake/b"])
    assert result.exit_code != 0
    assert "Analyze failed" in result.output


def test_compare_type_check_raises_type_error() -> None:
    from archex.cli.compare_cmd import render_comparison_markdown

    # Test that non-ComparisonResult raises TypeError
    with pytest.raises(TypeError, match="Expected ComparisonResult"):
        render_comparison_markdown({"not": "a_comparison_result"})


class TestMcpCmd:
    def test_mcp_import_error_raises_click_exception(self) -> None:
        from unittest.mock import patch

        runner = CliRunner()
        with patch.dict("sys.modules", {"archex.integrations.mcp": None}):
            result = runner.invoke(cli, ["mcp"])
        assert result.exit_code != 0
        assert "mcp" in result.output.lower()

    def test_mcp_runs_stdio_server(self) -> None:
        from unittest.mock import MagicMock, patch

        mock_run_stdio = MagicMock()
        mock_mcp_module = MagicMock()
        mock_mcp_module.run_stdio_server = mock_run_stdio

        runner = CliRunner()
        with (
            patch.dict("sys.modules", {"archex.integrations.mcp": mock_mcp_module}),
            patch("archex.cli.mcp_cmd.asyncio.run") as mock_asyncio_run,
        ):
            result = runner.invoke(cli, ["mcp"])
        assert result.exit_code == 0, result.output
        mock_asyncio_run.assert_called_once_with(mock_run_stdio())


class TestCacheList:
    def test_empty_cache(self, tmp_path: Path) -> None:
        runner = CliRunner()
        cache_dir = str(tmp_path / "empty_cache")
        result = runner.invoke(cli, ["cache", "list", "--cache-dir", cache_dir])
        assert result.exit_code == 0
        assert "No cached entries" in result.output

    def test_lists_entries(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Create a fake cache entry
        key = "a" * 64
        (cache_dir / f"{key}.db").write_text("fake")
        (cache_dir / f"{key}.meta").write_text("1234567890.0")

        runner = CliRunner()
        result = runner.invoke(cli, ["cache", "list", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert key[:12] in result.output


class TestCacheClean:
    def test_clean_removes_old(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        key = "b" * 64
        (cache_dir / f"{key}.db").write_text("fake")
        (cache_dir / f"{key}.meta").write_text("0")  # epoch = very old

        runner = CliRunner()
        result = runner.invoke(
            cli, ["cache", "clean", "--max-age", "1", "--cache-dir", str(cache_dir)]
        )
        assert result.exit_code == 0
        assert "Removed 1" in result.output

    def test_clean_keeps_recent(self, tmp_path: Path) -> None:
        import time

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        key = "c" * 64
        (cache_dir / f"{key}.db").write_text("fake")
        (cache_dir / f"{key}.meta").write_text(str(time.time()))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["cache", "clean", "--max-age", "24", "--cache-dir", str(cache_dir)]
        )
        assert result.exit_code == 0
        assert "Removed 0" in result.output


class TestCacheInfo:
    def test_info_output(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["cache", "info", "--cache-dir", str(cache_dir)])
        assert result.exit_code == 0
        assert "Cache directory" in result.output
        assert "Total entries" in result.output
        assert "Total size" in result.output


class TestTreeCmd:
    def test_tree_json(self, python_simple_repo: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["tree", str(python_simple_repo), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "entries" in data
        assert "total_files" in data

    def test_tree_human(self, python_simple_repo: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["tree", str(python_simple_repo)])
        assert result.exit_code == 0, result.output

    def test_tree_human_renders_entries(self) -> None:
        from unittest.mock import patch

        from archex.models import FileTree, FileTreeEntry

        entries = [
            FileTreeEntry(
                path="src",
                is_directory=True,
                children=[
                    FileTreeEntry(path="src/main.py", language="python", lines=50, symbol_count=5),
                ],
            ),
        ]
        tree = FileTree(root="/repo", entries=entries, total_files=1, languages={"python": 1})
        runner = CliRunner()
        with patch("archex.cli.tree_cmd.file_tree", return_value=tree):
            result = runner.invoke(cli, ["tree", "/repo"])
        assert result.exit_code == 0, result.output
        assert "/repo" in result.output
        assert "src/" in result.output
        assert "main.py" in result.output
        assert "python" in result.output
        assert "50 lines" in result.output

    def test_tree_timing(self) -> None:
        from unittest.mock import patch

        from archex.models import FileTree

        tree = FileTree(root="/r", entries=[], total_files=0, languages={})
        runner = CliRunner()
        with (
            patch("archex.cli.tree_cmd.file_tree", return_value=tree),
            patch("archex.cli.tree_cmd.get_repo_total_tokens", return_value=1000),
        ):
            result = runner.invoke(cli, ["tree", "/r", "--timing"])
        assert result.exit_code == 0
        assert "[savings]" in result.output
        assert "[timing]" in result.output
        # Phase timing present (cache hit or acquire)
        assert "Cache hit" in result.output or "timing" in result.output

    def test_tree_error_handling(self) -> None:
        from unittest.mock import patch

        from archex.exceptions import ArchexError

        runner = CliRunner()
        with patch("archex.cli.tree_cmd.file_tree", side_effect=ArchexError("tree fail")):
            result = runner.invoke(cli, ["tree", "/fake"])
        assert result.exit_code != 0
        assert "tree fail" in result.output


class TestOutlineCmd:
    def test_outline_error_for_missing_file(self, python_simple_repo: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["outline", str(python_simple_repo), "nonexistent.py"])
        # Should succeed but return empty outline
        assert result.exit_code == 0

    def test_outline_human_renders_symbols(self) -> None:
        from unittest.mock import patch

        from archex.models import FileOutline, SymbolKind, SymbolOutline, Visibility

        child = SymbolOutline(
            symbol_id="f.py::Foo.bar#method",
            name="bar",
            kind=SymbolKind.METHOD,
            file_path="f.py",
            start_line=5,
            end_line=10,
            signature="def bar(self)",
        )
        parent = SymbolOutline(
            symbol_id="f.py::Foo#class",
            name="Foo",
            kind=SymbolKind.CLASS,
            file_path="f.py",
            start_line=3,
            end_line=12,
            signature="class Foo",
            visibility=Visibility.PUBLIC,
            children=[child],
        )
        outline = FileOutline(
            file_path="f.py", language="python", lines=20, symbols=[parent], token_count_raw=200
        )
        runner = CliRunner()
        with patch("archex.cli.outline_cmd.file_outline", return_value=outline):
            result = runner.invoke(cli, ["outline", "/repo", "f.py"])
        assert result.exit_code == 0, result.output
        assert "file: f.py" in result.output
        assert "language: python" in result.output
        assert "lines: 20" in result.output
        assert "class Foo" in result.output
        assert "L3-12" in result.output
        assert "method bar" in result.output
        assert "def bar(self)" in result.output

    def test_outline_json(self) -> None:
        from unittest.mock import patch

        from archex.models import FileOutline

        outline = FileOutline(
            file_path="f.py", language="python", lines=10, symbols=[], token_count_raw=50
        )
        runner = CliRunner()
        with patch("archex.cli.outline_cmd.file_outline", return_value=outline):
            result = runner.invoke(cli, ["outline", "/repo", "f.py", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["file_path"] == "f.py"

    def test_outline_timing(self) -> None:
        from unittest.mock import patch

        from archex.models import FileOutline

        outline = FileOutline(
            file_path="f.py", language="python", lines=10, symbols=[], token_count_raw=50
        )
        runner = CliRunner()
        with patch("archex.cli.outline_cmd.file_outline", return_value=outline):
            result = runner.invoke(cli, ["outline", "/repo", "f.py", "--timing"])
        assert result.exit_code == 0
        assert "[savings]" in result.output
        assert "[timing]" in result.output
        assert "Cache hit" in result.output or "timing" in result.output

    def test_outline_error_handling(self) -> None:
        from unittest.mock import patch

        from archex.exceptions import ArchexError

        runner = CliRunner()
        with patch("archex.cli.outline_cmd.file_outline", side_effect=ArchexError("outline fail")):
            result = runner.invoke(cli, ["outline", "/fake", "f.py"])
        assert result.exit_code != 0
        assert "outline fail" in result.output


class TestSymbolsCmd:
    def test_symbols_json(self, python_simple_repo: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["symbols", str(python_simple_repo), "class", "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert isinstance(data, list)

    def test_symbols_no_match(self, python_simple_repo: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["symbols", str(python_simple_repo), "xyznonexistent123"])
        assert result.exit_code == 0, result.output

    def test_symbols_human_renders_table(self) -> None:
        from unittest.mock import patch

        from archex.models import SymbolKind, SymbolMatch

        matches = [
            SymbolMatch(
                symbol_id="a.py::foo#function",
                name="foo",
                kind=SymbolKind.FUNCTION,
                file_path="a.py",
                start_line=1,
            ),
            SymbolMatch(
                symbol_id="b.py::Bar#class",
                name="Bar",
                kind=SymbolKind.CLASS,
                file_path="b.py",
                start_line=10,
            ),
        ]
        runner = CliRunner()
        with patch("archex.cli.symbols_cmd.search_symbols", return_value=matches):
            result = runner.invoke(cli, ["symbols", "/repo", "test"])
        assert result.exit_code == 0, result.output
        assert "kind" in result.output
        assert "name" in result.output
        assert "file_path" in result.output
        assert "---" in result.output
        assert "foo" in result.output
        assert "function" in result.output
        assert "a.py" in result.output
        assert "Bar" in result.output
        assert "class" in result.output

    def test_symbols_human_no_results(self) -> None:
        from unittest.mock import patch

        runner = CliRunner()
        with patch("archex.cli.symbols_cmd.search_symbols", return_value=[]):
            result = runner.invoke(cli, ["symbols", "/repo", "nothing"])
        assert result.exit_code == 0
        assert "No symbols found." in result.output

    def test_symbols_timing(self) -> None:
        from unittest.mock import patch

        runner = CliRunner()
        with (
            patch("archex.cli.symbols_cmd.search_symbols", return_value=[]),
            patch("archex.cli.symbols_cmd.get_files_token_count", return_value=500),
        ):
            result = runner.invoke(cli, ["symbols", "/repo", "q", "--timing"])
        assert result.exit_code == 0
        assert "[savings]" in result.output
        assert "[timing]" in result.output

    def test_symbols_error_handling(self) -> None:
        from unittest.mock import patch

        from archex.exceptions import ArchexError

        runner = CliRunner()
        with patch("archex.cli.symbols_cmd.search_symbols", side_effect=ArchexError("search fail")):
            result = runner.invoke(cli, ["symbols", "/fake", "q"])
        assert result.exit_code != 0
        assert "search fail" in result.output


class TestSymbolCmd:
    def test_symbol_not_found(self, python_simple_repo: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["symbol", str(python_simple_repo), "fake::id#function"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_symbol_human_renders_source(self) -> None:
        from unittest.mock import patch

        from archex.models import SymbolKind, SymbolSource

        sym = SymbolSource(
            symbol_id="f.py::greet#function",
            name="greet",
            kind=SymbolKind.FUNCTION,
            file_path="f.py",
            start_line=1,
            end_line=3,
            source="def greet():\n    print('hi')",
        )
        runner = CliRunner()
        with patch("archex.cli.symbol_cmd.get_symbol", return_value=sym):
            result = runner.invoke(cli, ["symbol", "/repo", "f.py::greet#function"])
        assert result.exit_code == 0, result.output
        assert "# greet (function)" in result.output
        assert "f.py:1-3" in result.output
        assert "def greet():" in result.output
        assert "print('hi')" in result.output

    def test_symbol_json(self) -> None:
        from unittest.mock import patch

        from archex.models import SymbolKind, SymbolSource

        sym = SymbolSource(
            symbol_id="f.py::x#function",
            name="x",
            kind=SymbolKind.FUNCTION,
            file_path="f.py",
            start_line=1,
            end_line=2,
            source="def x(): pass",
        )
        runner = CliRunner()
        with patch("archex.cli.symbol_cmd.get_symbol", return_value=sym):
            result = runner.invoke(cli, ["symbol", "/repo", "f.py::x#function", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["source"] == "def x(): pass"

    def test_symbol_timing(self) -> None:
        from unittest.mock import patch

        from archex.models import SymbolKind, SymbolSource

        sym = SymbolSource(
            symbol_id="f.py::x#function",
            name="x",
            kind=SymbolKind.FUNCTION,
            file_path="f.py",
            start_line=1,
            end_line=2,
            source="pass",
        )
        runner = CliRunner()
        with (
            patch("archex.cli.symbol_cmd.get_symbol", return_value=sym),
            patch("archex.cli.symbol_cmd.get_file_token_count", return_value=200),
        ):
            result = runner.invoke(cli, ["symbol", "/repo", "f.py::x#function", "--timing"])
        assert result.exit_code == 0
        assert "[savings]" in result.output
        assert "[timing]" in result.output

    def test_symbol_error_handling(self) -> None:
        from unittest.mock import patch

        from archex.exceptions import ArchexError

        runner = CliRunner()
        with patch("archex.cli.symbol_cmd.get_symbol", side_effect=ArchexError("sym fail")):
            result = runner.invoke(cli, ["symbol", "/fake", "id"])
        assert result.exit_code != 0
        assert "sym fail" in result.output
