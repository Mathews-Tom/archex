from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from archex import __version__
from archex.cli.main import cli


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
    assert "init" in output
    assert "index" in output
    assert "status" in output
    assert "reset" in output


def test_init_command_creates_project_state(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["init", str(python_simple_repo)])

    assert result.exit_code == 0, result.output
    assert "Initialized archex project" in result.output
    assert (python_simple_repo / ".archex" / "settings.toml").exists()
    assert (python_simple_repo / ".archex" / "dogfood" / "history").is_dir()
    assert ".archex/" in (python_simple_repo / ".gitignore").read_text(encoding="utf-8")


def test_init_command_is_idempotent(python_simple_repo: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(cli, ["init", str(python_simple_repo)])
    second = runner.invoke(cli, ["init", str(python_simple_repo)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "already initialized" in second.output


def test_init_command_reset_requires_force(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["init", str(python_simple_repo), "--reset"])

    assert result.exit_code != 0
    assert "--reset requires --force" in result.output


class FakeIndexStore:
    db_path = Path("/tmp/archex-index.db")

    def __init__(self) -> None:
        self.closed = False

    def get_file_metadata(self) -> list[dict[str, str | int]]:
        return [
            {"file_path": "src/app.py", "language": "python", "lines": 10, "symbol_count": 2},
            {"file_path": "src/util.py", "language": "python", "lines": 8, "symbol_count": 1},
            {"file_path": "web/app.ts", "language": "typescript", "lines": 20, "symbol_count": 3},
        ]

    def get_metadata(self, key: str) -> str | None:
        if key == "commit_hash":
            return "abc123"
        return None

    def get_file_count(self) -> int:
        return 3

    def get_chunk_count(self) -> int:
        return 6

    def close(self) -> None:
        self.closed = True


def test_index_command_uses_project_config(python_simple_repo: Path) -> None:
    from archex.project import init_project

    init_project(python_simple_repo)
    store = FakeIndexStore()
    runner = CliRunner()

    with patch("archex.cli.index_cmd.index_repository", return_value=store) as index_mock:
        result = runner.invoke(cli, ["index", str(python_simple_repo)])

    assert result.exit_code == 0, result.output
    assert "Indexed repository:" in result.output
    assert "Strategy:" in result.output
    assert "python=2" in result.output
    config = index_mock.call_args.kwargs["config"]
    index_config = index_mock.call_args.kwargs["index_config"]
    assert config.cache_dir == str(python_simple_repo / ".archex")
    assert index_config.vector is False
    assert store.closed is True


def test_index_command_json_output(python_simple_repo: Path) -> None:
    store = FakeIndexStore()
    runner = CliRunner()

    with patch("archex.cli.index_cmd.index_repository", return_value=store):
        result = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["repo_root"] == str(python_simple_repo.resolve())
    assert data["index_path"] == "/tmp/archex-index.db"
    assert data["commit_hash"] == "abc123"
    assert data["files_indexed"] == 3
    assert data["chunks_indexed"] == 6
    assert data["languages"] == {"python": 2, "typescript": 1}


def test_index_command_writes_fixed_project_index_path(python_simple_repo: Path) -> None:
    from archex.project import init_project

    init_project(python_simple_repo)
    runner = CliRunner()

    result = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["index_path"] == str(python_simple_repo / ".archex" / "index.db")
    assert (python_simple_repo / ".archex" / "index.db").exists()
    assert not list((python_simple_repo / ".archex").glob("[0-9a-f]" * 64 + ".db"))


def test_status_command_reports_uninitialized(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["status", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["state"] == "uninitialized"
    assert data["initialized"] is False


def test_status_command_reports_missing_index(python_simple_repo: Path) -> None:
    from archex.project import init_project

    init_project(python_simple_repo)
    runner = CliRunner()

    result = runner.invoke(cli, ["status", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["state"] == "missing_index"
    assert data["initialized"] is True


def test_status_command_reports_fresh_index(python_simple_repo: Path) -> None:
    from archex.project import init_project

    init_project(python_simple_repo)
    runner = CliRunner()
    indexed = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])
    assert indexed.exit_code == 0, indexed.output

    result = runner.invoke(cli, ["status", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["state"] == "fresh"
    assert data["index_path"] == str(python_simple_repo / ".archex" / "index.db")
    assert data["files_indexed"] > 0
    assert data["chunks_indexed"] > 0
    assert data["languages"]["python"] > 0


def test_status_command_strict_fails_on_dirty_index(python_simple_repo: Path) -> None:
    from archex.project import init_project

    init_project(python_simple_repo)
    runner = CliRunner()
    indexed = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])
    assert indexed.exit_code == 0, indexed.output
    (python_simple_repo / "utils.py").write_text("def dirty_symbol(): return 1\n")

    result = runner.invoke(
        cli,
        ["status", str(python_simple_repo), "--format", "json", "--strict"],
    )

    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    assert data["state"] == "dirty"
    assert data["working_tree"] == "dirty"


def test_status_command_fails_on_corrupt_index(python_simple_repo: Path) -> None:
    from archex.project import init_project

    init_project(python_simple_repo)
    index_path = python_simple_repo / ".archex" / "index.db"
    index_path.write_text("not sqlite", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(cli, ["status", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 1, result.output
    data = json.loads(result.output)
    assert data["state"] == "corrupt"
    assert data["error"]


def test_reset_command_requires_force(python_simple_repo: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["reset", str(python_simple_repo)])

    assert result.exit_code != 0
    assert "reset requires --force" in result.output


def test_reset_command_preserves_settings(python_simple_repo: Path) -> None:
    from archex.project import init_project

    init_project(python_simple_repo)
    runner = CliRunner()
    indexed = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])
    assert indexed.exit_code == 0, indexed.output

    result = runner.invoke(cli, ["reset", str(python_simple_repo), "--force"])

    assert result.exit_code == 0, result.output
    assert (python_simple_repo / ".archex" / "settings.toml").exists()
    assert not (python_simple_repo / ".archex" / "index.db").exists()
    status = runner.invoke(cli, ["status", str(python_simple_repo), "--format", "json"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["state"] == "missing_index"


def test_reset_command_all_removes_project_state(python_simple_repo: Path) -> None:
    from archex.project import init_project

    init_project(python_simple_repo)
    runner = CliRunner()

    result = runner.invoke(cli, ["reset", str(python_simple_repo), "--all", "--force"])

    assert result.exit_code == 0, result.output
    assert not (python_simple_repo / ".archex").exists()


def test_lifecycle_commands_default_source_to_cwd(
    python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(python_simple_repo)
    runner = CliRunner()

    initialized = runner.invoke(cli, ["init"])
    indexed = runner.invoke(cli, ["index", "--format", "json"])
    status = runner.invoke(cli, ["status", "--format", "json"])
    reset = runner.invoke(cli, ["reset", "--force"])

    assert initialized.exit_code == 0, initialized.output
    assert indexed.exit_code == 0, indexed.output
    assert json.loads(indexed.output)["repo_root"] == str(python_simple_repo)
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["state"] == "fresh"
    assert reset.exit_code == 0, reset.output


def test_query_command_defaults_source_to_cwd(
    python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from archex.project import init_project

    class FakeBundle:
        chunks: list[object] = []
        token_count = 0

        def to_prompt(self, *, format: str) -> str:
            return f"format={format}"

    init_project(python_simple_repo)
    monkeypatch.chdir(python_simple_repo)
    runner = CliRunner()

    with patch("archex.cli.query_cmd.query", return_value=FakeBundle()) as query_mock:
        result = runner.invoke(cli, ["query", "How does the query pipeline work?"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "format=xml"
    assert query_mock.call_args.args[0].local_path == "."
    assert query_mock.call_args.args[1] == "How does the query pipeline work?"


def test_analyze_tree_and_symbols_default_source_to_cwd(
    python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from archex.models import FileTree
    from archex.project import init_project

    class FakeProfile:
        def to_json(self) -> str:
            return "{}"

        def to_markdown(self) -> str:
            return "# profile"

    init_project(python_simple_repo)
    monkeypatch.chdir(python_simple_repo)
    runner = CliRunner()

    with patch("archex.cli.analyze_cmd.analyze", return_value=FakeProfile()) as analyze_mock:
        analyze_result = runner.invoke(cli, ["analyze"])
    with patch(
        "archex.cli.tree_cmd.file_tree",
        return_value=FileTree(root=".", entries=[], total_files=0, languages={}),
    ) as tree_mock:
        tree_result = runner.invoke(cli, ["tree"])
    with patch("archex.cli.symbols_cmd.search_symbols", return_value=[]) as symbols_mock:
        symbols_result = runner.invoke(cli, ["symbols", "query"])

    assert analyze_result.exit_code == 0, analyze_result.output
    assert tree_result.exit_code == 0, tree_result.output
    assert symbols_result.exit_code == 0, symbols_result.output
    assert analyze_mock.call_args.args[0].local_path == "."
    assert tree_mock.call_args.args[0].local_path == "."
    assert symbols_mock.call_args.args[0].local_path == "."
    assert symbols_mock.call_args.kwargs["query"] == "query"


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


def test_query_uses_project_config_when_cli_args_omitted(python_simple_repo: Path) -> None:
    from archex.project import init_project

    class FakeBundle:
        chunks: list[object] = []
        token_count = 0

        def to_prompt(self, *, format: str) -> str:
            return f"format={format}"

    init_project(python_simple_repo)
    runner = CliRunner()

    with patch("archex.cli.query_cmd.query", return_value=FakeBundle()) as query_mock:
        result = runner.invoke(
            cli,
            ["query", str(python_simple_repo), "what functions exist?"],
        )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "format=xml"
    config = query_mock.call_args.kwargs["config"]
    index_config = query_mock.call_args.kwargs["index_config"]
    assert config.cache_dir == str(python_simple_repo / ".archex")
    assert config.languages is None
    assert index_config.vector is False


def test_query_cli_options_override_project_config(python_simple_repo: Path) -> None:
    from archex.project import init_project

    class FakeBundle:
        chunks: list[object] = []
        token_count = 0

        def to_prompt(self, *, format: str) -> str:
            return f"format={format}"

    init_project(python_simple_repo)
    runner = CliRunner()

    with patch("archex.cli.query_cmd.query", return_value=FakeBundle()) as query_mock:
        result = runner.invoke(
            cli,
            [
                "query",
                str(python_simple_repo),
                "what functions exist?",
                "--language",
                "python",
                "--strategy",
                "hybrid",
            ],
        )

    assert result.exit_code == 0, result.output
    config = query_mock.call_args.kwargs["config"]
    index_config = query_mock.call_args.kwargs["index_config"]
    assert config.languages == ["python"]
    assert index_config.vector is True


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


def test_query_metrics_flag(python_simple_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["query", str(python_simple_repo), "what functions exist?", "--metrics"],
    )
    assert result.exit_code == 0
    # Metrics JSON should be in stderr
    assert "strategy" in result.output


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
