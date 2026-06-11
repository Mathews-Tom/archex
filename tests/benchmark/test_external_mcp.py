"""Tests for external MCP benchmark strategy."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from archex.benchmark.external_mcp import normalize_external_search_result, run_external_mcp
from archex.benchmark.models import (
    BenchmarkTask,
    ExternalToolBenchmarkConfig,
    ExternalToolCommandConfig,
    Strategy,
)


def _write_fixture_repo(path: Path) -> None:
    source = path / "src"
    source.mkdir()
    (source / "main.py").write_text("def target() -> None:\n    pass\n", encoding="utf-8")
    (source / "support.py").write_text("class Helper:\n    pass\n", encoding="utf-8")


def test_normalize_external_json_result_keeps_existing_files(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    result = SimpleNamespace(
        content=[
            SimpleNamespace(
                text='{"results": [{"file_path": "src/main.py", "content": "def target"}]}'
            )
        ]
    )

    hits = normalize_external_search_result(tmp_path, result)

    assert [(hit.file_path, hit.text) for hit in hits] == [("src/main.py", "def target")]


def test_run_external_mcp_runs_bootstrap_commands(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    bootstrap_marker = tmp_path / "bootstrap.txt"
    server = tmp_path / "server.py"
    server.write_text(
        """\
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("benchmark-fixture", json_response=True)


@mcp.tool()
def search(query: str, limit: int = 10, paths: list[str] | None = None) -> list[dict[str, str]]:
    del query, limit, paths
    return [{"file_path": "src/main.py", "content": "def target() -> None:"}]


if __name__ == "__main__":
    mcp.run(transport="stdio")
""",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        task_id="external_fixture_bootstrap",
        repo="owner/repo",
        commit="abc123",
        question="Where is target defined?",
        expected_files=["src/main.py"],
        include_paths=["src"],
        languages=["python"],
    )
    bootstrap_script = (
        "from pathlib import Path; "
        f"Path({str(bootstrap_marker)!r}).write_text('ok', encoding='utf-8')"
    )
    config = ExternalToolBenchmarkConfig(
        name="fixture-mcp",
        version="1.0.0",
        command=sys.executable,
        args=[str(server)],
        embedder="fixture-local",
        timeout_seconds=20,
        bootstrap_commands=[
            ExternalToolCommandConfig(
                command=sys.executable,
                args=["-c", bootstrap_script],
                timeout_seconds=5,
            )
        ],
    )

    result = run_external_mcp(task, tmp_path, config)

    assert bootstrap_marker.read_text(encoding="utf-8") == "ok"
    assert result.tool_calls == 2
    assert result.cold_start_ms >= 0.0
    assert result.provenance["external_tool_bootstrap_count"] == "1"


def test_run_external_mcp_scores_fixture_server(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    server = tmp_path / "server.py"
    server.write_text(
        """\
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("benchmark-fixture", json_response=True)


@mcp.tool()
def search(query: str, limit: int = 10, paths: list[str] | None = None) -> list[dict[str, str]]:
    del query, limit, paths
    return [
        {"file_path": "src/main.py", "content": "def target() -> None:"},
        {"file_path": "src/support.py", "content": "class Helper:"},
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
""",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        task_id="external_fixture",
        repo="owner/repo",
        commit="abc123",
        question="Where is target defined?",
        expected_files=["src/main.py"],
        include_paths=["src"],
        languages=["python"],
    )
    config = ExternalToolBenchmarkConfig(
        name="fixture-mcp",
        version="1.0.0",
        command=sys.executable,
        args=[str(server)],
        embedder="fixture-local",
        timeout_seconds=20,
    )

    result = run_external_mcp(task, tmp_path, config)

    assert result.strategy is Strategy.EXTERNAL_MCP
    assert result.strategy_label == "fixture-mcp"
    assert result.result_files == ["src/main.py", "src/support.py"]
    assert result.recall == 1.0
    assert result.precision == 0.5
    assert result.tool_calls == 1
    assert result.warm_latency_ms >= 0.0
    assert result.provenance["external_tool_version"] == "1.0.0"
