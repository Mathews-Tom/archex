"""Tests for head-to-head runner and report rendering."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from archex.benchmark.headtohead import format_headtohead_markdown, run_headtohead
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    ExternalToolBenchmarkConfig,
    GraphifyLaneConfig,
    GraphifyLaneName,
    HeadToHeadManifest,
    Strategy,
)

if TYPE_CHECKING:
    import pytest


def _result(
    strategy: Strategy,
    *,
    label: str | None = None,
    provenance: dict[str, str] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="task_a",
        strategy=strategy,
        strategy_label=label,
        tokens_total=100,
        tokens_input=400,
        tokens_output=100,
        token_efficiency=0.75,
        bundle_completion_tokens=20,
        token_efficiency_with_completion=0.71,
        tool_calls=1,
        files_accessed=1,
        recall=0.5,
        precision=1.0,
        f1_score=0.67,
        savings_vs_raw=0.0,
        wall_time_ms=12.0,
        freshness_latency_ms=42.0,
        freshness_correct=True,
        warm_latency_ms=10.0,
        cold_start_ms=2.0,
        cached=True,
        timestamp="2026-06-11T00:00:00Z",
        provenance=provenance
        or {
            "external_tool": label or strategy.value,
            "external_tool_version": "0.2.35",
            "external_tool_embedder": "Snowflake/snowflake-arctic-embed-xs",
        },
    )


def test_format_headtohead_markdown_keeps_all_lanes_and_provenance() -> None:
    manifest = HeadToHeadManifest(
        name="comparison",
        task_subset=["task_a"],
        hardware_notes="M1 Pro",
        external_tools=[
            ExternalToolBenchmarkConfig(
                name="ccc",
                version="0.2.35",
                command="ccc",
                args=["mcp"],
                embedder="Snowflake/snowflake-arctic-embed-xs",
            )
        ],
    )
    report = BenchmarkReport(
        task_id="task_a",
        repo="owner/repo",
        question="How?",
        baseline_tokens=400,
        results=[
            _result(Strategy.ARCHEX_QUERY),
            _result(Strategy.EXTERNAL_MCP, label="ccc"),
            _result(Strategy.RAW_RIPGREP),
        ],
    )

    output = format_headtohead_markdown(manifest, [report])

    assert "| archex |" in output
    assert "| ccc |" in output
    assert "| raw-ripgrep/read |" in output
    assert "prov: manifest=comparison" in output
    assert "field=freshness_latency_ms" in output
    assert "field=freshness_correct" in output
    assert "field=bundle_completion_tokens" in output
    assert "No winner filtering" in output


def test_format_headtohead_markdown_ignores_graphify_followup_lanes() -> None:
    manifest = HeadToHeadManifest(
        name="comparison",
        task_subset=["task_a"],
        hardware_notes="M1 Pro",
        external_tools=[
            ExternalToolBenchmarkConfig(
                name="ccc",
                version="0.2.35",
                command="ccc",
                args=["mcp"],
                embedder="Snowflake/snowflake-arctic-embed-xs",
            )
        ],
        graphify_lanes=[
            GraphifyLaneConfig(
                name=GraphifyLaneName.GRAPHIFY_QUERY_WARM,
                version="0.8.44",
                command="graphify",
                includes_build_cost=False,
            )
        ],
    )
    report = BenchmarkReport(
        task_id="task_a",
        repo="owner/repo",
        question="How?",
        baseline_tokens=400,
        results=[
            _result(Strategy.ARCHEX_QUERY),
            _result(Strategy.EXTERNAL_MCP, label="ccc"),
            _result(
                Strategy.EXTERNAL_MCP,
                label="graphify_query_warm",
                provenance={
                    "external_tool": "graphify_query_warm",
                    "external_tool_version": "0.8.44",
                    "graphify_package": "graphifyy",
                    "graphify_run_mode": "artifact",
                },
            ),
            _result(Strategy.RAW_RIPGREP),
        ],
    )

    output = format_headtohead_markdown(manifest, [report])

    assert "| archex |" in output
    assert "| ccc |" in output
    assert "| raw-ripgrep/read |" in output
    assert "graphify_query_warm" not in output
    assert "Graphify rows are graph / memory-layer lanes" not in output


def test_run_headtohead_copies_manifest_and_uses_external_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "task.yaml").write_text(
        """\
task_id: task_a
repo: owner/repo
commit: abc
category: external-framework
question: "How?"
expected_files:
  - src/main.py
""",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        """\
manifest_version: 1
name: comparison
hardware_notes: M1 Pro
task_subset:
  - task_a
external_tools:
  - name: ccc
    version: "0.2.35"
    command: ccc
    args: [mcp]
    embedder: Snowflake/snowflake-arctic-embed-xs
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def replacement_run_all(**kwargs: object) -> list[BenchmarkReport]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr("archex.benchmark.headtohead.run_all", replacement_run_all)

    run_headtohead(manifest_path, tmp_path / "out", tasks_dir)

    assert (tmp_path / "out" / "manifest.yaml").is_file()
    assert captured["strategies"] == [
        Strategy.RAW_FILES,
        Strategy.RAW_RIPGREP,
        Strategy.ARCHEX_QUERY,
        Strategy.EXTERNAL_MCP,
    ]
