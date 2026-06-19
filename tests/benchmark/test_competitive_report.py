"""Tests for the competitive comparison report."""

from __future__ import annotations

import pytest

from archex.benchmark.competitive import format_competitive_markdown
from archex.benchmark.headtohead import HeadToHeadManifestError
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    CompressionLayerConfig,
    CompressionLayerMode,
    CompressionLayerResult,
    ExternalToolBenchmarkConfig,
    HeadToHeadArchexConfig,
    HeadToHeadManifest,
    Strategy,
    TaskCategory,
)


def _manifest() -> HeadToHeadManifest:
    return HeadToHeadManifest(
        name="competitive",
        task_subset=["task_a", "task_b"],
        hardware_notes="M1 Pro",
        archex=HeadToHeadArchexConfig(candidate_strategies=[Strategy.ARCHEX_QUERY_COMPRESSED]),
        external_tools=[
            ExternalToolBenchmarkConfig(
                name="ccc",
                version="0.2.35",
                command="ccc",
                args=["mcp"],
                embedder="Snowflake/snowflake-arctic-embed-xs",
            )
        ],
        compression_layers=[
            CompressionLayerConfig(name="headroom", version="0.4.1", command="headroom")
        ],
    )


def _result(
    strategy: Strategy,
    *,
    label: str | None = None,
    region_recall: float | None = None,
    compression_ratio: float | None = None,
    category: TaskCategory | None = TaskCategory.EXTERNAL_FRAMEWORK,
) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="task_a",
        strategy=strategy,
        strategy_label=label,
        tokens_total=100,
        tokens_input=400,
        tokens_output=100,
        token_efficiency=0.75,
        token_efficiency_with_completion=0.71,
        bundle_completion_tokens=20,
        tool_calls=1,
        files_accessed=1,
        recall=0.9,
        precision=0.5,
        f1_score=0.64,
        required_file_recall=0.9,
        region_recall=region_recall,
        line_recall=region_recall,
        context_noise_ratio=0.2 if region_recall is not None else None,
        bundle_compression_ratio=compression_ratio,
        category=category,
        savings_vs_raw=0.0,
        wall_time_ms=12.0,
        warm_latency_ms=10.0,
        cold_start_ms=2.0,
        cached=True,
        timestamp="2026-06-19T00:00:00Z",
        provenance={
            "external_tool": label or strategy.value,
            "external_tool_version": "0.2.35",
            "external_tool_embedder": "Snowflake/snowflake-arctic-embed-xs",
        },
    )


def _report(task_id: str, repo: str, results: list[BenchmarkResult]) -> BenchmarkReport:
    for result in results:
        result.task_id = task_id
    return BenchmarkReport(
        task_id=task_id,
        repo=repo,
        question="How?",
        baseline_tokens=400,
        results=results,
    )


def _compression(
    mode: CompressionLayerMode, source_lane: str, ratio: float
) -> CompressionLayerResult:
    return CompressionLayerResult(
        task_id="task_a",
        lane_label=mode.value,
        mode=mode,
        source_lane=source_lane,
        source_passthrough=ratio >= 1.0,
        bundle_tokens_uncompressed=1000,
        bundle_tokens_compressed=int(1000 * ratio),
        bundle_compression_ratio=ratio,
        provenance={
            "layer": "headroom",
            "version": "0.4.1",
            "source_lane": source_lane,
            "run_mode": "artifact",
            "command": "headroom compress",
        },
        timestamp="2026-06-19T00:00:00Z",
    )


def _reports_one_repo() -> list[BenchmarkReport]:
    return [
        _report(
            "task_a",
            "owner/repo",
            [
                _result(Strategy.ARCHEX_QUERY, region_recall=0.8),
                _result(Strategy.ARCHEX_QUERY_COMPRESSED, compression_ratio=0.85),
                _result(Strategy.EXTERNAL_MCP, label="ccc"),
                _result(Strategy.RAW_RIPGREP),
            ],
        )
    ]


def test_competitive_report_includes_all_lanes_and_layer_labels() -> None:
    compression = [
        _compression(CompressionLayerMode.HEADROOM_ONLY_ON_RAW_CONTEXT, "raw_files", 0.4),
        _compression(CompressionLayerMode.ARCHEX_PLUS_HEADROOM, "archex", 0.92),
    ]

    output = format_competitive_markdown(_manifest(), _reports_one_repo(), compression)

    # Retrieval, candidate, external, baseline, and both compression lanes present.
    assert "| archex | retrieval |" in output
    assert "| archex_query_compressed | retrieval |" in output
    assert "| ccc | retrieval |" in output
    assert "| raw-ripgrep/read | baseline |" in output
    assert "| headroom_only_on_raw_context | compression |" in output
    assert "| archex_plus_headroom | compression |" in output
    # p95 latency and token-efficiency-after-completion columns.
    assert "Warm p95 ms" in output
    assert "Token eff. (compl.)" in output
    # Provenance for retrieval, candidate, and compression lanes.
    assert "manifest=competitive; lane=archex; embedder=jina-v2" in output
    assert "manifest=competitive; lane=archex_query_compressed; embedder=jina-v2" in output
    assert "layer=headroom" in output


def test_competitive_report_renders_region_metrics_when_available() -> None:
    output = format_competitive_markdown(_manifest(), _reports_one_repo())

    assert "Region recall" in output
    # archex declared region_recall=0.8; rendered, not "n/a".
    assert "0.80" in output


def test_competitive_report_marks_compression_lane_not_retrieval() -> None:
    compression = [
        _compression(CompressionLayerMode.HEADROOM_ONLY_ON_RAW_CONTEXT, "raw_files", 0.4)
    ]

    output = format_competitive_markdown(_manifest(), _reports_one_repo(), compression)

    assert "not a retrieval engine" in output
    # Compression lane has n/a recall but a real compression ratio.
    compression_row = next(
        line
        for line in output.splitlines()
        if line.startswith("| headroom_only_on_raw_context |") and line.count("|") == 16
    )
    cells = [cell.strip() for cell in compression_row.strip("|").split("|")]
    assert cells[2] == "n/a"  # recall column
    assert cells[9] == "0.40"  # compression ratio column


def test_competitive_report_groups_by_repo_and_aggregate() -> None:
    reports = [
        _report("task_a", "owner/one", _standard_results()),
        _report("task_b", "owner/two", _standard_results()),
    ]

    output = format_competitive_markdown(_manifest(), reports)

    assert "## Aggregate (2 tasks)" in output
    assert "## By task family" in output
    assert "### `external-framework`" in output
    assert "## By repo" in output
    assert "### `owner/one`" in output
    assert "### `owner/two`" in output


def test_competitive_report_rejects_missing_required_lane() -> None:
    reports = [
        _report(
            "task_a",
            "owner/repo",
            [_result(Strategy.EXTERNAL_MCP, label="ccc"), _result(Strategy.RAW_RIPGREP)],
        )
    ]

    with pytest.raises(HeadToHeadManifestError, match="missing lane.*archex"):
        format_competitive_markdown(_manifest(), reports)


def test_competitive_report_handles_no_results() -> None:
    assert format_competitive_markdown(_manifest(), []) == "No competitive benchmark results."


def _standard_results() -> list[BenchmarkResult]:
    return [
        _result(Strategy.ARCHEX_QUERY),
        _result(Strategy.EXTERNAL_MCP, label="ccc"),
        _result(Strategy.RAW_RIPGREP),
    ]
