"""Tests for compression-artifact ingestion and checked-in artifact validation."""

from __future__ import annotations

import json
from pathlib import Path

from archex.benchmark.competitive import format_competitive_markdown, load_compression_results
from archex.benchmark.headtohead import load_headtohead_manifest, load_headtohead_results
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    CompressionLayerConfig,
    ExternalToolBenchmarkConfig,
    HeadToHeadManifest,
    Strategy,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "headroom_artifacts"
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "headtohead" / "results"


def _manifest_with_headroom() -> HeadToHeadManifest:
    return HeadToHeadManifest(
        name="competitive",
        task_subset=["httpx_pooling"],
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
        compression_layers=[
            CompressionLayerConfig(
                name="headroom",
                version="0.4.1",
                command="headroom",
                artifact_dir=str(_FIXTURE_DIR),
            )
        ],
    )


def _result(strategy: Strategy, *, label: str | None = None) -> BenchmarkResult:
    return BenchmarkResult(
        task_id="httpx_pooling",
        strategy=strategy,
        strategy_label=label,
        tokens_total=100,
        tokens_input=400,
        recall=0.9,
        precision=0.5,
        required_file_recall=0.9,
        token_efficiency_with_completion=0.7,
        savings_vs_raw=0.0,
        wall_time_ms=10.0,
        warm_latency_ms=10.0,
        cold_start_ms=0.0,
        tool_calls=1,
        files_accessed=1,
        cached=True,
        timestamp="2026-06-19T00:00:00Z",
        provenance={"external_tool": label or strategy.value},
    )


def test_load_compression_results_from_fixture() -> None:
    results = load_compression_results(_manifest_with_headroom(), ["httpx_pooling"])

    by_lane = {result.lane_label: result for result in results}
    assert set(by_lane) == {"headroom_only_on_raw_context", "archex_plus_headroom"}

    raw_lane = by_lane["headroom_only_on_raw_context"]
    assert raw_lane.source_lane == "raw_files"
    assert raw_lane.source_passthrough is False
    assert raw_lane.bundle_tokens_uncompressed == 18481
    assert raw_lane.bundle_tokens_compressed == 9120
    assert raw_lane.provenance["run_mode"] == "artifact"
    assert raw_lane.provenance["version"] == "0.4.1"
    assert len(raw_lane.provenance["artifact_sha256"]) == 64

    archex_lane = by_lane["archex_plus_headroom"]
    assert archex_lane.source_lane == "archex"
    assert archex_lane.source_passthrough is True
    assert archex_lane.bundle_compression_ratio == 1.0


def test_load_compression_results_skips_unknown_task() -> None:
    results = load_compression_results(_manifest_with_headroom(), ["task_without_artifact"])

    assert results == []


def test_competitive_report_includes_compression_lanes_from_artifacts() -> None:
    manifest = _manifest_with_headroom()
    reports = [
        BenchmarkReport(
            task_id="httpx_pooling",
            repo="encode/httpx",
            question="How?",
            baseline_tokens=400,
            results=[
                _result(Strategy.ARCHEX_QUERY),
                _result(Strategy.EXTERNAL_MCP, label="ccc"),
                _result(Strategy.RAW_RIPGREP),
            ],
        )
    ]
    compression = load_compression_results(manifest, ["httpx_pooling"])

    output = format_competitive_markdown(manifest, reports, compression)

    assert "| headroom_only_on_raw_context | compression |" in output
    assert "| archex_plus_headroom | compression |" in output
    assert "layer=headroom" in output


def test_checked_in_headtohead_artifacts_validate_and_render() -> None:
    reports = load_headtohead_results(_RESULTS_DIR)

    assert len(reports) == 19
    # The competitive report renders the checked-in artifacts without error and
    # carries every lane plus the per-repo and aggregate sections docs reference.
    manifest = load_headtohead_manifest(_RESULTS_DIR / "manifest.yaml")
    output = format_competitive_markdown(manifest, reports)

    assert "| archex | retrieval |" in output
    assert "| archex_query_compressed | retrieval |" in output
    assert "| archex_query_efficiency_packed | retrieval |" in output
    assert "| ccc | retrieval |" in output
    assert "| raw-ripgrep/read | baseline |" in output
    assert "## Aggregate (19 tasks)" in output
    assert "## By task family" in output
    assert "## By repo" in output


def test_checked_in_artifacts_have_no_absolute_path_leaks() -> None:
    for path in sorted(_RESULTS_DIR.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text, path.name
        assert "/home/" not in text, path.name
        # Result files must remain valid BenchmarkReport documents.
        BenchmarkReport.model_validate(json.loads(text))
