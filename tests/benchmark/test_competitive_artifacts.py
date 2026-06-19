"""Tests for compression-artifact ingestion and checked-in artifact validation."""

from __future__ import annotations

import json
from pathlib import Path

from archex.benchmark.competitive import format_competitive_markdown, load_compression_results
from archex.benchmark.headtohead import (
    load_headtohead_manifest,
    load_headtohead_results,
    reports_with_graphify_lanes,
)
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    CompressionLayerConfig,
    ExternalToolBenchmarkConfig,
    GraphifyLaneConfig,
    GraphifyLaneName,
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


def _write_graphify_artifact(
    artifact_dir: Path,
    *,
    lane: str,
    includes_build_cost: bool,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": "httpx_pooling",
        "lane": lane,
        "graphify_package": "graphifyy",
        "graphify_version": "0.8.44",
        "command": "graphify query --graph graphify-out/graph.json",
        "includes_build_cost": includes_build_cost,
        "tokens_total": 320,
        "tokens_input": 800,
        "tokens_output": 320,
        "tool_calls": 2 if includes_build_cost else 1,
        "files_accessed": 2,
        "recall": 0.9,
        "precision": 0.5,
        "f1_score": 0.64,
        "mrr": 1.0,
        "ndcg": 1.0,
        "map_score": 1.0,
        "required_file_recall": 0.9,
        "missed_required_file_rate": 0.0,
        "missed_required_task_rate": 0.0,
        "all_required_files_present": True,
        "required_files_present": ["src/main.py"],
        "required_files_missing": [],
        "result_files": ["src/main.py"],
        "task_completion_result": "pass",
        "bundle_completion_tokens": 0,
        "bundle_completion_files": [],
        "token_efficiency": 0.6,
        "token_efficiency_with_completion": 0.6,
        "cold_start_ms": 4200.0 if includes_build_cost else 0.0,
        "warm_latency_ms": 315.0,
        "wall_time_ms": 4515.0 if includes_build_cost else 315.0,
        "cache_state": "cold" if includes_build_cost else "warm",
        "cached": not includes_build_cost,
        "freshness_latency_ms": 0.0,
        "freshness_measured": False,
        "freshness_correct": False,
        "region_recall": None,
        "line_recall": None,
        "context_noise_ratio": None,
        "local_offline_posture": "local code graph only",
        "backend": "local-ast",
        "timestamp": "2026-06-19T00:00:00Z",
    }
    (artifact_dir / "httpx_pooling.json").write_text(json.dumps(payload), encoding="utf-8")


def test_reports_with_graphify_lanes_imports_artifacts() -> None:
    build_dir = _RESULTS_DIR.parent / "tmp-graphify-build"
    warm_dir = _RESULTS_DIR.parent / "tmp-graphify-warm"
    try:
        _write_graphify_artifact(
            build_dir,
            lane="graphify_build_plus_query",
            includes_build_cost=True,
        )
        _write_graphify_artifact(
            warm_dir,
            lane="graphify_query_warm",
            includes_build_cost=False,
        )
        manifest = HeadToHeadManifest(
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
            graphify_lanes=[
                GraphifyLaneConfig(
                    name=GraphifyLaneName.GRAPHIFY_BUILD_PLUS_QUERY,
                    version="0.8.44",
                    command="graphify",
                    includes_build_cost=True,
                    artifact_dir=str(build_dir),
                ),
                GraphifyLaneConfig(
                    name=GraphifyLaneName.GRAPHIFY_QUERY_WARM,
                    version="0.8.44",
                    command="graphify",
                    includes_build_cost=False,
                    artifact_dir=str(warm_dir),
                ),
            ],
        )
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

        augmented = reports_with_graphify_lanes(manifest, reports)
        output = format_competitive_markdown(manifest, augmented)

        assert "| graphify_build_plus_query | graph-memory |" in output
        assert "| graphify_query_warm | graph-memory |" in output
        assert "mode=build+query; run=artifact" in output
    finally:
        for path in (build_dir, warm_dir):
            if path.is_dir():
                for file in path.glob("*.json"):
                    file.unlink()
                path.rmdir()


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
    augmented = reports_with_graphify_lanes(manifest, reports)
    output = format_competitive_markdown(manifest, augmented)

    assert "| archex | retrieval |" in output
    assert "| archex_query_compressed | retrieval |" in output
    assert "| archex_query_efficiency_packed | retrieval |" in output
    assert "| ccc | retrieval |" in output
    assert "| graphify_build_plus_query | graph-memory |" in output
    assert "| graphify_query_warm | graph-memory |" in output
    assert "| raw-ripgrep/read | baseline |" in output
    assert "## Aggregate (19 tasks)" in output
    assert "## By task family" in output
    assert "## By repo" in output


def test_checked_in_artifacts_have_no_absolute_path_leaks() -> None:
    artifact_paths = sorted(_RESULTS_DIR.glob("*.json"))
    artifact_paths += sorted((_RESULTS_DIR / "graphify_build_plus_query").glob("*.json"))
    artifact_paths += sorted((_RESULTS_DIR / "graphify_query_warm").glob("*.json"))
    for path in artifact_paths:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text, path.name
        assert "/home/" not in text, path.name
        assert "/private/" not in text, path.name
        if path.parent == _RESULTS_DIR:
            BenchmarkReport.model_validate(json.loads(text))
