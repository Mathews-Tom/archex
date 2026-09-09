"""Tests for compression-artifact ingestion and checked-in artifact validation."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import pytest

from archex.benchmark.competitive import format_competitive_markdown, load_compression_results
from archex.benchmark.graft import (
    GRAFT_NPM_INTEGRITY,
    GRAFT_SOURCE_COMMIT,
    graft_extraction_tier,
)
from archex.benchmark.headtohead import (
    load_headtohead_manifest,
    load_headtohead_results,
    reports_with_graph_memory_lanes,
    select_headtohead_tasks,
)
from archex.benchmark.models import (
    BenchmarkReport,
    BenchmarkResult,
    CompressionLayerConfig,
    ExternalToolBenchmarkConfig,
    GraphMemoryLaneConfig,
    GraphMemoryLaneMode,
    GraphMemoryTool,
    HeadToHeadManifest,
    Strategy,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "headroom_artifacts"
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "headtohead" / "results"
_EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "evidence"
    / "r19-graft-graph-memory-comparison.json"
)
_GRAPH_MEMORY_LANE_DIRS = (
    "graphify_build_plus_query",
    "graphify_query_warm",
    "graft_build_plus_query",
    "graft_query_warm",
)


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


def test_reports_with_graph_memory_lanes_imports_artifacts() -> None:
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
            graph_memory_lanes=[
                GraphMemoryLaneConfig(
                    tool=GraphMemoryTool.GRAPHIFY,
                    mode=GraphMemoryLaneMode.BUILD_PLUS_QUERY,
                    package_name="graphifyy",
                    version="0.8.44",
                    command="graphify",
                    artifact_dir=str(build_dir),
                ),
                GraphMemoryLaneConfig(
                    tool=GraphMemoryTool.GRAPHIFY,
                    mode=GraphMemoryLaneMode.QUERY_WARM,
                    package_name="graphifyy",
                    version="0.8.44",
                    command="graphify",
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

        augmented = reports_with_graph_memory_lanes(manifest, reports)
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
    augmented = reports_with_graph_memory_lanes(manifest, reports)
    output = format_competitive_markdown(manifest, augmented)

    assert "| archex | retrieval |" in output
    assert "| archex_query_compressed | retrieval |" in output
    assert "| archex_query_efficiency_packed | retrieval |" in output
    assert "| ccc | retrieval |" in output
    assert "| graphify_build_plus_query | graph-memory |" in output
    assert "| graphify_query_warm | graph-memory |" in output
    assert "| graft_build_plus_query | graph-memory |" in output
    assert "| graft_query_warm | graph-memory |" in output
    assert "| raw-ripgrep/read | baseline |" in output
    assert "## Aggregate (19 tasks)" in output
    assert "## By task family" in output
    assert "## By repo" in output
    assert "package=@nanonets/graft; version=0.16.0; mode=build+query" in output
    assert "package=@nanonets/graft; version=0.16.0; mode=warm-query" in output


def test_checked_in_artifacts_have_no_absolute_path_leaks() -> None:
    artifact_paths = sorted(_RESULTS_DIR.glob("*.json"))
    for lane in _GRAPH_MEMORY_LANE_DIRS:
        artifact_paths += sorted((_RESULTS_DIR / lane).glob("*.json"))
    for path in artifact_paths:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text, path.name
        assert "/home/" not in text, path.name
        assert "/private/" not in text, path.name
        assert "/tmp/" not in text, path.name
        if path.parent == _RESULTS_DIR:
            BenchmarkReport.model_validate(json.loads(text))


def test_checked_in_graph_memory_lanes_cover_every_task() -> None:
    task_ids = {report.task_id for report in load_headtohead_results(_RESULTS_DIR)}

    for lane in _GRAPH_MEMORY_LANE_DIRS:
        covered = {path.stem for path in (_RESULTS_DIR / lane).glob("*.json")}
        assert covered == task_ids, lane


def test_checked_in_graft_artifacts_carry_pinned_provenance() -> None:
    manifest = load_headtohead_manifest(_RESULTS_DIR / "manifest.yaml")
    reports = reports_with_graph_memory_lanes(manifest, load_headtohead_results(_RESULTS_DIR))

    graft_results = [
        result
        for report in reports
        for result in report.results
        if (result.strategy_label or "").startswith("graft_")
    ]

    # 2 modes x 19 tasks, every cell recorded, none dropped.
    assert len(graft_results) == 38
    for result in graft_results:
        assert result.provenance["graph_memory_package"] == "@nanonets/graft"
        assert result.provenance["external_tool_version"] == "0.16.0"
        assert result.provenance["graft_npm_integrity"] == GRAFT_NPM_INTEGRITY
        assert result.provenance["graft_source_commit"] == GRAFT_SOURCE_COMMIT
        assert result.provenance["graft_rank_basis"] == "emitted_hits_order"
        assert result.provenance["graft_freshness_source"] == "check_json_graph"
        assert result.provenance["graft_no_refresh"] == "true"
        assert result.provenance["graft_deep_summaries"] == "false"
        assert result.provenance["graft_telemetry_disabled"] == "true"
        assert result.provenance["graft_result_cardinality"] == "10"
        assert result.provenance["graft_extraction_tier"] in {"depth", "breadth"}
        assert len(result.provenance["graft_output_digest"]) == 64

    tiers = [result.provenance["graft_extraction_tier"] for result in graft_results]
    # 17 depth-tier and 2 breadth-tier (Rust) tasks per mode, as the protocol froze.
    assert tiers.count("depth") == 34
    assert tiers.count("breadth") == 4


def test_checked_in_graft_artifact_tiers_match_each_task_language() -> None:
    """A swapped tier keeps the depth/breadth counts intact but corrupts the subset.

    The pre-declared secondary analysis is restricted to the depth-tier tasks, so each
    cell's tier has to match the languages its own task declares, not just aggregate.
    """
    manifest = load_headtohead_manifest(_RESULTS_DIR / "manifest.yaml")
    tasks_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "tasks"
    expected = {
        task.task_id: graft_extraction_tier(task.languages).value
        for task in select_headtohead_tasks(manifest, tasks_dir)
    }
    reports = reports_with_graph_memory_lanes(manifest, load_headtohead_results(_RESULTS_DIR))

    observed = {
        (report.task_id, result.strategy_label): result.provenance["graft_extraction_tier"]
        for report in reports
        for result in report.results
        if (result.strategy_label or "").startswith("graft_")
    }

    assert len(observed) == 38
    for (task_id, _lane), tier in observed.items():
        assert tier == expected[task_id]


def test_checked_in_graft_analysis_matches_the_artifacts() -> None:
    analysis = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    manifest = load_headtohead_manifest(_RESULTS_DIR / "manifest.yaml")
    reports = reports_with_graph_memory_lanes(manifest, load_headtohead_results(_RESULTS_DIR))

    treatment = [
        result
        for report in reports
        for result in report.results
        if result.strategy_label == "graft_query_warm"
    ]
    control = [
        result
        for report in reports
        for result in report.results
        if result.strategy is Strategy.ARCHEX_QUERY
    ]

    assert analysis["primary"]["tasks"] == len(treatment) == len(control) == 19
    assert analysis["primary"]["clusters"] == 15
    assert analysis["coverage"]["recorded_failures"] == 0
    assert analysis["bootstrap"] == {"resamples": 10000, "seed": 20260909, "unit": "repository"}
    assert analysis["margins"] == {
        "minimum_worthwhile_gain": 0.05,
        "non_inferiority": -0.05,
        "equivalence": 0.03,
    }
    assert analysis["primary"]["treatment_mean"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        mean(result.required_file_recall for result in treatment)
    )
    assert analysis["primary"]["control_mean"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        mean(result.required_file_recall for result in control)
    )
    assert analysis["primary"]["mean_difference"] == pytest.approx(  # pyright: ignore[reportUnknownMemberType]
        analysis["primary"]["treatment_mean"] - analysis["primary"]["control_mean"]
    )
