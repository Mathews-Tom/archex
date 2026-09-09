"""Tests for the pinned Graft graph-memory comparison-lane adapter."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from archex.benchmark.graft import (
    GRAFT_NPM_INTEGRITY,
    GRAFT_SOURCE_COMMIT,
    GraftExtractionTier,
    graft_extraction_tier,
    load_graft_artifact,
    parse_graft_ask_output,
    parse_graft_check_output,
)
from archex.benchmark.graph_memory import GraphMemoryAdapterError
from archex.benchmark.headtohead import load_graph_memory_results
from archex.benchmark.models import (
    BenchmarkTask,
    GraphMemoryLaneConfig,
    GraphMemoryLaneMode,
    GraphMemoryTool,
    Strategy,
)

_RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "run_graft_headtohead_lane.py"

# Two symbol hits and one whole-file hit, with scores deliberately out of order:
# the pinned release emits `hits` in rank order while `hits[].score` is not
# monotonically descending, so rank must come from the emitted order.
_ASK_JSON = json.dumps(
    {
        "query": "how is auth registered",
        "mode": "lexical",
        "hits": [
            {
                "kind": "symbol",
                "title": "register_auth · function",
                "pointer": "src/auth.py:L10-L20",
                "score": 1.5,
                "code": "def register_auth(app):\n    app.use(auth)\n",
            },
            {
                "kind": "symbol",
                "title": "server.py · file",
                "pointer": "src/server.py",
                "score": 0.97,
            },
            {
                "kind": "symbol",
                "title": "Auth · class",
                "pointer": "src/auth.py:L1-L8",
                "score": 1.42,
                "code": "class Auth:\n    pass\n",
            },
        ],
        "coverage": 1,
        "coverageStrong": 0.44,
    }
).encode("utf-8")

_CHECK_JSON = json.dumps(
    {
        "context": {"ok": False, "missing": True, "coverage": []},
        "graph": {"ok": True, "missing": False, "added": [], "nodes": 7},
    }
).encode("utf-8")


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="task_a",
        repo="owner/repo",
        commit="abc123",
        question="Where is the auth middleware registered?",
        expected_files=["src/auth.py", "src/server.py"],
        languages=["python"],
    )


def _config(
    *,
    mode: GraphMemoryLaneMode = GraphMemoryLaneMode.BUILD_PLUS_QUERY,
    version: str = "0.16.0",
    command: str = "graft",
    artifact_dir: Path | None = None,
) -> GraphMemoryLaneConfig:
    return GraphMemoryLaneConfig(
        tool=GraphMemoryTool.GRAFT,
        mode=mode,
        package_name="@nanonets/graft",
        version=version,
        command=command,
        artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
    )


def _artifact_payload(
    *,
    mode: GraphMemoryLaneMode = GraphMemoryLaneMode.BUILD_PLUS_QUERY,
) -> dict[str, object]:
    includes_build_cost = mode is GraphMemoryLaneMode.BUILD_PLUS_QUERY
    return {
        "task_id": "task_a",
        "lane": f"graft_{mode.value}",
        "command": "graft --dir <graph-dir> ask '...' -n 10 --source --no-refresh --json <repo>",
        "includes_build_cost": includes_build_cost,
        "graft_package": "@nanonets/graft",
        "graft_version": "0.16.0",
        "npm_integrity": GRAFT_NPM_INTEGRITY,
        "source_commit": GRAFT_SOURCE_COMMIT,
        "status": "ok",
        "timing_mode": mode.value,
        "extraction_tier": "depth",
        "query_mode": "lexical",
        "rank_basis": "emitted_hits_order",
        "freshness_source": "check_json_graph",
        "result_cardinality": 10,
        "deep_summaries": False,
        "no_refresh": True,
        "graph_dir_outside_repo": True,
        "telemetry_disabled": True,
        "returned_units": 3,
        "symbol_hits": 2,
        "whole_file_hits": 1,
        "returned_source_units": 2,
        "graph_ok": True,
        "graph_nodes": 7,
        "output_digest": "a" * 64,
        "tokens_total": 120,
        "tokens_input": 800,
        "tokens_output": 120,
        "tool_calls": 2 if includes_build_cost else 1,
        "files_accessed": 2,
        "recall": 1.0,
        "precision": 0.5,
        "f1_score": 0.67,
        "mrr": 1.0,
        "ndcg": 1.0,
        "map_score": 1.0,
        "required_file_recall": 1.0,
        "missed_required_file_rate": 0.0,
        "missed_required_task_rate": 0.0,
        "all_required_files_present": True,
        "required_files_present": ["src/auth.py", "src/server.py"],
        "required_files_missing": [],
        "result_files": ["src/auth.py", "src/server.py"],
        "task_completion_result": "pass",
        "bundle_completion_tokens": 0,
        "bundle_completion_files": [],
        "token_efficiency": 0.85,
        "token_efficiency_with_completion": 0.85,
        "cold_start_ms": 4200.0 if includes_build_cost else 0.0,
        "warm_latency_ms": 210.0,
        "wall_time_ms": 4410.0 if includes_build_cost else 210.0,
        "cache_state": "cold" if includes_build_cost else "warm",
        "cached": not includes_build_cost,
        "freshness_latency_ms": 30.0,
        "freshness_measured": True,
        "freshness_correct": True,
        "timestamp": "2026-09-09T00:00:00Z",
        "backend": "graft-structural",
        "local_offline_posture": "local structural graph only; no model call",
    }


def _write_artifact(
    artifact_dir: Path,
    *,
    mode: GraphMemoryLaneMode = GraphMemoryLaneMode.BUILD_PLUS_QUERY,
    overrides: dict[str, object] | None = None,
    task_id: str = "task_a",
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = _artifact_payload(mode=mode)
    payload["task_id"] = task_id
    if overrides:
        payload.update(overrides)
    path = artifact_dir / f"{task_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_graft_extraction_tier_separates_native_and_wasm_coverage() -> None:
    assert graft_extraction_tier(["python"]) is GraftExtractionTier.DEPTH
    assert graft_extraction_tier(["go"]) is GraftExtractionTier.DEPTH
    assert graft_extraction_tier(["javascript"]) is GraftExtractionTier.DEPTH
    # The two Rust tasks are covered only by the signature-only WASM breadth tier.
    assert graft_extraction_tier(["rust"]) is GraftExtractionTier.BREADTH
    assert graft_extraction_tier(["python", "rust"]) is GraftExtractionTier.BREADTH
    # An unlabeled task cannot claim depth coverage.
    assert graft_extraction_tier(None) is GraftExtractionTier.BREADTH


def test_parse_graft_ask_output_ranks_by_emitted_order_not_score() -> None:
    ask = parse_graft_ask_output(_ASK_JSON)

    assert [hit.rank for hit in ask.hits] == [1, 2, 3]
    assert [hit.pointer for hit in ask.hits] == [
        "src/auth.py:L10-L20",
        "src/server.py",
        "src/auth.py:L1-L8",
    ]
    # Returned files keep the emitted order and are deduplicated by path.
    assert ask.result_files == ("src/auth.py", "src/server.py")
    assert ask.query_mode == "lexical"


def test_parse_graft_ask_output_excludes_whole_file_hits_from_returned_source() -> None:
    ask = parse_graft_ask_output(_ASK_JSON)

    whole_file = ask.hits[1]
    assert whole_file.is_whole_file
    assert whole_file.code is None
    assert whole_file.path == "src/server.py"
    assert ask.symbol_hits == 2
    assert ask.whole_file_hits == 1
    # A whole-file hit names a required file but contributes no returned source.
    assert [hit.pointer for hit in ask.source_bearing_hits] == [
        "src/auth.py:L10-L20",
        "src/auth.py:L1-L8",
    ]


def test_parse_graft_ask_output_rejects_unattributable_hit() -> None:
    payload = json.dumps({"mode": "lexical", "hits": [{"title": "mystery", "score": 1.0}]})

    with pytest.raises(GraphMemoryAdapterError, match="never inferred"):
        parse_graft_ask_output(payload.encode("utf-8"))


def test_parse_graft_check_output_uses_graph_section_only() -> None:
    freshness = parse_graft_check_output(_CHECK_JSON)

    assert freshness.ok is True
    assert freshness.missing is False
    assert freshness.nodes == 7


def test_parse_graft_check_output_rejects_output_without_graph_section() -> None:
    payload = json.dumps({"context": {"ok": False, "missing": True}})

    with pytest.raises(GraphMemoryAdapterError, match="no 'graph' section"):
        parse_graft_check_output(payload.encode("utf-8"))


def test_load_graft_artifact_records_pinned_provenance(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "graft-build"
    _write_artifact(artifact_dir)
    config = _config(artifact_dir=artifact_dir)

    result = load_graft_artifact(config, task_id="task_a", artifact_dir=artifact_dir)

    assert result.strategy is Strategy.EXTERNAL_MCP
    assert result.strategy_label == "graft_build_plus_query"
    assert result.provenance["graph_memory_tool"] == "graft"
    assert result.provenance["graph_memory_mode"] == "build_plus_query"
    assert result.provenance["graph_memory_package"] == "@nanonets/graft"
    assert result.provenance["external_tool_version"] == "0.16.0"
    assert result.provenance["graft_npm_integrity"] == GRAFT_NPM_INTEGRITY
    assert result.provenance["graft_source_commit"] == GRAFT_SOURCE_COMMIT
    assert result.provenance["graft_extraction_tier"] == "depth"
    assert result.provenance["graft_rank_basis"] == "emitted_hits_order"
    assert result.provenance["graft_freshness_source"] == "check_json_graph"
    assert result.provenance["graft_no_refresh"] == "true"
    assert result.provenance["graft_deep_summaries"] == "false"
    assert result.provenance["graft_telemetry_disabled"] == "true"
    assert result.provenance["graft_symbol_hits"] == "2"
    assert result.provenance["graft_whole_file_hits"] == "1"
    assert result.provenance["graft_returned_source_units"] == "2"
    assert result.provenance["graft_status"] == "ok"
    assert len(result.provenance["graph_memory_artifact_sha256"]) == 64
    assert result.cold_start_ms == 4200.0
    assert result.warm_latency_ms == 210.0
    assert result.freshness_latency_ms == 30.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"graft_version": "0.15.0"}, "does not match pinned version"),
        ({"npm_integrity": "sha512-tampered"}, "does not match the pinned released artifact"),
        ({"source_commit": "0" * 40}, "does not match the pinned"),
        ({"timing_mode": "query_warm"}, "does not match lane mode"),
        ({"rank_basis": "score"}, "hits\\[\\].score is not monotonically descending"),
        ({"freshness_source": "check_json_context"}, "freshness_source must be"),
        ({"result_cardinality": 8}, "does not match the frozen"),
        ({"deep_summaries": True}, "paid --deep summaries"),
        ({"no_refresh": False}, "must be measured with --no-refresh"),
        ({"graph_dir_outside_repo": False}, "outside the"),
        ({"telemetry_disabled": False}, "telemetry disabled"),
        ({"symbol_hits": 1}, "does not equal symbol_hits"),
        ({"returned_source_units": 3}, "more returned_source_units than symbol"),
        ({"output_digest": "zz"}, "lowercase SHA-256 digest"),
        ({"failure_reason": "flaky"}, "records a failure_reason"),
        ({"result_files": []}, "returned file\\(s\\) disagree"),
        ({"cold_start_ms": 0.0}, "must report cold_start_ms > 0"),
    ],
)
def test_load_graft_artifact_fails_closed_on_contract_violation(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    artifact_dir = tmp_path / "graft-build"
    _write_artifact(artifact_dir, overrides=overrides)
    config = _config(artifact_dir=artifact_dir)

    with pytest.raises(GraphMemoryAdapterError, match=message):
        load_graft_artifact(config, task_id="task_a", artifact_dir=artifact_dir)


def test_load_graft_artifact_accepts_recorded_failure_cell(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "graft-build"
    _write_artifact(
        artifact_dir,
        overrides={
            "status": "failed",
            "failure_reason": "graft ask hit 1 has no pointer",
            "query_mode": "unattributable",
            "returned_units": 0,
            "symbol_hits": 0,
            "whole_file_hits": 0,
            "returned_source_units": 0,
            "result_files": [],
            "required_files_present": [],
            "required_files_missing": ["src/auth.py", "src/server.py"],
            "all_required_files_present": False,
            "recall": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "map_score": 0.0,
            "required_file_recall": 0.0,
            "missed_required_file_rate": 1.0,
            "missed_required_task_rate": 1.0,
            "token_efficiency": 0.0,
            "token_efficiency_with_completion": 0.0,
            "tokens_total": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "files_accessed": 0,
            "cold_start_ms": 0.0,
            "warm_latency_ms": 0.0,
            "wall_time_ms": 0.0,
        },
    )
    config = _config(artifact_dir=artifact_dir)

    result = load_graft_artifact(config, task_id="task_a", artifact_dir=artifact_dir)

    assert result.provenance["graft_status"] == "failed"
    assert result.provenance["graft_failure_reason"] == "graft ask hit 1 has no pointer"
    assert result.required_file_recall == 0.0


def test_load_graft_artifact_rejects_failure_cell_reporting_metrics(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "graft-build"
    _write_artifact(
        artifact_dir,
        overrides={"status": "failed", "failure_reason": "timeout"},
    )
    config = _config(artifact_dir=artifact_dir)

    with pytest.raises(GraphMemoryAdapterError, match="reports returned files"):
        load_graft_artifact(config, task_id="task_a", artifact_dir=artifact_dir)


def test_load_graft_artifact_rejects_mislabeled_extraction_tier(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "graft-build"
    _write_artifact(artifact_dir, overrides={"extraction_tier": "depth"})
    config = _config(artifact_dir=artifact_dir)

    with pytest.raises(GraphMemoryAdapterError, match="corrupts the depth-tier subset"):
        load_graft_artifact(
            config,
            task_id="task_a",
            artifact_dir=artifact_dir,
            expected_tier=GraftExtractionTier.BREADTH,
        )


def test_load_graph_memory_results_routes_graft_lanes(tmp_path: Path) -> None:
    build_dir = tmp_path / "graft-build"
    warm_dir = tmp_path / "graft-warm"
    _write_artifact(build_dir)
    _write_artifact(warm_dir, mode=GraphMemoryLaneMode.QUERY_WARM)

    results = load_graph_memory_results(
        [
            _config(artifact_dir=build_dir),
            _config(mode=GraphMemoryLaneMode.QUERY_WARM, artifact_dir=warm_dir),
        ],
        ["task_a"],
    )

    by_lane = {result.strategy_label: result for result in results}
    assert set(by_lane) == {"graft_build_plus_query", "graft_query_warm"}
    cold_start = by_lane["graft_build_plus_query"].cold_start_ms
    assert cold_start is not None and cold_start > 0.0
    assert by_lane["graft_query_warm"].cold_start_ms == 0.0
    assert by_lane["graft_query_warm"].cached is True


def _fake_graft(tmp_path: Path) -> Path:
    """A stand-in `graft` that replays the pinned release's observed output shapes."""
    binary = tmp_path / "fake-graft"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "argv = sys.argv[1:]\n"
        "graph_dir = pathlib.Path(argv[argv.index('--dir') + 1])\n"
        "command = argv[2]\n"
        "if command == 'build':\n"
        "    graph_dir.mkdir(parents=True, exist_ok=True)\n"
        "    (graph_dir / 'graph.json').write_text('{}')\n"
        "    sys.stdout.write('wiring: 7 nodes\\n')\n"
        "elif command == 'ask':\n"
        f"    sys.stdout.write({_ASK_JSON.decode('utf-8')!r})\n"
        "elif command == 'check':\n"
        "    built = (graph_dir / 'graph.json').is_file()\n"
        "    sys.stdout.write(json.dumps({'context': {'ok': False, 'missing': True},\n"
        "        'graph': {'ok': built, 'missing': not built, 'nodes': 7 if built else 0}}))\n"
        "else:\n"
        "    sys.stderr.write('unexpected command\\n')\n"
        "    raise SystemExit(2)\n",
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return binary


def _fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "auth.py").write_text(
        "class Auth:\n    pass\n\n\ndef register_auth(app):\n    app.use(auth)\n",
        encoding="utf-8",
    )
    (repo / "src" / "server.py").write_text("from .auth import register_auth\n", encoding="utf-8")
    return repo


def _run_runner(
    *,
    repo: Path,
    graph_dir: Path,
    mode: GraphMemoryLaneMode,
    binary: Path,
) -> subprocess.CompletedProcess[str]:
    payload = {
        "task": _task().model_dump(mode="json"),
        "repo_path": str(repo),
        "graph_dir": str(graph_dir),
        "lane": f"graft_{mode.value}",
        "tool": "graft",
        "mode": mode.value,
        "graft": {
            "package_name": "@nanonets/graft",
            "version": "0.16.0",
            "npm_integrity": GRAFT_NPM_INTEGRITY,
            "source_commit": GRAFT_SOURCE_COMMIT,
            "result_cardinality": 10,
        },
        "binary": str(binary),
    }
    return subprocess.run(
        [sys.executable, str(_RUNNER)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CI": "1", "DO_NOT_TRACK": "1"},
    )


def test_graft_runner_produces_importable_cold_and_warm_cells(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    graph_dir = tmp_path / "graft-index"
    binary = _fake_graft(tmp_path)

    cold = _run_runner(
        repo=repo,
        graph_dir=graph_dir,
        mode=GraphMemoryLaneMode.BUILD_PLUS_QUERY,
        binary=binary,
    )
    assert cold.returncode == 0, cold.stderr
    warm = _run_runner(
        repo=repo,
        graph_dir=graph_dir,
        mode=GraphMemoryLaneMode.QUERY_WARM,
        binary=binary,
    )
    assert warm.returncode == 0, warm.stderr

    build_dir = tmp_path / "cells" / "graft_build_plus_query"
    warm_dir = tmp_path / "cells" / "graft_query_warm"
    build_dir.mkdir(parents=True)
    warm_dir.mkdir(parents=True)
    (build_dir / "task_a.json").write_text(cold.stdout, encoding="utf-8")
    (warm_dir / "task_a.json").write_text(warm.stdout, encoding="utf-8")

    cold_result = load_graft_artifact(
        _config(artifact_dir=build_dir), task_id="task_a", artifact_dir=build_dir
    )
    warm_result = load_graft_artifact(
        _config(mode=GraphMemoryLaneMode.QUERY_WARM, artifact_dir=warm_dir),
        task_id="task_a",
        artifact_dir=warm_dir,
    )

    # Cold pays build cost; warm pays none, and neither folds the freshness probe in.
    assert cold_result.cold_start_ms is not None and cold_result.cold_start_ms > 0.0
    assert cold_result.warm_latency_ms is not None and cold_result.warm_latency_ms > 0.0
    assert warm_result.cold_start_ms == 0.0
    assert warm_result.warm_latency_ms is not None and warm_result.warm_latency_ms > 0.0
    assert warm_result.wall_time_ms == warm_result.warm_latency_ms
    assert warm_result.freshness_latency_ms > 0.0
    assert warm_result.freshness_measured is True
    assert warm_result.freshness_correct is True

    # Both required files are recovered, one by a symbol hit and one whole-file hit.
    assert cold_result.required_file_recall == 1.0
    assert cold_result.result_files == ["src/auth.py", "src/server.py"]
    assert cold_result.provenance["graft_returned_units"] == "3"
    assert cold_result.provenance["graft_symbol_hits"] == "2"
    assert cold_result.provenance["graft_whole_file_hits"] == "1"
    assert cold_result.provenance["graft_returned_source_units"] == "2"
    assert cold_result.provenance["graft_query_mode"] == "lexical"
    assert cold_result.provenance["graft_graph_nodes"] == "7"
    assert cold_result.tokens_total > 0
    assert "--no-refresh" in cold_result.provenance["graph_memory_command"]
    assert "--deep" not in cold_result.provenance["graph_memory_command"]


def test_graft_runner_refuses_warm_lane_without_prebuilt_graph(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    binary = _fake_graft(tmp_path)

    warm = _run_runner(
        repo=repo,
        graph_dir=tmp_path / "empty-index",
        mode=GraphMemoryLaneMode.QUERY_WARM,
        binary=binary,
    )

    assert warm.returncode != 0
    assert "no prebuilt graph" in warm.stderr


def test_graft_runner_refuses_graph_dir_inside_task_repository(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    binary = _fake_graft(tmp_path)

    cold = _run_runner(
        repo=repo,
        graph_dir=repo / "graft",
        mode=GraphMemoryLaneMode.BUILD_PLUS_QUERY,
        binary=binary,
    )

    assert cold.returncode != 0
    assert "inside the task repository" in cold.stderr
