"""Tests for the Graphify competitive-lane adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from archex.benchmark.graphify import (
    GraphifyAdapterError,
    GraphifyUnavailableError,
    load_graphify_artifact,
    load_graphify_results,
    run_graphify_lane,
)
from archex.benchmark.models import (
    BenchmarkTask,
    GraphifyLaneConfig,
    GraphifyLaneName,
    Strategy,
    TaskCompletionResult,
)


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
    lane: str = "graphify_build_plus_query",
    includes_build_cost: bool = True,
    command: str,
    args: list[str] | None = None,
    artifact_dir: Path | None = None,
) -> GraphifyLaneConfig:
    return GraphifyLaneConfig(
        name=GraphifyLaneName(lane),
        version="0.8.44",
        command=command,
        args=args or [],
        includes_build_cost=includes_build_cost,
        artifact_dir=str(artifact_dir) if artifact_dir is not None else None,
    )


def _payload(
    *,
    lane: str,
    includes_build_cost: bool,
    version: str = "0.8.44",
) -> dict[str, object]:
    return {
        "task_id": "task_a",
        "lane": lane,
        "graphify_package": "graphifyy",
        "graphify_version": version,
        "command": (
            f"uvx --from graphifyy=={version} graphify query --graph graphify-out/graph.json"
        ),
        "includes_build_cost": includes_build_cost,
        "tokens_total": 320,
        "tokens_input": 800,
        "tokens_output": 320,
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
        "token_efficiency": 0.6,
        "token_efficiency_with_completion": 0.6,
        "cold_start_ms": 4200.0 if includes_build_cost else 0.0,
        "warm_latency_ms": 315.0,
        "wall_time_ms": 4515.0 if includes_build_cost else 315.0,
        "cache_state": "cold" if includes_build_cost else "warm",
        "cached": not includes_build_cost,
        "receipt_accuracy": None,
        "freshness_latency_ms": 0.0,
        "freshness_measured": False,
        "freshness_correct": False,
        "region_recall": 0.75,
        "line_recall": 0.5,
        "context_noise_ratio": 0.25,
        "bundle_compression_ratio": None,
        "operational_notes": "code-only graph build on local checkout",
        "local_offline_posture": "local code graph only",
        "backend": "local-ast",
        "timestamp": "2026-06-19T00:00:00Z",
    }


def _write_artifact(
    artifact_dir: Path,
    *,
    lane: str,
    includes_build_cost: bool,
    version: str = "0.8.44",
    overrides: dict[str, object] | None = None,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "task_a.json"
    payload = _payload(
        lane=lane,
        includes_build_cost=includes_build_cost,
        version=version,
    )
    if overrides:
        payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_graphify_lane_local_records_provenance_and_build_label(tmp_path: Path) -> None:
    script = tmp_path / "graphify_fixture.py"
    script.write_text(
        "import json, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "lane = payload['lane']\n"
        "includes = payload['graphify']['includes_build_cost']\n"
        "result = "
        + repr(_payload(lane="graphify_build_plus_query", includes_build_cost=True))
        + "\n"
        "result['lane'] = lane\n"
        "result['includes_build_cost'] = includes\n"
        "sys.stdout.write(json.dumps(result))\n",
        encoding="utf-8",
    )
    config = _config(command=sys.executable, args=[str(script)])

    result = run_graphify_lane(config, task=_task(), repo_path=tmp_path)

    assert result.strategy is Strategy.EXTERNAL_MCP
    assert result.strategy_label == "graphify_build_plus_query"
    assert result.cold_start_ms == 4200.0
    assert result.warm_latency_ms == 315.0
    assert result.provenance["external_tool"] == "graphify_build_plus_query"
    assert result.provenance["external_tool_version"] == "0.8.44"
    assert result.provenance["graphify_run_mode"] == "local"
    assert result.provenance["graphify_includes_build_cost"] == "true"
    assert "graphify query" in result.provenance["graphify_command"]


def test_run_graphify_lane_raises_when_unavailable_without_artifact_mode() -> None:
    config = _config(
        command="archex-graphify-missing-binary",
        lane="graphify_query_warm",
        includes_build_cost=False,
    )

    with pytest.raises(GraphifyUnavailableError, match="artifact_dir"):
        run_graphify_lane(config, task=_task(), repo_path=Path("."))


def test_run_graphify_lane_uses_artifact_mode_without_binary(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "graphify-warm"
    _write_artifact(
        artifact_dir,
        lane="graphify_query_warm",
        includes_build_cost=False,
    )
    config = _config(
        command="archex-graphify-missing-binary",
        lane="graphify_query_warm",
        includes_build_cost=False,
        artifact_dir=artifact_dir,
    )

    result = run_graphify_lane(config, task=_task(), repo_path=tmp_path)

    assert result.strategy_label == "graphify_query_warm"
    assert result.cold_start_ms == 0.0
    assert result.warm_latency_ms == 315.0
    assert result.provenance["graphify_run_mode"] == "artifact"
    assert result.provenance["graphify_includes_build_cost"] == "false"
    assert len(result.provenance["graphify_artifact_sha256"]) == 64


def test_load_graphify_artifact_rejects_version_mismatch(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "graphify-build"
    _write_artifact(
        artifact_dir,
        lane="graphify_build_plus_query",
        includes_build_cost=True,
        version="9.9.9",
    )
    config = _config(command="graphify", artifact_dir=artifact_dir)

    with pytest.raises(GraphifyAdapterError, match="does not match pinned version"):
        load_graphify_artifact(config, task_id="task_a", artifact_dir=artifact_dir)


def test_load_graphify_artifact_rejects_warm_lane_with_cold_start_cost(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "graphify-warm"
    _write_artifact(
        artifact_dir,
        lane="graphify_query_warm",
        includes_build_cost=False,
        overrides={
            "cold_start_ms": 99.0,
            "wall_time_ms": 414.0,
        },
    )
    config = _config(
        command="graphify",
        lane="graphify_query_warm",
        includes_build_cost=False,
        artifact_dir=artifact_dir,
    )

    with pytest.raises(GraphifyAdapterError, match="must report cold_start_ms == 0"):
        load_graphify_artifact(config, task_id="task_a", artifact_dir=artifact_dir)


def test_load_graphify_artifact_rejects_build_lane_without_cold_start_cost(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "graphify-build"
    _write_artifact(
        artifact_dir,
        lane="graphify_build_plus_query",
        includes_build_cost=True,
        overrides={
            "cold_start_ms": 0.0,
        },
    )
    config = _config(command="graphify", artifact_dir=artifact_dir)

    with pytest.raises(GraphifyAdapterError, match="must report cold_start_ms > 0"):
        load_graphify_artifact(config, task_id="task_a", artifact_dir=artifact_dir)


def test_load_graphify_results_import_is_deterministic(tmp_path: Path) -> None:
    build_dir = tmp_path / "graphify-build"
    warm_dir = tmp_path / "graphify-warm"
    _write_artifact(
        build_dir,
        lane="graphify_build_plus_query",
        includes_build_cost=True,
    )
    _write_artifact(
        warm_dir,
        lane="graphify_query_warm",
        includes_build_cost=False,
    )
    results = load_graphify_results(
        [
            _config(command="graphify", artifact_dir=build_dir),
            _config(
                command="graphify",
                lane="graphify_query_warm",
                includes_build_cost=False,
                artifact_dir=warm_dir,
            ),
        ],
        ["task_a", "missing_task"],
    )

    by_lane = {result.strategy_label: result for result in results}
    assert set(by_lane) == {"graphify_build_plus_query", "graphify_query_warm"}
    assert by_lane["graphify_build_plus_query"].provenance["graphify_artifact_sha256"]
    assert by_lane["graphify_query_warm"].provenance["graphify_artifact_sha256"]
    assert (
        by_lane["graphify_build_plus_query"].provenance["graphify_artifact_sha256"]
        != by_lane["graphify_query_warm"].provenance["graphify_artifact_sha256"]
    )
    assert all(result.task_completion_result is TaskCompletionResult.PASS for result in results)
