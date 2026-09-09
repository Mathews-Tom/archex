"""Run one pinned Graft graph-memory lane from the adapter stdin contract.

Reads a JSON payload on stdin with:
- ``task``: BenchmarkTask JSON
- ``repo_path``: checked-out repository path (never modified)
- ``graph_dir``: graph directory, which must live outside ``repo_path``
- ``lane``: graft_build_plus_query | graft_query_warm
- ``tool``: graph-memory tool id (``graft``)
- ``mode``: build_plus_query | query_warm
- ``graft``: {package_name, version, npm_integrity, source_commit, result_cardinality}
- ``binary`` (optional): graft executable, default ``graft``

Writes one GraftArtifact JSON document to stdout.

This script owns the frozen protocol in
`benchmarks/preregistrations/R19-graft-graph-memory-comparison.md`: structural
`build` (no ``--deep``), a graph directory outside the repository with
``--no-gitignore --no-ignore``, ``ask ... --no-refresh`` for every measured
query, and a freshness probe that is measured separately and never folded into
warm latency. The cold lane builds then asks; the warm lane asks only, against
the graph the cold lane already built.

A cell that cannot be measured is written as a recorded failure artifact
(``status: failed``) instead of being dropped. A protocol violation the operator
must fix — a graph directory inside the repository, or a warm lane with no
prebuilt graph — exits non-zero instead.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from archex.benchmark.graft import (
    GraftArtifact,
    GraftAskOutput,
    GraftCellStatus,
    GraftGraphFreshness,
    graft_extraction_tier,
    parse_graft_ask_output,
    parse_graft_check_output,
)
from archex.benchmark.graph_memory import GraphMemoryAdapterError, artifact_digest, now_iso
from archex.benchmark.models import BenchmarkTask, GraphMemoryLaneMode
from archex.benchmark.region_metrics import ReturnedRegion, compute_region_metrics
from archex.benchmark.strategies import (
    completion_result_from_missing,
    compute_bundle_completion_penalty,
    compute_f1,
    compute_map,
    compute_mrr,
    compute_ndcg,
    compute_precision,
    compute_recall,
    compute_required_file_metrics,
    compute_token_efficiency,
    count_file_tokens,
)
from archex.reporting import count_tokens


class GraftProtocolError(RuntimeError):
    """Raised when the operator's invocation violates the frozen protocol."""


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["CI"] = "1"
    env["DO_NOT_TRACK"] = "1"
    return env


def _run(command: list[str]) -> tuple[subprocess.CompletedProcess[bytes], float]:
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, check=False, env=_env())
    return completed, (time.perf_counter() - started) * 1000


def _require_success(completed: subprocess.CompletedProcess[bytes], command: list[str]) -> None:
    if completed.returncode != 0:
        detail = (
            completed.stderr.decode("utf-8", "replace").strip()
            or completed.stdout.decode("utf-8", "replace").strip()
            or "no output"
        )
        raise GraphMemoryAdapterError(
            f"{shlex.join(command)} failed with exit {completed.returncode}: {detail}"
        )


def _sanitized_reason(exc: Exception, *, repo_path: Path, graph_dir: Path) -> str:
    """A recorded failure is published, so its reason carries no machine paths.

    The artifact's ``command`` is already written with ``<repo>``/``<graph-dir>``
    placeholders; a failure reason quoting the resolved command or a tool's stderr would
    otherwise leak the operator's home directory into checked-in evidence.
    """
    reason = str(exc).replace(str(repo_path), "<repo>").replace(str(graph_dir), "<graph-dir>")
    home = str(Path.home())
    return reason.replace(home, "<home>")


def _returned_regions(ask: GraftAskOutput) -> list[ReturnedRegion]:
    """Ranked returned units. Whole-file hits carry no source, so they contribute none."""
    regions: list[ReturnedRegion] = []
    for hit in ask.hits:
        if hit.start_line is None or hit.end_line is None or not hit.code:
            continue
        regions.append(
            ReturnedRegion(
                path=hit.path,
                start_line=hit.start_line,
                end_line=hit.end_line,
                symbol=hit.title,
                tokens=max(1, count_tokens(hit.code)),
            )
        )
    return regions


def _command_text(*, question: str, cardinality: int, includes_build_cost: bool) -> str:
    """The sanitized command shape, naming the pinned tool rather than a local path."""
    ask = (
        f"graft --dir <graph-dir> ask {shlex.quote(question)} -n {cardinality} "
        "--source --no-refresh --json <repo>"
    )
    if includes_build_cost:
        return f"graft --dir <graph-dir> build --no-gitignore --no-ignore <repo> && {ask}"
    return ask


def _failure_artifact(
    *,
    task: BenchmarkTask,
    lane: str,
    mode: GraphMemoryLaneMode,
    graft: dict[str, object],
    command_text: str,
    reason: str,
    output_digest: str,
    query_mode: str,
) -> GraftArtifact:
    includes_build_cost = mode is GraphMemoryLaneMode.BUILD_PLUS_QUERY
    return GraftArtifact(
        task_id=task.task_id,
        lane=lane,
        command=command_text,
        includes_build_cost=includes_build_cost,
        graft_package=str(graft["package_name"]),
        graft_version=str(graft["version"]),
        npm_integrity=str(graft["npm_integrity"]),
        source_commit=str(graft["source_commit"]),
        status=GraftCellStatus.FAILED,
        failure_reason=reason,
        timing_mode=mode.value,
        extraction_tier=graft_extraction_tier(task.languages),
        query_mode=query_mode,
        result_cardinality=int(str(graft["result_cardinality"])),
        returned_units=0,
        symbol_hits=0,
        whole_file_hits=0,
        returned_source_units=0,
        output_digest=output_digest,
        tokens_total=0,
        tokens_input=0,
        tokens_output=0,
        tool_calls=2 if includes_build_cost else 1,
        files_accessed=0,
        recall=0.0,
        precision=0.0,
        f1_score=0.0,
        mrr=0.0,
        ndcg=0.0,
        map_score=0.0,
        required_file_recall=0.0,
        missed_required_file_rate=1.0,
        missed_required_task_rate=1.0,
        all_required_files_present=False,
        token_efficiency=0.0,
        token_efficiency_with_completion=0.0,
        cold_start_ms=0.0,
        warm_latency_ms=0.0,
        wall_time_ms=0.0,
        cache_state="cold" if includes_build_cost else "warm",
        cached=not includes_build_cost,
        timestamp=now_iso(),
        operational_notes="recorded failure; retained and counted, never dropped",
        local_offline_posture="local structural graph only; no model call",
        backend="graft-structural",
    )


def main() -> int:
    payload = json.loads(sys.stdin.read())
    task = BenchmarkTask.model_validate(payload["task"])
    repo_path = Path(payload["repo_path"]).resolve()
    graph_dir = Path(payload["graph_dir"]).resolve()
    lane = str(payload["lane"])
    mode = GraphMemoryLaneMode(str(payload["mode"]))
    graft = dict(payload["graft"])
    binary = str(payload.get("binary", "graft"))
    cardinality = int(str(graft["result_cardinality"]))
    includes_build_cost = mode is GraphMemoryLaneMode.BUILD_PLUS_QUERY

    if graph_dir == repo_path or repo_path in graph_dir.parents:
        raise GraftProtocolError(
            f"graph_dir {graph_dir} is inside the task repository {repo_path}; the measured "
            "checkout must stay pristine"
        )

    command_text = _command_text(
        question=task.question,
        cardinality=cardinality,
        includes_build_cost=includes_build_cost,
    )

    build_command = [
        binary,
        "--dir",
        str(graph_dir),
        "build",
        "--no-gitignore",
        "--no-ignore",
        str(repo_path),
    ]
    ask_command = [
        binary,
        "--dir",
        str(graph_dir),
        "ask",
        task.question,
        "-n",
        str(cardinality),
        "--source",
        "--no-refresh",
        "--json",
        str(repo_path),
    ]
    check_command = [binary, "--dir", str(graph_dir), "check", "--json", str(repo_path)]

    build_ms = 0.0
    if includes_build_cost:
        graph_dir.parent.mkdir(parents=True, exist_ok=True)
        completed, build_ms = _run(build_command)
        _require_success(completed, build_command)
    else:
        probe, _ = _run(check_command)
        _require_success(probe, check_command)
        if parse_graft_check_output(probe.stdout).missing:
            raise GraftProtocolError(
                f"warm lane {lane!r} found no prebuilt graph in {graph_dir}; run the "
                "build_plus_query lane for this task first"
            )

    ask_completed, ask_ms = _run(ask_command)
    ask_digest_source = ask_completed.stdout
    try:
        _require_success(ask_completed, ask_command)
        ask = parse_graft_ask_output(ask_completed.stdout)
    except GraphMemoryAdapterError as exc:
        artifact = _failure_artifact(
            task=task,
            lane=lane,
            mode=mode,
            graft=graft,
            command_text=command_text,
            reason=_sanitized_reason(exc, repo_path=repo_path, graph_dir=graph_dir),
            output_digest=artifact_digest(ask_digest_source),
            query_mode="unattributable",
        )
        sys.stdout.write(artifact.model_dump_json())
        return 0

    # The freshness probe is measured separately and never folded into query latency. It is an
    # exploratory signal, so a probe failure is recorded rather than allowed to discard an
    # already-attributable primary-metric measurement.
    check_completed, check_ms = _run(check_command)
    freshness: GraftGraphFreshness | None = None
    if check_completed.returncode == 0:
        try:
            freshness = parse_graft_check_output(check_completed.stdout)
        except GraphMemoryAdapterError:
            freshness = None

    result_files = list(ask.result_files)
    result_file_set = set(result_files)
    source_hits = ask.source_bearing_hits
    tokens_output = sum(count_tokens(hit.code or "") for hit in source_hits)
    # Whole-file hits return no source, so per the frozen protocol they contribute to neither
    # the returned-source count nor the token-efficiency baseline.
    source_files = sorted({hit.path for hit in source_hits})
    tokens_input = count_file_tokens(repo_path, source_files) if source_files else 0
    completion_tokens, completion_files = compute_bundle_completion_penalty(
        repo_path,
        result_file_set,
        task.expected_files,
    )
    (
        required_file_recall,
        missed_required_file_rate,
        missed_required_task_rate,
        all_required_files_present,
        present,
        missing,
    ) = compute_required_file_metrics(result_file_set, task.expected_files)
    recall = compute_recall(result_file_set, task.expected_files)
    precision = compute_precision(result_file_set, task.expected_files)
    region_metrics = (
        compute_region_metrics(_returned_regions(ask), task.expected_regions)
        if task.expected_regions
        else None
    )

    artifact = GraftArtifact(
        task_id=task.task_id,
        lane=lane,
        command=command_text,
        includes_build_cost=includes_build_cost,
        graft_package=str(graft["package_name"]),
        graft_version=str(graft["version"]),
        npm_integrity=str(graft["npm_integrity"]),
        source_commit=str(graft["source_commit"]),
        status=GraftCellStatus.OK,
        timing_mode=mode.value,
        extraction_tier=graft_extraction_tier(task.languages),
        query_mode=ask.query_mode,
        result_cardinality=cardinality,
        returned_units=len(ask.hits),
        symbol_hits=ask.symbol_hits,
        whole_file_hits=ask.whole_file_hits,
        returned_source_units=len(source_hits),
        graph_ok=freshness.ok if freshness is not None else False,
        graph_nodes=freshness.nodes if freshness is not None else 0,
        output_digest=ask.digest,
        tokens_total=tokens_output,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tool_calls=2 if includes_build_cost else 1,
        files_accessed=len(result_file_set),
        recall=recall,
        precision=precision,
        f1_score=compute_f1(recall, precision),
        mrr=compute_mrr(result_files, task.expected_files),
        ndcg=compute_ndcg(result_files, task.expected_files),
        map_score=compute_map(result_files, task.expected_files),
        required_file_recall=required_file_recall,
        missed_required_file_rate=missed_required_file_rate,
        missed_required_task_rate=missed_required_task_rate,
        all_required_files_present=all_required_files_present,
        required_files_present=present,
        required_files_missing=missing,
        result_files=result_files,
        task_completion_result=completion_result_from_missing(completion_files),
        bundle_completion_tokens=completion_tokens,
        bundle_completion_files=completion_files,
        token_efficiency=compute_token_efficiency(tokens_output, tokens_input),
        token_efficiency_with_completion=compute_token_efficiency(
            tokens_output + completion_tokens,
            tokens_input + completion_tokens,
        ),
        cold_start_ms=build_ms,
        warm_latency_ms=ask_ms,
        wall_time_ms=build_ms + ask_ms,
        cache_state="cold" if includes_build_cost else "warm",
        cached=not includes_build_cost,
        freshness_latency_ms=check_ms,
        freshness_measured=freshness is not None,
        freshness_correct=freshness is not None and freshness.ok and not freshness.missing,
        region_recall=region_metrics.region_recall if region_metrics is not None else None,
        line_recall=region_metrics.line_recall if region_metrics is not None else None,
        context_noise_ratio=(
            region_metrics.context_noise_ratio if region_metrics is not None else None
        ),
        timestamp=now_iso(),
        operational_notes=(
            "pinned released @nanonets/graft structural graph; "
            "build without --deep, ask with --no-refresh, freshness probed separately"
        ),
        local_offline_posture="local structural graph only; no model call",
        backend="graft-structural",
    )
    sys.stdout.write(artifact.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
