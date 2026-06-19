"""Run one Graphify competitive lane from the PR2 stdin contract.

Reads a JSON payload on stdin with:
- ``task``: BenchmarkTask JSON
- ``repo_path``: checked-out repository path
- ``lane``: graphify_build_plus_query | graphify_query_warm
- ``graphify``: {package_name, version, includes_build_cost}

Writes one GraphifyArtifact JSON document to stdout.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from archex.benchmark.graphify import GraphifyArtifact
from archex.benchmark.models import BenchmarkTask, GraphifyLaneName
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

_NODE_RE = re.compile(r"^NODE (?P<label>.+?) \[src=(?P<src>.*?) loc=(?P<loc>.*?) community=.*\]$")
_LINE_RE = re.compile(r"L(?P<line>\d+)")

_GRAPHIFY_IGNORE_MARKER = "# archex graphify benchmark code-only filter"
_GRAPHIFY_IGNORE_BLOCK = "\n".join(
    [
        _GRAPHIFY_IGNORE_MARKER,
        "*.md",
        "*.mdx",
        "*.txt",
        "*.rst",
        "*.html",
        "*.pdf",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "*.webp",
        "*.svg",
        "docs/",
        "documentation/",
        "",
    ]
)


def _ensure_code_only_graphifyignore(repo_path: Path) -> None:
    ignore_path = repo_path / ".graphifyignore"
    existing = ignore_path.read_text(encoding="utf-8") if ignore_path.exists() else ""
    if _GRAPHIFY_IGNORE_MARKER in existing:
        return
    content = (
        f"{existing.rstrip()}\n\n{_GRAPHIFY_IGNORE_BLOCK}" if existing else _GRAPHIFY_IGNORE_BLOCK
    )
    ignore_path.write_text(content, encoding="utf-8")


def _run(
    command: list[str], *, env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(f"{' '.join(command)} failed with exit {completed.returncode}: {detail}")
    return completed, elapsed_ms


def _parse_query_output(text: str) -> tuple[list[str], list[ReturnedRegion]]:
    result_files: list[str] = []
    seen_files: set[str] = set()
    returned_regions: list[ReturnedRegion] = []
    for line in text.splitlines():
        match = _NODE_RE.match(line)
        if match is None:
            continue
        path = match.group("src").strip()
        if not path:
            continue
        if path not in seen_files:
            seen_files.add(path)
            result_files.append(path)
        line_match = _LINE_RE.search(match.group("loc"))
        if line_match is None:
            continue
        line_no = int(line_match.group("line"))
        returned_regions.append(
            ReturnedRegion(
                path=path,
                start_line=line_no,
                end_line=line_no,
                symbol=match.group("label").strip() or None,
                tokens=max(1, count_tokens(line)),
            )
        )
    return result_files, returned_regions


def main() -> int:
    payload = json.loads(sys.stdin.read())
    task = BenchmarkTask.model_validate(payload["task"])
    repo_path = Path(payload["repo_path"]).resolve()
    lane = GraphifyLaneName(str(payload["lane"]))
    graphify = payload["graphify"]
    package_name = str(graphify["package_name"])
    version = str(graphify["version"])
    includes_build_cost = bool(graphify["includes_build_cost"])

    graphify_out = repo_path / "graphify-out"
    graph_path = graphify_out / "graph.json"
    if graphify_out.exists():
        shutil.rmtree(graphify_out)
    _ensure_code_only_graphifyignore(repo_path)

    env = dict(os.environ)
    env["GRAPHIFY_QUERY_LOG_DISABLE"] = "1"

    build_command = [
        "uvx",
        "--from",
        f"{package_name}=={version}",
        "graphify",
        str(repo_path),
        "--no-viz",
    ]
    query_command = [
        "uvx",
        "--from",
        f"{package_name}=={version}",
        "graphify",
        "query",
        task.question,
        "--graph",
        str(graph_path),
        "--budget",
        str(task.token_budget),
    ]

    build_ms = 0.0
    if includes_build_cost:
        _, build_ms = _run(build_command, env=env)
    else:
        _run(build_command, env=env)
    query_completed, query_ms = _run(query_command, env=env)

    result_files, returned_regions = _parse_query_output(query_completed.stdout)
    result_file_set = set(result_files)
    tokens_input = count_file_tokens(repo_path, result_files) if result_files else 0
    tokens_output = count_tokens(query_completed.stdout)
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
    region_metrics = (
        compute_region_metrics(returned_regions, task.expected_regions)
        if task.expected_regions
        else None
    )

    question_arg = shlex.quote(task.question)
    command_text = (
        f"uvx --from {package_name}=={version} graphify query {question_arg} "
        f"--graph <repo>/graphify-out/graph.json --budget {task.token_budget}"
    )
    if includes_build_cost:
        command_text = (
            f"uvx --from {package_name}=={version} graphify <repo> --no-viz && {command_text}"
        )

    artifact = GraphifyArtifact(
        task_id=task.task_id,
        lane=lane,
        graphify_package=package_name,
        graphify_version=version,
        command=command_text,
        includes_build_cost=includes_build_cost,
        tokens_total=tokens_output,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tool_calls=2 if includes_build_cost else 1,
        files_accessed=len(result_file_set),
        recall=compute_recall(result_file_set, task.expected_files),
        precision=compute_precision(result_file_set, task.expected_files),
        f1_score=compute_f1(
            compute_recall(result_file_set, task.expected_files),
            compute_precision(result_file_set, task.expected_files),
        ),
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
        warm_latency_ms=query_ms,
        wall_time_ms=build_ms + query_ms,
        cache_state="cold" if includes_build_cost else "warm",
        cached=not includes_build_cost,
        region_recall=region_metrics.region_recall if region_metrics is not None else None,
        line_recall=region_metrics.line_recall if region_metrics is not None else None,
        context_noise_ratio=(
            region_metrics.context_noise_ratio if region_metrics is not None else None
        ),
        operational_notes="local AST-only Graphify code graph",
        local_offline_posture="local AST code graph only",
        backend="local-ast",
    )
    sys.stdout.write(artifact.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
