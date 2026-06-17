"""External MCP-backed benchmark strategy."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from archex.benchmark.models import (
    BenchmarkResult,
    BenchmarkTask,
    ExternalToolBenchmarkConfig,
    Strategy,
)
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
    now_iso,
)
from archex.reporting import count_tokens

_EXTERNAL_TOOL_CONFIG: ContextVar[ExternalToolBenchmarkConfig | None] = ContextVar(
    "external_tool_config", default=None
)

_CODE_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.@+-]+/)*[A-Za-z0-9_.@+-]+\."
    r"(?:py|js|jsx|ts|tsx|go|rs|java|kt|cs|swift|md|yaml|yml|toml|json))"
)


class ExternalToolBenchmarkError(RuntimeError):
    """Raised when an external MCP benchmark lane cannot produce a valid result."""


@dataclass(frozen=True)
class ExternalSearchHit:
    """Normalized file-scoped result returned by an external MCP search tool."""

    file_path: str
    text: str


def _deduplicate_paths(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(paths))


def set_external_tool_config(
    config: ExternalToolBenchmarkConfig,
) -> Token[ExternalToolBenchmarkConfig | None]:
    return _EXTERNAL_TOOL_CONFIG.set(config)


def reset_external_tool_config(token: Token[ExternalToolBenchmarkConfig | None]) -> None:
    _EXTERNAL_TOOL_CONFIG.reset(token)


def current_external_tool_config() -> ExternalToolBenchmarkConfig:
    config = _EXTERNAL_TOOL_CONFIG.get()
    if config is None:
        raise NotImplementedError("external_mcp requires an ExternalToolBenchmarkConfig")
    return config


def _run_bootstrap_commands(config: ExternalToolBenchmarkConfig, repo_path: Path) -> float:
    if not config.bootstrap_commands:
        return 0.0

    total_elapsed_ms = 0.0
    for step in config.bootstrap_commands:
        command = [step.command, *step.args]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=repo_path,
                env={**os.environ, **config.env} if config.env else None,
                capture_output=True,
                text=True,
                timeout=step.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalToolBenchmarkError(
                f"bootstrap command timed out after {step.timeout_seconds}s: {command}"
            ) from exc

        total_elapsed_ms += (time.perf_counter() - started) * 1000
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
            raise ExternalToolBenchmarkError(
                f"bootstrap command failed with exit {completed.returncode}: {detail}"
            )
    return total_elapsed_ms


def _search_arguments(
    config: ExternalToolBenchmarkConfig, task: BenchmarkTask
) -> dict[str, object]:
    arguments: dict[str, object] = {config.query_argument: task.question}
    if config.limit_argument:
        arguments[config.limit_argument] = config.limit
    if config.path_argument and task.include_paths:
        arguments[config.path_argument] = list(task.include_paths)
    if config.language_argument and task.languages:
        arguments[config.language_argument] = list(task.languages)
    if config.extra_arguments:
        arguments.update(config.extra_arguments)
    return arguments


async def _call_mcp_search(
    config: ExternalToolBenchmarkConfig,
    task: BenchmarkTask,
    repo_path: Path,
) -> tuple[object, float]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cwd = Path(config.cwd) if config.cwd else repo_path
    server_params = StdioServerParameters(
        command=config.command,
        args=list(config.args),
        env={**config.env} if config.env else None,
        cwd=cwd,
    )
    try:
        async with (
            stdio_client(server_params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await asyncio.wait_for(session.initialize(), timeout=config.timeout_seconds)
            started = time.perf_counter()
            result = await asyncio.wait_for(
                session.call_tool(config.search_tool, _search_arguments(config, task)),
                timeout=config.timeout_seconds,
            )
            warm_ms = (time.perf_counter() - started) * 1000
            return result, warm_ms
    except TimeoutError as exc:
        raise ExternalToolBenchmarkError(
            f"MCP tool {config.search_tool!r} timed out after {config.timeout_seconds}s"
        ) from exc


def _relative_existing_path(repo_path: Path, value: str) -> str | None:
    candidate = value.strip().strip("`'\" ,;()[]{}")
    if "#L" in candidate:
        candidate = candidate.split("#L", 1)[0]
    if ":" in candidate:
        prefix = candidate.split(":", 1)[0]
        if (repo_path / prefix).is_file() or Path(prefix).is_file():
            candidate = prefix
    path = Path(candidate)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(repo_path.resolve())
        except ValueError:
            return None
    if path.is_absolute() or ".." in path.parts:
        return None
    normalized = path.as_posix().lstrip("./")
    return normalized if (repo_path / normalized).is_file() else None


def _string_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str):
            return value
    return ""


def _hits_from_json_value(repo_path: Path, value: object) -> list[ExternalSearchHit]:
    hits: list[ExternalSearchHit] = []
    if isinstance(value, list):
        for item in cast("list[object]", value):
            hits.extend(_hits_from_json_value(repo_path, item))
        return hits
    if not isinstance(value, dict):
        return hits

    mapping = cast("dict[str, Any]", value)
    path_value = _string_value(mapping, ("file_path", "filepath", "path", "file", "filename"))
    normalized = _relative_existing_path(repo_path, path_value) if path_value else None
    if normalized is not None:
        text = _string_value(mapping, ("content", "code", "text", "snippet", "chunk"))
        hits.append(ExternalSearchHit(file_path=normalized, text=text))
        return hits

    for child in mapping.values():
        hits.extend(_hits_from_json_value(repo_path, child))
    return hits


def _hits_from_text(repo_path: Path, text: str) -> list[ExternalSearchHit]:
    stripped = text.strip()
    if stripped:
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = None
        if decoded is not None:
            hits = _hits_from_json_value(repo_path, decoded)
            if hits:
                return hits

    hits: list[ExternalSearchHit] = []
    seen: set[str] = set()
    for match in _CODE_PATH_RE.finditer(text):
        normalized = _relative_existing_path(repo_path, match.group("path"))
        if normalized is not None and normalized not in seen:
            seen.add(normalized)
            hits.append(ExternalSearchHit(file_path=normalized, text=""))
    return hits


def _content_texts(result: object) -> list[str]:
    content = getattr(result, "content", None)
    if content is None:
        return [str(result)]
    texts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            texts.append(text)
        elif isinstance(item, str):
            texts.append(item)
        elif hasattr(item, "model_dump"):
            texts.append(json.dumps(item.model_dump(mode="json")))
        else:
            texts.append(str(item))
    return texts


def normalize_external_search_result(repo_path: Path, result: object) -> list[ExternalSearchHit]:
    """Normalize MCP search output into ordered, file-scoped hits."""
    hits: list[ExternalSearchHit] = []
    for text in _content_texts(result):
        hits.extend(_hits_from_text(repo_path, text))
    return hits


def _external_output_tokens(repo_path: Path, hits: list[ExternalSearchHit]) -> tuple[int, str]:
    total = 0
    token_mode = "chunk_text"
    full_file_paths: list[str] = []
    for hit in hits:
        if hit.text:
            total += count_tokens(hit.text)
        else:
            token_mode = "chunk_text_with_full_file_fallback"
            full_file_paths.append(hit.file_path)
    if full_file_paths:
        total += count_file_tokens(repo_path, _deduplicate_paths(full_file_paths))
    return total, token_mode


def run_external_mcp(
    task: BenchmarkTask,
    repo_path: Path,
    config: ExternalToolBenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Run a configured external MCP search tool and score file-scoped results."""
    config = config or current_external_tool_config()
    total_started = time.perf_counter()
    cold_start_ms = _run_bootstrap_commands(config, repo_path)
    result, warm_latency_ms = asyncio.run(_call_mcp_search(config, task, repo_path))
    hits = normalize_external_search_result(repo_path, result)
    ranked_files = [hit.file_path for hit in hits]
    unique_files = _deduplicate_paths(ranked_files)
    result_files = set(unique_files)
    tokens_input = count_file_tokens(repo_path, unique_files)
    tokens_output, token_mode = _external_output_tokens(repo_path, hits)
    completion_tokens, completion_files = compute_bundle_completion_penalty(
        repo_path, result_files, task.expected_files
    )
    required_metrics = compute_required_file_metrics(
        result_files,
        task.expected_files,
    )
    (
        required_file_recall,
        missed_required_file_rate,
        all_required_files_present,
        present,
        missing,
    ) = required_metrics
    recall = compute_recall(result_files, task.expected_files)
    precision = compute_precision(result_files, task.expected_files)
    wall_ms = (time.perf_counter() - total_started) * 1000

    return BenchmarkResult(
        task_id=task.task_id,
        strategy=Strategy.EXTERNAL_MCP,
        strategy_label=config.name,
        tokens_total=tokens_output,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        token_efficiency=compute_token_efficiency(tokens_output, tokens_input),
        result_files=unique_files,
        required_file_recall=required_file_recall,
        missed_required_file_rate=missed_required_file_rate,
        missed_required_task_rate=0.0 if all_required_files_present else 1.0,
        all_required_files_present=all_required_files_present,
        required_files_present=present,
        required_files_missing=missing,
        post_bundle_read_turns=len(completion_files),
        task_completion_result=completion_result_from_missing(completion_files),
        bundle_completion_tokens=completion_tokens,
        bundle_completion_files=completion_files,
        token_efficiency_with_completion=compute_token_efficiency(
            tokens_output + completion_tokens, tokens_input + completion_tokens
        ),
        tokens_raw_baseline=count_file_tokens(repo_path, task.expected_files),
        tool_calls=1 + len(config.bootstrap_commands),
        files_accessed=len(result_files),
        recall=recall,
        precision=precision,
        f1_score=compute_f1(recall, precision),
        mrr=compute_mrr(ranked_files, task.expected_files),
        ndcg=compute_ndcg(ranked_files, task.expected_files),
        map_score=compute_map(ranked_files, task.expected_files),
        savings_vs_raw=0.0,
        wall_time_ms=wall_ms,
        cached=not bool(config.bootstrap_commands),
        timestamp=now_iso(),
        unique_ranked_files=len(unique_files),
        seed_files=unique_files,
        cold_start_ms=cold_start_ms,
        warm_latency_ms=warm_latency_ms,
        cache_state="cold" if config.bootstrap_commands else "warm",
        provenance={
            "external_tool": config.name,
            "external_tool_version": config.version,
            "external_tool_embedder": config.embedder,
            "external_tool_command": " ".join([config.command, *config.args]),
            "external_tool_search_tool": config.search_tool,
            "external_tool_token_mode": token_mode,
            "external_tool_result_count": str(len(hits)),
            "external_tool_bootstrap_count": str(len(config.bootstrap_commands)),
        },
    )
