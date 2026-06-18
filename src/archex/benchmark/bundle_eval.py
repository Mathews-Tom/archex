"""Optional bundle-only benchmark evaluator runner."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from archex.benchmark.models import (  # noqa: TC001 — Pydantic needs at runtime
    BenchmarkReport,
    BenchmarkResult,
    Strategy,
    TaskCompletionResult,
)

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkTask, BundleOnlyEvaluatorCommand
    from archex.models import ContextBundle


class BundleOnlyResultFields(TypedDict):
    bundle_only_success: TaskCompletionResult
    needed_files_outside_returned: list[str]
    needed_files_in_frontier_cut: list[str]
    needed_files_in_top_candidates: list[str]
    safe_to_act_false_positive: bool
    post_bundle_read_turns: int


class BundleOnlyEvaluatorError(RuntimeError):
    """Raised when a bundle-only evaluator command cannot produce valid output."""


class BundleOnlyEvaluatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    question: str
    rendered_bundle: str
    receipt_json: str
    allowed_context_policy: str
    output_schema: dict[str, Any]


class BundleOnlyEvaluatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    needed_files: list[str]
    attempted_more_context: bool
    post_bundle_read_turns: int = Field(default=0, ge=0)
    bundle_only_success: TaskCompletionResult | None = None

    @model_validator(mode="after")
    def _validate_needed_files(self) -> BundleOnlyEvaluatorOutput:
        for path in self.needed_files:
            if not path or path.startswith("/") or ".." in path.split("/"):
                msg = f"needed_files entries must be relative paths: {path!r}"
                raise ValueError(msg)
        return self


def bundle_only_output_schema() -> dict[str, Any]:
    """Return the structured JSON contract evaluator commands must print."""
    return {
        "type": "object",
        "required": ["answer", "confidence", "needed_files", "attempted_more_context"],
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "needed_files": {"type": "array", "items": {"type": "string"}},
            "attempted_more_context": {"type": "boolean"},
            "post_bundle_read_turns": {"type": "integer", "minimum": 0},
            "bundle_only_success": {"enum": ["pass", "fail", "unknown", None]},
        },
        "additionalProperties": False,
    }


def build_bundle_only_evaluator_input(
    task: BenchmarkTask,
    bundle: ContextBundle,
    *,
    bundle_format: str = "markdown",
) -> BundleOnlyEvaluatorInput:
    if task.bundle_only_eval is None:
        msg = "bundle-only evaluation is not configured for this task"
        raise BundleOnlyEvaluatorError(msg)
    receipt_json = "null"
    if bundle.receipt is not None:
        receipt_json = bundle.receipt.model_dump_json()
    return BundleOnlyEvaluatorInput(
        task_id=task.task_id,
        question=task.question,
        rendered_bundle=bundle.to_prompt(format=bundle_format),
        receipt_json=receipt_json,
        allowed_context_policy=task.bundle_only_eval.allowed_context_policy.value,
        output_schema=bundle_only_output_schema(),
    )


def parse_bundle_only_evaluator_output(stdout: str) -> BundleOnlyEvaluatorOutput:
    if not stdout.strip():
        msg = "bundle-only evaluator produced no JSON output"
        raise BundleOnlyEvaluatorError(msg)
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        msg = "bundle-only evaluator produced invalid JSON output"
        raise BundleOnlyEvaluatorError(msg) from exc
    try:
        return BundleOnlyEvaluatorOutput.model_validate(payload)
    except ValidationError as exc:
        msg = "bundle-only evaluator output does not match the required schema"
        raise BundleOnlyEvaluatorError(msg) from exc


def _returned_files(bundle: ContextBundle) -> set[str]:
    if bundle.receipt is not None and bundle.receipt.returned_context:
        return {item.file_path for item in bundle.receipt.returned_context if item.file_path}
    return {chunk.chunk.file_path for chunk in bundle.chunks if chunk.chunk.file_path}


def _frontier_files(bundle: ContextBundle) -> set[str]:
    if bundle.receipt is None:
        return set()
    files = {
        edge.target_path for edge in bundle.receipt.omitted_edges if edge.target_path is not None
    }
    files.update(
        candidate.file_path
        for candidate in bundle.receipt.skipped_candidates
        if candidate.reason == "dependency_frontier_cut" and candidate.file_path
    )
    return files


def _top_candidate_files(bundle: ContextBundle) -> set[str]:
    if bundle.receipt is None:
        return set()
    return {
        candidate.file_path
        for candidate in bundle.receipt.skipped_candidates
        if candidate.file_path
    }


def bundle_only_result_fields(
    bundle: ContextBundle,
    output: BundleOnlyEvaluatorOutput,
) -> BundleOnlyResultFields:
    """Derive benchmark result fields from evaluator output and receipt metadata."""
    needed_files = list(dict.fromkeys(output.needed_files))
    returned_files = _returned_files(bundle)
    frontier_files = _frontier_files(bundle)
    top_candidate_files = _top_candidate_files(bundle)
    outside_returned = [path for path in needed_files if path not in returned_files]
    in_frontier = [path for path in outside_returned if path in frontier_files]
    in_top_candidates = [path for path in outside_returned if path in top_candidate_files]
    receipt_claims_complete = (
        bundle.receipt is not None and bundle.receipt.context_complete == "complete"
    )
    return {
        "bundle_only_success": output.bundle_only_success or TaskCompletionResult.UNKNOWN,
        "needed_files_outside_returned": outside_returned,
        "needed_files_in_frontier_cut": in_frontier,
        "needed_files_in_top_candidates": in_top_candidates,
        "safe_to_act_false_positive": bool(receipt_claims_complete and outside_returned),
        "post_bundle_read_turns": output.post_bundle_read_turns,
    }


def _with_expected_answer_success(
    task: BenchmarkTask,
    output: BundleOnlyEvaluatorOutput,
) -> BundleOnlyEvaluatorOutput:
    if output.bundle_only_success is not None or task.bundle_only_eval is None:
        return output
    expected_answer = task.bundle_only_eval.expected_answer
    if expected_answer is None:
        return output.model_copy(update={"bundle_only_success": TaskCompletionResult.UNKNOWN})
    expected = expected_answer.strip()
    actual = output.answer.strip()
    success = TaskCompletionResult.PASS if actual == expected else TaskCompletionResult.FAIL
    return output.model_copy(update={"bundle_only_success": success})


def _resolve_evaluator_command(
    task: BenchmarkTask,
    command: BundleOnlyEvaluatorCommand | None,
) -> BundleOnlyEvaluatorCommand:
    if command is not None:
        return command
    if task.bundle_only_eval is not None and task.bundle_only_eval.evaluator_command is not None:
        return task.bundle_only_eval.evaluator_command
    msg = "bundle-only evaluation requires an explicit evaluator command"
    raise BundleOnlyEvaluatorError(msg)


def run_bundle_only_evaluator(
    task: BenchmarkTask,
    bundle: ContextBundle,
    *,
    command: BundleOnlyEvaluatorCommand | None = None,
    cwd: Path | None = None,
    bundle_format: str = "markdown",
) -> BundleOnlyEvaluatorOutput:
    """Run a user-supplied local evaluator command with bundle-only inputs."""
    evaluator_command = _resolve_evaluator_command(task, command)
    evaluator_input = build_bundle_only_evaluator_input(
        task,
        bundle,
        bundle_format=bundle_format,
    )
    try:
        completed = subprocess.run(
            [evaluator_command.command, *evaluator_command.args],
            input=evaluator_input.model_dump_json(),
            text=True,
            capture_output=True,
            cwd=cwd,
            timeout=evaluator_command.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        msg = (
            "bundle-only evaluator command timed out after "
            f"{evaluator_command.timeout_seconds:g} seconds"
        )
        raise BundleOnlyEvaluatorError(msg) from exc
    except OSError as exc:
        msg = f"bundle-only evaluator command could not be started: {exc}"
        raise BundleOnlyEvaluatorError(msg) from exc
    if completed.returncode != 0:
        msg = f"bundle-only evaluator command failed with exit code {completed.returncode}"
        raise BundleOnlyEvaluatorError(msg)
    return _with_expected_answer_success(task, parse_bundle_only_evaluator_output(completed.stdout))


def _unique_ranked_files(bundle: ContextBundle) -> list[str]:
    seen: set[str] = set()
    files: list[str] = []
    for chunk in bundle.chunks:
        path = chunk.chunk.file_path
        if path not in seen:
            seen.add(path)
            files.append(path)
    return files


def _repo_path_for_task(task: BenchmarkTask) -> tuple[Path, bool]:
    from archex.benchmark.runner import clone_at_commit

    if task.repo == ".":
        return Path.cwd(), False
    return clone_at_commit(task.repo, task.commit)


def run_bundle_only_eval_task(
    task: BenchmarkTask,
    *,
    command: BundleOnlyEvaluatorCommand,
    bundle_format: str = "markdown",
) -> BenchmarkReport:
    """Run the opt-in bundle-only eval lane for a single task."""
    from archex.api import query
    from archex.benchmark.strategies import (
        benchmark_repo_source,
        compute_f1,
        compute_precision,
        compute_recall,
        compute_receipt_accuracy,
        compute_required_file_metrics,
    )
    from archex.models import Config, IndexConfig

    repo_path, needs_cleanup = _repo_path_for_task(task)
    try:
        t0 = time.perf_counter()
        source = benchmark_repo_source(task, repo_path, strategy=Strategy.ARCHEX_QUERY)
        bundle = query(
            source,
            task.question,
            token_budget=task.token_budget,
            explicit_token_budget=True,
            config=Config(cache=False, languages=task.languages),
            index_config=IndexConfig(vector=False),
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        ranked_files = _unique_ranked_files(bundle)
        result_files = set(ranked_files)
        (
            required_file_recall,
            missed_required_file_rate,
            missed_required_task_rate,
            all_required_files_present,
            present,
            missing,
        ) = compute_required_file_metrics(result_files, task.expected_files)
        evaluator_output = run_bundle_only_evaluator(
            task,
            bundle,
            command=command,
            cwd=repo_path,
            bundle_format=bundle_format,
        )
        bundle_fields = bundle_only_result_fields(bundle, evaluator_output)
        result = BenchmarkResult(
            task_id=task.task_id,
            strategy=Strategy.ARCHEX_QUERY,
            strategy_label="bundle-only eval",
            tokens_total=bundle.token_count,
            tokens_input=bundle.token_count,
            tokens_output=bundle.token_count,
            tool_calls=1,
            files_accessed=len(result_files),
            recall=compute_recall(result_files, task.expected_files),
            precision=compute_precision(result_files, task.expected_files),
            f1_score=compute_f1(
                compute_recall(result_files, task.expected_files),
                compute_precision(result_files, task.expected_files),
            ),
            savings_vs_raw=0.0,
            wall_time_ms=wall_ms,
            cached=False,
            timestamp=datetime.now(UTC).isoformat(),
            result_files=ranked_files,
            required_file_recall=required_file_recall,
            missed_required_file_rate=missed_required_file_rate,
            missed_required_task_rate=missed_required_task_rate,
            all_required_files_present=all_required_files_present,
            required_files_present=present,
            required_files_missing=missing,
            receipt_accuracy=compute_receipt_accuracy(
                bundle,
                all_required_files_present=all_required_files_present,
            ),
            bundle_only_success=bundle_fields["bundle_only_success"],
            needed_files_outside_returned=bundle_fields["needed_files_outside_returned"],
            needed_files_in_frontier_cut=bundle_fields["needed_files_in_frontier_cut"],
            needed_files_in_top_candidates=bundle_fields["needed_files_in_top_candidates"],
            safe_to_act_false_positive=bundle_fields["safe_to_act_false_positive"],
            post_bundle_read_turns=bundle_fields["post_bundle_read_turns"],
        )
        return BenchmarkReport(
            task_id=task.task_id,
            repo=task.repo,
            question=task.question,
            results=[result],
            baseline_tokens=0,
            median_latency_ms=wall_ms,
            p95_latency_ms=wall_ms,
        )
    finally:
        if needs_cleanup:
            shutil.rmtree(repo_path, ignore_errors=True)


def run_bundle_only_eval_all(
    tasks: list[BenchmarkTask],
    output_dir: Path,
    *,
    command: BundleOnlyEvaluatorCommand,
    bundle_format: str = "markdown",
) -> list[BenchmarkReport]:
    """Run bundle-only eval for selected tasks and write BenchmarkReport JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[BenchmarkReport] = []
    for task in tasks:
        report = run_bundle_only_eval_task(
            task,
            command=command,
            bundle_format=bundle_format,
        )
        reports.append(report)
        (output_dir / f"{task.task_id}.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
    return reports
