"""Optional bundle-only benchmark evaluator runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from archex.benchmark.models import TaskCompletionResult  # noqa: TC001 — Pydantic needs at runtime

if TYPE_CHECKING:
    from archex.benchmark.models import BenchmarkTask, BundleOnlyEvaluatorCommand
    from archex.models import ContextBundle


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
