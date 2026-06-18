"""Tests for optional bundle-only evaluator commands."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from archex.benchmark.bundle_eval import (
    BundleOnlyEvaluatorError,
    BundleOnlyEvaluatorOutput,
    build_bundle_only_evaluator_input,
    bundle_only_result_fields,
    parse_bundle_only_evaluator_output,
    run_bundle_only_evaluator,
)
from archex.benchmark.models import (
    BenchmarkTask,
    BundleOnlyAllowedContext,
    BundleOnlyEvaluation,
    BundleOnlyEvaluatorCommand,
    TaskCompletionResult,
)
from archex.models import (
    ContextBundle,
    ContextCompletenessStatus,
    ContextReceipt,
    ContextReceiptEdge,
    ContextReceiptItem,
    ContextReceiptTokenBudget,
    ContextSkippedCandidate,
    ContextSkippedReason,
    EdgeKind,
)


def _fixture_command(mode: str) -> BundleOnlyEvaluatorCommand:
    command_path = Path(__file__).parents[1] / "fixtures" / "bundle_eval_command.py"
    return BundleOnlyEvaluatorCommand(
        command=sys.executable,
        args=[str(command_path), mode],
        timeout_seconds=10,
    )


def _task(command: BundleOnlyEvaluatorCommand | None = None) -> BenchmarkTask:
    return BenchmarkTask(
        task_id="bundle_eval_task",
        repo="owner/repo",
        commit="abc123",
        question="Can the bundle answer this?",
        expected_files=["src/main.py"],
        bundle_only_eval=BundleOnlyEvaluation(
            expected_answer="bundle contains receipt",
            allowed_context_policy=BundleOnlyAllowedContext.BUNDLE_PLUS_FRONTIER,
            evaluator_command=command,
        ),
    )


def _bundle() -> ContextBundle:
    receipt = ContextReceipt(
        query="Can the bundle answer this?",
        token_budget=ContextReceiptTokenBudget(requested=1000, consumed=20),
        index_revision="sha256:test",
        context_complete=ContextCompletenessStatus.COMPLETE,
    )
    return ContextBundle(
        query="Can the bundle answer this?",
        token_count=20,
        token_budget=1000,
        receipt=receipt,
    )


def test_build_evaluator_input_contains_bundle_receipt_and_policy() -> None:
    evaluator_input = build_bundle_only_evaluator_input(_task(), _bundle())

    assert evaluator_input.question == "Can the bundle answer this?"
    assert "Can the bundle answer this?" in evaluator_input.rendered_bundle
    assert "sha256:test" in evaluator_input.receipt_json
    assert evaluator_input.allowed_context_policy == "bundle-plus-frontier"
    assert evaluator_input.output_schema["required"] == [
        "answer",
        "confidence",
        "needed_files",
        "attempted_more_context",
    ]


def test_result_fields_attribute_needed_files_and_false_positive() -> None:
    receipt = ContextReceipt(
        query="Can the bundle answer this?",
        token_budget=ContextReceiptTokenBudget(requested=1000, consumed=20),
        index_revision="sha256:test",
        context_complete=ContextCompletenessStatus.COMPLETE,
        returned_context=[
            ContextReceiptItem(
                handle="src/main.py#1-10",
                file_path="src/main.py",
                start_line=1,
                end_line=10,
                content_hash="sha256:main",
            )
        ],
        omitted_edges=[
            ContextReceiptEdge(
                source="src/main.py",
                target="src/frontier.py",
                kind=EdgeKind.IMPORTS,
                target_path="src/frontier.py",
            )
        ],
        skipped_candidates=[
            ContextSkippedCandidate(
                file_path="src/frontier.py",
                reason=ContextSkippedReason.DEPENDENCY_FRONTIER_CUT,
            ),
            ContextSkippedCandidate(
                file_path="src/skipped.py",
                reason=ContextSkippedReason.BELOW_THRESHOLD,
            ),
        ],
    )
    output = BundleOnlyEvaluatorOutput(
        answer="bundle contains receipt",
        confidence=1.0,
        needed_files=[
            "src/main.py",
            "src/frontier.py",
            "src/skipped.py",
            "src/absent.py",
        ],
        attempted_more_context=True,
        post_bundle_read_turns=3,
        bundle_only_success=TaskCompletionResult.FAIL,
    )

    fields = bundle_only_result_fields(
        ContextBundle(query="q", token_count=20, token_budget=1000, receipt=receipt),
        output,
    )

    assert fields["bundle_only_success"] is TaskCompletionResult.FAIL
    assert fields["needed_files_outside_returned"] == [
        "src/frontier.py",
        "src/skipped.py",
        "src/absent.py",
    ]
    assert fields["needed_files_in_frontier_cut"] == ["src/frontier.py"]
    assert fields["needed_files_in_top_candidates"] == ["src/frontier.py", "src/skipped.py"]
    assert fields["safe_to_act_false_positive"] is True
    assert fields["post_bundle_read_turns"] == 3


def test_run_evaluator_command_returns_structured_output(tmp_path: Path) -> None:
    output = run_bundle_only_evaluator(
        _task(_fixture_command("pass")),
        _bundle(),
        cwd=tmp_path,
    )

    assert output.answer == "bundle contains receipt"
    assert output.confidence == 0.95
    assert output.needed_files == ["src/frontier.py"]
    assert output.attempted_more_context is False
    assert output.post_bundle_read_turns == 1
    assert output.bundle_only_success is TaskCompletionResult.PASS


def test_expected_answer_grades_missing_success_field(tmp_path: Path) -> None:
    output = run_bundle_only_evaluator(
        _task(_fixture_command("no-success")),
        _bundle(),
        cwd=tmp_path,
    )

    assert output.bundle_only_success is TaskCompletionResult.PASS


def test_explicit_command_overrides_task_command(tmp_path: Path) -> None:
    output = run_bundle_only_evaluator(
        _task(_fixture_command("exit-2")),
        _bundle(),
        command=_fixture_command("pass"),
        cwd=tmp_path,
    )

    assert output.bundle_only_success is TaskCompletionResult.PASS


def test_invalid_evaluator_json_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(BundleOnlyEvaluatorError, match="invalid JSON"):
        run_bundle_only_evaluator(
            _task(_fixture_command("invalid-json")),
            _bundle(),
            cwd=tmp_path,
        )


def test_invalid_evaluator_schema_fails_loudly() -> None:
    with pytest.raises(BundleOnlyEvaluatorError, match="required schema"):
        parse_bundle_only_evaluator_output('{"answer": "missing fields"}')


def test_missing_required_evaluator_fields_fail_loudly() -> None:
    with pytest.raises(BundleOnlyEvaluatorError, match="required schema"):
        parse_bundle_only_evaluator_output(
            """{
                "answer": "a",
                "confidence": 1.0
            }"""
        )


def test_invalid_needed_file_path_fails_loudly() -> None:
    with pytest.raises(BundleOnlyEvaluatorError, match="required schema"):
        parse_bundle_only_evaluator_output(
            """{
                "answer": "a",
                "confidence": 1.0,
                "needed_files": ["../secret.py"],
                "attempted_more_context": false
            }"""
        )


def test_nonzero_evaluator_exit_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(BundleOnlyEvaluatorError, match="exit code 2"):
        run_bundle_only_evaluator(
            _task(_fixture_command("exit-2")),
            _bundle(),
            cwd=tmp_path,
        )


def test_nonexistent_evaluator_command_fails_loudly(tmp_path: Path) -> None:
    command = BundleOnlyEvaluatorCommand(command="definitely-not-archex-bundle-eval")

    with pytest.raises(BundleOnlyEvaluatorError, match="could not be started"):
        run_bundle_only_evaluator(
            _task(command),
            _bundle(),
            cwd=tmp_path,
        )


def test_evaluator_timeout_fails_loudly(tmp_path: Path) -> None:
    command = _fixture_command("sleep").model_copy(update={"timeout_seconds": 0.01})

    with pytest.raises(BundleOnlyEvaluatorError, match="timed out"):
        run_bundle_only_evaluator(
            _task(command),
            _bundle(),
            cwd=tmp_path,
        )


def test_bundle_eval_repo_path_slices_include_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import archex.benchmark.bundle_eval as bundle_eval_mod
    import archex.benchmark.runner as runner_mod

    clone_dir = tmp_path / "clone"
    (clone_dir / "pkg" / "sub").mkdir(parents=True)
    (clone_dir / "pkg" / "sub" / "kept.py").write_text("kept = True\n")
    (clone_dir / "pkg" / "discarded.py").write_text("discarded = True\n")

    def clone_repo(_repo: str, _commit: str) -> tuple[Path, bool]:
        return clone_dir, True

    monkeypatch.setattr(runner_mod, "clone_at_commit", clone_repo)
    task = BenchmarkTask(
        task_id="slice",
        repo="owner/repo",
        commit="abc",
        question="How?",
        expected_files=["pkg/sub/kept.py"],
        include_paths=["pkg/sub"],
        bundle_only_eval=BundleOnlyEvaluation(expected_answer="answer"),
    )

    repo_path, cleanup_paths = bundle_eval_mod._repo_path_for_task(task)  # pyright: ignore[reportPrivateUsage]

    try:
        assert (repo_path / "pkg" / "sub" / "kept.py").exists()
        assert not (repo_path / "pkg" / "discarded.py").exists()
        assert cleanup_paths == [clone_dir, repo_path]
    finally:
        for path in cleanup_paths:
            shutil.rmtree(path, ignore_errors=True)


def test_missing_command_fails_before_execution() -> None:
    with pytest.raises(BundleOnlyEvaluatorError, match="explicit evaluator command"):
        run_bundle_only_evaluator(_task(), _bundle())
