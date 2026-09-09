"""Tests for the R20 paired product-loop harness.

These cover the parts of the protocol that a hostile reader would attack: that
the measured prompt matches the merged pre-registration byte for byte, that the
recall metric cannot be won by breadth, that path normalization does not
systematically penalise one arm, that a cell which cannot be measured becomes a
retained zero rather than a silent gap, and that the hook recorder does not
change what a product's hook does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


from archex.benchmark.product_loop import (  # noqa: E402
    AGENT_MODEL,
    AGENT_NAME,
    AGENT_VERSION,
    ANSWER_PATH_CAP,
    BILLING_MODE,
    COST_CEILING_USD,
    INSTRUCTION_BLOCK,
    INSTRUCTION_BLOCK_SHA256,
    PREREGISTRATION_PATH,
    REPETITIONS,
    ProductLoopAnswerFlag,
    ProductLoopArm,
    ProductLoopCellArtifact,
    ProductLoopCellStatus,
    ProductLoopError,
    ProductLoopFailureReason,
    TranscriptSummary,
    ambient_tool_fingerprint,
    assert_frozen_prompt,
    build_cell_artifact,
    build_prompt,
    cell_filename,
    classify_cell_failure,
    extract_answer_paths,
    instruction_block_digest,
    normalize_answer_path,
    preregistered_instruction_block,
    read_hook_records,
    sanitize_document,
    score_completeness,
    summarize_transcript,
    validate_product_loop_directory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A minimal repository-shaped tree for path normalization."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "service.py").write_text("x = 1\n", encoding="utf-8")
    (root / "pkg" / "models.py").write_text("y = 2\n", encoding="utf-8")
    (root / "README.md").write_text("readme\n", encoding="utf-8")
    return root


def _cell_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "instruction_block_sha256": INSTRUCTION_BLOCK_SHA256,
        "task_id": "celery_task_dispatch",
        "repo": "celery/celery",
        "commit": "v5.4.0",
        "arm": ProductLoopArm.ARCHEX.value,
        "repetition": 1,
        "status": ProductLoopCellStatus.OK.value,
        "agent_name": AGENT_NAME,
        "agent_version": AGENT_VERSION,
        "model_requested": AGENT_MODEL,
        "models_observed": [f"{AGENT_MODEL}-20260101"],
        "billing_mode": BILLING_MODE,
        "mcp_server": "archex",
        "mcp_status": "connected",
        "mcp_dropped": False,
        "tools_advertised": sorted(ProductLoopArm.ARCHEX.expected_tools),
        "ambient_tool_fingerprint": ambient_tool_fingerprint(
            sorted(ProductLoopArm.ARCHEX.expected_tools)
        ),
        "expected_file_count": 3,
        "matched_file_count": 3,
        "completeness": 1.0,
        "answer_flag": ProductLoopAnswerFlag.SCORED.value,
        "answer_paths": ["a.py", "b.py", "c.py"],
        "answer_path_count": 3,
        "answer_precision": 1.0,
        "tool_calls": 4,
        "tool_call_mix": {"Grep": 3, "mcp__archex__query_repo": 1},
        "product_tool_calls": 1,
        "no_product_use": False,
        "input_tokens": 1200,
        "output_tokens": 90,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
        "modelled_cost_usd": 0.05,
        "setup_seconds": 3.5,
        "wall_seconds": 21.0,
        "num_turns": 3,
    }
    payload.update(overrides)
    return payload


class TestFrozenPrompt:
    def test_code_matches_the_merged_preregistration_byte_for_byte(self) -> None:
        # The whole pre-registration rests on the prompt being unchangeable
        # after data exists, so document and code must never drift apart.
        assert_frozen_prompt(REPO_ROOT / PREREGISTRATION_PATH)

    def test_declared_hash_is_the_hash_of_the_block(self) -> None:
        assert instruction_block_digest() == INSTRUCTION_BLOCK_SHA256

    def test_drifted_document_block_is_rejected(self, tmp_path: Path) -> None:
        drifted = tmp_path / "prereg.md"
        drifted.write_text(
            "## Appendix A\n\n```text\nName the single most relevant file.\n```\n",
            encoding="utf-8",
        )
        with pytest.raises(ProductLoopError, match="Appendix A"):
            assert_frozen_prompt(drifted)

    def test_prompt_is_question_then_blank_line_then_block(self) -> None:
        prompt = build_prompt("  How does Celery dispatch tasks?  ")
        assert prompt == f"How does Celery dispatch tasks?\n\n{INSTRUCTION_BLOCK}"

    def test_extracted_block_round_trips(self) -> None:
        extracted = preregistered_instruction_block(REPO_ROOT / PREREGISTRATION_PATH)
        assert extracted == INSTRUCTION_BLOCK


class TestAnswerExtraction:
    def test_reads_the_last_files_block(self, checkout: Path) -> None:
        text = "FILES:\nREADME.md\n\nOn reflection:\n\nFILES:\npkg/service.py\n"
        paths, flag = extract_answer_paths(text, repo_root=checkout)
        assert flag is ProductLoopAnswerFlag.SCORED
        assert paths == ["pkg/service.py"]

    def test_missing_marker_scores_zero_as_unparsed(self, checkout: Path) -> None:
        paths, flag = extract_answer_paths("The answer is pkg/service.py.", repo_root=checkout)
        assert flag is ProductLoopAnswerFlag.ANSWER_UNPARSED
        assert paths == []

    def test_over_broad_list_scores_zero(self, checkout: Path) -> None:
        # Breadth alone must not reach completeness 1.0; this is the hole the
        # cardinality cap exists to close.
        listed = "\n".join(f"pkg/f{index}.py" for index in range(ANSWER_PATH_CAP + 1))
        paths, flag = extract_answer_paths(f"FILES:\n{listed}\n", repo_root=checkout)
        assert flag is ProductLoopAnswerFlag.ANSWER_OVER_BROAD
        assert paths == []

    def test_exactly_the_cap_is_still_scored(self, checkout: Path) -> None:
        listed = "\n".join(["pkg/service.py"] * ANSWER_PATH_CAP)
        paths, flag = extract_answer_paths(f"FILES:\n{listed}\n", repo_root=checkout)
        assert flag is ProductLoopAnswerFlag.SCORED
        assert paths == ["pkg/service.py"]

    def test_comma_separated_paths_are_split(self, checkout: Path) -> None:
        paths, flag = extract_answer_paths(
            "FILES:\npkg/service.py, pkg/models.py\n", repo_root=checkout
        )
        assert flag is ProductLoopAnswerFlag.SCORED
        assert paths == ["pkg/service.py", "pkg/models.py"]

    def test_duplicates_collapse(self, checkout: Path) -> None:
        paths, flag = extract_answer_paths(
            "FILES:\npkg/service.py\n./pkg/service.py\n", repo_root=checkout
        )
        assert flag is ProductLoopAnswerFlag.SCORED
        assert paths == ["pkg/service.py"]


class TestPathNormalization:
    @pytest.mark.parametrize(
        "token",
        [
            "pkg/service.py",
            "./pkg/service.py",
            "`pkg/service.py`",
            "pkg/service.py:L10-L20",
            "pkg/service.py:42",
            "pkg\\service.py",
            "[service](pkg/service.py)",
            "pkg/service.py,",
        ],
    )
    def test_shapes_an_agent_actually_emits_all_normalize(self, checkout: Path, token: str) -> None:
        # Graft's native pointers carry a :L<start>-L<end> suffix. Exact matching
        # without this normalization would penalise the Graft arm for echoing
        # its own tool's output format.
        assert normalize_answer_path(token, repo_root=checkout) == "pkg/service.py"

    def test_absolute_path_inside_the_checkout_is_rebased(self, checkout: Path) -> None:
        absolute = str(checkout / "pkg" / "models.py")
        assert normalize_answer_path(absolute, repo_root=checkout) == "pkg/models.py"

    def test_path_escaping_the_checkout_is_discarded(self, checkout: Path) -> None:
        assert normalize_answer_path("../outside.py", repo_root=checkout) is None

    def test_nonexistent_path_is_discarded(self, checkout: Path) -> None:
        assert normalize_answer_path("pkg/nope.py", repo_root=checkout) is None

    def test_directory_is_discarded(self, checkout: Path) -> None:
        assert normalize_answer_path("pkg/", repo_root=checkout) is None


class TestScoring:
    def test_matching_is_exact_not_by_basename(self) -> None:
        assert score_completeness(["celery/app/task.py"], ["celery/app/task.py"]) == 1
        assert score_completeness(["task.py"], ["celery/app/task.py"]) == 0
        assert score_completeness(["app/task.py"], ["celery/app/task.py"]) == 0

    def test_counts_distinct_required_files_only(self) -> None:
        expected = ["a.py", "b.py", "c.py"]
        assert score_completeness(["a.py", "a.py", "b.py"], expected) == 2


class TestTranscriptSummary:
    def _transcript(self, *events: dict[str, Any]) -> list[str]:
        return [json.dumps(event) for event in events]

    def test_reads_identity_tools_usage_and_calls(self) -> None:
        lines = self._transcript(
            {
                "type": "system",
                "subtype": "init",
                "tools": sorted(ProductLoopArm.ARCHEX.expected_tools),
                "mcp_servers": [{"name": "archex", "status": "connected"}],
            },
            {
                "type": "assistant",
                "message": {
                    "model": "claude-haiku-4-5-20260101",
                    "content": [{"type": "tool_use", "name": "Grep", "input": {}}],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "model": "claude-haiku-4-5-20260101",
                    "content": [
                        {"type": "tool_use", "name": "mcp__archex__query_repo", "input": {}}
                    ],
                },
            },
            {
                "type": "assistant",
                "message": {
                    "model": "claude-haiku-4-5-20260101",
                    "content": [{"type": "text", "text": "FILES:\npkg/service.py"}],
                },
            },
            {
                "type": "result",
                "is_error": False,
                "num_turns": 3,
                "total_cost_usd": 0.0412,
                "usage": {
                    "input_tokens": 2400,
                    "output_tokens": 120,
                    "cache_read_input_tokens": 800,
                    "cache_creation_input_tokens": 64,
                },
            },
        )
        summary = summarize_transcript(lines, arm=ProductLoopArm.ARCHEX)
        assert summary.tools_advertised == sorted(ProductLoopArm.ARCHEX.expected_tools)
        assert summary.mcp_status == "connected"
        assert summary.tool_calls == 2
        assert summary.product_tool_calls == 1
        assert summary.models_observed == ["claude-haiku-4-5-20260101"]
        assert summary.input_tokens == 2400
        assert summary.cache_read_tokens == 800
        assert summary.modelled_cost_usd == pytest.approx(0.0412)  # pyright: ignore[reportUnknownMemberType]
        assert summary.final_text.endswith("pkg/service.py")
        assert summary.saw_result

    def test_routine_quota_telemetry_is_not_a_rate_limit(self) -> None:
        # Claude Code emits this event on healthy turns with status "allowed".
        # Treating every occurrence as a failure would turn an entirely healthy
        # run into 114 recorded failures.
        lines = self._transcript(
            {"type": "system", "subtype": "init", "tools": [], "mcp_servers": []},
            {
                "type": "rate_limit_event",
                "rate_limit_info": {
                    "status": "allowed",
                    "unifiedWindows": {"five_hour": {"utilization": 0.04}},
                },
            },
            {"type": "result", "is_error": False, "total_cost_usd": 0.01},
        )
        summary = summarize_transcript(lines, arm=ProductLoopArm.ARCHEX)
        assert summary.rate_limited is False
        assert summary.quota_utilization == pytest.approx(0.04)  # pyright: ignore[reportUnknownMemberType]

    def test_a_real_rate_limit_is_visible(self) -> None:
        lines = self._transcript(
            {"type": "system", "subtype": "init", "tools": [], "mcp_servers": []},
            {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}},
            {"type": "result", "is_error": True, "result": "stopped", "total_cost_usd": 0.0},
        )
        summary = summarize_transcript(lines, arm=ProductLoopArm.ARCHEX)
        assert summary.rate_limited

    def test_mcp_error_after_init_marks_the_server_dropped(self) -> None:
        lines = self._transcript(
            {
                "type": "system",
                "subtype": "init",
                "tools": [],
                "mcp_servers": [{"name": "graft", "status": "connected"}],
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "is_error": True,
                            "content": "mcp__graft__graft_find_code failed: server exited",
                        }
                    ]
                },
            },
            {"type": "result", "is_error": False, "total_cost_usd": 0.0},
        )
        summary = summarize_transcript(lines, arm=ProductLoopArm.GRAFT)
        assert summary.mcp_dropped

    def test_malformed_line_is_a_hard_error(self) -> None:
        # A transcript the harness cannot read must become a recorded failure,
        # never a silently short-counted cell.
        with pytest.raises(ProductLoopError, match="not JSON"):
            summarize_transcript(["{oops"], arm=ProductLoopArm.ARCHEX)


class TestArtifactContract:
    def test_valid_cell_round_trips(self) -> None:
        artifact = ProductLoopCellArtifact.model_validate(_cell_payload())
        assert artifact.completeness == pytest.approx(1.0)  # pyright: ignore[reportUnknownMemberType]

    def test_wrong_model_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="model_requested"):
            ProductLoopCellArtifact.model_validate(_cell_payload(model_requested="claude-opus-5"))

    def test_wrong_agent_version_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="agent identity"):
            ProductLoopCellArtifact.model_validate(_cell_payload(agent_version="2.1.215"))

    def test_altered_prompt_hash_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="instruction_block_sha256"):
            ProductLoopCellArtifact.model_validate(_cell_payload(instruction_block_sha256="0" * 64))

    def test_tool_parity_violation_cannot_be_a_scored_cell(self) -> None:
        payload = _cell_payload(
            tools_advertised=[*sorted(ProductLoopArm.ARCHEX.expected_tools), "Bash"]
        )
        with pytest.raises(ValidationError, match="advertised"):
            ProductLoopCellArtifact.model_validate(payload)

    def test_fingerprint_must_describe_the_recorded_tool_list(self) -> None:
        with pytest.raises(ValidationError, match="ambient_tool_fingerprint"):
            ProductLoopCellArtifact.model_validate(_cell_payload(ambient_tool_fingerprint="0" * 64))

    def test_fingerprint_ignores_mcp_tools(self) -> None:
        # The arms advertise different MCP tools by design, so only the shared
        # non-MCP surface may enter the fingerprint the validator compares.
        assert ambient_tool_fingerprint(
            ["Read", "Grep", "Glob", "mcp__archex__context"]
        ) == ambient_tool_fingerprint(["Glob", "Grep", "Read", "mcp__graft__graft_find_code"])

    def test_disconnected_mcp_cannot_be_a_scored_cell(self) -> None:
        with pytest.raises(ValidationError, match="connected MCP"):
            ProductLoopCellArtifact.model_validate(_cell_payload(mcp_status="failed"))

    def test_graft_arm_must_register_the_graft_server(self) -> None:
        payload = _cell_payload(arm=ProductLoopArm.GRAFT.value)
        with pytest.raises(ValidationError, match="MCP server"):
            ProductLoopCellArtifact.model_validate(payload)

    def test_completeness_must_equal_the_recorded_ratio(self) -> None:
        with pytest.raises(ValidationError, match="does not equal"):
            ProductLoopCellArtifact.model_validate(
                _cell_payload(matched_file_count=1, completeness=1.0)
            )

    def test_failed_cell_enters_the_mean_as_zero(self) -> None:
        artifact = ProductLoopCellArtifact.model_validate(
            _cell_payload(
                status=ProductLoopCellStatus.FAILED.value,
                failure_reason=ProductLoopFailureReason.DEADLINE_EXCEEDED.value,
                failure_detail="agent exceeded the 300s cell deadline",
                matched_file_count=0,
                completeness=0.0,
                answer_flag=ProductLoopAnswerFlag.ANSWER_UNPARSED.value,
                answer_paths=[],
                answer_path_count=0,
                answer_precision=0.0,
            )
        )
        assert artifact.completeness == 0.0
        assert artifact.status is ProductLoopCellStatus.FAILED

    def test_failed_cell_may_not_claim_matches(self) -> None:
        with pytest.raises(ValidationError, match="enters the mean as 0.0"):
            ProductLoopCellArtifact.model_validate(
                _cell_payload(
                    status=ProductLoopCellStatus.FAILED.value,
                    failure_reason=ProductLoopFailureReason.AGENT_ERROR.value,
                )
            )

    def test_failed_cell_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="failure reason"):
            ProductLoopCellArtifact.model_validate(
                _cell_payload(
                    status=ProductLoopCellStatus.FAILED.value,
                    matched_file_count=0,
                    completeness=0.0,
                )
            )

    def test_unparsed_answer_cannot_score(self) -> None:
        with pytest.raises(ValidationError, match="score 0.0"):
            ProductLoopCellArtifact.model_validate(
                _cell_payload(answer_flag=ProductLoopAnswerFlag.ANSWER_UNPARSED.value)
            )

    def test_no_product_use_must_match_the_call_count(self) -> None:
        with pytest.raises(ValidationError, match="no_product_use"):
            ProductLoopCellArtifact.model_validate(
                _cell_payload(product_tool_calls=0, no_product_use=False)
            )

    def test_repetition_beyond_the_frozen_count_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exceeds the frozen"):
            ProductLoopCellArtifact.model_validate(_cell_payload(repetition=REPETITIONS + 1))


class TestCellFailurePaths:
    """A cell that cannot be measured must be retained, never abort the run."""

    def _task(self) -> Any:
        from archex.benchmark.models import BenchmarkTask

        return BenchmarkTask(
            task_id="celery_task_dispatch",
            repo="celery/celery",
            commit="v5.4.0",
            question="q",
            expected_files=["a.py", "b.py", "c.py"],
        )

    @pytest.mark.parametrize(
        "reason",
        [
            ProductLoopFailureReason.DEADLINE_EXCEEDED,
            ProductLoopFailureReason.TRANSCRIPT_UNPARSABLE,
        ],
    )
    def test_summaryless_failure_still_yields_a_valid_artifact(
        self, tmp_path: Path, reason: ProductLoopFailureReason
    ) -> None:
        # Killing the process group at the deadline truncates the last
        # stream-json line, so the summary is absent. If that path cannot build
        # an artifact, the first timed-out cell aborts an unregenerable run.
        artifact = build_cell_artifact(
            task=self._task(),
            arm=ProductLoopArm.ARCHEX,
            repetition=1,
            summary=None,
            reason=reason,
            detail="truncated",
            repo_path=tmp_path,
            setup_seconds=1.0,
            wall_seconds=300.0,
            hook_records=[],
            freshness_state=None,
            stale_index_event=False,
            graft_cards_greppable=None,
            mcp_command_original=None,
            mcp_command_used=None,
            provider_endpoint_overridden=False,
        )
        assert artifact.status is ProductLoopCellStatus.FAILED
        assert artifact.failure_reason is reason
        assert artifact.completeness == 0.0
        assert artifact.modelled_cost_usd == 0.0
        assert artifact.no_product_use is True

    def test_partial_hook_log_line_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "hooks.jsonl").write_text(
            '{"event": "PreToolUse", "stdin_bytes": 1, "stdout_bytes": 2, '
            '"exit_code": 0, "latency_ms": 1.0, "augmented": true}\n'
            '{"event": "PostToolUse", "stdin_by',
            encoding="utf-8",
        )
        records = read_hook_records(tmp_path)
        assert len(records) == 1
        assert records[0].event == "PreToolUse"

    @pytest.mark.parametrize(
        ("timed_out", "tools", "mcp_status", "dropped", "is_error", "saw_result", "expected"),
        [
            (
                True,
                None,
                "connected",
                False,
                False,
                True,
                ProductLoopFailureReason.DEADLINE_EXCEEDED,
            ),
            (False, None, "connected", False, True, True, ProductLoopFailureReason.AGENT_ERROR),
            (False, None, "connected", False, False, False, ProductLoopFailureReason.AGENT_ERROR),
            (False, None, "failed", False, False, True, ProductLoopFailureReason.MCP_DISCONNECTED),
            (
                False,
                None,
                "connected",
                True,
                False,
                True,
                ProductLoopFailureReason.MCP_DISCONNECTED,
            ),
            (
                False,
                [],
                "connected",
                False,
                False,
                True,
                ProductLoopFailureReason.TOOL_PARITY_VIOLATION,
            ),
            (False, None, "connected", False, False, True, None),
        ],
    )
    def test_failure_classification(
        self,
        timed_out: bool,
        tools: list[str] | None,
        mcp_status: str,
        dropped: bool,
        is_error: bool,
        saw_result: bool,
        expected: ProductLoopFailureReason | None,
    ) -> None:
        summary = TranscriptSummary(
            tools_advertised=(
                sorted(ProductLoopArm.ARCHEX.expected_tools) if tools is None else tools
            ),
            mcp_status=mcp_status,
            mcp_dropped=dropped,
            models_observed=[],
            tool_call_mix={},
            tool_calls=0,
            product_tool_calls=0,
            final_text="",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            modelled_cost_usd=0.0,
            num_turns=1,
            is_error=is_error,
            error_text=None,
            saw_result=saw_result,
            rate_limited=False,
        )
        actual = classify_cell_failure(
            timed_out=timed_out, summary=summary, arm=ProductLoopArm.ARCHEX
        )
        assert actual is expected

    def test_rate_limit_outranks_a_generic_agent_error(self) -> None:
        summary = TranscriptSummary(
            tools_advertised=[],
            mcp_status=None,
            mcp_dropped=False,
            models_observed=[],
            tool_call_mix={},
            tool_calls=0,
            product_tool_calls=0,
            final_text="",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            modelled_cost_usd=0.0,
            num_turns=0,
            is_error=True,
            error_text="rate limit",
            saw_result=True,
            rate_limited=True,
        )
        assert (
            classify_cell_failure(timed_out=False, summary=summary, arm=ProductLoopArm.ARCHEX)
            is ProductLoopFailureReason.RATE_LIMITED
        )


class TestSanitization:
    def test_workspace_paths_are_replaced(self) -> None:
        document = '{"failure_detail": "/scratch/cell/repo/pkg/x.py missing"}'
        sanitized = sanitize_document(
            document, replacements=[(Path("/scratch/cell/repo"), "<repo>")]
        )
        assert sanitized == '{"failure_detail": "<repo>/pkg/x.py missing"}'

    def test_residual_absolute_path_aborts(self) -> None:
        # A leaked path is a privacy-contract violation, and the artifact is the
        # published surface, so this fails the run rather than the check.
        with pytest.raises(ProductLoopError, match="leaks an absolute path"):
            sanitize_document('{"detail": "/Users/someone/repo"}', replacements=[])


class TestDirectoryValidation:
    def _write(self, directory: Path, task_ids: list[str], **overrides: Any) -> None:
        for arm in ProductLoopArm:
            arm_dir = directory / arm.value
            arm_dir.mkdir(parents=True, exist_ok=True)
            for task_id in task_ids:
                for repetition in range(1, REPETITIONS + 1):
                    payload = _cell_payload(
                        task_id=task_id,
                        arm=arm.value,
                        repetition=repetition,
                        mcp_server=arm.mcp_server,
                        tools_advertised=sorted(arm.expected_tools),
                        ambient_tool_fingerprint=ambient_tool_fingerprint(
                            sorted(arm.expected_tools)
                        ),
                        tool_call_mix={f"mcp__{arm.mcp_server}__x": 1},
                        **overrides,
                    )
                    path = arm_dir / cell_filename(task_id, repetition)
                    path.write_text(json.dumps(payload), encoding="utf-8")

    def test_complete_population_validates(self, tmp_path: Path) -> None:
        self._write(tmp_path, ["alpha", "beta"], modelled_cost_usd=0.01)
        coverage = validate_product_loop_directory(tmp_path, task_ids=["alpha", "beta"])
        assert coverage.complete
        assert coverage.cells == 2 * 2 * REPETITIONS
        assert coverage.ok_cells == coverage.cells

    def test_missing_cell_is_never_dropped(self, tmp_path: Path) -> None:
        self._write(tmp_path, ["alpha"], modelled_cost_usd=0.01)
        (tmp_path / ProductLoopArm.GRAFT.value / cell_filename("alpha", 2)).unlink()
        with pytest.raises(ProductLoopError, match="planned cells are missing"):
            validate_product_loop_directory(tmp_path, task_ids=["alpha"])

    def test_cell_outside_the_frozen_population_is_rejected(self, tmp_path: Path) -> None:
        self._write(tmp_path, ["alpha", "smuggled"], modelled_cost_usd=0.01)
        with pytest.raises(ProductLoopError, match="outside the frozen population"):
            validate_product_loop_directory(tmp_path, task_ids=["alpha"])

    def test_cost_beyond_the_ceiling_is_rejected(self, tmp_path: Path) -> None:
        self._write(tmp_path, ["alpha"], modelled_cost_usd=COST_CEILING_USD)
        with pytest.raises(ProductLoopError, match="exceeds the pre-registered"):
            validate_product_loop_directory(tmp_path, task_ids=["alpha"])

    def test_filename_must_match_its_cell(self, tmp_path: Path) -> None:
        self._write(tmp_path, ["alpha"], modelled_cost_usd=0.01)
        arm_dir = tmp_path / ProductLoopArm.ARCHEX.value
        (arm_dir / cell_filename("alpha", 1)).rename(arm_dir / "renamed.json")
        with pytest.raises(ProductLoopError, match="filename does not match"):
            validate_product_loop_directory(tmp_path, task_ids=["alpha"])

    def test_stub_endpoint_cell_is_never_publishable(self, tmp_path: Path) -> None:
        # The harness can be exercised end to end against a local stub endpoint;
        # nothing produced that way may ever reach published evidence.
        self._write(tmp_path, ["alpha"], modelled_cost_usd=0.01)
        path = tmp_path / ProductLoopArm.ARCHEX.value / cell_filename("alpha", 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["provider_endpoint_overridden"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ProductLoopError, match="overridden provider endpoint"):
            validate_product_loop_directory(tmp_path, task_ids=["alpha"])

    def test_a_changed_client_surface_mid_run_is_rejected(self, tmp_path: Path) -> None:
        # Subscription auth forces the operator's real home, so the built-in and
        # plugin surface is not fully excludable. An install that changes
        # mid-campaign must be a visible failure, not a silent confound.
        self._write(tmp_path, ["alpha"], modelled_cost_usd=0.01)
        path = tmp_path / ProductLoopArm.ARCHEX.value / cell_filename("alpha", 1)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["tools_advertised"] = [*payload["tools_advertised"], "Bash"]
        payload["ambient_tool_fingerprint"] = ambient_tool_fingerprint(payload["tools_advertised"])
        payload["status"] = ProductLoopCellStatus.FAILED.value
        payload["failure_reason"] = ProductLoopFailureReason.TOOL_PARITY_VIOLATION.value
        payload["matched_file_count"] = 0
        payload["completeness"] = 0.0
        path.write_text(json.dumps(payload), encoding="utf-8")
        # A failed cell does not contribute a fingerprint, so this still passes.
        validate_product_loop_directory(tmp_path, task_ids=["alpha"])

        other = tmp_path / ProductLoopArm.GRAFT.value / cell_filename("alpha", 1)
        payload = json.loads(other.read_text(encoding="utf-8"))
        payload["tools_advertised"] = ["Glob", "Grep", *sorted(ProductLoopArm.GRAFT.mcp_tools)]
        payload["ambient_tool_fingerprint"] = ambient_tool_fingerprint(payload["tools_advertised"])
        payload["status"] = ProductLoopCellStatus.FAILED.value
        payload["failure_reason"] = ProductLoopFailureReason.TOOL_PARITY_VIOLATION.value
        payload["matched_file_count"] = 0
        payload["completeness"] = 0.0
        other.write_text(json.dumps(payload), encoding="utf-8")
        validate_product_loop_directory(tmp_path, task_ids=["alpha"])

    def test_missing_arm_directory_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / ProductLoopArm.ARCHEX.value).mkdir(parents=True)
        with pytest.raises(ProductLoopError, match="missing the"):
            validate_product_loop_directory(tmp_path, task_ids=["alpha"])


class TestHookRecorder:
    """The recorder must be invisible to the product whose hook it wraps."""

    def _hook_script(self, tmp_path: Path) -> Path:
        script = tmp_path / "fake_hook.py"
        script.write_text(
            "import json, sys\n"
            "payload = json.loads(sys.stdin.read())\n"
            'sys.stdout.write(json.dumps({"hookSpecificOutput": '
            '{"additionalContext": payload["tool_name"]}}))\n'
            'sys.stderr.write("diag\\n")\n'
            "sys.exit(0)\n",
            encoding="utf-8",
        )
        return script

    def _run(self, argv: list[str], payload: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *argv],
            input=payload,
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )

    def test_stdout_and_exit_code_pass_through_unchanged(self, tmp_path: Path) -> None:
        script = self._hook_script(tmp_path)
        payload = json.dumps({"tool_name": "Grep", "tool_input": {"pattern": "x"}})
        direct = self._run([str(script)], payload)
        log = tmp_path / "hooks.jsonl"
        wrapped = self._run(
            [
                "-m",
                "archex.benchmark.product_loop_hook_recorder",
                "--log",
                str(log),
                "--event",
                "PreToolUse",
                "--matcher",
                "Glob|Grep",
                "--",
                sys.executable,
                str(script),
            ],
            payload,
        )
        assert wrapped.stdout == direct.stdout
        assert wrapped.returncode == direct.returncode

    def test_one_row_per_invocation_with_the_observed_fields(self, tmp_path: Path) -> None:
        script = self._hook_script(tmp_path)
        log = tmp_path / "hooks.jsonl"
        payload = json.dumps({"tool_name": "Glob", "tool_input": {"pattern": "*.py"}})
        for _ in range(2):
            self._run(
                [
                    "-m",
                    "archex.benchmark.product_loop_hook_recorder",
                    "--log",
                    str(log),
                    "--event",
                    "PreToolUse",
                    "--matcher",
                    "Glob|Grep",
                    "--",
                    sys.executable,
                    str(script),
                ],
                payload,
            )
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        assert rows[0]["event"] == "PreToolUse"
        assert rows[0]["matcher"] == "Glob|Grep"
        assert rows[0]["tool_name"] == "Glob"
        assert rows[0]["exit_code"] == 0
        assert rows[0]["augmented"] is True

    def test_a_hook_that_augments_nothing_is_recorded_as_such(self, tmp_path: Path) -> None:
        silent = tmp_path / "silent_hook.py"
        silent.write_text("import sys\nsys.stdin.read()\nsys.exit(0)\n", encoding="utf-8")
        log = tmp_path / "hooks.jsonl"
        self._run(
            [
                "-m",
                "archex.benchmark.product_loop_hook_recorder",
                "--log",
                str(log),
                "--event",
                "UserPromptSubmit",
                "--",
                sys.executable,
                str(silent),
            ],
            "{}",
        )
        row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert row["augmented"] is False
        assert row["stdout_bytes"] == 0

    def test_nonzero_exit_is_reproduced(self, tmp_path: Path) -> None:
        failing = tmp_path / "failing_hook.py"
        failing.write_text("import sys\nsys.stdin.read()\nsys.exit(3)\n", encoding="utf-8")
        log = tmp_path / "hooks.jsonl"
        result = self._run(
            [
                "-m",
                "archex.benchmark.product_loop_hook_recorder",
                "--log",
                str(log),
                "--event",
                "PostToolUse",
                "--",
                sys.executable,
                str(failing),
            ],
            "{}",
        )
        assert result.returncode == 3
        assert json.loads(log.read_text(encoding="utf-8").splitlines()[0])["exit_code"] == 3

    def test_unwritable_log_never_breaks_the_hook(self, tmp_path: Path) -> None:
        # A recorder failure must not become a product failure: the hook
        # contract is fail-open at the client boundary.
        script = self._hook_script(tmp_path)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory\n", encoding="utf-8")
        payload = json.dumps({"tool_name": "Grep"})
        result = self._run(
            [
                "-m",
                "archex.benchmark.product_loop_hook_recorder",
                "--log",
                str(blocker / "hooks.jsonl"),
                "--event",
                "PreToolUse",
                "--",
                sys.executable,
                str(script),
            ],
            payload,
        )
        assert result.returncode == 0
        assert "additionalContext" in result.stdout

    def test_shell_shaped_hook_command_is_executed_and_recorded(self, tmp_path: Path) -> None:
        # Graft ships its hooks as a single quoted command string, which Claude
        # Code runs through a shell. Exec'ing it directly silently records zero
        # invocations for the whole arm, so the shell path must be exercised.
        script = self._hook_script(tmp_path)
        log = tmp_path / "hooks.jsonl"
        payload = json.dumps({"tool_name": "Grep"})
        result = self._run(
            [
                "-m",
                "archex.benchmark.product_loop_hook_recorder",
                "--log",
                str(log),
                "--event",
                "UserPromptSubmit",
                "--shell",
                "--",
                f'{sys.executable} "{script}"',
            ],
            payload,
        )
        assert result.returncode == 0
        assert "additionalContext" in result.stdout
        row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert row["event"] == "UserPromptSubmit"
        assert row["augmented"] is True

    def test_unrunnable_command_is_recorded_not_silent(self, tmp_path: Path) -> None:
        # A recorder that cannot start the wrapped command must leave a visible
        # row; otherwise the arm looks like a product whose hook never fired.
        log = tmp_path / "hooks.jsonl"
        result = self._run(
            [
                "-m",
                "archex.benchmark.product_loop_hook_recorder",
                "--log",
                str(log),
                "--event",
                "PreToolUse",
                "--",
                str(tmp_path / "definitely-not-a-binary"),
            ],
            "{}",
        )
        assert result.returncode == 0
        row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert "recorder_error" in row
