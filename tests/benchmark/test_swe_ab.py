"""Tests for the SWE A/B harness core: channels, compounding, parsing, cells, validator."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from archex.benchmark.swe_ab import (
    BASE_TOOLS,
    BROKER_ENV_NAMES,
    CHANNELS,
    ISOLATION_FLAGS,
    MODELS,
    OMP_VERSION,
    QUOTA_BLOCKED_DIR,
    SUBSCRIPTION_PROVIDERS,
    CellKey,
    QuotaStatus,
    SweAbArm,
    SweAbCell,
    SweAbError,
    SweAbPlan,
    compound,
    credential_files_in_profile,
    is_emulated,
    is_quota_error,
    localize,
    observations,
    omp_argv,
    omp_channel,
    parse_omp_events,
    parse_omp_session,
    provider_endpoint_overridden,
    provider_logins,
    quota_blocked_relative_path,
    quota_reading,
    swe_agent_channel_of,
    tool_fingerprint,
    validate_swe_ab_directory,
)

# --- channels --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "channel"),
    [
        ("grep", {"pattern": "x"}, "search"),
        ("glob", {"path": "**/*.py"}, "search"),
        ("read", {"path": "a.py"}, "read"),
        ("edit", {"input": "[a.py#1A2B]"}, "edit"),
        ("write", {"path": "a.py"}, "write"),
        ("todo", {}, "other"),
        ("bash", {"command": "archex scout . 'where is auth'"}, "archex-CLI"),
        ("bash", {"command": "cd /app && archex symbol . 'symbol:a.py::f#function'"}, "archex-CLI"),
        ("bash", {"command": "cd /app && python -m pytest tests/x.py -q"}, "test"),
        ("bash", {"command": "go test ./..."}, "test"),
        ("bash", {"command": "rg -n needle src"}, "search"),
        ("bash", {"command": "git grep -n needle"}, "search"),
        ("bash", {"command": "ls -R src"}, "search"),
        ("bash", {"command": "ls src"}, "other"),
        ("bash", {"command": "cat src/a.py"}, "read"),
        ("bash", {"command": "sed -n 1,40p src/a.py"}, "read"),
        ("bash", {"command": "git status"}, "other"),
    ],
)
def test_omp_channel_follows_the_frozen_rule_table(
    tool: str, arguments: dict[str, Any], channel: str
) -> None:
    assert omp_channel(tool, arguments) == channel


@pytest.mark.parametrize(
    ("action", "channel"),
    [
        ("str_replace_editor view /app/a.py", "read_file"),
        ("str_replace_editor str_replace /app/a.py", "edit"),
        ("find . -name '*.go'", "search"),
        ("cat a.py | grep foo", "search"),
        ("pytest tests", "test_build"),
        ("submit", "submit_vcs"),
        ("echo hi", "other"),
    ],
)
def test_swe_agent_channel_rules_are_unchanged(action: str, channel: str) -> None:
    assert swe_agent_channel_of(action) == channel


# --- compounding -----------------------------------------------------------------


def test_an_observation_is_charged_once_per_later_request() -> None:
    once, compounded = compound([(0, "search", 100), (2, "read", 10), (3, "read", 5)], 4)

    assert once["search"] == 100 and once["read"] == 15
    # request 0 of 4 -> 3 later requests; request 2 -> 1; request 3 (last) -> 0
    assert compounded["search"] == 300
    assert compounded["read"] == 10
    assert set(once) == set(CHANNELS)


def test_compounding_rejects_a_channel_outside_the_table() -> None:
    with pytest.raises(SweAbError):
        compound([(0, "grep", 1)], 2)


# --- session parsing and attribution ------------------------------------------------


def _session_file(tmp_path: Path, *, error: bool = False) -> Path:
    def assistant(content: list[dict[str, Any]], tokens: int, stop: str) -> dict[str, Any]:
        return {
            "type": "message",
            "message": {
                "role": "assistant",
                "provider": "amazon-bedrock",
                "model": "m",
                "content": content,
                "usage": {
                    "input": tokens,
                    "output": 10,
                    "cacheRead": 5,
                    "cacheWrite": 1,
                    "cost": {"total": 0.01},
                },
                "stopReason": stop,
                "contextSnapshot": {"nonMessageTokens": 4101},
            },
        }

    def result(call_id: str, *blocks: str) -> dict[str, Any]:
        return {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": call_id,
                "toolName": "x",
                "content": [{"type": "text", "text": b} for b in blocks],
                "isError": False,
            },
        }

    grep = {"type": "toolCall", "id": "c1", "name": "grep", "arguments": {"pattern": "f"}}
    read = {
        "type": "toolCall",
        "id": "c2",
        "name": "read",
        "arguments": {"path": "/app/pkg/a.py:10-40"},
    }
    edit = {
        "type": "toolCall",
        "id": "c3",
        "name": "edit",
        "arguments": {"input": "[pkg/b.py#1A2B]\n"},
    }
    lines = [
        {"type": "session", "cwd": "/app"},
        assistant([grep], 100, "toolUse"),
        result(
            "c1", "# pkg/\n## a.py\n*3|def f():", "\n\n[archex receipt] index_revision=abc units=2"
        ),
        assistant([read], 200, "toolUse"),
        result("c2", "10:def f():"),
        assistant([edit], 300, "toolUse"),
        result("c3", "ok"),
        assistant([{"type": "text", "text": "done"}], 400, "error" if error else "stop"),
    ]
    path = tmp_path / "session.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_session_parser_joins_calls_to_results_and_sums_nothing_twice(tmp_path: Path) -> None:
    session = parse_omp_session(_session_file(tmp_path))

    assert [r.input for r in session.requests] == [100, 200, 300, 400]
    assert [(e.tool, e.request_index) for e in session.exchanges] == [
        ("grep", 0),
        ("read", 1),
        ("edit", 2),
    ]
    assert session.exchanges[0].result_text.endswith("units=2")
    assert session.requests[0].non_message_tokens == 4101
    assert session.error_message is None


def test_session_parser_surfaces_a_provider_error(tmp_path: Path) -> None:
    assert parse_omp_session(_session_file(tmp_path, error=True)).error_message is not None


def test_annotation_tokens_move_from_search_to_their_own_channel(tmp_path: Path) -> None:
    session = parse_omp_session(_session_file(tmp_path))

    rows = observations(session, {"c1": 7}, lambda text: len(text.split()))

    by_channel = {(index, channel): tokens for index, channel, tokens in rows}
    grep_total = len(session.exchanges[0].result_text.split())
    assert by_channel[(0, "archex-annotation")] == 7
    assert by_channel[(0, "search")] == grep_total - 7
    assert (1, "archex-annotation") not in by_channel


def test_localization_uses_normalized_read_and_edit_targets(tmp_path: Path) -> None:
    session = parse_omp_session(_session_file(tmp_path))

    loc = localize(session, ["pkg/a.py"], ["pkg/b.py"], ["/app"])

    assert loc.first_gold_read_request == 1
    assert loc.first_gold_edit_request is None
    assert loc.read_all_gold is True
    assert loc.edited_non_gold is True


# --- the frozen command line -----------------------------------------------------------


@pytest.mark.parametrize("arm", list(SweAbArm))
def test_omp_argv_differs_across_arms_only_by_hook_and_guide(arm: SweAbArm) -> None:
    argv = omp_argv(
        ["omp"],
        arm=arm,
        model="m",
        prompt_path="/task/prompt.md",
        session_dir="/out/s",
        hook_module_path="/opt/archex/omp-annotate.ts" if arm.hook else None,
        cli_guide_path="/task/cli-guide.md" if arm.cli else None,
    )

    assert argv[argv.index("--tools") + 1] == ",".join(BASE_TOOLS)
    assert all(flag in argv for flag in ISOLATION_FLAGS)
    assert ("-e" in argv) is arm.hook
    assert ("--append-system-prompt" in argv) is arm.cli
    assert argv[-1] == "@/task/prompt.md"


def test_omp_argv_refuses_a_hook_outside_the_hook_arms() -> None:
    with pytest.raises(SweAbError):
        omp_argv(
            ["omp"],
            arm=SweAbArm.A0,
            model="m",
            prompt_path="p",
            session_dir="s",
            hook_module_path="/opt/archex/omp-annotate.ts",
            cli_guide_path=None,
        )


# --- endpoint override detection --------------------------------------------------------


def test_a_profile_provider_with_its_own_base_url_is_an_override(tmp_path: Path) -> None:
    (tmp_path / "models.yml").write_text(
        "providers:\n  stub:\n    baseUrl: http://127.0.0.1:4000/v1\n    api: openai-completions\n"
        "    auth: none\n    models:\n      - id: stub-model\n",
        encoding="utf-8",
    )

    assert provider_endpoint_overridden(tmp_path, "stub/stub-model") is True
    assert provider_endpoint_overridden(tmp_path, "global.openai.gpt-6-sol") is False
    assert provider_endpoint_overridden(tmp_path / "absent", "stub/stub-model") is False


# --- cells and the directory validator --------------------------------------------------


MODEL = "anthropic/claude-sonnet-5-5"


def _cell(arm: SweAbArm = SweAbArm.A0, **overrides: Any) -> dict[str, Any]:
    zero = dict.fromkeys(CHANNELS, 0)
    cell: dict[str, Any] = {
        "task_id": "t1",
        "repo": "org/repo",
        "model": MODEL,
        "arm": arm.value,
        "repetition": 1,
        "status": "ok",
        "omp_version": OMP_VERSION,
        "archex_version": "0.33.0" if arm.archex_installed else None,
        "archex_wheel_sha256": "w" if arm.archex_installed else None,
        "hook_module_sha256": "h" if arm.hook else None,
        "cli_guide_sha256": "g" if arm.cli else None,
        "image": "img",
        "tool_fingerprint": tool_fingerprint(BASE_TOOLS),
        "provider": "anthropic",
        "provider_endpoint_overridden": False,
        "emulated": False,
        "network": "bridge",
        "credential_env_names": sorted(BROKER_ENV_NAMES),
        "usage": {"input": 10, "output": 1, "cache_read": 0, "cache_write": 0, "cost_usd": 1.0},
        "requests": 1,
        "tool_calls": 0,
        "channel_tokens_once": zero,
        "channel_tokens_compounded": dict(zero),
        "hook_ledger": (
            {"results": 0, "eligible": 0, "annotated": 0, "units": 0, "tokens": 0}
            if arm.hook
            else None
        ),
        "archex_cli_calls": 0,
        "isolation": {
            "tools_source": "declared",
            "tools_advertised": sorted(BASE_TOOLS),
            "system_prompt_checked": False,
            "compressor_marker_seen": False,
            "annotation_seen": False,
        },
        "localization": {"gold_files": []},
        "patch_sha256": "p",
        "patch_bytes": 0,
        "resolved": True,
        "score_source": "pro_verifier",
        "wall_seconds": 1.0,
        "setup_seconds": 1.0,
        "quota": {},
    }
    cell.update(overrides)
    return cell


@pytest.mark.parametrize(
    ("arm", "overrides"),
    [
        (SweAbArm.A0, {"isolation": {**_cell()["isolation"], "annotation_seen": True}}),
        (SweAbArm.H, {"hook_ledger": None}),
        (SweAbArm.C, {"cli_guide_sha256": None}),
        (SweAbArm.A0, {"isolation": {**_cell()["isolation"], "tools_advertised": ["read"]}}),
        (SweAbArm.A0, {"status": "failed"}),
        (SweAbArm.A0, {"status": "failed", "failure_reason": "timeout", "resolved": True}),
        (SweAbArm.A0, {"omp_version": "18.5.0"}),
    ],
)
def test_cell_schema_rejects_protocol_violations(arm: SweAbArm, overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SweAbCell.model_validate(_cell(arm, **overrides))


def _plan(**overrides: Any) -> SweAbPlan:
    raw: dict[str, Any] = {
        "name": "p",
        "tasks": [{"task_id": "t1", "repo": "org/repo"}],
        "models": [MODEL],
        "repetitions": {"A0": 2, "H": 1},
        "cost_ceiling_usd": 10.0,
    }
    raw.update(overrides)
    return SweAbPlan.model_validate(raw)


def _write(directory: Path, cell: dict[str, Any]) -> None:
    parsed = SweAbCell.model_validate(cell)
    path = directory / parsed.key.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(parsed.model_dump_json(), encoding="utf-8")


def _complete(directory: Path) -> None:
    _write(directory, _cell(SweAbArm.A0, repetition=1))
    _write(directory, _cell(SweAbArm.A0, repetition=2))
    _write(directory, _cell(SweAbArm.H))


def test_plan_declares_every_repetition_of_every_arm() -> None:
    assert _plan().cells() == [
        CellKey("t1", MODEL, SweAbArm.A0, 1),
        CellKey("t1", MODEL, SweAbArm.A0, 2),
        CellKey("t1", MODEL, SweAbArm.H, 1),
    ]


def test_complete_directory_validates_and_keeps_failures(tmp_path: Path) -> None:
    _complete(tmp_path)
    failed = _cell(SweAbArm.A0, repetition=2, status="failed", failure_reason="timeout")
    _write(tmp_path, {**failed, "resolved": False})

    coverage = validate_swe_ab_directory(tmp_path, _plan())

    assert (coverage.present, coverage.ok, coverage.failed) == (3, 2, 1)


def test_validator_refuses_stub_endpoint_cells_for_publication(tmp_path: Path) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.H, provider_endpoint_overridden=True))

    with pytest.raises(SweAbError, match="overridden provider endpoint"):
        validate_swe_ab_directory(tmp_path, _plan())


def test_validator_reports_a_missing_declared_cell(tmp_path: Path) -> None:
    _write(tmp_path, _cell(SweAbArm.A0, repetition=1))

    with pytest.raises(SweAbError, match="missing"):
        validate_swe_ab_directory(tmp_path, _plan())
    assert validate_swe_ab_directory(tmp_path, _plan(), require_complete=False).present == 1


def test_validator_rejects_an_undeclared_cell(tmp_path: Path) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.HC))

    with pytest.raises(SweAbError, match="not a declared cell"):
        validate_swe_ab_directory(tmp_path, _plan())


def test_validator_rejects_mixed_identities_within_an_arm(tmp_path: Path) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.H, repetition=2, hook_module_sha256="other"))
    plan = _plan(repetitions={"A0": 2, "H": 2})

    with pytest.raises(SweAbError, match="hook module differs"):
        validate_swe_ab_directory(tmp_path, plan)


def test_validator_enforces_the_cost_ceiling(tmp_path: Path) -> None:
    usage = {"input": 1, "output": 1, "cache_read": 0, "cache_write": 0, "cost_usd": 6.0}
    _write(tmp_path, _cell(SweAbArm.A0, repetition=1, usage=usage))
    _write(tmp_path, _cell(SweAbArm.A0, repetition=2, usage=usage))
    _write(tmp_path, _cell(SweAbArm.H, usage=usage))

    with pytest.raises(SweAbError, match="ceiling"):
        validate_swe_ab_directory(tmp_path, _plan())


# --- subscription route, emulation, and quota blocks in the validator -----------------------


def _complete_with(directory: Path, **overrides: Any) -> None:
    _write(directory, _cell(SweAbArm.A0, repetition=1, **overrides))
    _write(directory, _cell(SweAbArm.A0, repetition=2, **overrides))
    _write(directory, _cell(SweAbArm.H, **overrides))


def _blocked(arm: SweAbArm = SweAbArm.A0, **overrides: Any) -> dict[str, Any]:
    return _cell(
        arm,
        status="failed",
        failure_reason="quota_block",
        resolved=False,
        quota={"block_phase": "mid_run"},
        **overrides,
    )


def test_every_campaign_model_is_a_subscription_selector() -> None:
    assert all(model.split("/", 1)[0] in SUBSCRIPTION_PROVIDERS for model in MODELS)


def test_validator_refuses_a_stage_that_mixes_emulated_and_native_cells(tmp_path: Path) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.A0, repetition=2, emulated=True))

    with pytest.raises(SweAbError, match="emulated and native"):
        validate_swe_ab_directory(tmp_path, _plan())


def test_validator_accepts_a_stage_that_is_uniformly_emulated(tmp_path: Path) -> None:
    _complete_with(tmp_path, emulated=True)

    assert validate_swe_ab_directory(tmp_path, _plan()).ok == 3


def test_validator_refuses_cells_that_did_not_authenticate_through_the_broker(
    tmp_path: Path,
) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.H, credential_env_names=[]))

    with pytest.raises(SweAbError, match="auth broker"):
        validate_swe_ab_directory(tmp_path, _plan())


@pytest.mark.parametrize("provider", ["amazon-bedrock", "openai-codex", "openrouter"])
def test_validator_refuses_a_cell_that_left_its_subscription_route(
    tmp_path: Path, provider: str
) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.H, provider=provider))

    with pytest.raises(SweAbError, match="subscription route"):
        validate_swe_ab_directory(tmp_path, _plan())


def test_a_quota_block_phase_is_recorded_iff_the_reason_is_quota_block() -> None:
    with pytest.raises(ValueError, match="quota block phase"):
        SweAbCell.model_validate(
            _cell(status="failed", failure_reason="quota_block", resolved=False)
        )
    with pytest.raises(ValueError, match="quota block phase"):
        SweAbCell.model_validate(
            _cell(
                status="failed",
                failure_reason="timeout",
                resolved=False,
                quota={"block_phase": "before_first_tool_call"},
            )
        )


def test_validator_refuses_a_quota_blocked_cell_left_as_the_record(tmp_path: Path) -> None:
    _complete(tmp_path)
    _write(tmp_path, _blocked(SweAbArm.A0, repetition=2))

    with pytest.raises(SweAbError, match="quota block"):
        validate_swe_ab_directory(tmp_path, _plan())


def test_filed_quota_blocked_attempts_are_counted_in_cost_and_never_scored(
    tmp_path: Path,
) -> None:
    _complete(tmp_path)
    attempt = SweAbCell.model_validate(_blocked(SweAbArm.A0, repetition=2))
    filed = tmp_path / quota_blocked_relative_path(attempt.key, 1)
    filed.parent.mkdir(parents=True)
    filed.write_text(attempt.model_dump_json(), encoding="utf-8")

    coverage = validate_swe_ab_directory(tmp_path, _plan())

    assert (coverage.present, coverage.ok, coverage.quota_blocked) == (3, 3, 1)
    assert coverage.total_cost_usd == 4.0


def test_a_cell_that_was_not_quota_blocked_cannot_be_filed_as_one(tmp_path: Path) -> None:
    _complete(tmp_path)
    ordinary = SweAbCell.model_validate(_cell(SweAbArm.A0, repetition=2))
    filed = tmp_path / QUOTA_BLOCKED_DIR / "any" / "A0" / "t1__rep2.attempt1.json"
    filed.parent.mkdir(parents=True)
    filed.write_text(ordinary.model_dump_json(), encoding="utf-8")

    with pytest.raises(SweAbError, match="not one"):
        validate_swe_ab_directory(tmp_path, _plan())


# --- quota headroom from the broker's usage report --------------------------------------------


def _limit(
    *,
    used: float | None = None,
    percent: float | None = None,
    status: str | None = None,
    reset: int | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    amount: dict[str, Any] = {"unit": "percent" if percent is not None else "requests"}
    if used is not None:
        amount["usedFraction"] = used
    if percent is not None:
        amount["used"] = percent
    limit: dict[str, Any] = {"id": "x", "amount": amount, "scope": {"provider": "anthropic"}}
    if model:
        limit["scope"]["modelId"] = model
    if reset is not None:
        limit["window"] = {"id": "5h", "resetsAt": reset}
    if status:
        limit["status"] = status
    return limit


def _usage(*reports: dict[str, Any]) -> dict[str, Any]:
    return {"generatedAt": 0, "reports": list(reports)}


def _report(*limits: dict[str, Any], provider: str = "anthropic") -> dict[str, Any]:
    return {"provider": provider, "fetchedAt": 0, "limits": list(limits)}


def _reading(usage: dict[str, Any], model_id: str = "claude-sonnet-5-5") -> Any:
    return quota_reading(usage, "anthropic", model_id=model_id, min_headroom=0.10)


def test_a_provider_with_headroom_on_every_window_is_clear() -> None:
    reading = _reading(_usage(_report(_limit(used=0.5), _limit(used=0.2))))

    assert reading.status is QuotaStatus.CLEAR
    assert reading.headroom == pytest.approx(0.5)  # pyright: ignore[reportUnknownMemberType]


def test_the_tightest_window_blocks_and_its_reset_is_reported() -> None:
    usage = _usage(_report(_limit(used=0.95, reset=1_000), _limit(used=0.2, reset=9_000)))

    reading = _reading(usage)

    assert (reading.status, reading.resets_at_ms) == (QuotaStatus.BLOCKED, 1_000)


def test_a_login_frees_only_when_its_last_blocking_window_resets() -> None:
    usage = _usage(_report(_limit(used=0.99, reset=1_000), _limit(used=0.97, reset=9_000)))

    assert _reading(usage).resets_at_ms == 9_000


def test_percent_units_and_exhausted_status_block_without_a_used_fraction() -> None:
    assert _reading(_usage(_report(_limit(percent=96.0)))).status is QuotaStatus.BLOCKED
    assert _reading(_usage(_report(_limit(status="exhausted")))).status is QuotaStatus.BLOCKED


def test_a_limit_scoped_to_another_model_does_not_gate_this_model() -> None:
    usage = _usage(_report(_limit(used=0.2), _limit(used=1.0, model="claude-opus-5-5")))

    assert _reading(usage, "claude-sonnet-5-5").status is QuotaStatus.CLEAR
    assert _reading(usage, "claude-opus-5-5").status is QuotaStatus.BLOCKED


def test_one_login_with_headroom_keeps_the_provider_clear() -> None:
    usage = _usage(_report(_limit(used=1.0, reset=1_000)), _report(_limit(used=0.4)))

    reading = _reading(usage)

    assert reading.status is QuotaStatus.CLEAR
    assert reading.headroom == pytest.approx(0.6)  # pyright: ignore[reportUnknownMemberType]


def test_with_every_login_exhausted_the_earliest_reset_wins() -> None:
    usage = _usage(_report(_limit(used=1.0, reset=5_000)), _report(_limit(used=1.0, reset=2_000)))

    reading = _reading(usage)

    assert (reading.status, reading.resets_at_ms) == (QuotaStatus.BLOCKED, 2_000)


def test_a_blocking_window_without_a_reset_time_leaves_the_reset_unknown() -> None:
    reading = _reading(_usage(_report(_limit(used=1.0))))

    assert (reading.status, reading.resets_at_ms) == (QuotaStatus.BLOCKED, None)


def test_a_provider_the_broker_says_nothing_usable_about_is_unknown() -> None:
    assert _reading(_usage()).status is QuotaStatus.UNKNOWN
    assert _reading(_usage(_report(provider="openai-codex"))).status is QuotaStatus.UNKNOWN
    assert _reading(_usage(_report(_limit()))).status is QuotaStatus.UNKNOWN


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 rate limit exceeded", True),
        ("You have hit your usage limit. Try again in 3 hours.", True),
        ("Claude usage limit reached; your limit will reset at 5pm", True),
        ("Quota exceeded for this plan", True),
        ("Your subscription plan cap has been reached", True),
        ("Too Many Requests", True),
        ("overloaded_error: the service is temporarily overloaded", False),
        ("connection reset by peer", False),
        ("400 invalid request: messages.0.content", False),
        ("", False),
        (None, False),
    ],
)
def test_quota_errors_are_told_from_other_provider_errors(
    message: str | None, expected: bool
) -> None:
    assert is_quota_error(message) is expected


def test_omp_events_count_quota_retries_and_the_retry_it_gave_up_on() -> None:
    events = "\n".join(
        [
            "not json",
            json.dumps({"type": "tool_execution_start", "toolName": "read"}),
            json.dumps(
                {"type": "auto_retry_start", "delayMs": 30_000, "errorMessage": "429 rate limit"}
            ),
            json.dumps(
                {"type": "auto_retry_start", "delayMs": 500, "errorMessage": "connection reset"}
            ),
            json.dumps(
                {"type": "auto_retry_start", "delayMs": 5_000, "errorMessage": "usage limit"}
            ),
            json.dumps({"type": "auto_retry_end", "success": False, "finalError": "usage limit"}),
            json.dumps({"type": "tool_execution_start", "toolName": "grep"}),
        ]
    )

    parsed = parse_omp_events(events)

    assert (parsed.retries, parsed.quota_retries) == (3, 2)
    assert parsed.quota_wait_seconds == 35.0
    assert parsed.gave_up == "usage limit"
    assert parsed.tool_calls_started == 2


def test_a_retried_error_turn_does_not_fail_the_run_but_a_final_one_does(tmp_path: Path) -> None:
    def line(stop: str, error: str | None = None) -> str:
        message: dict[str, Any] = {
            "role": "assistant",
            "provider": "anthropic",
            "model": "x",
            "content": [],
            "usage": {},
            "stopReason": stop,
        }
        if error:
            message["errorMessage"] = error
        return json.dumps({"type": "message", "message": message})

    recovered = tmp_path / "recovered.jsonl"
    recovered.write_text(
        "\n".join([line("error", "429 rate limit"), line("stop")]), encoding="utf-8"
    )
    ended = tmp_path / "ended.jsonl"
    ended.write_text("\n".join([line("stop"), line("error", "usage limit")]), encoding="utf-8")

    assert parse_omp_session(recovered).error_message is None
    assert parse_omp_session(ended).error_message == "usage limit"


# --- credential hygiene and emulation ----------------------------------------------------------


def test_credential_stores_in_a_profile_are_found_and_nothing_else(tmp_path: Path) -> None:
    (tmp_path / "config.yml").write_text("x: 1")
    (tmp_path / "models.yml").write_text("providers: {}")
    (tmp_path / "agent.db").write_text("")
    (tmp_path / "agent.db-wal").write_text("")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "auth-broker-snapshot.enc").write_text("")
    (tmp_path / "auth-broker.token").write_text("")

    assert credential_files_in_profile(tmp_path) == [
        "agent.db",
        "agent.db-wal",
        "auth-broker.token",
        "cache/auth-broker-snapshot.enc",
    ]
    assert credential_files_in_profile(tmp_path / "absent") == []


def _login_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE auth_credentials (id INTEGER PRIMARY KEY, provider TEXT, "
            "credential_type TEXT, data TEXT, disabled_cause TEXT DEFAULT NULL)"
        )
        db.executemany(
            "INSERT INTO auth_credentials (provider, credential_type, data, disabled_cause) "
            "VALUES (?, ?, 'opaque', ?)",
            [
                ("anthropic", "oauth", None),
                ("anthropic", "oauth", "revoked"),
                ("openai-codex", "api_key", None),
            ],
        )


def test_login_metadata_counts_enabled_and_disabled_rows_without_touching_the_vault(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "agent.db"
    _login_db(db_path)
    before = db_path.read_bytes()

    logins = provider_logins(db_path)

    assert (logins["anthropic"].enabled, logins["anthropic"].disabled) == (1, 1)
    assert logins["anthropic"].types == ("oauth",)
    assert (logins["openai-codex"].enabled, logins["openai-codex"].types) == (1, ("api_key",))
    assert db_path.read_bytes() == before


def test_login_metadata_of_a_missing_or_foreign_database_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SweAbError, match="does not exist"):
        provider_logins(tmp_path / "absent.db")
    foreign = tmp_path / "other.db"
    with sqlite3.connect(foreign) as db:
        db.execute("CREATE TABLE unrelated (x)")
    with pytest.raises(SweAbError, match="cannot read login metadata"):
        provider_logins(foreign)


@pytest.mark.parametrize(
    ("runtime", "machine", "emulated"),
    [
        ("docker", "arm64", True),
        ("docker", "aarch64", True),
        ("docker", "x86_64", False),
        ("docker", "AMD64", False),
        ("local", "arm64", False),
    ],
)
def test_docker_cells_are_emulated_exactly_when_the_host_is_not_x86(
    runtime: str, machine: str, emulated: bool
) -> None:
    assert is_emulated(runtime, machine) is emulated
