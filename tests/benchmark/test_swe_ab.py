"""Tests for the SWE A/B harness core: channels, compounding, parsing, cells, validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from archex.benchmark.swe_ab import (
    BASE_TOOLS,
    CHANNELS,
    CONFIGURATIONS,
    CREDENTIAL_ENV_NAMES,
    ISOLATION_FLAGS,
    MUNA_BASE_URL,
    OMP_VERSION,
    QUOTA_BLOCKED_DIR,
    CellKey,
    Configuration,
    SweAbArm,
    SweAbCell,
    SweAbError,
    SweAbPlan,
    compound,
    configuration,
    credential_files_in_profile,
    is_credit_error,
    is_emulated,
    is_quota_error,
    localize,
    observations,
    omp_argv,
    omp_channel,
    out_of_patch_read_tokens_compounded,
    parse_omp_events,
    parse_omp_session,
    provider_base_url,
    quota_blocked_relative_path,
    summarize_ledger,
    swe_agent_channel_of,
    tool_fingerprint,
    unpriced_models,
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


def test_out_of_patch_reads_exclude_files_either_accepted_patch_touches(tmp_path: Path) -> None:
    session = parse_omp_session(_session_file(tmp_path))

    def words(text: str) -> int:
        return len(text.split())

    # The read of pkg/a.py is request 1 of 4, so its 2-word result is re-sent twice.
    assert out_of_patch_read_tokens_compounded(session, ["pkg/b.py"], ["/app"], words) == 4
    assert out_of_patch_read_tokens_compounded(session, ["pkg/a.py"], ["/app"], words) == 0


def test_ledger_counts_stale_declines_only_after_the_first_edit() -> None:
    rows = [
        {"toolCallId": "c0", "eligible": True, "annotated": False, "reason": "index_not_fresh"},
        {"toolCallId": "c1", "eligible": True, "annotated": True, "reason": None},
        {"toolCallId": "c4", "eligible": True, "annotated": False, "reason": "index_not_fresh"},
        {"toolCallId": "c5", "eligible": True, "annotated": False, "reason": "no_hits"},
    ]

    summary = summarize_ledger(rows, {"c4", "c5"})

    assert summary.not_fresh_after_first_edit == 1
    assert (summary.eligible, summary.annotated) == (4, 1)


# --- the frozen command line -----------------------------------------------------------


@pytest.mark.parametrize("arm", list(SweAbArm))
def test_omp_argv_differs_across_arms_only_by_hook_and_guide(arm: SweAbArm) -> None:
    argv = omp_argv(
        ["omp"],
        arm=arm,
        model=CONFIGURATIONS[0].label,
        prompt_path="/task/prompt.md",
        session_dir="/out/s",
        hook_module_path="/opt/archex/omp-annotate.ts" if arm.hook else None,
        cli_guide_path="/task/cli-guide.md" if arm.cli else None,
        omp_config_path="/task/omp-campaign.yml",
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
            model=CONFIGURATIONS[0].label,
            prompt_path="p",
            session_dir="s",
            hook_module_path="/opt/archex/omp-annotate.ts",
            cli_guide_path=None,
            omp_config_path="c",
        )


def _argv_for(label: str) -> list[str]:
    return omp_argv(
        ["omp"],
        arm=SweAbArm.A0,
        model=label,
        prompt_path="p",
        session_dir="s",
        hook_module_path=None,
        cli_guide_path=None,
        omp_config_path="/task/omp-campaign.yml",
    )


@pytest.mark.parametrize("entry", CONFIGURATIONS, ids=lambda entry: entry.label)
def test_omp_argv_carries_the_configurations_selector_effort_and_config(
    entry: Configuration,
) -> None:
    argv = _argv_for(entry.label)

    assert argv[argv.index("--model") + 1] == entry.selector
    assert argv[argv.index("--thinking") + 1] == entry.thinking
    assert argv[argv.index("--config") + 1] == "/task/omp-campaign.yml"


def test_the_qwen_configurations_differ_only_in_thinking() -> None:
    low, high = _argv_for("qwen-3.8-27b@low"), _argv_for("qwen-3.8-27b@high")

    differing = [(a, b) for a, b in zip(low, high, strict=True) if a != b]
    assert differing == [("low", "high")]
    assert low[low.index("--thinking") + 1] == "low"


def test_an_unknown_configuration_label_is_a_protocol_error() -> None:
    with pytest.raises(SweAbError):
        _argv_for("qwen-3.8-27b")
    with pytest.raises(SweAbError):
        configuration("muna/@qwen/qwen-3.8-27b")


# --- provider config ------------------------------------------------------------------------


def _models_yml(tmp_path: Path, cost: str) -> Path:
    path = tmp_path / "models.yml"
    path.write_text(
        "providers:\n  muna:\n    baseUrl: https://inference.muna.ai/v1\n    models:\n"
        f"      - id: m1\n        cost: {cost}\n",
        encoding="utf-8",
    )
    return path


def test_provider_base_url_is_read_from_the_models_file(tmp_path: Path) -> None:
    path = _models_yml(tmp_path, "{input: 1}")

    assert provider_base_url(path, "muna") == "https://inference.muna.ai/v1"
    assert provider_base_url(path, "other") is None
    assert provider_base_url(tmp_path / "absent.yml", "muna") is None


def test_unpriced_models_flags_zero_and_absent_prices(tmp_path: Path) -> None:
    path = _models_yml(tmp_path, "{input: 0.5, output: 0, cacheWrite: 0}")

    assert unpriced_models(path) == ["m1: output", "m1: cacheRead"]


def test_unpriced_models_accepts_a_fully_priced_config_without_cache_write(tmp_path: Path) -> None:
    path = _models_yml(tmp_path, "{input: 0.5, output: 1.5, cacheRead: 0.1, cacheWrite: 0}")

    assert unpriced_models(path) == []


# --- cells and the directory validator --------------------------------------------------


MODEL = CONFIGURATIONS[0].label


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
        "provider": "muna",
        "provider_base_url": MUNA_BASE_URL,
        "provider_config_sha256": "pc",
        "omp_config_sha256": "oc",
        "emulated": False,
        "network": "bridge",
        "credential_env_names": list(CREDENTIAL_ENV_NAMES),
        "usage": {"input": 10, "output": 1, "cache_read": 0, "cache_write": 0, "cost_usd": 1.0},
        "requests": 1,
        "tool_calls": 0,
        "channel_tokens_once": zero,
        "channel_tokens_compounded": dict(zero),
        "out_of_patch_read_tokens_compounded": 0,
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
    cell.setdefault(
        "thinking", dict((c.label, c.thinking) for c in CONFIGURATIONS).get(cell["model"])
    )
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
        (SweAbArm.A0, {"thinking": "high"}),
        (SweAbArm.A0, {"model": "anthropic/claude-sonnet-5-5", "thinking": "high"}),
        (
            SweAbArm.A0,
            {"failure_reason": "credit_exhausted", "status": "failed", "resolved": False},
        ),
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


def test_a_plan_refuses_models_that_are_not_distinct_configuration_labels() -> None:
    with pytest.raises(ValueError, match="distinct campaign configurations"):
        _plan(models=["anthropic/claude-sonnet-5-5"])
    with pytest.raises(ValueError, match="distinct campaign configurations"):
        _plan(models=[MODEL, MODEL])


def test_scheduled_cells_run_one_model_family_at_a_time_in_task_order() -> None:
    plan = _plan(
        tasks=[{"task_id": "t2", "repo": "org/repo"}, {"task_id": "t1", "repo": "org/repo"}],
        models=["gemma-4-26b-a4b-it@high", "qwen-3.8-27b@high", "qwen-3.8-27b@low"],
        repetitions={"A0": 1},
    )

    assert [(key.task_id, key.model) for key in plan.scheduled_cells()] == [
        ("t2", "gemma-4-26b-a4b-it@high"),
        ("t1", "gemma-4-26b-a4b-it@high"),
        ("t2", "qwen-3.8-27b@high"),
        ("t2", "qwen-3.8-27b@low"),
        ("t1", "qwen-3.8-27b@high"),
        ("t1", "qwen-3.8-27b@low"),
    ]
    assert sorted(plan.scheduled_cells()) == sorted(plan.cells())


def test_complete_directory_validates_and_keeps_failures(tmp_path: Path) -> None:
    _complete(tmp_path)
    failed = _cell(SweAbArm.A0, repetition=2, status="failed", failure_reason="timeout")
    _write(tmp_path, {**failed, "resolved": False})

    coverage = validate_swe_ab_directory(tmp_path, _plan())

    assert (coverage.present, coverage.ok, coverage.failed) == (3, 2, 1)


def test_validator_refuses_stub_endpoint_cells_for_publication(tmp_path: Path) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.H, provider_base_url="http://127.0.0.1:4000/v1"))

    with pytest.raises(SweAbError, match="127.0.0.1:4000"):
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


def test_validator_refuses_a_stage_that_mixes_emulated_and_native_cells(tmp_path: Path) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.A0, repetition=2, emulated=True))

    with pytest.raises(SweAbError, match="emulated and native"):
        validate_swe_ab_directory(tmp_path, _plan())


def test_validator_accepts_a_stage_that_is_uniformly_emulated(tmp_path: Path) -> None:
    _complete_with(tmp_path, emulated=True)

    assert validate_swe_ab_directory(tmp_path, _plan()).ok == 3


def test_validator_refuses_a_cell_that_resolved_another_endpoint(tmp_path: Path) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.H, provider_base_url="http://127.0.0.1:4000/v1"))

    with pytest.raises(SweAbError, match="a local stub or another endpoint"):
        validate_swe_ab_directory(tmp_path, _plan())


@pytest.mark.parametrize("names", [[], ["MUNA_ACCESS_KEY", "OPENAI_API_KEY"], ["OPENAI_API_KEY"]])
def test_validator_refuses_cells_that_did_not_authenticate_with_only_the_muna_key(
    tmp_path: Path, names: list[str]
) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.H, credential_env_names=names))

    with pytest.raises(SweAbError, match="MUNA_ACCESS_KEY"):
        validate_swe_ab_directory(tmp_path, _plan())


@pytest.mark.parametrize("provider", ["amazon-bedrock", "openai-codex", "openrouter"])
def test_validator_refuses_a_cell_that_ran_through_another_provider(
    tmp_path: Path, provider: str
) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.H, provider=provider))

    with pytest.raises(SweAbError, match="provider"):
        validate_swe_ab_directory(tmp_path, _plan())


@pytest.mark.parametrize(
    ("field", "label"),
    [("provider_config_sha256", "provider config"), ("omp_config_sha256", "omp config")],
)
def test_validator_refuses_a_stage_that_mixes_config_hashes(
    tmp_path: Path, field: str, label: str
) -> None:
    _complete(tmp_path)
    _write(tmp_path, _cell(SweAbArm.H, **{field: "other"}))

    with pytest.raises(SweAbError, match=f"{label} differs"):
        validate_swe_ab_directory(tmp_path, _plan())


def test_a_quota_block_phase_is_recorded_iff_the_reason_is_quota_block() -> None:
    with pytest.raises(ValueError, match="block phase"):
        SweAbCell.model_validate(
            _cell(status="failed", failure_reason="quota_block", resolved=False)
        )
    with pytest.raises(ValueError, match="block phase"):
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

    with pytest.raises(SweAbError, match="quota_block"):
        validate_swe_ab_directory(tmp_path, _plan())


def _credit_exhausted(arm: SweAbArm = SweAbArm.A0, **overrides: Any) -> dict[str, Any]:
    return _cell(
        arm,
        status="failed",
        failure_reason="credit_exhausted",
        resolved=False,
        quota={"block_phase": "mid_run"},
        **overrides,
    )


def test_validator_refuses_a_credit_exhausted_cell_left_as_the_record(tmp_path: Path) -> None:
    _complete(tmp_path)
    _write(tmp_path, _credit_exhausted(SweAbArm.A0, repetition=2))

    with pytest.raises(SweAbError, match="credit_exhausted"):
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


def test_filed_credit_exhausted_attempts_are_counted_in_cost_and_never_scored(
    tmp_path: Path,
) -> None:
    _complete(tmp_path)
    attempt = SweAbCell.model_validate(_credit_exhausted(SweAbArm.A0, repetition=2))
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


# --- provider error classification ----------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "quota", "credit"),
    [
        ("429 rate limit exceeded", True, False),
        ("429 Too Many Requests", True, False),
        ('{"error": {"type": "rate_limit_error"}}', True, False),
        ('429 {"error": {"code": "model_loading"}}', True, False),
        ('429 {"error": {"code": "model_capacity_exhausted"}}', True, False),
        ("Quota exceeded for this plan", True, False),
        ("402 Payment Required", False, True),
        ("insufficient credits to run this request", False, True),
        ("Insufficient balance", False, True),
        ('{"error": {"code": "credits_required"}}', False, True),
        ("you are out of credits", False, True),
        ("overloaded_error: the service is temporarily overloaded", False, False),
        ("connection reset by peer", False, False),
        ("400 invalid request: messages.0.content", False, False),
        ("", False, False),
        (None, False, False),
    ],
)
def test_quota_and_credit_errors_are_told_from_each_other_and_from_other_errors(
    message: str | None, quota: bool, credit: bool
) -> None:
    assert is_quota_error(message) is quota
    assert is_credit_error(message) is credit


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
