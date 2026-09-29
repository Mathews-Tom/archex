"""Tests for the SWE A/B harness core: channels, compounding, parsing, cells, validator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from archex.benchmark.swe_ab import (
    BASE_TOOLS,
    CHANNELS,
    ISOLATION_FLAGS,
    CellKey,
    SweAbArm,
    SweAbCell,
    SweAbError,
    SweAbPlan,
    compound,
    localize,
    observations,
    omp_argv,
    omp_channel,
    parse_omp_session,
    provider_endpoint_overridden,
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


def _cell(arm: SweAbArm = SweAbArm.A0, **overrides: Any) -> dict[str, Any]:
    zero = dict.fromkeys(CHANNELS, 0)
    cell: dict[str, Any] = {
        "task_id": "t1",
        "repo": "org/repo",
        "model": "m",
        "arm": arm.value,
        "repetition": 1,
        "status": "ok",
        "omp_version": "18.4.2",
        "archex_version": "0.33.0" if arm.archex_installed else None,
        "archex_wheel_sha256": "w" if arm.archex_installed else None,
        "hook_module_sha256": "h" if arm.hook else None,
        "cli_guide_sha256": "g" if arm.cli else None,
        "image": "img",
        "tool_fingerprint": tool_fingerprint(BASE_TOOLS),
        "provider": "amazon-bedrock",
        "provider_endpoint_overridden": False,
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
        "models": ["m"],
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
        CellKey("t1", "m", SweAbArm.A0, 1),
        CellKey("t1", "m", SweAbArm.A0, 2),
        CellKey("t1", "m", SweAbArm.H, 1),
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
