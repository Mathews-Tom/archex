"""Runs the Codex CLI `PostToolUse` annotation hook as Codex does.

Each call spawns `python -m archex.integrations.codex_annotate_hook` with a
payload shaped like Codex 0.153.4's shell `PostToolUse` input
(`codex-rs/hooks/schema/generated/post-tool-use.command.input.schema.json`:
`tool_name` is `Bash`, `tool_input` is `{"command": ...}`, `tool_response` is the
command output as a bare string), against a real index, then reads back stdout,
the exit code, the per-call ledger, and the diagnostics log.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from archex.cli.main import cli
from archex.integrations.codex_annotate_hook import HOOK_MATCHER
from archex.project import init_project

MODULE = "archex.integrations.codex_annotate_hook"

LOGIN_LINE = "[archex] services/auth.py::AuthService.login method L15-18 · importers 1"
HASH_LINE = "[archex] utils.py::hash_password function L9-10 · importers 2"

RG_STDOUT = (
    'services/auth.py:16:        token = hash_password(f"{user.id}:{password}")\n'
    "utils.py:9:def hash_password(password: str) -> str:"
)
# `grep -rn hash_password .` output: paths carry the `./` of the operand.
GREP_DOT_STDOUT = "./" + RG_STDOUT.replace("\n", "\n./")
LEDGER_KEYS = {
    "timestamp",
    "host",
    "toolCallId",
    "tool",
    "eligible",
    "annotated",
    "units",
    "tokens",
    "freshness",
    "reason",
    "latency_ms",
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def indexed_repo(python_simple_repo: Path) -> Path:
    init_project(python_simple_repo)
    result = CliRunner().invoke(cli, ["index", str(python_simple_repo)])
    assert result.exit_code == 0, result.output
    return python_simple_repo


class Hook:
    """One hook environment: a ledger, a diagnostics log, and a way to run the module."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.ledger = tmp_path / "ledger.jsonl"
        self.diagnostics = tmp_path / "diagnostics.log"
        monkeypatch.setenv("ARCHEX_ANNOTATION_LEDGER", str(self.ledger))
        monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(self.diagnostics))
        # The 0.5 s production budget is exercised by its own test; a cold interpreter
        # or a loaded machine must not turn the other tests' searches into silent no-ops.
        monkeypatch.setenv("ARCHEX_HOOK_TIMEOUT_SECONDS", "60")

    def run(self, stdin: str | dict[str, Any]) -> subprocess.CompletedProcess[str]:
        raw = stdin if isinstance(stdin, str) else json.dumps(stdin)
        return subprocess.run(
            [sys.executable, "-m", MODULE],
            input=raw,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def ledger_rows(self) -> list[dict[str, Any]]:
        if not self.ledger.exists():
            return []
        return [json.loads(line) for line in self.ledger.read_text("utf-8").splitlines()]

    def diagnostic_rows(self) -> list[dict[str, Any]]:
        if not self.diagnostics.exists():
            return []
        return [json.loads(line) for line in self.diagnostics.read_text("utf-8").splitlines()]


@pytest.fixture
def hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Hook:
    return Hook(tmp_path, monkeypatch)


def shell_call(
    repo: Path,
    command: str,
    output: Any,
    use_id: str = "call_bash_1",
    tool: str = "Bash",
) -> dict[str, Any]:
    return {
        "session_id": "019e0a4c-6f1b-7c52-9d5e-2a6b8c1f4e07",
        "turn_id": "019e0a4c-70aa-7e13-8b3c-5d1e9f2a7c60",
        "transcript_path": "/tmp/rollout-2026-09-30T10-00-00-019e0a4c.jsonl",
        "cwd": str(repo),
        "hook_event_name": "PostToolUse",
        "model": "gpt-5.5",
        "permission_mode": "default",
        "tool_name": tool,
        "tool_input": {"command": command},
        "tool_response": output,
        "tool_use_id": use_id,
    }


def _unit_lines(context: str) -> list[str]:
    return [line for line in context.split("\n") if line.startswith("[archex] ")]


def _context(completed: subprocess.CompletedProcess[str]) -> str:
    output = json.loads(completed.stdout)
    return cast("str", output["hookSpecificOutput"]["additionalContext"])


# --- output shape ---------------------------------------------------------------


def test_matcher_is_the_codex_shell_tool_only() -> None:
    assert HOOK_MATCHER == "^Bash$"
    assert re.fullmatch(HOOK_MATCHER, "Bash")
    assert not re.fullmatch(HOOK_MATCHER, "apply_patch")


def test_search_result_gets_only_additional_context_beside_the_original(
    indexed_repo: Path, hook: Hook
) -> None:
    """Augment only: the hook prints `hookSpecificOutput.additionalContext` and nothing that
    could block or rewrite the tool result (`decision`, `continue`, `updatedMCPToolOutput`, ...).
    """
    payload = shell_call(indexed_repo, "rg -n hash_password", RG_STDOUT)

    completed = hook.run(payload)

    assert completed.returncode == 0
    assert completed.stderr == ""
    output = json.loads(completed.stdout)
    assert set(output) == {"hookSpecificOutput"}
    assert set(output["hookSpecificOutput"]) == {"hookEventName", "additionalContext"}
    assert output["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    context = output["hookSpecificOutput"]["additionalContext"]
    assert context.startswith("[archex receipt] index_revision=")
    assert _unit_lines(context) == [LOGIN_LINE, HASH_LINE]


@pytest.mark.parametrize(
    ("command", "output"),
    [
        pytest.param("rg -n hash_password", RG_STDOUT, id="rg"),
        pytest.param("grep -rn hash_password .", GREP_DOT_STDOUT, id="grep-dot"),
        pytest.param("cd . && rg -n hash_password", RG_STDOUT, id="cd-then-rg"),
        pytest.param("git grep -n hash_password", RG_STDOUT, id="git-grep"),
    ],
)
def test_each_shell_search_shape_is_annotated(
    indexed_repo: Path, hook: Hook, command: str, output: str
) -> None:
    completed = hook.run(shell_call(indexed_repo, command, output))

    assert completed.returncode == 0
    assert any("utils.py" in line for line in _unit_lines(_context(completed)))


def test_unknown_payload_fields_do_not_break_parsing(indexed_repo: Path, hook: Hook) -> None:
    payload = shell_call(indexed_repo, "rg -n hash_password", RG_STDOUT)
    payload["field_from_a_newer_codex"] = {"nested": [1, 2, 3]}
    payload["agent_id"] = "agent-1"
    payload["agent_type"] = "worker"
    payload["tool_input"]["workdir"] = str(indexed_repo)
    payload["tool_input"]["yield_time_ms"] = 1000

    completed = hook.run(payload)

    assert completed.returncode == 0
    assert _unit_lines(_context(completed)) == [LOGIN_LINE, HASH_LINE]
    assert hook.diagnostic_rows() == []


# --- three calls in one session --------------------------------------------------


def test_three_calls_in_one_session_are_all_annotated_with_three_ledger_lines(
    indexed_repo: Path, hook: Hook
) -> None:
    calls = [
        shell_call(indexed_repo, "rg -n hash_password", RG_STDOUT, "call_01ABC"),
        shell_call(indexed_repo, "grep -rn hash_password .", GREP_DOT_STDOUT, "call_02DEF"),
        shell_call(indexed_repo, "rg -n hash_password", RG_STDOUT, "call_03GHI"),
    ]

    completed = [hook.run(call) for call in calls]

    assert [c.returncode for c in completed] == [0, 0, 0]
    assert all(_unit_lines(_context(c)) for c in completed)
    rows = hook.ledger_rows()
    assert [row["toolCallId"] for row in rows] == ["call_01ABC", "call_02DEF", "call_03GHI"]
    for row in rows:
        assert set(row) == LEDGER_KEYS
        assert row["host"] == "codex"
        assert row["tool"] == "Bash"
        assert row["eligible"] is True
        assert row["annotated"] is True
        assert row["freshness"] == "fresh"
        assert row["reason"] is None
        assert row["units"] >= 1
        assert row["tokens"] > 0
        assert row["latency_ms"] >= 0
    assert hook.diagnostic_rows() == []


# --- fail open -------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   \n",
        "not json",
        "[]",
        '"a string"',
        "{}",
        json.dumps({"tool_name": "Bash"}),
        json.dumps({"tool_name": "Bash", "tool_input": "rg x"}),
        json.dumps({"tool_name": 7, "tool_input": {"command": "rg x"}}),
    ],
)
def test_malformed_payloads_print_nothing_exit_zero_and_log(hook: Hook, raw: str) -> None:
    completed = hook.run(raw)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""
    kinds = [row["kind"] for row in hook.diagnostic_rows()]
    assert kinds == ["malformed_payload"]
    assert hook.ledger_rows() == []


@pytest.mark.parametrize(
    "response",
    [None, ["output"], {"stdout": "rg output"}, 5],
    ids=["null", "list", "claude-shaped-object", "number"],
)
def test_search_call_with_an_unusable_tool_response_adds_nothing_and_is_ledgered(
    indexed_repo: Path, hook: Hook, response: Any
) -> None:
    payload = shell_call(indexed_repo, "rg -n hash_password", RG_STDOUT)
    payload["tool_response"] = response

    completed = hook.run(payload)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert [row["kind"] for row in hook.diagnostic_rows()] == ["malformed_payload"]
    (row,) = hook.ledger_rows()
    assert row["annotated"] is False
    assert row["reason"] == "malformed_tool_response"
    assert row["toolCallId"] == "call_bash_1"


def test_stale_index_produces_nothing(indexed_repo: Path, hook: Hook) -> None:
    (indexed_repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    _git(indexed_repo, "add", "new.py")
    _git(indexed_repo, "commit", "-qm", "advance")

    completed = hook.run(shell_call(indexed_repo, "rg -n hash_password", RG_STDOUT))

    assert completed.returncode == 0
    assert completed.stdout == ""
    (row,) = hook.ledger_rows()
    assert row["annotated"] is False
    assert row["freshness"] == "stale"
    assert row["reason"] == "index_not_fresh"
    assert [d["kind"] for d in hook.diagnostic_rows()] == ["annotate_declined"]


def test_dirty_index_produces_nothing(indexed_repo: Path, hook: Hook) -> None:
    (indexed_repo / "utils.py").write_text("x = 1\n", encoding="utf-8")

    completed = hook.run(shell_call(indexed_repo, "rg -n hash_password", RG_STDOUT))

    assert completed.returncode == 0
    assert completed.stdout == ""
    (row,) = hook.ledger_rows()
    assert row["annotated"] is False
    assert row["freshness"] == "dirty"


def test_search_without_hits_is_ledgered_quietly(indexed_repo: Path, hook: Hook) -> None:
    completed = hook.run(shell_call(indexed_repo, "rg -n nonexistent_symbol", ""))

    assert completed.returncode == 0
    assert completed.stdout == ""
    (row,) = hook.ledger_rows()
    assert row["reason"] == "no_hits"
    assert row["eligible"] is True
    assert hook.diagnostic_rows() == []


_SLOW_ANNOTATE = """
import sys, time
import archex.integrations.codex_annotate_hook as hook
import archex.integrations.post_tool_use_annotate as runner
runner.run_request = lambda request: time.sleep(30)
hook.main()
"""


def test_over_budget_call_exits_zero_with_no_stdout_and_is_ledgered(
    indexed_repo: Path, hook: Hook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARCHEX_HOOK_TIMEOUT_SECONDS", "0.3")
    started = time.monotonic()

    completed = subprocess.run(
        [sys.executable, "-c", _SLOW_ANNOTATE],
        input=json.dumps(shell_call(indexed_repo, "rg -n hash_password", RG_STDOUT, "call_slow")),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert time.monotonic() - started < 15
    assert completed.returncode == 0
    assert completed.stdout == ""
    assert [d["kind"] for d in hook.diagnostic_rows()] == ["annotate_timeout"]
    (row,) = hook.ledger_rows()
    assert row["reason"] == "timeout"
    assert row["toolCallId"] == "call_slow"
    assert row["host"] == "codex"


# --- non-search commands ---------------------------------------------------------

_HEAVY_MODULES = ("archex.index.store", "archex.reporting", "archex.status", "pydantic", "click")

_IMPORT_PROBE = """
import io, json, os, sys
heavy = {heavy!r}
real_exit = os._exit
def fake_exit(code):
    sys.stdout.write("\\n" + json.dumps(sorted(m for m in heavy if m in sys.modules)))
    sys.stdout.flush()  # os._exit skips the flush a buffered pipe needs
    real_exit(code)
os._exit = fake_exit
sys.stdin = io.StringIO({payload!r})
from archex.integrations.codex_annotate_hook import main
main()
"""


@pytest.mark.parametrize("command", ["ls -la", "git status", "npm test 2>&1 | tail -5", "echo rg"])
def test_non_search_command_exits_without_output_ledger_or_heavy_imports(
    hook: Hook, tmp_path: Path, command: str
) -> None:
    payload = json.dumps(shell_call(tmp_path, command, "some output"))

    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE.format(heavy=_HEAVY_MODULES, payload=payload)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "[]", "index/tokenizer/pydantic/click were imported"
    assert hook.ledger_rows() == []
    assert hook.diagnostic_rows() == []


def test_non_search_command_through_the_real_entry_point_prints_nothing(
    indexed_repo: Path, hook: Hook
) -> None:
    completed = hook.run(shell_call(indexed_repo, "ls -la", "total 0"))

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert hook.ledger_rows() == []
    assert hook.diagnostic_rows() == []


@pytest.mark.parametrize("tool", ["apply_patch", "Grep", "Read", "mcp__srv__tool"])
def test_tool_outside_the_shell_tool_is_ignored_silently(
    indexed_repo: Path, hook: Hook, tool: str
) -> None:
    payload = shell_call(indexed_repo, "rg -n hash_password", RG_STDOUT, "call_x", tool=tool)

    completed = hook.run(payload)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert hook.ledger_rows() == []
    assert hook.diagnostic_rows() == []


# --- workdir recovered from the session rollout ---------------------------------------

# In `services/`, `rg -n hash_password` prints paths relative to that directory.
SERVICES_RG_STDOUT = 'auth.py:16:        token = hash_password(f"{user.id}:{password}")'


def _rollout(path: Path, records: list[dict[str, Any]], *, leading_noise: str = "") -> str:
    """Write a Codex rollout file (one JSON record per line) and return its path."""
    path.write_text(
        leading_noise + "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return str(path)


def _function_call(call_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": "2026-09-30T04:04:48.209Z",
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "exec_command",
            "arguments": json.dumps(arguments),
            "call_id": call_id,
        },
    }


def _services_call(repo: Path, transcript: str | None, use_id: str = "call_wd_1") -> dict[str, Any]:
    payload = shell_call(repo, "rg -n hash_password", SERVICES_RG_STDOUT, use_id)
    payload["transcript_path"] = transcript
    return payload


def test_workdir_is_read_back_from_the_rollout_and_resolves_relative_paths(
    indexed_repo: Path, hook: Hook, tmp_path: Path
) -> None:
    rollout = _rollout(
        tmp_path / "rollout.jsonl",
        [
            _function_call("call_other", {"cmd": "ls", "workdir": str(indexed_repo / "elsewhere")}),
            _function_call(
                "call_wd_1",
                {"cmd": "rg -n hash_password", "workdir": str(indexed_repo / "services")},
            ),
            {
                "type": "event_msg",
                "payload": {"type": "exec_command_begin", "call_id": "call_wd_1"},
            },
        ],
        leading_noise='{"cut": "off mid-rec',
    )

    completed = hook.run(_services_call(indexed_repo, rollout))

    assert completed.returncode == 0
    assert _unit_lines(_context(completed)) == [LOGIN_LINE]


def test_a_relative_workdir_in_the_rollout_is_taken_from_the_session_cwd(
    indexed_repo: Path, hook: Hook, tmp_path: Path
) -> None:
    rollout = _rollout(
        tmp_path / "rollout.jsonl",
        [_function_call("call_wd_1", {"cmd": "rg -n hash_password", "workdir": "services"})],
    )

    completed = hook.run(_services_call(indexed_repo, rollout))

    assert _unit_lines(_context(completed)) == [LOGIN_LINE]


@pytest.mark.parametrize("transcript", ["missing", "unrelated-call", "no-workdir", "null"])
def test_without_a_recorded_workdir_the_base_is_the_session_cwd(
    indexed_repo: Path, hook: Hook, tmp_path: Path, transcript: str
) -> None:
    """`auth.py` is not at the repo root, so nothing is annotated: never a guess."""
    rollouts = {
        "missing": str(tmp_path / "absent.jsonl"),
        "unrelated-call": _rollout(
            tmp_path / "a.jsonl", [_function_call("call_zzz", {"workdir": "services"})]
        ),
        "no-workdir": _rollout(
            tmp_path / "b.jsonl", [_function_call("call_wd_1", {"cmd": "rg -n hash_password"})]
        ),
        "null": None,
    }

    completed = hook.run(_services_call(indexed_repo, rollouts[transcript]))

    assert completed.returncode == 0
    assert completed.stdout == ""
    (row,) = hook.ledger_rows()
    assert row["annotated"] is False
    assert row["reason"] == "no_code_units"
    assert [d["kind"] for d in hook.diagnostic_rows()] == ["annotate_declined"]


def test_a_workdir_in_tool_input_wins_over_the_rollout(
    indexed_repo: Path, hook: Hook, tmp_path: Path
) -> None:
    payload = _services_call(indexed_repo, str(tmp_path / "absent.jsonl"))
    payload["tool_input"]["workdir"] = str(indexed_repo / "services")

    completed = hook.run(payload)

    assert _unit_lines(_context(completed)) == [LOGIN_LINE]
