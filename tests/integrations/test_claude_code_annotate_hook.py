"""Runs the Claude Code `PostToolUse` annotation hook as Claude Code does.

Each call spawns `python -m archex.integrations.claude_code_annotate_hook` with a
payload shaped like the ones recorded at a live Claude Code 2.1.285 hook, against
a real index, then reads back stdout, the exit code, the per-call ledger, and the
diagnostics log.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from archex.cli.main import cli
from archex.integrations.claude_code_annotate_hook import HOOK_MATCHER
from archex.project import init_project

MODULE = "archex.integrations.claude_code_annotate_hook"

LOGIN_LINE = "[archex] services/auth.py::AuthService.login method L15-18 · importers 1"
HASH_LINE = "[archex] utils.py::hash_password function L9-10 · importers 2"

# Claude Code Bash `rg -n hash_password` stdout, and Grep content-mode `content`.
RG_STDOUT = (
    'services/auth.py:16:        token = hash_password(f"{user.id}:{password}")\n'
    "utils.py:9:def hash_password(password: str) -> str:"
)
# Claude Code Bash `grep -rn hash_password .` stdout: paths carry the `./` of the operand.
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


def _payload(
    repo: Path, tool: str, tool_input: dict[str, Any], response: Any, use_id: str
) -> dict[str, Any]:
    return {
        "session_id": "a825de89-868b-4cb9-b47b-c6870847f0a4",
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": str(repo),
        "prompt_id": "6695d8a2-a5a0-4d99-bb95-f051d1440d35",
        "permission_mode": "bypassPermissions",
        "hook_event_name": "PostToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
        "tool_response": response,
        "tool_use_id": use_id,
        "duration_ms": 15,
    }


def bash_call(
    repo: Path, command: str, stdout: str, use_id: str = "toolu_bash_1"
) -> dict[str, Any]:
    response = {
        "stdout": stdout,
        "stderr": "",
        "interrupted": False,
        "isImage": False,
        "noOutputExpected": False,
    }
    return _payload(repo, "Bash", {"command": command, "description": "search"}, response, use_id)


def grep_content_call(repo: Path, use_id: str = "toolu_grep_1") -> dict[str, Any]:
    response: dict[str, Any] = {
        "mode": "content",
        "numFiles": 0,
        "filenames": [],
        "content": RG_STDOUT,
        "numLines": 2,
        "totalLines": 2,
    }
    tool_input = {"pattern": "hash_password", "output_mode": "content", "-n": True}
    return _payload(repo, "Grep", tool_input, response, use_id)


def grep_files_call(repo: Path, use_id: str = "toolu_grep_2") -> dict[str, Any]:
    response = {
        "mode": "files_with_matches",
        "filenames": ["utils.py", "services/auth.py"],
        "numFiles": 2,
        "totalFiles": 2,
    }
    return _payload(repo, "Grep", {"pattern": "hash_password"}, response, use_id)


def glob_call(repo: Path, use_id: str = "toolu_glob_1") -> dict[str, Any]:
    response = {
        "filenames": ["utils.py", "services/auth.py"],
        "durationMs": 11,
        "numFiles": 2,
        "truncated": False,
        "totalMatches": 2,
        "countIsComplete": True,
    }
    return _payload(repo, "Glob", {"pattern": "**/*.py"}, response, use_id)


def _unit_lines(context: str) -> list[str]:
    return [line for line in context.split("\n") if line.startswith("[archex] ")]


def _context(completed: subprocess.CompletedProcess[str]) -> str:
    output = json.loads(completed.stdout)
    return cast("str", output["hookSpecificOutput"]["additionalContext"])


# --- output shape ---------------------------------------------------------------


def test_matcher_is_the_claude_search_tools() -> None:
    assert HOOK_MATCHER == "Bash|Grep|Glob"


def test_search_result_gets_only_additional_context_beside_the_original(
    indexed_repo: Path, hook: Hook
) -> None:
    """Augment only: the hook prints `additionalContext` and nothing that could replace
    the tool result (`updatedToolOutput`, `decision`, `continue`, ...).
    """
    payload = bash_call(indexed_repo, "rg -n hash_password", RG_STDOUT)

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


def _bash_rg_call(repo: Path) -> dict[str, Any]:
    return bash_call(repo, "rg -n hash_password", RG_STDOUT)


def _bash_grep_dot_call(repo: Path) -> dict[str, Any]:
    return bash_call(repo, "grep -rn hash_password .", GREP_DOT_STDOUT)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(_bash_rg_call, id="bash-rg"),
        pytest.param(_bash_grep_dot_call, id="bash-grep-dot"),
        pytest.param(grep_content_call, id="grep-content"),
        pytest.param(grep_files_call, id="grep-files-with-matches"),
        pytest.param(glob_call, id="glob"),
    ],
)
def test_each_search_tool_shape_is_annotated(
    indexed_repo: Path, hook: Hook, call: Callable[[Path], dict[str, Any]]
) -> None:
    completed = hook.run(call(indexed_repo))

    assert completed.returncode == 0
    units = _unit_lines(_context(completed))
    assert units and all(line.startswith("[archex] ") for line in units)
    assert any("utils.py" in line for line in units)


def test_unknown_payload_fields_do_not_break_parsing(indexed_repo: Path, hook: Hook) -> None:
    payload = bash_call(indexed_repo, "rg -n hash_password", RG_STDOUT)
    payload["field_from_a_newer_claude_code"] = {"nested": [1, 2, 3]}
    payload["tool_input"]["run_in_background"] = False
    payload["tool_response"]["backgroundTaskId"] = None
    payload["tool_response"]["persistedOutputPath"] = None

    completed = hook.run(payload)

    assert completed.returncode == 0
    assert _unit_lines(_context(completed)) == [LOGIN_LINE, HASH_LINE]
    assert hook.diagnostic_rows() == []


# --- three calls in one session --------------------------------------------------


def test_three_calls_in_one_session_are_all_annotated_with_three_ledger_lines(
    indexed_repo: Path, hook: Hook
) -> None:
    calls = [
        bash_call(indexed_repo, "rg -n hash_password", RG_STDOUT, "toolu_01ABC"),
        grep_content_call(indexed_repo, "toolu_02DEF"),
        glob_call(indexed_repo, "toolu_03GHI"),
    ]

    completed = [hook.run(call) for call in calls]

    assert [c.returncode for c in completed] == [0, 0, 0]
    assert all(_unit_lines(_context(c)) for c in completed)
    rows = hook.ledger_rows()
    assert [row["toolCallId"] for row in rows] == ["toolu_01ABC", "toolu_02DEF", "toolu_03GHI"]
    assert [row["tool"] for row in rows] == ["Bash", "Grep", "Glob"]
    for row in rows:
        assert set(row) == LEDGER_KEYS
        assert row["host"] == "claude-code"
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
    ["a bare string", None, ["stdout"], {"stderr": "", "interrupted": False}, {"stdout": 5}],
)
def test_search_call_with_an_unusable_tool_response_adds_nothing_and_is_ledgered(
    indexed_repo: Path, hook: Hook, response: Any
) -> None:
    payload = bash_call(indexed_repo, "rg -n hash_password", RG_STDOUT)
    payload["tool_response"] = response

    completed = hook.run(payload)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert [row["kind"] for row in hook.diagnostic_rows()] == ["malformed_payload"]
    rows = hook.ledger_rows()
    assert len(rows) == 1
    assert rows[0]["annotated"] is False
    assert rows[0]["reason"] == "malformed_tool_response"
    assert rows[0]["toolCallId"] == "toolu_bash_1"


def test_stale_index_produces_nothing(indexed_repo: Path, hook: Hook) -> None:
    (indexed_repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    _git(indexed_repo, "add", "new.py")
    _git(indexed_repo, "commit", "-qm", "advance")

    completed = hook.run(bash_call(indexed_repo, "rg -n hash_password", RG_STDOUT))

    assert completed.returncode == 0
    assert completed.stdout == ""
    (row,) = hook.ledger_rows()
    assert row["annotated"] is False
    assert row["freshness"] == "stale"
    assert row["reason"] == "index_not_fresh"
    assert [d["kind"] for d in hook.diagnostic_rows()] == ["annotate_declined"]


def test_dirty_index_produces_nothing(indexed_repo: Path, hook: Hook) -> None:
    (indexed_repo / "utils.py").write_text("x = 1\n", encoding="utf-8")

    completed = hook.run(grep_content_call(indexed_repo))

    assert completed.returncode == 0
    assert completed.stdout == ""
    (row,) = hook.ledger_rows()
    assert row["annotated"] is False
    assert row["freshness"] == "dirty"


def test_search_without_hits_is_ledgered_quietly(indexed_repo: Path, hook: Hook) -> None:
    completed = hook.run(bash_call(indexed_repo, "rg -n nonexistent_symbol", ""))

    assert completed.returncode == 0
    assert completed.stdout == ""
    (row,) = hook.ledger_rows()
    assert row["reason"] == "no_hits"
    assert row["eligible"] is True
    assert hook.diagnostic_rows() == []


_SLOW_ANNOTATE = """
import sys, time
import archex.integrations.claude_code_annotate_hook as hook
hook.run_request = lambda request: time.sleep(30)
hook.main()
"""


def test_over_budget_call_exits_zero_with_no_stdout_and_is_ledgered(
    indexed_repo: Path, hook: Hook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ARCHEX_HOOK_TIMEOUT_SECONDS", "0.3")
    started = time.monotonic()

    completed = subprocess.run(
        [sys.executable, "-c", _SLOW_ANNOTATE],
        input=json.dumps(bash_call(indexed_repo, "rg -n hash_password", RG_STDOUT, "toolu_slow")),
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
    assert row["toolCallId"] == "toolu_slow"
    assert row["annotated"] is False


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
from archex.integrations.claude_code_annotate_hook import main
main()
"""


@pytest.mark.parametrize("command", ["ls -la", "git status", "npm test 2>&1 | tail -5", "echo rg"])
def test_non_search_bash_exits_without_output_ledger_or_heavy_imports(
    hook: Hook, tmp_path: Path, command: str
) -> None:
    payload = json.dumps(bash_call(tmp_path, command, "some output"))

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


def test_non_search_bash_through_the_real_entry_point_prints_nothing(
    indexed_repo: Path, hook: Hook
) -> None:
    completed = hook.run(bash_call(indexed_repo, "ls -la", "total 0"))

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert hook.ledger_rows() == []
    assert hook.diagnostic_rows() == []


def test_tool_outside_the_matcher_is_ignored_silently(indexed_repo: Path, hook: Hook) -> None:
    payload = _payload(
        indexed_repo, "Read", {"file_path": "utils.py"}, {"file": {"content": "x"}}, "toolu_r"
    )

    completed = hook.run(payload)

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert hook.ledger_rows() == []
    assert hook.diagnostic_rows() == []
