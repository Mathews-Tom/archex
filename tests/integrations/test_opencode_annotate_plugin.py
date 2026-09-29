"""Runs the generated OpenCode `archex-hook.ts` plugin under Bun against a real index.

The driver loads the plugin the way OpenCode does (`ArchexHookPlugin({directory})`
returns the hooks table), then feeds the `tool.execute.after` hook recorded-shape
`(input, output)` pairs one after another in a single process, as one agent
session would. The shapes follow OpenCode v1.14.33's own tool sources:
`tool/grep.ts` (`Found N matches`, absolute `path:` headers, `  Line N: text`),
`tool/glob.ts`, and `tool/bash.ts`, wrapped in the `{title, output, metadata}`
object `session/prompt.ts` passes to the hook. Each output is compared with its
pre-call snapshot, and the per-call ledger is read back.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from archex.cli.main import cli
from archex.client_setup import build_hook_install_plan, write_hook_install_plan
from archex.project import init_project

pytestmark = pytest.mark.skipif(shutil.which("bun") is None, reason="bun is not installed")

_DRIVER = r"""
import { ArchexHookPlugin } from "./archex-hook.ts";
import { readFileSync } from "node:fs";

const calls = JSON.parse(readFileSync(process.env.ARCHEX_TEST_CALLS!, "utf-8"));
const hooks: Record<string, (input: unknown, output: unknown) => Promise<unknown>> =
  await (ArchexHookPlugin as any)({ directory: process.env.ARCHEX_TEST_DIRECTORY });
const hook = hooks["tool.execute.after"];
const results = [];
for (const { input, output } of calls) {
  const inputBefore = JSON.stringify(input);
  const outputBefore = JSON.parse(JSON.stringify(output));
  const returned = await hook(input, output);
  results.push({
    returned: returned === undefined ? null : returned,
    input_unchanged: JSON.stringify(input) === inputBefore,
    before: outputBefore,
    after: output,
  });
}
console.log(JSON.stringify({ hook_names: Object.keys(hooks), results }));
"""


def _index(repo: Path) -> None:
    init_project(repo)
    result = CliRunner().invoke(cli, ["index", str(repo)])
    assert result.exit_code == 0, result.output


@pytest.fixture
def indexed_repo(python_simple_repo: Path) -> Path:
    _index(python_simple_repo)
    return python_simple_repo


def _grep_output(repo: Path) -> str:
    """OpenCode's `grep` text as recorded live from 1.14.33: the tool joins one
    element per match with newlines, and each match's own text keeps its
    trailing newline, so every `Line N:` entry is followed by a blank line.
    """
    root = repo.resolve()
    return "\n".join(
        [
            "Found 3 matches",
            f"{root}/services/auth.py:",
            "  Line 5: from utils import hash_password\n",
            '  Line 16:         token = hash_password(f"{user.id}:{password}")\n',
            "",
            f"{root}/utils.py:",
            "  Line 9: def hash_password(password: str) -> str:\n",
        ]
    )


def _call(
    tool: str, call_id: str, args: object, text: object, **extra: object
) -> dict[str, dict[str, Any]]:
    """One recorded `(input, output)` pair, shaped as `session/prompt.ts` passes it."""
    output: dict[str, Any] = {
        "title": "hash_password",
        "output": text,
        "metadata": {"matches": 3, "truncated": False},
    }
    output.update(extra)
    return {
        "input": {"tool": tool, "sessionID": "ses_test", "callID": call_id, "args": args},
        "output": output,
    }


def _grep_call(repo: Path, call_id: str) -> dict[str, dict[str, Any]]:
    return _call("grep", call_id, {"pattern": "hash_password"}, _grep_output(repo))


class _Run:
    def __init__(self, output: dict[str, Any], ledger: Path, diagnostics: Path) -> None:
        self.hook_names = cast("list[str]", output["hook_names"])
        self.results = cast("list[dict[str, Any]]", output["results"])
        self._ledger = ledger
        self._diagnostics = diagnostics

    @property
    def texts(self) -> list[Any]:
        return [r["after"].get("output") for r in self.results]

    def ledger(self) -> list[dict[str, Any]]:
        return _read_jsonl(self._ledger)

    def diagnostics(self) -> list[dict[str, Any]]:
        return _read_jsonl(self._diagnostics)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _drive(
    tmp_path: Path,
    repo: Path,
    calls: list[dict[str, dict[str, Any]]],
    *,
    timeout_seconds: str = "30",
    python_command: Path | None = None,
) -> _Run:
    module_dir = tmp_path / "module"
    plan = build_hook_install_plan("opencode", str(module_dir), scope="project", action="install")
    installed = write_hook_install_plan(plan)
    if python_command is not None:
        source = installed.read_text(encoding="utf-8")
        assert json.dumps(sys.executable) in source
        installed.write_text(
            source.replace(json.dumps(sys.executable), json.dumps(str(python_command))),
            encoding="utf-8",
        )
    driver = installed.parent / "driver.ts"
    driver.write_text(_DRIVER, encoding="utf-8")
    calls_path = tmp_path / "calls.json"
    calls_path.write_text(json.dumps(calls), encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    diagnostics = tmp_path / "diagnostics.log"
    result = subprocess.run(
        ["bun", "run", str(driver)],
        cwd=str(tmp_path),  # not the repo: the plugin must use its `directory`
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:" + str(Path.home()) + "/.bun/bin",
            "HOME": str(Path.home()),
            "ARCHEX_TEST_CALLS": str(calls_path),
            "ARCHEX_TEST_DIRECTORY": str(repo),
            "ARCHEX_ANNOTATION_LEDGER": str(ledger),
            "ARCHEX_HOOK_DIAGNOSTICS_LOG": str(diagnostics),
            "ARCHEX_HOOK_TIMEOUT_SECONDS": timeout_seconds,
        },
    )
    assert result.returncode == 0, result.stderr
    return _Run(json.loads(result.stdout.strip().splitlines()[-1]), ledger, diagnostics)


def test_annotation_is_appended_and_the_original_text_stays_a_byte_identical_prefix(
    tmp_path: Path, indexed_repo: Path
) -> None:
    original = _grep_output(indexed_repo)
    call = _grep_call(indexed_repo, "call_1")
    call["output"]["attachments"] = [{"type": "file", "url": "data:x/y;base64,AA=="}]
    call["output"]["hostOnly"] = {"kept": [1, 2]}

    run = _drive(tmp_path, indexed_repo, [call])

    text = run.texts[0]
    assert text.startswith(original)
    appended = text[len(original) :]
    assert appended.startswith("\n\n[archex receipt] index_revision=")
    assert "utils.py::hash_password function L9-10" in appended
    # Only `output.output` changed; title, metadata, attachments, and fields
    # this plugin does not know about are exactly as OpenCode built them.
    after = dict(run.results[0]["after"])
    before = dict(run.results[0]["before"])
    del after["output"], before["output"]
    assert after == before
    assert run.results[0]["input_unchanged"] is True
    assert run.results[0]["returned"] is None
    assert run.hook_names == ["tool.execute.after"]


def test_three_calls_in_one_session_are_all_annotated_with_three_distinct_ledger_lines(
    tmp_path: Path, indexed_repo: Path
) -> None:
    calls = [_grep_call(indexed_repo, f"call_{n}") for n in range(3)]

    run = _drive(tmp_path, indexed_repo, calls)

    original = _grep_output(indexed_repo)
    assert all(text.startswith(original) and len(text) > len(original) for text in run.texts)
    ledger = run.ledger()
    assert [entry["toolCallId"] for entry in ledger] == ["call_0", "call_1", "call_2"]
    assert {entry["host"] for entry in ledger} == {"opencode"}
    assert {entry["tool"] for entry in ledger} == {"grep"}
    assert all(entry["annotated"] and entry["eligible"] for entry in ledger)
    assert all(entry["units"] == 3 and entry["tokens"] > 0 for entry in ledger)
    assert all(entry["freshness"] == "fresh" and entry["reason"] is None for entry in ledger)
    assert all(isinstance(entry["latency_ms"], int) for entry in ledger)


def test_glob_and_bash_search_are_annotated_and_other_calls_are_not(
    tmp_path: Path, indexed_repo: Path
) -> None:
    root = indexed_repo.resolve()
    glob_text = f"{root}/utils.py\n{root}/main.py"
    bash_text = "utils.py:9:def hash_password(password: str) -> str:\n"
    calls = [
        _call("glob", "g-1", {"pattern": "**/*.py"}, glob_text),
        _call("bash", "b-1", {"command": "rg -n hash_password .", "description": "s"}, bash_text),
        _call("bash", "b-2", {"command": "ls -la", "description": "list"}, "utils.py:9:x\n"),
        _call("read", "r-1", {"filePath": "utils.py"}, "utils.py:9:x\n"),
        # An MCP-routed call reaches the same hook with the raw MCP result,
        # not `{title, output, metadata}`; its id can never match the table.
        {
            "input": {
                "tool": "archex_query_repo",
                "sessionID": "ses_test",
                "callID": "m-1",
                "args": {"query": "hash_password"},
            },
            "output": {"content": [{"type": "text", "text": "utils.py:9:x\n"}]},
        },
    ]

    run = _drive(tmp_path, indexed_repo, calls)

    assert run.texts[0].startswith(glob_text + "\n\n[archex receipt] index_revision=")
    assert "\n[archex] utils.py · units 2" in run.texts[0]
    assert run.texts[1].startswith(bash_text + "\n\n[archex receipt]")
    assert run.texts[2] == "utils.py:9:x\n"
    assert run.texts[3] == "utils.py:9:x\n"
    assert run.results[4]["after"] == run.results[4]["before"]
    ledger = run.ledger()
    assert [entry["toolCallId"] for entry in ledger] == ["g-1", "b-1", "b-2"]
    assert [entry["annotated"] for entry in ledger] == [True, True, False]
    assert ledger[2]["eligible"] is False
    assert ledger[2]["reason"] == "not_search_command"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"args": None}, "malformed_input"),
        ({"args": ["not", "an", "object"]}, "malformed_input"),
        ({"output": None}, "malformed_output"),
        ({"output": {"parts": ["not a string"]}}, "malformed_output"),
        ({"output": "not grep output"}, "no_hits"),
    ],
)
def test_malformed_input_leaves_the_output_untouched(
    tmp_path: Path, indexed_repo: Path, overrides: dict[str, object], reason: str
) -> None:
    call = _grep_call(indexed_repo, "m-1")
    if "args" in overrides:
        call["input"]["args"] = overrides["args"]
    if "output" in overrides:
        call["output"]["output"] = overrides["output"]

    run = _drive(tmp_path, indexed_repo, [call])

    assert run.results[0]["after"] == run.results[0]["before"]
    entry = run.ledger()[0]
    assert entry["reason"] == reason
    assert entry["annotated"] is False
    assert entry["toolCallId"] == "m-1"


def test_dirty_index_adds_nothing(tmp_path: Path, indexed_repo: Path) -> None:
    (indexed_repo / "utils.py").write_text("x = 1\n", encoding="utf-8")

    run = _drive(tmp_path, indexed_repo, [_grep_call(indexed_repo, "d-1")])

    assert run.results[0]["after"] == run.results[0]["before"]
    assert run.ledger()[0]["freshness"] == "dirty"
    assert run.ledger()[0]["reason"] == "index_not_fresh"


def test_budget_overrun_kills_the_subprocess_and_adds_nothing(
    tmp_path: Path, indexed_repo: Path
) -> None:
    """Point the plugin at an interpreter that records its pid and then sleeps
    for 30s: past the budget the plugin must SIGKILL it and move on. The budget
    is 2s, not shorter, because the first exec of a fresh script can take
    hundreds of ms on macOS and the pid must be recorded before the kill.
    """
    pid_file = tmp_path / "child.pid"
    sleeper = tmp_path / "slow-python"
    sleeper.write_text(f'#!/bin/sh\necho $$ > "{pid_file}"\nexec sleep 30\n', encoding="utf-8")
    sleeper.chmod(sleeper.stat().st_mode | stat.S_IXUSR)

    run = _drive(
        tmp_path,
        indexed_repo,
        [_grep_call(indexed_repo, "t-1")],
        timeout_seconds="2",
        python_command=sleeper,
    )

    assert run.results[0]["after"] == run.results[0]["before"]
    assert run.ledger()[0]["reason"] == "timeout"
    assert [entry["kind"] for entry in run.diagnostics()] == ["ts_timeout"]
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_real_annotate_process_past_a_tiny_budget_adds_nothing(
    tmp_path: Path, indexed_repo: Path
) -> None:
    run = _drive(tmp_path, indexed_repo, [_grep_call(indexed_repo, "t-2")], timeout_seconds="0.001")

    assert run.results[0]["after"] == run.results[0]["before"]
    assert run.ledger()[0]["reason"] == "timeout"


def test_missing_interpreter_fails_open(tmp_path: Path, indexed_repo: Path) -> None:
    run = _drive(
        tmp_path,
        indexed_repo,
        [_grep_call(indexed_repo, "s-1")],
        python_command=tmp_path / "no-such-python",
    )

    assert run.results[0]["after"] == run.results[0]["before"]
    assert run.ledger()[0]["reason"] == "spawn_error"
    assert {entry["kind"] for entry in run.diagnostics()} == {"ts_spawn_error"}
