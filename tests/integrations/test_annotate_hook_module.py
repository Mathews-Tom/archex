"""Runs the generated omp/Pi `archex-hook.ts` module under Bun against a real index.

The driver registers the module with a stand-in host that records the
`tool_result` handler, then feeds it recorded-shape events one after another
in a single process, as one agent session would. Each patch is compared with
the event it came from, and the per-call ledger is read back.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from archex.cli.main import cli
from archex.client_setup import build_hook_install_plan, write_hook_install_plan
from archex.project import init_project

pytestmark = pytest.mark.skipif(shutil.which("bun") is None, reason="bun is not installed")

OMP_GREP_TREE = """# services/
## auth.py#8C5F
 15|    def login(self, user: User, password: str) -> str:
*16|        token = hash_password(f"{user.id}:{password}")
 17|        self._sessions[token] = user

# utils.py#171E
*9|def hash_password(password: str) -> str:
"""

_DRIVER = r"""
import archexHook from "./archex-hook.ts";
import { readFileSync } from "node:fs";

const events = JSON.parse(readFileSync(process.env.ARCHEX_TEST_EVENTS!, "utf-8"));
let handler: ((event: unknown, ctx: unknown) => Promise<unknown>) | undefined;
let registrations = 0;
archexHook({
  on(name: string, fn: (event: unknown, ctx: unknown) => Promise<unknown>) {
    registrations += 1;
    if (name === "tool_result") handler = fn;
  },
});
const results = [];
for (const event of events) {
  const before = JSON.stringify(event);
  const patch = await handler!(event, {});
  results.push({
    patch: patch === undefined ? null : patch,
    event_unchanged: JSON.stringify(event) === before,
  });
}
console.log(JSON.stringify({ registrations, results }));
"""


def _index(repo: Path) -> None:
    init_project(repo)
    result = CliRunner().invoke(cli, ["index", str(repo)])
    assert result.exit_code == 0, result.output


@pytest.fixture
def indexed_repo(python_simple_repo: Path) -> Path:
    _index(python_simple_repo)
    return python_simple_repo


class _Run:
    def __init__(self, output: dict[str, Any], ledger: Path) -> None:
        self.registrations = cast("int", output["registrations"])
        self.results = cast("list[dict[str, Any]]", output["results"])
        self._ledger = ledger

    @property
    def patches(self) -> list[dict[str, Any] | None]:
        return [cast("dict[str, Any] | None", r["patch"]) for r in self.results]

    def ledger(self) -> list[dict[str, Any]]:
        if not self._ledger.exists():
            return []
        lines = self._ledger.read_text(encoding="utf-8").splitlines()
        return [cast("dict[str, Any]", json.loads(line)) for line in lines]


def _drive(
    tmp_path: Path, repo: Path, events: list[dict[str, Any]], *, timeout_seconds: str = "30"
) -> _Run:
    module_dir = tmp_path / "module"
    plan = build_hook_install_plan("omp", str(module_dir), scope="project", action="install")
    installed = write_hook_install_plan(plan)
    driver = installed.parent / "driver.ts"
    driver.write_text(_DRIVER, encoding="utf-8")
    events_path = tmp_path / "events.json"
    events_path.write_text(json.dumps(events), encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    result = subprocess.run(
        ["bun", "run", str(driver)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:" + str(Path.home()) + "/.bun/bin",
            "HOME": str(Path.home()),
            "ARCHEX_TEST_EVENTS": str(events_path),
            "ARCHEX_ANNOTATION_LEDGER": str(ledger),
            "ARCHEX_HOOK_DIAGNOSTICS_LOG": str(tmp_path / "diagnostics.log"),
            "ARCHEX_HOOK_TIMEOUT_SECONDS": timeout_seconds,
        },
    )
    assert result.returncode == 0, result.stderr
    return _Run(json.loads(result.stdout.strip().splitlines()[-1]), ledger)


def _grep_event(call_id: str, text: str = OMP_GREP_TREE) -> dict[str, Any]:
    return {
        "toolName": "grep",
        "toolCallId": call_id,
        "input": {"pattern": "hash_password"},
        "content": [{"type": "text", "text": text}],
        "details": {"matchCount": 2},
        "isError": False,
    }


def test_patch_is_the_original_content_plus_one_text_block(
    tmp_path: Path, indexed_repo: Path
) -> None:
    event = _grep_event("call-1")
    original_block = {"type": "text", "text": OMP_GREP_TREE, "hostOnly": {"kept": [1, 2]}}
    event["content"] = [original_block, {"type": "image", "data": "AA==", "mimeType": "x/y"}]
    event["hostEnvelope"] = "unknown event field"

    run = _drive(tmp_path, indexed_repo, [event])

    patch = run.patches[0]
    assert patch is not None
    assert set(patch) == {"content"}
    assert patch["content"][:2] == event["content"]
    assert patch["content"][0]["text"] == OMP_GREP_TREE
    appended = patch["content"][2]
    assert appended["type"] == "text"
    assert appended["text"].startswith("\n\n[archex receipt] index_revision=")
    assert "utils.py::hash_password function L9-10" in appended["text"]
    assert run.results[0]["event_unchanged"] is True
    assert run.registrations == 1


def test_three_calls_in_one_session_are_all_annotated_and_ledgered(
    tmp_path: Path, indexed_repo: Path
) -> None:
    run = _drive(tmp_path, indexed_repo, [_grep_event(f"call-{n}") for n in range(3)])

    assert all(patch is not None for patch in run.patches)
    ledger = run.ledger()
    assert [entry["toolCallId"] for entry in ledger] == ["call-0", "call-1", "call-2"]
    assert all(entry["annotated"] and entry["eligible"] for entry in ledger)
    assert all(entry["units"] == 2 and entry["tokens"] > 0 for entry in ledger)
    assert all(entry["freshness"] == "fresh" and entry["reason"] is None for entry in ledger)


def test_bash_search_is_annotated_and_other_bash_is_not_sent(
    tmp_path: Path, indexed_repo: Path
) -> None:
    search = {
        "toolName": "bash",
        "toolCallId": "b-1",
        "input": {"command": "rg -n hash_password ."},
        "content": [{"type": "text", "text": "utils.py:9:def hash_password(password: str):\n"}],
    }
    other = {
        "toolName": "bash",
        "toolCallId": "b-2",
        "input": {"command": "ls -la"},
        "content": [{"type": "text", "text": "utils.py:9:x\n"}],
    }
    read = {
        "toolName": "read",
        "toolCallId": "r-1",
        "input": {"path": "utils.py"},
        "content": [{"type": "text", "text": "utils.py:9:x\n"}],
    }

    run = _drive(tmp_path, indexed_repo, [search, other, read])

    assert run.patches[0] is not None
    assert run.patches[1] is None
    assert run.patches[2] is None
    ledger = run.ledger()
    assert [entry["toolCallId"] for entry in ledger] == ["b-1", "b-2"]
    assert ledger[1]["eligible"] is False
    assert ledger[1]["reason"] == "not_search_command"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"content": "not an array"}, "malformed_content"),
        ({"input": None}, "malformed_input"),
        ({"content": [{"type": "image", "data": "AA=="}]}, "no_text_content"),
        ({"isError": True}, "tool_error"),
        ({"content": [{"type": "text", "text": "not grep output"}]}, "unrecognized_format"),
    ],
)
def test_malformed_events_produce_no_patch(
    tmp_path: Path, indexed_repo: Path, overrides: dict[str, object], reason: str
) -> None:
    event = {**_grep_event("m-1"), **overrides}

    run = _drive(tmp_path, indexed_repo, [event])

    assert run.patches == [None]
    assert run.ledger()[0]["reason"] == reason
    assert run.ledger()[0]["annotated"] is False


def test_dirty_index_produces_no_patch(tmp_path: Path, indexed_repo: Path) -> None:
    (indexed_repo / "utils.py").write_text("x = 1\n", encoding="utf-8")

    run = _drive(tmp_path, indexed_repo, [_grep_event("d-1")])

    assert run.patches == [None]
    assert run.ledger()[0]["freshness"] == "dirty"
    assert run.ledger()[0]["reason"] == "index_not_fresh"


def test_budget_overrun_kills_the_subprocess_and_produces_no_patch(
    tmp_path: Path, indexed_repo: Path
) -> None:
    run = _drive(tmp_path, indexed_repo, [_grep_event("t-1")], timeout_seconds="0.001")

    assert run.patches == [None]
    assert run.ledger()[0]["reason"] == "timeout"
