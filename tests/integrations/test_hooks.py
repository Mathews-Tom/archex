"""Tests for the shared symbol-lookup engine and the Codex and Cursor hooks.

Covers `lookup_with_timeout`'s non-blocking contract against a real
project-local index (happy path, missing/stale index, timeout, internal error,
each with its diagnostics log line), then the Codex and Cursor hooks built on
it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from archex.cli.main import cli
from archex.integrations.cursor_hook import (
    _ALWAYS_CONTINUE,  # pyright: ignore[reportPrivateUsage]
    _parse_cursor_payload,  # pyright: ignore[reportPrivateUsage]
    handle_before_submit_prompt,
)
from archex.integrations.diagnostics import DEFAULT_HOOK_TIMEOUT_SECONDS
from archex.integrations.hook import lookup_with_timeout
from archex.integrations.session_hook import handle_session_start
from archex.project import init_project
from archex.session import SessionRecordKind, capture_session_record
from archex.status import inspect_project_status

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def indexed_repo(python_simple_repo: Path) -> Path:
    """A `python_simple_repo` that has been `archex init`'d and freshly indexed."""
    init_project(python_simple_repo)
    runner = CliRunner()
    result = runner.invoke(cli, ["index", str(python_simple_repo)])
    assert result.exit_code == 0, result.output
    return python_simple_repo


@pytest.fixture
def diagnostics_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the hook's diagnostics log to a throwaway file for this test."""
    log_path = tmp_path / "hook-diagnostics.log"
    monkeypatch.setenv("ARCHEX_HOOK_DIAGNOSTICS_LOG", str(log_path))
    return log_path


def _read_diagnostics(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_session_hook_subprocess(
    payload: dict[str, Any], *, cwd: Path, diagnostics_log: Path
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["ARCHEX_HOOK_DIAGNOSTICS_LOG"] = str(diagnostics_log)
    return subprocess.run(
        [sys.executable, "-m", "archex.integrations.session_hook"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=30,
    )


def test_session_start_hook_injects_only_fresh_explicit_context(
    indexed_repo: Path, diagnostics_log: Path
) -> None:
    capture_session_record(
        indexed_repo,
        kind=SessionRecordKind.ACTIVE_TASK,
        content="Repair the parser boundary.",
        creator="test",
    )

    for source in ("startup", "resume"):
        completed = _run_session_hook_subprocess(
            {"source": source, "cwd": str(indexed_repo)},
            cwd=indexed_repo,
            diagnostics_log=diagnostics_log,
        )
        assert completed.returncode == 0, completed.stderr
        output = json.loads(completed.stdout)
        context = output["hookSpecificOutput"]["additionalContext"]
        assert "Repair the parser boundary." in context
        assert "Index revision:" in context

    assert handle_session_start({"source": "clear", "cwd": str(indexed_repo)}) is None
    assert handle_session_start({"source": [], "cwd": str(indexed_repo)}) is None

    (indexed_repo / "main.py").write_text("changed = True\n", encoding="utf-8")
    assert handle_session_start({"source": "resume", "cwd": str(indexed_repo)}) is None


# ---------------------------------------------------------------------------
# `lookup_with_timeout`: the shared engine behind the Cursor and Codex hooks
#
# The retired pattern-search `PreToolUse` process that used to wrap it is
# gone; these pin the engine's own non-blocking contract directly.
# ---------------------------------------------------------------------------


def test_lookup_returns_receipt_stamped_context(indexed_repo: Path) -> None:
    context = lookup_with_timeout(str(indexed_repo), "AuthService")

    assert context is not None
    assert "AuthService" in context
    assert re.search(r"index_revision=\S+", context)
    assert re.search(r"generated_at=\S+", context)


def test_lookup_without_a_match_returns_none(indexed_repo: Path) -> None:
    assert lookup_with_timeout(str(indexed_repo), "NoSuchSymbolAnywhere") is None


def test_missing_index_degrades_silently_and_logs_diagnostic(
    python_simple_repo: Path, diagnostics_log: Path
) -> None:
    """`python_simple_repo` is git-init'd but never `archex init`'d/indexed."""
    assert lookup_with_timeout(str(python_simple_repo), "AuthService") is None

    entry = _read_diagnostics(diagnostics_log)[-1]
    assert entry["kind"] in {"index_not_fresh", "status_error"}
    assert entry.get("detail")
    assert entry.get("cwd")


def test_stale_index_degrades_silently_and_logs_diagnostic(
    indexed_repo: Path, diagnostics_log: Path
) -> None:
    (indexed_repo / "utils.py").write_text("def dirty_symbol(): return 1\n", encoding="utf-8")
    status = inspect_project_status(indexed_repo)
    assert status.state in {"dirty", "stale"}

    assert lookup_with_timeout(str(indexed_repo), "AuthService") is None

    entry = _read_diagnostics(diagnostics_log)[-1]
    assert entry["kind"] == "index_not_fresh"
    assert f"state={status.state}" in entry["detail"]


def test_lookup_timeout_degrades_silently_and_logs_diagnostic(
    indexed_repo: Path, diagnostics_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 0.1ms budget reliably beats even a real SQLite-backed lookup — verified
    deterministic across repeated local runs (no thread/sleep mocking needed).
    """
    monkeypatch.setenv("ARCHEX_HOOK_TIMEOUT_SECONDS", "0.0001")

    assert lookup_with_timeout(str(indexed_repo), "AuthService") is None

    assert _read_diagnostics(diagnostics_log)[-1]["kind"] == "timeout"


def test_internal_error_opening_store_degrades_silently(
    indexed_repo: Path, diagnostics_log: Path
) -> None:
    assert inspect_project_status(indexed_repo).state == "fresh"

    with patch("archex.integrations.hook.IndexStore", side_effect=RuntimeError("boom")):
        assert lookup_with_timeout(str(indexed_repo), "AuthService") is None

    assert _read_diagnostics(diagnostics_log)[-1]["kind"] == "lookup_error"


def test_default_timeout_budget_holds_on_realistic_fixture(
    monorepo_simple_repo: Path,
) -> None:
    """The default budget is generous next to a real lookup on a small index.

    Measured locally at ~44-55ms per call, a tenth of the 500ms budget; the
    `* 0.5` threshold keeps a wide margin for a loaded CI runner while still
    failing loudly if the lookup path regressed toward the actual timeout.
    """
    init_project(monorepo_simple_repo)
    result = CliRunner().invoke(cli, ["index", str(monorepo_simple_repo)])
    assert result.exit_code == 0, result.output

    start = time.perf_counter()
    context = lookup_with_timeout(str(monorepo_simple_repo), "initialize")
    elapsed = time.perf_counter() - start

    # A `None` here would mean a timeout no-op, and the timing check below
    # would be measuring nothing.
    assert context is not None, "lookup timed out against a small, freshly-indexed fixture"
    assert "initialize" in context
    budget = DEFAULT_HOOK_TIMEOUT_SECONDS
    assert elapsed < budget * 0.5, (
        f"lookup took {elapsed * 1000:.1f}ms, more than half of the {budget * 1000:.0f}ms "
        "default timeout budget on a small fixture"
    )


# ---------------------------------------------------------------------------
# Cursor `beforeSubmitPrompt` diagnostics-only hook (M23)
#
# Confirmation spike (read against Cursor's own official docs directly --
# `https://cursor.com/docs/hooks` and `https://cursor.com/docs/reference/
# third-party-hooks`, not secondary sources): `beforeSubmitPrompt`'s output
# schema is `{"continue": bool, "user_message": str | None}` ONLY -- unlike
# `sessionStart`/`postToolUse`, it has no `additional_context`/
# `additionalContext` output field, nested or flat, and Cursor's documented
# Claude Code `UserPromptSubmit` -> `beforeSubmitPrompt` compatibility
# mapping does not add one either. Cursor also has no Grep/Glob-equivalent
# tool-call hook at all. `archex.integrations.cursor_hook` therefore ships
# the plan's own diagnostics-only fallback: it always returns
# `{"continue": true}` and logs what an archex lookup for the submitted
# prompt would have surfaced, instead of injecting it.
# ---------------------------------------------------------------------------


def _run_cursor_hook_subprocess(
    raw_stdin: str, *, cwd: Path, diagnostics_log: Path
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["ARCHEX_HOOK_DIAGNOSTICS_LOG"] = str(diagnostics_log)
    return subprocess.run(
        [sys.executable, "-m", "archex.integrations.cursor_hook"],
        input=raw_stdin,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=30,
    )


def test_cursor_prompt_with_matches_logs_withheld_diagnostic_not_injection(
    indexed_repo: Path, diagnostics_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(indexed_repo)
    payload: dict[str, Any] = {
        "prompt": "How does AuthService handle login?",
        "attachments": [],
    }

    result = handle_before_submit_prompt(payload)

    assert result is None  # never returns context injection, ever
    entries = _read_diagnostics(diagnostics_log)
    assert entries, "expected a diagnostic line for a prompt with archex matches"
    entry = entries[-1]
    assert entry["kind"] == "cursor_context_injection_unsupported"
    assert "AuthService" in entry["detail"]
    assert "no context-injection output field" in entry["detail"]


def test_cursor_prompt_without_identifier_tokens_is_noop(
    indexed_repo: Path, diagnostics_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(indexed_repo)
    payload: dict[str, Any] = {"prompt": "?? .. --", "attachments": []}

    assert handle_before_submit_prompt(payload) is None
    assert _read_diagnostics(diagnostics_log) == []


@pytest.mark.parametrize("payload", [{}, {"prompt": None}, {"prompt": 42}, {"prompt": "   "}])
def test_cursor_missing_or_non_string_prompt_short_circuits_before_lookup(
    payload: dict[str, Any], diagnostics_log: Path
) -> None:
    lookup_mock = MagicMock(side_effect=AssertionError("must not be called"))

    with patch("archex.integrations.cursor_hook.lookup_with_timeout", lookup_mock):
        assert handle_before_submit_prompt(payload) is None

    lookup_mock.assert_not_called()
    assert _read_diagnostics(diagnostics_log) == []


def test_cursor_missing_index_degrades_silently_and_logs_diagnostic(
    python_simple_repo: Path, diagnostics_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`python_simple_repo` is git-init'd but never `archex init`'d/indexed."""
    monkeypatch.chdir(python_simple_repo)
    payload: dict[str, Any] = {"prompt": "Where is compute_delta defined?"}

    assert handle_before_submit_prompt(payload) is None

    entries = _read_diagnostics(diagnostics_log)
    assert entries
    assert entries[-1]["kind"] in {"status_error", "index_not_fresh"}


@pytest.mark.parametrize("raw", ["not json", "[]", "42", '"a string"'])
def test_cursor_parse_payload_rejects_non_object_input_and_logs_diagnostic(
    raw: str, diagnostics_log: Path
) -> None:
    assert _parse_cursor_payload(raw) is None

    entries = _read_diagnostics(diagnostics_log)
    assert entries[-1]["kind"] == "cursor_malformed_payload"


def test_cursor_internal_error_degrades_silently_and_logs_diagnostic(
    indexed_repo: Path, diagnostics_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(indexed_repo)
    payload: dict[str, Any] = {"prompt": "How does AuthService handle login?"}

    with patch(
        "archex.integrations.cursor_hook.lookup_with_timeout",
        side_effect=RuntimeError("boom"),
    ):
        assert handle_before_submit_prompt(payload) is None

    entries = _read_diagnostics(diagnostics_log)
    assert entries[-1]["kind"] == "cursor_internal_error"


def test_cursor_subprocess_prompt_with_matches_exits_zero_with_continue_true(
    indexed_repo: Path, tmp_path: Path
) -> None:
    diagnostics_log = tmp_path / "cursor-subprocess-diag.log"
    stdin_payload = json.dumps({"prompt": "How does AuthService handle login?"})

    result = _run_cursor_hook_subprocess(
        stdin_payload, cwd=indexed_repo, diagnostics_log=diagnostics_log
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"continue": True}
    entries = _read_diagnostics(diagnostics_log)
    assert entries[-1]["kind"] == "cursor_context_injection_unsupported"


def test_cursor_subprocess_garbage_stdin_exits_zero_with_continue_true(tmp_path: Path) -> None:
    diagnostics_log = tmp_path / "cursor-subprocess-diag.log"

    result = _run_cursor_hook_subprocess("not json", cwd=tmp_path, diagnostics_log=diagnostics_log)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"continue": True}


def test_cursor_subprocess_empty_stdin_exits_zero_with_continue_true(tmp_path: Path) -> None:
    diagnostics_log = tmp_path / "cursor-subprocess-diag.log"

    result = _run_cursor_hook_subprocess("", cwd=tmp_path, diagnostics_log=diagnostics_log)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"continue": True}


def test_cursor_always_continue_constant_never_carries_context_or_blocks() -> None:
    assert _ALWAYS_CONTINUE == {"continue": True}
