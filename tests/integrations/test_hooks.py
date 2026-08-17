"""Tests for the Claude Code PreToolUse hook (M19).

Covers the hook's non-blocking contract end to end: the happy path against a
real project-local index, every degradation branch (missing/stale index,
malformed payload, timeout, internal error) with its diagnostics log line, the
tool-name filter that keeps `Read` untouched, the `_extract_query` glob/grep
token extraction, and the `python -m archex.integrations.hook` subprocess
exit-0 contract.
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
from archex.integrations.codex_hook import (
    _SEARCH_COMMAND_RE,  # pyright: ignore[reportPrivateUsage]
    DIAGNOSED_TOOL_NAME,
    _parse_codex_payload,  # pyright: ignore[reportPrivateUsage]
    handle_codex_pre_tool_use,
)
from archex.integrations.codex_hook import (
    HOOK_MATCHER as CODEX_HOOK_MATCHER,
)
from archex.integrations.cursor_hook import (
    _ALWAYS_CONTINUE,  # pyright: ignore[reportPrivateUsage]
    _parse_cursor_payload,  # pyright: ignore[reportPrivateUsage]
    handle_before_submit_prompt,
)
from archex.integrations.hook import (
    AUGMENTED_TOOLS,
    DEFAULT_HOOK_TIMEOUT_SECONDS,
    HOOK_MATCHER,
    _extract_query,  # pyright: ignore[reportPrivateUsage]
    _parse_payload,  # pyright: ignore[reportPrivateUsage]
    handle_pre_tool_use,
)
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


def _run_hook_subprocess(
    raw_stdin: str, *, cwd: Path, diagnostics_log: Path
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["ARCHEX_HOOK_DIAGNOSTICS_LOG"] = str(diagnostics_log)
    return subprocess.run(
        [sys.executable, "-m", "archex.integrations.hook"],
        input=raw_stdin,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=30,
    )


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
# Happy path (Grep / Glob) against a real project-local index
# ---------------------------------------------------------------------------


def test_grep_happy_path_returns_receipt_stamped_context(indexed_repo: Path) -> None:
    payload: dict[str, Any] = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "AuthService"},
        "cwd": str(indexed_repo),
    }

    result = handle_pre_tool_use(payload)

    assert result is not None
    hook_output = result["hookSpecificOutput"]
    assert hook_output["hookEventName"] == "PreToolUse"
    context = hook_output["additionalContext"]
    assert "AuthService" in context
    assert re.search(r"index_revision=\S+", context)
    assert re.search(r"generated_at=\S+", context)


def test_glob_happy_path_degrades_gracefully_without_crashing(indexed_repo: Path) -> None:
    """Glob patterns route through `_extract_query`'s identifier-token extraction
    rather than a literal search term. For "**/*_service.py" the only token is
    "_service", which has no exact FTS-token match in this fixture (FTS5's
    unicode61 tokenizer keeps "AuthService" as a single token and splits
    "services/auth.py" into "services", not "service") — so the lookup finds
    nothing. Verified empirically this is a deterministic `None`, not a race;
    the disjunction below still accepts a match in case indexing behavior ever
    changes, per the "both outcomes acceptable" contract for Glob.
    """
    payload: dict[str, Any] = {
        "tool_name": "Glob",
        "tool_input": {"pattern": "**/*_service.py"},
        "cwd": str(indexed_repo),
    }

    result = handle_pre_tool_use(payload)

    assert result is None or (
        result["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
        and isinstance(result["hookSpecificOutput"]["additionalContext"], str)
        and result["hookSpecificOutput"]["additionalContext"]
    )


def test_glob_pattern_without_identifier_tokens_returns_none(indexed_repo: Path) -> None:
    payload: dict[str, Any] = {
        "tool_name": "Glob",
        "tool_input": {"pattern": "**/*.py"},
        "cwd": str(indexed_repo),
    }

    assert handle_pre_tool_use(payload) is None


# ---------------------------------------------------------------------------
# M20 client-shim payload translation (oh-my-pi / Pi)
#
# The TS hook module installed for omp/pi (`archex.client_setup`) translates
# each host's native grep/glob-equivalent tool_result event into exactly this
# subprocess's existing Grep/Glob contract before ever invoking it -- this
# module gains no client-specific branches. These tests exercise that
# contract with the payload shapes the shim actually sends for each host.
# ---------------------------------------------------------------------------


def test_omp_grep_shim_payload_returns_receipt_stamped_context(indexed_repo: Path) -> None:
    """oh-my-pi's `grep` tool already carries its query in a `pattern`-named
    field, so the shim's translation is an identity mapping onto Grep.
    """
    payload: dict[str, Any] = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "AuthService"},
        "cwd": str(indexed_repo),
    }

    result = handle_pre_tool_use(payload)

    assert result is not None
    assert "AuthService" in result["hookSpecificOutput"]["additionalContext"]


def test_omp_glob_shim_translates_path_field_to_subprocess_glob_payload(
    indexed_repo: Path,
) -> None:
    """oh-my-pi has no `pattern` field on its `glob` tool -- the query lives in
    `path` (e.g. `{"path": "**/*_service.py"}`). The shim reads that field and
    sends it to the subprocess as `tool_input.pattern`, exactly like Claude
    Code's own Glob tool would -- proving the translation, not a client-aware
    branch in `archex.integrations.hook`, is what bridges the field-name gap.
    """
    payload: dict[str, Any] = {
        "tool_name": "Glob",
        "tool_input": {"pattern": "**/*_service.py"},
        "cwd": str(indexed_repo),
    }

    result = handle_pre_tool_use(payload)

    assert result is None or (
        isinstance(result["hookSpecificOutput"]["additionalContext"], str)
        and result["hookSpecificOutput"]["additionalContext"]
    )


def test_pi_find_shim_translates_pattern_field_to_subprocess_glob_payload(
    indexed_repo: Path,
) -> None:
    """Pi has no `glob` tool at all -- its glob-equivalent is `find`, whose
    query already lives in a field named `pattern`. The shim maps `find` to
    the subprocess's `Glob` tool_name, carrying the `pattern` value through
    unchanged.
    """
    payload: dict[str, Any] = {
        "tool_name": "Glob",
        "tool_input": {"pattern": "AuthService"},
        "cwd": str(indexed_repo),
    }

    result = handle_pre_tool_use(payload)

    assert result is not None
    assert "AuthService" in result["hookSpecificOutput"]["additionalContext"]


def test_hook_py_stays_client_agnostic_after_m20(indexed_repo: Path) -> None:
    """M20 reuses this subprocess contract unmodified (DEVELOPMENT_PLAN.md
    Section G assumption): it still recognizes only the capitalized Claude
    Code tool names, never a native omp/Pi tool name directly. Translation is
    entirely the TS shim's responsibility (see
    `tests/cli/test_install_client_hooks.py`'s `test_omp_ts_hook_module_*`
    tests), not something smuggled into this module.
    """
    assert {"Grep", "Glob"} == AUGMENTED_TOOLS
    for native_tool_name in ("grep", "glob", "find"):
        payload: dict[str, Any] = {
            "tool_name": native_tool_name,
            "tool_input": {"pattern": "AuthService"},
            "cwd": str(indexed_repo),
        }
        assert handle_pre_tool_use(payload) is None


# ---------------------------------------------------------------------------
# M22 client-shim payload translation (OpenCode)
#
# Unlike omp's `glob` (field `path`) or Pi's `find` (a different tool name),
# OpenCode's native `grep` and `glob` tools both already carry their query in
# a field named `pattern` -- confirmed against the installed `opencode-ai`
# 1.14.33's own tool definitions. The OpenCode plugin's translation is
# therefore an identity mapping onto this subprocess's own Grep/Glob
# contract, exercised here with the exact payload shape the plugin sends.
# ---------------------------------------------------------------------------


def test_opencode_grep_shim_payload_returns_receipt_stamped_context(indexed_repo: Path) -> None:
    payload: dict[str, Any] = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "AuthService"},
        "cwd": str(indexed_repo),
    }

    result = handle_pre_tool_use(payload)

    assert result is not None
    assert "AuthService" in result["hookSpecificOutput"]["additionalContext"]


def test_opencode_glob_shim_payload_returns_receipt_stamped_context(indexed_repo: Path) -> None:
    """OpenCode's `glob` tool carries its query in a field already named
    `pattern`, so the plugin's translation onto the subprocess's Glob
    contract is an identity mapping -- exercised here with a glob-shaped
    pattern, mirroring the same lenient assertion M20's equivalent omp test
    uses, since whether a specific glob-derived identifier token happens to
    match any indexed symbol is a BM25/corpus concern, not a translation one.
    """
    payload: dict[str, Any] = {
        "tool_name": "Glob",
        "tool_input": {"pattern": "**/*_service.py"},
        "cwd": str(indexed_repo),
    }

    result = handle_pre_tool_use(payload)

    assert result is None or (
        isinstance(result["hookSpecificOutput"]["additionalContext"], str)
        and result["hookSpecificOutput"]["additionalContext"]
    )


def test_hook_py_stays_client_agnostic_after_m22(indexed_repo: Path) -> None:
    """M22 reuses this subprocess contract unmodified, same as M20/M21: it
    still recognizes only the capitalized Claude Code tool names, never
    OpenCode's native lowercase `grep`/`glob` (already proven by M20's
    `test_hook_py_stays_client_agnostic_after_m20`) nor an MCP-routed,
    `{server}_{tool}`-prefixed id such as an archex MCP tool call would use.
    """
    assert {"Grep", "Glob"} == AUGMENTED_TOOLS
    for native_tool_name in ("grep", "glob", "archex_query_repo", "archex_scout_repo"):
        payload: dict[str, Any] = {
            "tool_name": native_tool_name,
            "tool_input": {"pattern": "AuthService"},
            "cwd": str(indexed_repo),
        }
        assert handle_pre_tool_use(payload) is None


# ---------------------------------------------------------------------------
# Read (and other non-augmented tools) are never intercepted
# ---------------------------------------------------------------------------


def test_read_tool_returns_none(indexed_repo: Path) -> None:
    payload: dict[str, Any] = {
        "tool_name": "Read",
        "tool_input": {"file_path": "/x"},
        "cwd": str(indexed_repo),
    }

    assert handle_pre_tool_use(payload) is None


@pytest.mark.parametrize("tool_name", ["Read", "Bash", "Write"])
def test_non_augmented_tools_short_circuit_before_lookup(tool_name: str) -> None:
    """Prove the tool-name filter runs before any lookup is attempted.

    `handle_pre_tool_use` catches `Exception` broadly, so a raising lookup
    would itself be swallowed into a `None` return and prove nothing; a mock
    call-count assertion is the only way to actually verify the short circuit.
    """
    lookup_mock = MagicMock(side_effect=AssertionError("must not be called"))
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_input": {"file_path": "/x", "pattern": "whatever"},
        "cwd": "/tmp",
    }

    with patch("archex.integrations.hook.lookup_with_timeout", lookup_mock):
        result = handle_pre_tool_use(payload)

    assert result is None
    lookup_mock.assert_not_called()


def test_hook_matcher_and_augmented_tools_constants() -> None:
    assert "Read" not in AUGMENTED_TOOLS
    assert {"Grep", "Glob"} == AUGMENTED_TOOLS
    assert HOOK_MATCHER == "Glob|Grep"


# ---------------------------------------------------------------------------
# Missing / stale index degrades silently and logs diagnostics
# ---------------------------------------------------------------------------


def test_missing_index_degrades_silently_and_logs_diagnostic(
    python_simple_repo: Path, diagnostics_log: Path
) -> None:
    """`python_simple_repo` is git-init'd but never `archex init`'d/indexed."""
    payload: dict[str, Any] = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "AuthService"},
        "cwd": str(python_simple_repo),
    }

    result = handle_pre_tool_use(payload)

    assert result is None
    entries = _read_diagnostics(diagnostics_log)
    assert entries
    entry = entries[-1]
    assert entry["kind"] in {"index_not_fresh", "status_error"}
    assert entry.get("detail")
    assert entry.get("cwd")


def test_stale_index_degrades_silently_and_logs_diagnostic(
    indexed_repo: Path, diagnostics_log: Path
) -> None:
    (indexed_repo / "utils.py").write_text("def dirty_symbol(): return 1\n", encoding="utf-8")
    status = inspect_project_status(indexed_repo)
    assert status.state in {"dirty", "stale"}

    payload: dict[str, Any] = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "AuthService"},
        "cwd": str(indexed_repo),
    }
    result = handle_pre_tool_use(payload)

    assert result is None
    entries = _read_diagnostics(diagnostics_log)
    assert entries
    entry = entries[-1]
    assert entry["kind"] == "index_not_fresh"
    assert f"state={status.state}" in entry["detail"]


# ---------------------------------------------------------------------------
# Malformed payload degradation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["not json at all", "[1, 2, 3]", '"just a string"', "42", "null", "true"],
    ids=["invalid_json", "list", "string", "number", "null", "boolean"],
)
def test_parse_payload_rejects_non_object_input_and_logs_diagnostic(
    raw: str, diagnostics_log: Path
) -> None:
    assert _parse_payload(raw) is None
    entries = _read_diagnostics(diagnostics_log)
    assert entries
    assert entries[-1]["kind"] == "malformed_payload"


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_name": "Grep", "cwd": "/tmp"},
        {"tool_name": "Grep", "tool_input": "not-a-dict", "cwd": "/tmp"},
        {"tool_name": "Grep", "tool_input": ["a", "list"], "cwd": "/tmp"},
    ],
    ids=["missing_tool_input", "string_tool_input", "list_tool_input"],
)
def test_handle_pre_tool_use_rejects_non_object_tool_input(
    payload: dict[str, Any], diagnostics_log: Path
) -> None:
    assert handle_pre_tool_use(payload) is None
    entries = _read_diagnostics(diagnostics_log)
    assert entries
    assert entries[-1]["kind"] == "malformed_payload"


# ---------------------------------------------------------------------------
# Timeout degradation
# ---------------------------------------------------------------------------


def test_lookup_timeout_degrades_silently_and_logs_diagnostic(
    indexed_repo: Path, diagnostics_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 0.1ms budget reliably beats even a real SQLite-backed lookup — verified
    deterministic across repeated local runs (no thread/sleep mocking needed).
    """
    monkeypatch.setenv("ARCHEX_HOOK_TIMEOUT_SECONDS", "0.0001")
    payload: dict[str, Any] = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "AuthService"},
        "cwd": str(indexed_repo),
    }

    result = handle_pre_tool_use(payload)

    assert result is None
    entries = _read_diagnostics(diagnostics_log)
    assert entries
    assert entries[-1]["kind"] == "timeout"


# ---------------------------------------------------------------------------
# Internal error degradation
# ---------------------------------------------------------------------------


def test_internal_error_opening_store_degrades_silently(
    indexed_repo: Path, diagnostics_log: Path
) -> None:
    assert inspect_project_status(indexed_repo).state == "fresh"
    payload: dict[str, Any] = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "AuthService"},
        "cwd": str(indexed_repo),
    }

    with patch("archex.integrations.hook.IndexStore", side_effect=RuntimeError("boom")):
        result = handle_pre_tool_use(payload)

    assert result is None
    entries = _read_diagnostics(diagnostics_log)
    assert entries
    assert entries[-1]["kind"] in {"lookup_error", "internal_error"}


# ---------------------------------------------------------------------------
# Full subprocess exit-0 contract
# ---------------------------------------------------------------------------


def test_subprocess_grep_payload_exits_zero_with_additional_context(
    indexed_repo: Path, tmp_path: Path
) -> None:
    diagnostics_log = tmp_path / "subprocess-diag.log"
    raw_stdin = json.dumps(
        {
            "tool_name": "Grep",
            "tool_input": {"pattern": "AuthService"},
            "cwd": str(indexed_repo),
        }
    )

    result = _run_hook_subprocess(raw_stdin, cwd=indexed_repo, diagnostics_log=diagnostics_log)

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert "AuthService" in output["hookSpecificOutput"]["additionalContext"]


def test_subprocess_read_payload_exits_zero_with_empty_stdout(tmp_path: Path) -> None:
    diagnostics_log = tmp_path / "subprocess-diag.log"
    raw_stdin = json.dumps({"tool_name": "Read", "tool_input": {"file_path": "/x"}})

    result = _run_hook_subprocess(raw_stdin, cwd=tmp_path, diagnostics_log=diagnostics_log)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_subprocess_garbage_stdin_exits_zero(tmp_path: Path) -> None:
    diagnostics_log = tmp_path / "subprocess-diag.log"

    result = _run_hook_subprocess("not json at all", cwd=tmp_path, diagnostics_log=diagnostics_log)

    assert result.returncode == 0, result.stderr


def test_subprocess_empty_stdin_exits_zero(tmp_path: Path) -> None:
    diagnostics_log = tmp_path / "subprocess-diag.log"

    result = _run_hook_subprocess("", cwd=tmp_path, diagnostics_log=diagnostics_log)

    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# `_extract_query` unit tests
# ---------------------------------------------------------------------------


def test_extract_query_grep_returns_pattern_verbatim_stripped() -> None:
    assert _extract_query("Grep", {"pattern": "  AuthService  "}) == "AuthService"


def test_extract_query_glob_extracts_identifier_tokens() -> None:
    result = _extract_query("Glob", {"pattern": "**/*_service.py"})

    assert "service" in result


def test_extract_query_glob_pattern_without_identifier_tokens_returns_empty_string() -> None:
    assert _extract_query("Glob", {"pattern": "**/*.py"}) == ""


@pytest.mark.parametrize(
    "tool_input",
    [{}, {"pattern": 123}, {"pattern": None}, {"pattern": "   "}],
    ids=["missing", "non_string", "none", "blank"],
)
def test_extract_query_missing_or_non_string_pattern_returns_empty_string(
    tool_input: dict[str, Any],
) -> None:
    assert _extract_query("Grep", tool_input) == ""


# ---------------------------------------------------------------------------
# Measured latency: the default timeout budget holds with real margin
# ---------------------------------------------------------------------------


def test_default_timeout_budget_holds_on_realistic_fixture(
    monorepo_simple_repo: Path,
) -> None:
    """Real-world latency evidence that the *default* 0.5s budget is generous.

    `test_lookup_timeout_degrades_silently_and_logs_diagnostic` above proves
    the timeout fires under an artificially tiny budget; this test proves the
    complementary fact — under realistic conditions the timeout does *not*
    fire, with a wide real margin. Uses `monorepo_simple_repo` (a multi-package
    fixture, larger than the single-package `python_simple_repo`/`indexed_repo`
    used elsewhere in this file) indexed for real via the same
    `init_project` + `cli index` pipeline, then calls `handle_pre_tool_use`
    in-process with `ARCHEX_HOOK_TIMEOUT_SECONDS` deliberately left unset so
    the real default budget (`DEFAULT_HOOK_TIMEOUT_SECONDS`) is exercised.

    Measured locally (8 back-to-back runs, this fixture): ~44-55ms per call —
    roughly a tenth of the 500ms budget. The `* 0.5` threshold below keeps a
    wide margin above that measured ceiling (well over 4x) so a loaded CI
    runner has ample headroom before this test could ever flake, while still
    failing loudly if a regression pushed the real lookup path anywhere close
    to the actual timeout.
    """
    init_project(monorepo_simple_repo)
    runner = CliRunner()
    result = runner.invoke(cli, ["index", str(monorepo_simple_repo)])
    assert result.exit_code == 0, result.output

    payload: dict[str, Any] = {
        "tool_name": "Grep",
        "tool_input": {"pattern": "initialize"},
        "cwd": str(monorepo_simple_repo),
    }

    start = time.perf_counter()
    output = handle_pre_tool_use(payload)
    elapsed = time.perf_counter() - start

    # Must not have degraded to a timeout no-op — a `None` here would mean the
    # timing assertion below is measuring nothing.
    assert output is not None, "lookup timed out against a small, freshly-indexed fixture"
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "initialize" in context

    budget = DEFAULT_HOOK_TIMEOUT_SECONDS
    assert elapsed < budget * 0.5, (
        f"lookup took {elapsed * 1000:.1f}ms, more than half of the {budget * 1000:.0f}ms "
        "default timeout budget on a small fixture"
    )


# ---------------------------------------------------------------------------
# Codex CLI diagnostics-only fallback (M21)
#
# Confirmation spike (read against `openai/codex`'s Rust source, not
# secondary docs): `PreToolUseHookSpecificOutputWire.additionalContext` DOES
# exist on Codex's wire schema (`codex-rs/hooks/src/schema.rs`) -- Codex
# hooks support content/context augmentation. But Codex has no
# Grep/Glob-equivalent `PreToolUse` tool name: its only hookable tool names
# are `Bash` (every shell invocation, `HookToolName::bash()` in
# `codex-rs/core/src/tools/hook_names.rs`), `apply_patch`, `spawn_agent`, and
# MCP tools under their own names. There is no way to scope augmentation to
# "search calls only" the way Claude Code's Grep/Glob or oh-my-pi/Pi's
# grep/glob/find do -- hooking `Bash` unconditionally would intercept every
# shell command, including destructive ones. `archex.integrations.codex_hook`
# therefore ships the plan's own diagnostics-only fallback: it detects
# search-shaped `Bash` commands and logs what archex could have surfaced,
# but never returns or applies any `hookSpecificOutput`.
# ---------------------------------------------------------------------------


def _run_codex_hook_subprocess(
    raw_stdin: str, *, cwd: Path, diagnostics_log: Path
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["ARCHEX_HOOK_DIAGNOSTICS_LOG"] = str(diagnostics_log)
    return subprocess.run(
        [sys.executable, "-m", "archex.integrations.codex_hook"],
        input=raw_stdin,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=30,
    )


def test_codex_search_shaped_bash_command_logs_withheld_diagnostic_not_augmentation(
    indexed_repo: Path, diagnostics_log: Path
) -> None:
    payload: dict[str, Any] = {
        "tool_name": "Bash",
        "tool_input": {"command": "grep -rn AuthService ."},
        "cwd": str(indexed_repo),
    }

    result = handle_codex_pre_tool_use(payload)

    assert result is None  # never returns augmented output, ever
    entries = _read_diagnostics(diagnostics_log)
    assert entries, "expected a diagnostic line for a search-shaped Bash command"
    entry = entries[-1]
    assert entry["kind"] == "codex_augmentation_withheld"
    assert "AuthService" in entry["detail"]
    assert "Grep/Glob-equivalent" in entry["detail"]


@pytest.mark.parametrize(
    "command",
    ["ls -la", "npm test", "git status", "echo grepping along nicely"],
)
def test_codex_non_search_bash_commands_never_log_a_diagnostic(
    indexed_repo: Path, diagnostics_log: Path, command: str
) -> None:
    payload: dict[str, Any] = {
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "cwd": str(indexed_repo),
    }

    assert handle_codex_pre_tool_use(payload) is None
    assert _read_diagnostics(diagnostics_log) == []


@pytest.mark.parametrize("tool_name", ["Read", "apply_patch", "spawn_agent", "Grep", "Glob"])
def test_codex_non_bash_tool_names_short_circuit_before_lookup(
    tool_name: str, diagnostics_log: Path
) -> None:
    """Codex exposes no `Read` hook at all, but this asserts defense in depth:
    the dispatch table only ever matches `Bash`, never `Read` or anything else,
    regardless of what a payload claims.
    """
    lookup_mock = MagicMock(side_effect=AssertionError("must not be called"))
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_input": {"command": "grep -rn foo ."},
        "cwd": "/tmp",
    }

    with patch("archex.integrations.codex_hook.lookup_with_timeout", lookup_mock):
        assert handle_codex_pre_tool_use(payload) is None

    lookup_mock.assert_not_called()
    assert _read_diagnostics(diagnostics_log) == []


def test_codex_missing_index_degrades_silently_and_logs_diagnostic(
    python_simple_repo: Path, diagnostics_log: Path
) -> None:
    """`python_simple_repo` is git-init'd but never `archex init`'d/indexed."""
    payload: dict[str, Any] = {
        "tool_name": "Bash",
        "tool_input": {"command": "grep -rn foo ."},
        "cwd": str(python_simple_repo),
    }

    assert handle_codex_pre_tool_use(payload) is None

    entries = _read_diagnostics(diagnostics_log)
    assert entries
    assert entries[-1]["kind"] in {"status_error", "index_not_fresh"}


@pytest.mark.parametrize("raw", ["not json", "[]", "42", '"a string"'])
def test_codex_parse_payload_rejects_non_object_input_and_logs_diagnostic(
    raw: str, diagnostics_log: Path
) -> None:
    assert _parse_codex_payload(raw) is None

    entries = _read_diagnostics(diagnostics_log)
    assert entries[-1]["kind"] == "codex_malformed_payload"


def test_codex_handle_pre_tool_use_rejects_non_object_tool_input(diagnostics_log: Path) -> None:
    payload: dict[str, Any] = {"tool_name": "Bash", "tool_input": "grep foo", "cwd": "/tmp"}

    assert handle_codex_pre_tool_use(payload) is None
    assert _read_diagnostics(diagnostics_log) == []


def test_codex_internal_error_degrades_silently_and_logs_diagnostic(
    indexed_repo: Path, diagnostics_log: Path
) -> None:
    payload: dict[str, Any] = {
        "tool_name": "Bash",
        "tool_input": {"command": "grep -rn foo ."},
        "cwd": str(indexed_repo),
    }

    with patch(
        "archex.integrations.codex_hook.lookup_with_timeout",
        side_effect=RuntimeError("boom"),
    ):
        assert handle_codex_pre_tool_use(payload) is None

    entries = _read_diagnostics(diagnostics_log)
    assert entries[-1]["kind"] == "codex_internal_error"


def test_codex_subprocess_search_command_exits_zero_with_empty_object_stdout(
    indexed_repo: Path, tmp_path: Path
) -> None:
    diagnostics_log = tmp_path / "codex-subprocess-diag.log"
    stdin_payload = json.dumps(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "grep -rn AuthService ."},
            "cwd": str(indexed_repo),
        }
    )

    result = _run_codex_hook_subprocess(
        stdin_payload, cwd=indexed_repo, diagnostics_log=diagnostics_log
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {}
    entries = _read_diagnostics(diagnostics_log)
    assert entries[-1]["kind"] == "codex_augmentation_withheld"


def test_codex_subprocess_garbage_stdin_exits_zero_with_empty_object_stdout(tmp_path: Path) -> None:
    diagnostics_log = tmp_path / "codex-subprocess-diag.log"

    result = _run_codex_hook_subprocess("not json", cwd=tmp_path, diagnostics_log=diagnostics_log)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {}


def test_codex_subprocess_empty_stdin_exits_zero_with_empty_object_stdout(tmp_path: Path) -> None:
    diagnostics_log = tmp_path / "codex-subprocess-diag.log"

    result = _run_codex_hook_subprocess("", cwd=tmp_path, diagnostics_log=diagnostics_log)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {}


def test_codex_hook_matcher_targets_bash_only() -> None:
    assert CODEX_HOOK_MATCHER == "^Bash$"
    assert DIAGNOSED_TOOL_NAME == "Bash"
    matcher_re = re.compile(CODEX_HOOK_MATCHER)
    assert matcher_re.fullmatch("Bash")
    for other in ["Read", "Grep", "Glob", "apply_patch", "spawn_agent", "bash", "Bashing"]:
        assert not matcher_re.fullmatch(other)


@pytest.mark.parametrize(
    "command",
    [
        "grep -rn AuthService .",
        "rg AuthService",
        "ag AuthService",
        "git grep AuthService",
        "find . -name '*.py'",
        "fd '.py$'",
        "sudo grep -rn AuthService .",
        "cat file.txt | grep foo",
        "cd src && grep -rn foo .",
        "ls; grep -rn foo .",
    ],
)
def test_codex_search_command_regex_matches_known_search_tools(command: str) -> None:
    assert _SEARCH_COMMAND_RE.search(command)


@pytest.mark.parametrize(
    "command",
    ["ls -la", "npm test", "git status", "echo grepping along nicely", "python -c 'print(1)'"],
)
def test_codex_search_command_regex_does_not_match_non_search_commands(command: str) -> None:
    assert not _SEARCH_COMMAND_RE.search(command)


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
