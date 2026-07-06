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
from archex.integrations.hook import (
    AUGMENTED_TOOLS,
    DEFAULT_HOOK_TIMEOUT_SECONDS,
    HOOK_MATCHER,
    _extract_query,  # pyright: ignore[reportPrivateUsage]
    _parse_payload,  # pyright: ignore[reportPrivateUsage]
    handle_pre_tool_use,
)
from archex.project import init_project
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

    with patch("archex.integrations.hook._lookup_with_timeout", lookup_mock):
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
