"""Post-edit hook adapters: dispatch, safety, and real installed-hook smokes.

Tests that spawn `python -m archex.integrations...` exercise the module
exactly as a client does — same argv, same stdin JSON, same stdout contract —
so the non-blocking exit-code guarantee is measured rather than asserted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from archex.cli.main import cli
from archex.integrations.codex_post_edit_hook import (
    _patch_paths,  # pyright: ignore[reportPrivateUsage]
    handle_codex_post_tool_use,
)
from archex.integrations.post_edit_hook import (
    AUGMENTED_TOOLS,
    POST_EDIT_MATCHER,
    SHIM_CLIENTS,
    _client_of,  # pyright: ignore[reportPrivateUsage]
    extract_paths,
    handle_post_tool_use,
    parse_payload,
)
from archex.post_edit import PostEditStatus, read_state
from archex.project import init_project


def _indexed(repo: Path) -> Path:
    init_project(repo)
    result = CliRunner().invoke(cli, ["index", str(repo)])
    assert result.exit_code == 0, result.output
    return repo


def _run_hook(
    module: str,
    payload: dict[str, Any],
    *,
    cwd: Path,
    diagnostics: Path,
    env_extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["ARCHEX_HOOK_DIAGNOSTICS_LOG"] = str(diagnostics)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", module],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=120,
        check=False,
    )


def _kinds(diagnostics: Path) -> list[str]:
    if not diagnostics.exists():
        return []
    return [
        json.loads(line)["kind"]
        for line in diagnostics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source(repo: Path) -> Path:
    candidates = sorted(path for path in repo.rglob("*.py") if ".archex" not in path.parts)
    assert candidates
    return candidates[0]


# ---------------------------------------------------------------------------
# Claude Code dispatch
# ---------------------------------------------------------------------------


def test_matcher_selects_exactly_the_edit_tools() -> None:
    import re

    pattern = re.compile(POST_EDIT_MATCHER)
    for tool in AUGMENTED_TOOLS:
        assert pattern.search(tool), tool
    for tool in ("Read", "Grep", "Glob", "Bash", "Task", "WebFetch"):
        assert not pattern.search(tool), tool


def test_non_edit_tools_are_ignored() -> None:
    for tool in ("Read", "Grep", "Glob", "Bash"):
        assert (
            handle_post_tool_use({"tool_name": tool, "tool_input": {"file_path": "/x.py"}}) is None
        )


def test_an_explicitly_failed_edit_is_not_recorded(tmp_path: Path) -> None:
    payload = {
        "tool_name": "Write",
        "cwd": str(tmp_path),
        "tool_input": {"file_path": str(tmp_path / "a.py")},
        "tool_response": {"success": False},
    }

    assert handle_post_tool_use(payload) is None
    assert not (tmp_path / ".archex" / "post-edit-state.json").exists()


def test_an_errored_edit_is_not_recorded(tmp_path: Path) -> None:
    payload = {
        "tool_name": "Edit",
        "cwd": str(tmp_path),
        "tool_input": {"file_path": str(tmp_path / "a.py")},
        "tool_response": {"error": "permission denied"},
    }

    assert handle_post_tool_use(payload) is None


def test_a_payload_without_a_path_is_ignored(tmp_path: Path) -> None:
    assert handle_post_tool_use({"tool_name": "Edit", "cwd": str(tmp_path)}) is None


def test_paths_are_collected_from_input_and_response() -> None:
    payload = {
        "tool_input": {"file_path": "/a.py", "notebook_path": "/b.ipynb"},
        "tool_response": {"filePath": "/c.py"},
    }

    assert extract_paths(payload) == ["/a.py", "/b.ipynb", "/c.py"]


def test_malformed_payloads_are_discarded(tmp_path: Path) -> None:
    log = tmp_path / "diag.log"
    os.environ["ARCHEX_HOOK_DIAGNOSTICS_LOG"] = str(log)
    try:
        assert parse_payload("") is None
        assert parse_payload("{not json") is None
        assert parse_payload("[1, 2]") is None
        assert parse_payload('{"tool_name": "Edit"}') == {"tool_name": "Edit"}
    finally:
        del os.environ["ARCHEX_HOOK_DIAGNOSTICS_LOG"]
    assert _kinds(log) == ["post_edit_malformed_payload"] * 3


def test_shim_clients_are_allowlisted_and_default_to_claude_code() -> None:
    assert _client_of({}) == "claude-code"
    assert _client_of({"archex_client": "not-a-client"}) == "claude-code"
    for client in SHIM_CLIENTS:
        assert _client_of({"archex_client": client}) == client


def test_an_unmanaged_repository_produces_no_output(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    payload = {
        "tool_name": "Edit",
        "cwd": str(tmp_path),
        "tool_input": {"file_path": str(tmp_path / "a.py")},
    }

    assert handle_post_tool_use(payload) is None


# ---------------------------------------------------------------------------
# Codex dispatch and patch parsing
# ---------------------------------------------------------------------------


def test_v4a_patch_markers_yield_every_edited_path() -> None:
    command = (
        "*** Begin Patch\n"
        "*** Add File: pkg/new.py\n"
        "+x = 1\n"
        "*** Update File: pkg/core.py\n"
        "@@\n"
        "-a\n"
        "+b\n"
        "*** Delete File: pkg/old.py\n"
        "*** Move to: pkg/renamed.py\n"
        "*** End Patch\n"
    )

    assert _patch_paths({"command": command}) == [
        "pkg/new.py",
        "pkg/core.py",
        "pkg/old.py",
        "pkg/renamed.py",
    ]


def test_marker_text_outside_the_patch_envelope_is_not_a_path() -> None:
    command = (
        "*** Update File: outside.py\n"
        "*** Begin Patch\n"
        "*** Update File: inside.py\n"
        "*** End Patch\n"
        "*** Add File: after.py\n"
    )

    assert _patch_paths({"command": command}) == ["inside.py"]


def test_a_non_patch_tool_input_yields_no_paths() -> None:
    assert _patch_paths(None) == []
    assert _patch_paths({}) == []
    assert _patch_paths({"command": "rm -rf /"}) == []
    assert _patch_paths({"command": 17}) == []


def test_codex_ignores_every_tool_but_apply_patch() -> None:
    assert handle_codex_post_tool_use({"tool_name": "Bash", "tool_input": {"command": "ls"}}) == {}


def test_codex_returns_no_decision_when_no_path_parses(tmp_path: Path) -> None:
    payload = {
        "tool_name": "apply_patch",
        "cwd": str(tmp_path),
        "tool_input": {"command": "not a patch"},
    }

    assert handle_codex_post_tool_use(payload) == {}


# ---------------------------------------------------------------------------
# Real installed-hook subprocess smokes
# ---------------------------------------------------------------------------


def test_installed_claude_hook_emits_fresh_bounded_impact(python_simple_repo: Path) -> None:
    repo = _indexed(python_simple_repo)
    diagnostics = repo.parent / "diag.log"
    target = _source(repo)
    target.write_text(target.read_text() + "\n# post-edit smoke\n")

    result = _run_hook(
        "archex.integrations.post_edit_hook",
        {
            "cwd": str(repo),
            "hook_event_name": "PostToolUse",
            "tool_name": "Edit",
            "tool_input": {"file_path": str(target)},
            "tool_response": {"filePath": str(target), "success": True},
        },
        cwd=repo,
        diagnostics=diagnostics,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    context = payload["hookSpecificOutput"]["additionalContext"]
    assert "[archex post-edit receipt]" in context
    assert "client=claude-code" in context
    assert "file-scoped" in context
    assert target.relative_to(repo).as_posix() in context
    assert read_state(repo).status is PostEditStatus.CLEAN


def test_installed_hook_attributes_a_shim_client(python_simple_repo: Path) -> None:
    repo = _indexed(python_simple_repo)
    target = _source(repo)
    target.write_text(target.read_text() + "\n# shim smoke\n")

    result = _run_hook(
        "archex.integrations.post_edit_hook",
        {
            "cwd": str(repo),
            "tool_name": "Edit",
            "tool_input": {"file_path": str(target)},
            "archex_client": "opencode",
        },
        cwd=repo,
        diagnostics=repo.parent / "diag.log",
    )

    assert result.returncode == 0
    assert "client=opencode" in json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]


def test_installed_codex_hook_emits_impact_from_a_real_patch(python_simple_repo: Path) -> None:
    repo = _indexed(python_simple_repo)
    target = _source(repo)
    target.write_text(target.read_text() + "\n# codex smoke\n")
    relative = target.relative_to(repo).as_posix()
    command = f"*** Begin Patch\n*** Update File: {relative}\n@@\n+# codex smoke\n*** End Patch\n"

    result = _run_hook(
        "archex.integrations.codex_post_edit_hook",
        {
            "cwd": str(repo),
            "hook_event_name": "PostToolUse",
            "tool_name": "apply_patch",
            "tool_input": {"command": command},
            "tool_response": {"output": "Success"},
        },
        cwd=repo,
        diagnostics=repo.parent / "diag.log",
    )

    assert result.returncode == 0
    context = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "client=codex" in context
    assert relative in context


def test_a_forced_timeout_exits_successfully_and_records_a_diagnostic(
    python_simple_repo: Path,
) -> None:
    repo = _indexed(python_simple_repo)
    diagnostics = repo.parent / "timeout.log"
    target = _source(repo)
    target.write_text(target.read_text() + "\n# timeout smoke\n")

    started = time.monotonic()
    result = _run_hook(
        "archex.integrations.post_edit_hook",
        {
            "cwd": str(repo),
            "tool_name": "Edit",
            "tool_input": {"file_path": str(target)},
        },
        cwd=repo,
        diagnostics=diagnostics,
        env_extra={"ARCHEX_POST_EDIT_TIMEOUT_SECONDS": "0.001"},
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0
    assert result.stdout.strip() == ""
    assert elapsed < 30
    assert "post_edit_timeout" in _kinds(diagnostics)
    # The edit stays recorded so the next event retries rather than losing it.
    assert read_state(repo).status is PostEditStatus.DIRTY


def test_a_garbage_payload_exits_successfully(python_simple_repo: Path) -> None:
    diagnostics = python_simple_repo.parent / "garbage.log"
    env = dict(os.environ)
    env["ARCHEX_HOOK_DIAGNOSTICS_LOG"] = str(diagnostics)

    result = subprocess.run(
        [sys.executable, "-m", "archex.integrations.post_edit_hook"],
        input="}{ not json",
        capture_output=True,
        text=True,
        cwd=str(python_simple_repo),
        env=env,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == ""
    assert "post_edit_malformed_payload" in _kinds(diagnostics)


def test_codex_hook_always_writes_an_object_even_on_garbage(python_simple_repo: Path) -> None:
    diagnostics = python_simple_repo.parent / "codex-garbage.log"
    env = dict(os.environ)
    env["ARCHEX_HOOK_DIAGNOSTICS_LOG"] = str(diagnostics)

    result = subprocess.run(
        [sys.executable, "-m", "archex.integrations.codex_post_edit_hook"],
        input="}{ not json",
        capture_output=True,
        text=True,
        cwd=str(python_simple_repo),
        env=env,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {}
    assert "codex_post_edit_malformed_payload" in _kinds(diagnostics)


def test_an_edit_outside_the_repository_is_rejected(python_simple_repo: Path) -> None:
    repo = _indexed(python_simple_repo)
    diagnostics = repo.parent / "escape.log"
    outside = repo.parent / "outside.py"
    outside.write_text("secret = 1\n")

    result = _run_hook(
        "archex.integrations.post_edit_hook",
        {"cwd": str(repo), "tool_name": "Write", "tool_input": {"file_path": str(outside)}},
        cwd=repo,
        diagnostics=diagnostics,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == ""
    assert "post_edit_path_rejected" in _kinds(diagnostics)
    assert read_state(repo).pending_paths == []


def test_an_indented_context_line_is_not_read_as_a_file_header() -> None:
    """V4A body lines carry a leading ' ', '+', or '-'; headers are at column 0.

    Left-stripping before matching would let a context line that happens to
    reproduce a marker inject a path the patch never touched.
    """
    command = (
        "*** Begin Patch\n"
        "*** Update File: real.py\n"
        "@@\n"
        " *** Update File: smuggled.py\n"
        "+changed\n"
        "*** End Patch\n"
    )

    assert _patch_paths({"command": command}) == ["real.py"]


def test_an_indented_end_marker_does_not_truncate_the_scan() -> None:
    command = (
        "*** Begin Patch\n"
        "*** Update File: first.py\n"
        "@@\n"
        " *** End Patch\n"
        "+changed\n"
        "*** Update File: second.py\n"
        "@@\n"
        "+more\n"
        "*** End Patch\n"
    )

    assert _patch_paths({"command": command}) == ["first.py", "second.py"]


def test_crlf_patch_text_still_yields_paths() -> None:
    command = "*** Begin Patch\r\n*** Update File: pkg/core.py\r\n@@\r\n+x\r\n*** End Patch\r\n"

    assert _patch_paths({"command": command}) == ["pkg/core.py"]


def test_a_heredoc_wrapped_patch_is_still_recognized() -> None:
    command = (
        "apply_patch <<'EOF'\n"
        "*** Begin Patch\n"
        "*** Update File: pkg/core.py\n"
        "@@\n"
        "+x\n"
        "*** End Patch\n"
        "EOF\n"
    )

    assert _patch_paths({"command": command}) == ["pkg/core.py"]


def test_a_filename_containing_spaces_survives_extraction() -> None:
    command = "*** Begin Patch\n*** Add File: docs/my notes.md\n+hi\n*** End Patch\n"

    assert _patch_paths({"command": command}) == ["docs/my notes.md"]


def test_both_hooks_exit_zero_when_the_client_closes_the_pipe_early() -> None:
    """`os._exit` does not flush, so an unguarded flush on a closed pipe
    would skip it and exit non-zero with a traceback."""
    for module in ("post_edit_hook", "codex_post_edit_hook"):
        process = subprocess.Popen(
            [sys.executable, "-m", f"archex.integrations.{module}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(b'{"tool_name": "Edit", "tool_input": {"file_path": "/nope.py"}}')
        process.stdin.close()
        process.stdout.close()  # client stops reading before the hook writes
        assert process.wait(timeout=60) == 0, module
        if process.stderr is not None:
            process.stderr.close()


def test_indented_headers_outside_an_update_hunk_are_accepted() -> None:
    """Upstream trims both ends when scanning for headers outside a hunk.

    Matching those at column zero would silently drop a patch upstream
    applies, skipping impact for a real edit.
    """
    command = (
        "  *** Begin Patch\n"
        "  *** Delete File: pkg/gone.py\n"
        "  *** Add File: pkg/new.py\n"
        "  +created\n"
        "  *** End Patch\n"
    )

    assert _patch_paths({"command": command}) == ["pkg/gone.py", "pkg/new.py"]


def test_a_tab_indented_header_is_accepted() -> None:
    command = "\t*** Begin Patch\n\t*** Delete File: pkg/gone.py\n\t*** End Patch\n"

    assert _patch_paths({"command": command}) == ["pkg/gone.py"]


def test_a_move_to_marker_does_not_leave_the_update_hunk_mode() -> None:
    """`*** Move to:` appears inside an update hunk, so the stricter
    column-zero rule must stay in force after it."""
    command = (
        "*** Begin Patch\n"
        "*** Update File: pkg/old.py\n"
        "*** Move to: pkg/new.py\n"
        "@@\n"
        " *** Update File: smuggled.py\n"
        "+changed\n"
        "*** End Patch\n"
    )

    assert _patch_paths({"command": command}) == ["pkg/old.py", "pkg/new.py"]
