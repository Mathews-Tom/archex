"""`install-client --post-edit-hooks` across every supported client (R21).

The post-edit hook is a second, independent surface beside the PreToolUse
search hook and the SessionStart primer. These tests pin the two properties
that make that safe: each surface owns its own event/marker/filename so
installing one never disturbs another, and an unsupported client is refused
explicitly rather than silently no-op'ing.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.cli.main import cli
from archex.client_setup import (
    POST_EDIT_UNSUPPORTED_CLIENTS,
    CodexPostEditHookInstallPlan,
    TsPostEditHookInstallPlan,
    build_hook_install_plan,
    build_post_edit_hook_install_plan,
    build_session_primer_install_plan,
    render_post_edit_hook_install_preview,
    write_hook_install_plan,
    write_post_edit_hook_install_plan,
    write_session_primer_install_plan,
)
from archex.integrations.codex_post_edit_hook import POST_EDIT_MATCHER as CODEX_POST_EDIT_MATCHER
from archex.integrations.post_edit_hook import POST_EDIT_MATCHER

if TYPE_CHECKING:
    from typing import Any

    import pytest

SUPPORTED = ("claude-code", "codex", "omp", "pi", "opencode")

_EXPECTED_TARGET = {
    "claude-code": Path(".claude") / "settings.json",
    "codex": Path(".codex") / "config.toml",
    "omp": Path(".omp") / "extensions" / "archex-post-edit-hook.ts",
    "pi": Path(".pi") / "extensions" / "archex-post-edit-hook.ts",
    "opencode": Path(".opencode") / "plugins" / "archex-post-edit-hook.ts",
}


def _install(repo: Path, client: str) -> Path:
    plan = build_post_edit_hook_install_plan(client, str(repo), action="install")  # pyright: ignore[reportArgumentType]
    return write_post_edit_hook_install_plan(plan)


def _remove(repo: Path, client: str) -> Path:
    plan = build_post_edit_hook_install_plan(client, str(repo), action="remove")  # pyright: ignore[reportArgumentType]
    return write_post_edit_hook_install_plan(plan)


# ---------------------------------------------------------------------------
# Per-client install/remove
# ---------------------------------------------------------------------------


def test_every_supported_client_writes_its_own_target(tmp_path: Path) -> None:
    for client in SUPPORTED:
        target = _install(tmp_path, client)
        assert target == tmp_path / _EXPECTED_TARGET[client], client
        assert target.exists(), client


def test_claude_code_installs_a_post_tool_use_group(tmp_path: Path) -> None:
    target = _install(tmp_path, "claude-code")

    payload = json.loads(target.read_text(encoding="utf-8"))
    group = payload["hooks"]["PostToolUse"][0]
    assert group["matcher"] == POST_EDIT_MATCHER
    assert group["hooks"][0]["command"] == sys.executable
    assert group["hooks"][0]["args"] == ["-m", "archex.integrations.post_edit_hook"]


def test_codex_installs_a_marker_delimited_post_tool_use_block(tmp_path: Path) -> None:
    target = _install(tmp_path, "codex")

    raw = target.read_text(encoding="utf-8")
    assert "# archex:codex-post-edit-hook start" in raw
    hook = tomllib.loads(raw)["hooks"]["PostToolUse"][0]
    assert hook["matcher"] == CODEX_POST_EDIT_MATCHER
    assert "archex.integrations.codex_post_edit_hook" in hook["hooks"][0]["command"]
    assert hook["hooks"][0]["timeout"] > 0


def test_ts_clients_bake_their_own_identity_into_the_module(tmp_path: Path) -> None:
    for client in ("omp", "pi", "opencode"):
        module = _install(tmp_path, client).read_text(encoding="utf-8")
        assert f'const ARCHEX_CLIENT = "{client}"' in module, client
        assert "__ARCHEX_CLIENT__" not in module, client
        assert "__ARCHEX_PYTHON_COMMAND__" not in module, client
        assert "archex.integrations.post_edit_hook" in module, client


def test_omp_and_pi_share_one_module_and_opencode_differs(tmp_path: Path) -> None:
    omp = _install(tmp_path, "omp").read_text(encoding="utf-8")
    pi = _install(tmp_path, "pi").read_text(encoding="utf-8")
    opencode = _install(tmp_path, "opencode").read_text(encoding="utf-8")

    assert omp.replace('"omp"', '"pi"') == pi
    assert 'pi.on("tool_result"' in omp
    assert '"tool.execute.after"' in opencode
    assert 'pi.on("tool_result"' not in opencode


def test_reinstall_is_an_idempotent_no_op(tmp_path: Path) -> None:
    for client in SUPPORTED:
        first = _install(tmp_path, client).read_text(encoding="utf-8")
        second = _install(tmp_path, client).read_text(encoding="utf-8")
        assert first == second, client


def test_remove_restores_the_pre_install_state(tmp_path: Path) -> None:
    """Remove must leave no archex post-edit hook and no other trace.

    A JSON settings file archex itself created is left behind holding `{}`
    rather than being deleted, matching the existing search-hook installer:
    removing a file the user may since have taken ownership of is not the
    uninstaller's call. The standalone TS modules archex fully owns are
    unlinked.
    """
    for client in SUPPORTED:
        target = tmp_path / _EXPECTED_TARGET[client]
        _install(tmp_path, client)
        _remove(tmp_path, client)
        if client in {"omp", "pi", "opencode"}:
            assert not target.exists(), client
            continue
        remaining = target.read_text(encoding="utf-8") if target.exists() else ""
        assert "post_edit_hook" not in remaining, client
        assert "PostToolUse" not in remaining, client


def test_remove_without_a_prior_install_is_harmless(tmp_path: Path) -> None:
    for client in SUPPORTED:
        _remove(tmp_path, client)


# ---------------------------------------------------------------------------
# Independence from the other archex surfaces
# ---------------------------------------------------------------------------


def _seed_unrelated_claude_settings(repo: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "otherTopLevelKey": "unrelated",
        "hooks": {
            "PostToolUse": [
                {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo mine"}]}
            ]
        },
    }
    target = repo / ".claude" / "settings.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def test_claude_install_preserves_unrelated_post_tool_use_hooks(tmp_path: Path) -> None:
    _seed_unrelated_claude_settings(tmp_path)

    target = _install(tmp_path, "claude-code")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["otherTopLevelKey"] == "unrelated"
    matchers = [group["matcher"] for group in payload["hooks"]["PostToolUse"]]
    assert "Bash" in matchers
    assert POST_EDIT_MATCHER in matchers


def test_claude_remove_leaves_unrelated_hooks_intact(tmp_path: Path) -> None:
    seeded = _seed_unrelated_claude_settings(tmp_path)

    _install(tmp_path, "claude-code")
    target = _remove(tmp_path, "claude-code")

    assert json.loads(target.read_text(encoding="utf-8")) == seeded


def test_the_three_claude_surfaces_coexist_and_uninstall_independently(tmp_path: Path) -> None:
    write_hook_install_plan(build_hook_install_plan("claude-code", str(tmp_path), action="install"))
    write_session_primer_install_plan(
        build_session_primer_install_plan(str(tmp_path), action="install")
    )
    target = _install(tmp_path, "claude-code")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload["hooks"]) == {"PreToolUse", "SessionStart", "PostToolUse"}

    _remove(tmp_path, "claude-code")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload["hooks"]) == {"PreToolUse", "SessionStart"}


def test_codex_post_edit_block_coexists_with_the_search_hook_block(tmp_path: Path) -> None:
    write_hook_install_plan(build_hook_install_plan("codex", str(tmp_path), action="install"))
    target = _install(tmp_path, "codex")

    parsed = tomllib.loads(target.read_text(encoding="utf-8"))
    assert "PreToolUse" in parsed["hooks"]
    assert "PostToolUse" in parsed["hooks"]

    _remove(tmp_path, "codex")

    raw = target.read_text(encoding="utf-8")
    assert "PostToolUse" not in tomllib.loads(raw)["hooks"]
    assert "PreToolUse" in tomllib.loads(raw)["hooks"]


def test_ts_post_edit_module_is_a_different_file_from_the_search_module(tmp_path: Path) -> None:
    write_hook_install_plan(build_hook_install_plan("omp", str(tmp_path), action="install"))
    post_edit_target = _install(tmp_path, "omp")

    search_target = tmp_path / ".omp" / "extensions" / "archex-hook.ts"
    assert search_target.exists()
    assert post_edit_target != search_target

    _remove(tmp_path, "omp")

    assert search_target.exists()
    assert not post_edit_target.exists()


# ---------------------------------------------------------------------------
# Unsupported clients
# ---------------------------------------------------------------------------


def test_cursor_is_refused_with_its_upstream_reason(tmp_path: Path) -> None:
    assert "cursor" in POST_EDIT_UNSUPPORTED_CLIENTS

    try:
        build_post_edit_hook_install_plan("cursor", str(tmp_path), action="install")
    except ValueError as exc:
        assert "afterFileEdit" in str(exc)
        assert "no output fields" in str(exc)
    else:
        raise AssertionError("cursor must not produce a post-edit plan")


def test_cursor_refusal_writes_nothing(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli, ["install-client", "cursor", str(tmp_path), "--scope", "project", "--post-edit-hooks"]
    )

    assert result.exit_code != 0
    assert "no supported post-edit event" in result.output
    assert not (tmp_path / ".cursor").exists()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_dry_run_previews_without_writing(tmp_path: Path) -> None:
    for client in SUPPORTED:
        result = CliRunner().invoke(
            cli,
            [
                "install-client",
                client,
                str(tmp_path),
                "--scope",
                "project",
                "--post-edit-hooks",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, (client, result.output)
        assert "Dry run" in result.output, client
        assert not (tmp_path / _EXPECTED_TARGET[client]).exists(), client


def test_cli_install_then_remove_round_trips(tmp_path: Path) -> None:
    runner = CliRunner()
    base = ["install-client", "claude-code", str(tmp_path), "--scope", "project"]

    installed = runner.invoke(cli, [*base, "--post-edit-hooks"])
    assert installed.exit_code == 0, installed.output
    assert "Installed archex post-edit hook for claude-code" in installed.output

    removed = runner.invoke(cli, [*base, "--remove-post-edit-hooks"])
    assert removed.exit_code == 0, removed.output
    assert "Removed archex post-edit hook for claude-code" in removed.output
    assert not (tmp_path / ".claude" / "settings.json").exists() or "PostToolUse" not in json.loads(
        (tmp_path / ".claude" / "settings.json").read_text(encoding="utf-8")
    ).get("hooks", {})


def test_opposing_post_edit_flags_are_rejected(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "install-client",
            "claude-code",
            str(tmp_path),
            "--post-edit-hooks",
            "--remove-post-edit-hooks",
        ],
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_post_edit_cannot_be_combined_with_another_surface(tmp_path: Path) -> None:
    for other in ("--hooks", "--session-primer"):
        result = CliRunner().invoke(
            cli, ["install-client", "claude-code", str(tmp_path), "--post-edit-hooks", other]
        )
        assert result.exit_code != 0, other
        assert "separately" in result.output, other


def test_post_edit_requires_an_explicit_client(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["install-client", str(tmp_path), "--post-edit-hooks"])

    assert result.exit_code != 0
    assert "Must specify a client" in result.output


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------


def test_preview_reports_an_idempotent_reinstall(tmp_path: Path) -> None:
    for client in SUPPORTED:
        _install(tmp_path, client)
        plan = build_post_edit_hook_install_plan(client, str(tmp_path), action="install")  # pyright: ignore[reportArgumentType]
        assert "idempotent no-op" in render_post_edit_hook_install_preview(plan), client


def test_preview_reports_nothing_to_remove(tmp_path: Path) -> None:
    for client in SUPPORTED:
        plan = build_post_edit_hook_install_plan(client, str(tmp_path), action="remove")  # pyright: ignore[reportArgumentType]
        assert "No change" in render_post_edit_hook_install_preview(plan), client


def test_plan_types_match_each_client_family(tmp_path: Path) -> None:
    codex = build_post_edit_hook_install_plan("codex", str(tmp_path), action="install")
    opencode = build_post_edit_hook_install_plan("opencode", str(tmp_path), action="install")

    assert isinstance(codex, CodexPostEditHookInstallPlan)
    assert isinstance(opencode, TsPostEditHookInstallPlan)


def test_user_scope_targets_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    plan = build_post_edit_hook_install_plan("claude-code", action="install")

    assert plan.scope == "user"
    assert plan.target_path == tmp_path / ".claude" / "settings.json"
