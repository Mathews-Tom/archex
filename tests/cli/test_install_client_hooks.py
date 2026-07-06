from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from click.testing import CliRunner

from archex.cli.main import cli
from archex.client_setup import (
    build_hook_install_plan,
    render_hook_install_preview,
    write_hook_install_plan,
)
from archex.integrations.hook import HOOK_MATCHER

if TYPE_CHECKING:
    from typing import Any

    from archex.client_setup import ClientName


def _seed_payload() -> dict[str, Any]:
    """Unrelated pre-existing settings.json content the installer must never touch."""
    return {
        "otherTopLevelKey": "unrelated-value",
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "echo bash-hook"}],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": "echo post-hook"}],
                }
            ],
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _hook_group_has_archex_entry(group: object) -> bool:
    if not isinstance(group, dict):
        return False
    handlers = cast("dict[str, object]", group).get("hooks")
    if not isinstance(handlers, list):
        return False
    for handler in cast("list[object]", handlers):
        if not isinstance(handler, dict):
            continue
        args = cast("dict[str, object]", handler).get("args")
        if not isinstance(args, list):
            continue
        items = cast("list[object]", args)
        if any(isinstance(item, str) and "archex.integrations.hook" in item for item in items):
            return True
    return False


# --- build_hook_install_plan / write_hook_install_plan / render_hook_install_preview ---


def test_build_hook_install_plan_project_scope_produces_expected_shape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_hook_install_plan("claude-code", str(repo), action="install")
    target = write_hook_install_plan(plan)

    assert plan.scope == "project"
    assert target == repo / ".claude" / "settings.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": HOOK_MATCHER,
                    "hooks": [
                        {
                            "type": "command",
                            "command": sys.executable,
                            "args": ["-m", "archex.integrations.hook"],
                        }
                    ],
                }
            ]
        }
    }


def test_build_hook_install_plan_user_scope_produces_expected_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    plan = build_hook_install_plan("claude-code", action="install")
    target = write_hook_install_plan(plan)

    assert plan.scope == "user"
    assert target == tmp_path / ".claude" / "settings.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    group = payload["hooks"]["PreToolUse"][0]
    assert group["matcher"] == HOOK_MATCHER
    assert group["hooks"][0]["args"] == ["-m", "archex.integrations.hook"]
    assert group["hooks"][0]["command"] == sys.executable


def test_write_hook_install_plan_idempotent_on_reinstall(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_hook_install_plan("claude-code", str(repo), action="install")
    target = write_hook_install_plan(plan)
    after_first = target.read_text(encoding="utf-8")
    write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action="install"))
    after_second = target.read_text(encoding="utf-8")

    assert after_first == after_second


def test_write_hook_install_plan_preserves_unrelated_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    _write_json(target, _seed_payload())

    write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action="install"))

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["otherTopLevelKey"] == "unrelated-value"
    assert payload["hooks"]["PostToolUse"] == _seed_payload()["hooks"]["PostToolUse"]
    pre_tool_use = payload["hooks"]["PreToolUse"]
    assert {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "echo bash-hook"}],
    } in pre_tool_use
    archex_groups = [g for g in pre_tool_use if _hook_group_has_archex_entry(g)]
    assert len(archex_groups) == 1
    assert archex_groups[0]["matcher"] == HOOK_MATCHER


def test_write_hook_install_plan_config_assertion_matcher_excludes_read(tmp_path: Path) -> None:
    """M19 acceptance criterion: the archex hook entry is reachable ONLY from a
    PreToolUse group matched exactly by Glob|Grep -- never from a group whose
    matcher includes Read, and never from any other hook event.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    seed = _seed_payload()
    # A pre-existing group that matches Read must never end up carrying the
    # archex entry.
    seed["hooks"]["PreToolUse"].append(
        {"matcher": "Read", "hooks": [{"type": "command", "command": "echo read-hook"}]}
    )
    _write_json(target, seed)

    write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action="install"))
    payload = json.loads(target.read_text(encoding="utf-8"))
    hooks_root = payload["hooks"]
    pre_tool_use = hooks_root["PreToolUse"]

    archex_groups = [g for g in pre_tool_use if _hook_group_has_archex_entry(g)]
    assert len(archex_groups) == 1, "exactly one PreToolUse group should carry the archex entry"
    archex_group = archex_groups[0]

    # (a) the archex-carrying group's matcher is exactly Glob|Grep, matching Grep
    # and Glob only, and nothing else.
    assert HOOK_MATCHER == "Glob|Grep"
    assert archex_group["matcher"] == HOOK_MATCHER

    # (b) no matcher that includes "Read" ever carries an archex entry.
    for group in pre_tool_use:
        if "Read" in str(group["matcher"]):
            assert not _hook_group_has_archex_entry(group)

    # (c) the archex entry lives only under PreToolUse -- no other hook event
    # (e.g. the pre-existing PostToolUse) gained an archex entry from install.
    for event_name, groups in hooks_root.items():
        if event_name == "PreToolUse":
            continue
        for group in groups:
            assert not _hook_group_has_archex_entry(group)


def test_write_hook_install_plan_remove_reduces_to_empty_object(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action="install"))
    assert target.exists()

    write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action="remove"))

    assert json.loads(target.read_text(encoding="utf-8")) == {}


def test_write_hook_install_plan_remove_preserves_unrelated_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    _write_json(target, _seed_payload())
    write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action="install"))

    write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action="remove"))

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["otherTopLevelKey"] == "unrelated-value"
    assert payload["hooks"]["PostToolUse"] == _seed_payload()["hooks"]["PostToolUse"]
    pre_tool_use = payload["hooks"]["PreToolUse"]
    assert len(pre_tool_use) == 1
    assert pre_tool_use[0]["matcher"] == "Bash"
    assert not any(_hook_group_has_archex_entry(g) for g in pre_tool_use)


def test_write_hook_install_plan_remove_missing_file_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"

    result_target = write_hook_install_plan(
        build_hook_install_plan("claude-code", str(repo), action="remove")
    )

    assert result_target == target
    assert not target.exists()
    assert not target.parent.exists()


def test_write_hook_install_plan_remove_without_archex_entry_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    _write_json(target, _seed_payload())
    before = target.read_text(encoding="utf-8")

    write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action="remove"))

    assert target.read_text(encoding="utf-8") == before


def test_render_hook_install_preview_install_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    plan = build_hook_install_plan("claude-code", str(repo), action="install")

    preview = render_hook_install_preview(plan)

    assert "Install" in preview
    assert not target.exists()


def test_render_hook_install_preview_remove_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action="install"))
    before = target.read_text(encoding="utf-8")

    preview = render_hook_install_preview(
        build_hook_install_plan("claude-code", str(repo), action="remove")
    )

    assert "Remove" in preview
    assert target.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("client", ["codex", "cursor", "opencode", "pi", "omp"])
def test_build_hook_install_plan_rejects_non_claude_code_clients(client: ClientName) -> None:
    with pytest.raises(ValueError) as exc_info:
        build_hook_install_plan(client, action="install")

    message = str(exc_info.value)
    assert "claude-code" in message
    assert "M19" in message


# --- CLI wiring: install-client --hooks / --remove-hooks ---


def test_cli_hooks_installs_and_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "claude-code", "--hooks"])

    assert result.exit_code == 0, result.output
    assert "Installed" in result.output
    target = tmp_path / ".claude" / "settings.json"
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["hooks"]["PreToolUse"][0]["matcher"] == HOOK_MATCHER


def test_cli_hooks_dry_run_previews_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "claude-code", "--hooks", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run." in result.output
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_cli_remove_hooks_removes_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CliRunner().invoke(cli, ["install-client", "claude-code", "--hooks"])

    result = CliRunner().invoke(cli, ["install-client", "claude-code", "--remove-hooks"])

    assert result.exit_code == 0, result.output
    assert "Removed" in result.output
    target = tmp_path / ".claude" / "settings.json"
    assert json.loads(target.read_text(encoding="utf-8")) == {}


def test_cli_hooks_and_remove_hooks_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "claude-code", "--hooks", "--remove-hooks"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_cli_hooks_rejects_non_claude_code_client_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "codex", "--hooks"])

    assert result.exit_code != 0
    assert "claude-code" in result.output
    assert "M19" in result.output
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".codex").exists()


def test_cli_plain_install_client_still_writes_mcp_config_not_hook_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "claude-code"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".claude.json").exists()
    assert not (tmp_path / ".claude" / "settings.json").exists()
