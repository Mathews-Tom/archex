"""`install-client --statusline` contract (R23).

`statusLine` is a scalar settings key, not a matcher group, so unlike the hook
installers there is no way to coexist with another status line. These tests
pin the consequences: a foreign status line is refused rather than replaced,
removal only touches archex's own entry, and the surface installs and removes
independently of the three hook surfaces that share `settings.json`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.cli.main import cli
from archex.client_setup import (
    STATUSLINE_SCRIPT_FILENAME,
    STATUSLINE_SCRIPT_MARKER,
    build_hook_install_plan,
    build_statusline_install_plan,
    render_statusline_install_preview,
    statusline_interpreter,
    write_hook_install_plan,
    write_statusline_install_plan,
)

if TYPE_CHECKING:
    import pytest


def _settings(repo: Path) -> Path:
    return repo / ".claude" / "settings.json"


def _script(repo: Path) -> Path:
    return repo / ".claude" / STATUSLINE_SCRIPT_FILENAME


def _install(repo: Path) -> Path:
    plan = build_statusline_install_plan("claude-code", repo, scope="project", action="install")
    return write_statusline_install_plan(plan)


def _remove(repo: Path) -> Path:
    plan = build_statusline_install_plan("claude-code", repo, scope="project", action="remove")
    return write_statusline_install_plan(plan)


def test_install_writes_the_renderer_and_an_owned_statusline_entry(tmp_path: Path) -> None:
    target = _install(tmp_path)

    payload = json.loads(target.read_text(encoding="utf-8"))
    entry = payload["statusLine"]
    assert entry["type"] == "command"
    assert STATUSLINE_SCRIPT_FILENAME in entry["command"]
    assert str(_script(tmp_path)) in entry["command"]
    script = _script(tmp_path)
    assert script.exists()
    assert STATUSLINE_SCRIPT_MARKER in script.read_text(encoding="utf-8")
    assert script.stat().st_mode & 0o111


def test_installed_command_quotes_a_path_containing_spaces(tmp_path: Path) -> None:
    repo = tmp_path / "my repo"
    repo.mkdir()

    target = _install(repo)

    entry = json.loads(target.read_text(encoding="utf-8"))["statusLine"]
    script = repo / ".claude" / STATUSLINE_SCRIPT_FILENAME
    assert entry["command"] == f'{statusline_interpreter()} "{script}"'


def test_installed_interpreter_can_read_a_clock_when_the_host_has_one() -> None:
    interpreter = statusline_interpreter()

    if interpreter != "sh":
        probe = subprocess.run(
            [interpreter, "-c", "zmodload zsh/datetime 2>/dev/null; echo ${EPOCHSECONDS:-}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert probe.stdout.strip().isdigit(), "a chosen interpreter must expose a clock"


def test_a_refused_install_leaves_no_renderer_script(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"statusLine": {"type": "command", "command": "~/.claude/mine.sh"}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli, ["install-client", "claude-code", str(tmp_path), "--scope", "project", "--statusline"]
    )

    assert result.exit_code != 0
    assert not _script(tmp_path).exists(), "a refused install must write nothing"


def test_reinstall_is_an_idempotent_no_op(tmp_path: Path) -> None:
    target = _install(tmp_path)
    first = target.read_text(encoding="utf-8")

    _install(tmp_path)

    assert target.read_text(encoding="utf-8") == first


def test_install_preserves_unrelated_settings(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"model": "opus", "permissions": {"allow": ["Read"]}}), encoding="utf-8"
    )

    _install(tmp_path)

    payload = json.loads(settings.read_text(encoding="utf-8"))
    assert payload["model"] == "opus"
    assert payload["permissions"] == {"allow": ["Read"]}
    assert "statusLine" in payload


def test_a_foreign_statusline_is_refused_rather_than_replaced(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.parent.mkdir(parents=True)
    foreign = {"type": "command", "command": "~/.claude/my-own-statusline.sh"}
    settings.write_text(json.dumps({"statusLine": foreign}), encoding="utf-8")

    result = CliRunner().invoke(
        cli, ["install-client", "claude-code", str(tmp_path), "--scope", "project", "--statusline"]
    )

    assert result.exit_code != 0
    assert "will not overwrite" in result.output
    assert json.loads(settings.read_text(encoding="utf-8"))["statusLine"] == foreign


def test_remove_deletes_the_entry_and_the_renderer(tmp_path: Path) -> None:
    _install(tmp_path)

    target = _remove(tmp_path)

    assert "statusLine" not in json.loads(target.read_text(encoding="utf-8"))
    assert not _script(tmp_path).exists()


def test_remove_leaves_a_foreign_statusline_and_script_alone(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.parent.mkdir(parents=True)
    foreign = {"type": "command", "command": "~/.claude/my-own-statusline.sh"}
    settings.write_text(json.dumps({"statusLine": foreign}), encoding="utf-8")
    unowned_script = _script(tmp_path)
    unowned_script.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")

    _remove(tmp_path)

    assert json.loads(settings.read_text(encoding="utf-8"))["statusLine"] == foreign
    assert unowned_script.read_text(encoding="utf-8") == "#!/bin/sh\necho mine\n"


def test_remove_without_an_install_is_a_no_op(tmp_path: Path) -> None:
    target = _remove(tmp_path)

    assert not target.exists()
    assert not _script(tmp_path).exists()


def test_dry_run_previews_the_entry_without_writing(tmp_path: Path) -> None:
    plan = build_statusline_install_plan("claude-code", tmp_path, scope="project", action="install")

    preview = render_statusline_install_preview(plan)

    assert "Dry run" in preview
    assert STATUSLINE_SCRIPT_FILENAME in preview
    assert not _settings(tmp_path).exists()
    assert not _script(tmp_path).exists()


def test_statusline_and_search_hook_install_independently(tmp_path: Path) -> None:
    write_hook_install_plan(
        build_hook_install_plan("claude-code", tmp_path, scope="project", action="install")
    )

    _install(tmp_path)
    payload = json.loads(_settings(tmp_path).read_text(encoding="utf-8"))
    assert "statusLine" in payload
    assert payload["hooks"]["PreToolUse"]

    _remove(tmp_path)
    after = json.loads(_settings(tmp_path).read_text(encoding="utf-8"))
    assert "statusLine" not in after
    assert after["hooks"]["PreToolUse"], "removing the status line must not disturb hooks"


def test_clients_without_a_persistent_status_surface_are_refused(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli, ["install-client", "codex", str(tmp_path), "--scope", "project", "--statusline"]
    )

    assert result.exit_code != 0
    assert "no persistent status surface" in result.output


def test_statusline_cannot_be_combined_with_another_surface(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "install-client",
            "claude-code",
            str(tmp_path),
            "--scope",
            "project",
            "--statusline",
            "--hooks",
        ],
    )

    assert result.exit_code != 0
    assert "separately" in result.output


def test_install_and_remove_are_mutually_exclusive(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "install-client",
            "claude-code",
            str(tmp_path),
            "--scope",
            "project",
            "--statusline",
            "--remove-statusline",
        ],
    )

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_cli_install_reports_both_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        cli, ["install-client", "claude-code", ".", "--scope", "project", "--statusline"]
    )

    assert result.exit_code == 0, result.output
    assert "Installed archex status surface for claude-code" in result.output
    assert STATUSLINE_SCRIPT_FILENAME in result.output
