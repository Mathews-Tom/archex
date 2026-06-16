from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.cli.main import cli

if TYPE_CHECKING:
    import pytest


def test_install_client_preview_claude_code_project(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["install-client", "claude-code", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f"Target: {tmp_path / '.mcp.json'}" in result.output
    assert '"mcpServers"' in result.output
    assert '"command": "archex"' in result.output
    assert "Preview only." in result.output


def test_install_client_write_claude_code_project(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["install-client", "claude-code", str(tmp_path), "--write"],
    )

    assert result.exit_code == 0, result.output
    config_path = tmp_path / ".mcp.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["mcpServers"]["archex"]["args"] == ["mcp"]


def test_install_client_refuses_existing_entry(tmp_path: Path) -> None:
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"archex": {"command": "archex", "args": ["mcp"]}}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli,
        ["install-client", "claude-code", str(tmp_path), "--write"],
    )

    assert result.exit_code != 0
    assert "already configured" in result.output


def test_install_client_writes_codex_project_toml(tmp_path: Path) -> None:
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir(parents=True)
    config_path = codex_dir / "config.toml"
    config_path.write_text('default_model = "gpt-5"\n', encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        ["install-client", "codex", str(tmp_path), "--write"],
    )

    assert result.exit_code == 0, result.output
    content = config_path.read_text(encoding="utf-8")
    assert 'default_model = "gpt-5"' in content
    assert "[mcp_servers.archex]" in content
    assert 'args = ["mcp"]' in content


def test_install_client_pi_user_scope_uses_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "pi", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f"Target: {tmp_path / '.pi' / 'agent' / 'mcp.json'}" in result.output


def test_install_client_pi_rejects_project_scope(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["install-client", "pi", str(tmp_path), "--scope", "project"],
    )

    assert result.exit_code != 0
    assert "supports only --scope user" in result.output
