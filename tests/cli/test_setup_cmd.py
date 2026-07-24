import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from archex.cli import setup_cmd
from archex.cli.main import cli


def test_setup_dry_run(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    runner = CliRunner()
    result = runner.invoke(cli, ["setup", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "--- Setup Preflight ---" in result.output
    assert "has_dot_archex: False" in result.output


def test_setup_json_format(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    runner = CliRunner()
    result = runner.invoke(cli, ["setup", str(tmp_path), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "preflight" in data
    assert "planned_actions" in data
    assert data["preflight"]["has_dot_archex"] is False
    assert data["preflight"]["has_index"] is False


def test_setup_non_tty_error(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    runner = CliRunner()
    # CliRunner simulates non-TTY
    result = runner.invoke(cli, ["setup", str(tmp_path)])
    assert result.exit_code == 1
    assert "setup is interactive by default" in result.output


def test_setup_init_index(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    runner = CliRunner()
    result = runner.invoke(cli, ["setup", str(tmp_path), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "planned_actions" in data
    actions = data["planned_actions"]
    init_action = next(a for a in actions if a["type"] == "init")
    assert init_action["status"] == "planned"

    index_action = next(a for a in actions if a["type"] == "index")
    assert index_action["status"] == "planned"


def test_setup_yes_executes(tmp_path: Path) -> None:
    subprocess.check_call(["git", "init", str(tmp_path)])
    # Create at least one file so indexing works
    (tmp_path / "hello.py").write_text("print('hello')")
    runner = CliRunner()
    result = runner.invoke(cli, ["setup", str(tmp_path), "--yes"])
    assert result.exit_code == 0
    assert "Executing Setup" in result.output
    assert "init: executed" in result.output
    assert "index: executed" in result.output


def _mcp_runtime_available_stub(repo_root: Path) -> bool:
    """Deterministic stand-in for archex.doctor.mcp_runtime_available in tests."""
    return True


def test_setup_yes_without_clients_skips_client_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", str(repo)])
    (repo / "hello.py").write_text("print('hello')")
    codex_config = repo / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text('existing = "value"\n', encoding="utf-8")
    original_codex_config = codex_config.read_text(encoding="utf-8")
    monkeypatch.setattr(setup_cmd, "mcp_runtime_available", _mcp_runtime_available_stub)

    result = CliRunner().invoke(cli, ["setup", str(repo), "--yes"])

    assert result.exit_code == 0
    assert "init: executed" in result.output
    assert "index: executed" in result.output
    assert "client_install: skipped_by_flag" in result.output
    assert codex_config.read_text(encoding="utf-8") == original_codex_config


def test_setup_yes_with_clients_writes_codex_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", str(repo)])
    (repo / "hello.py").write_text("print('hello')")
    codex_config = repo / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text('existing = "value"\n', encoding="utf-8")
    monkeypatch.setattr(setup_cmd, "mcp_runtime_available", _mcp_runtime_available_stub)

    result = CliRunner().invoke(cli, ["setup", str(repo), "--yes", "--clients"])

    assert result.exit_code == 0
    assert "codex: executed" in result.output
    written = codex_config.read_text(encoding="utf-8")
    assert 'existing = "value"' in written
    assert "[mcp_servers.archex]" in written


def test_setup_yes_without_clients_does_not_append_agent_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", str(repo)])
    (repo / "hello.py").write_text("print('hello')")
    agents_file = repo / "AGENTS.md"
    agents_file.write_text("# Existing project notes\n", encoding="utf-8")
    original_agents_file = agents_file.read_text(encoding="utf-8")
    monkeypatch.setattr(setup_cmd, "mcp_runtime_available", _mcp_runtime_available_stub)

    result = CliRunner().invoke(cli, ["setup", str(repo), "--yes"])

    assert result.exit_code == 0
    assert agents_file.read_text(encoding="utf-8") == original_agents_file


def test_setup_yes_with_clients_appends_agent_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", str(repo)])
    (repo / "hello.py").write_text("print('hello')")
    agents_file = repo / "AGENTS.md"
    original_content = "# Existing project notes\n"
    agents_file.write_text(original_content, encoding="utf-8")
    monkeypatch.setattr(setup_cmd, "mcp_runtime_available", _mcp_runtime_available_stub)

    result = CliRunner().invoke(cli, ["setup", str(repo), "--yes", "--clients"])

    assert result.exit_code == 0
    updated = agents_file.read_text(encoding="utf-8")
    assert updated.startswith(original_content)
    assert "<!-- archex:mcp-guidance start -->" in updated
    assert updated != original_content


def test_setup_yes_with_clients_and_tool_scope_writes_scoped_codex_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", str(repo)])
    (repo / "hello.py").write_text("print('hello')")
    codex_config = repo / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text("", encoding="utf-8")
    monkeypatch.setattr(setup_cmd, "mcp_runtime_available", _mcp_runtime_available_stub)

    result = CliRunner().invoke(
        cli, ["setup", str(repo), "--yes", "--clients", "--tool-scope", "core"]
    )

    assert result.exit_code == 0
    written = codex_config.read_text(encoding="utf-8")
    assert 'args = ["mcp", "--tools", "core"]' in written


def test_setup_unknown_tool_scope_fails_cleanly_not_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: setup --tool-scope must validate up front like
    install-client/mcp do, not let resolve_tool_scope's ValueError escape
    uncaught from deep inside build_discovered_install_plans."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", str(repo)])
    (repo / "hello.py").write_text("print('hello')")
    codex_config = repo / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text("", encoding="utf-8")
    monkeypatch.setattr(setup_cmd, "mcp_runtime_available", _mcp_runtime_available_stub)

    result = CliRunner().invoke(
        cli, ["setup", str(repo), "--yes", "--clients", "--tool-scope", "not_a_real_tool"]
    )

    assert result.exit_code != 0
    assert "Unknown MCP tool name" in result.output
    assert "Traceback" not in result.output
    assert codex_config.read_text(encoding="utf-8") == ""
