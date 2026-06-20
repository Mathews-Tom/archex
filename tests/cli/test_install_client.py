from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.cli.main import cli

if TYPE_CHECKING:
    import pytest


def test_install_client_default_scope_is_global(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "claude-code", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Scope: user" in result.output
    assert f"Target: {tmp_path / '.claude.json'}" in result.output
    assert '"command": "archex"' in result.output


def test_install_client_source_path_selects_project_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()

    result = CliRunner().invoke(cli, ["install-client", "claude-code", str(repo), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Scope: project" in result.output
    assert f"Target: {repo / '.mcp.json'}" in result.output


def test_install_client_scope_project_flag_selects_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    result = CliRunner().invoke(
        cli,
        ["install-client", "claude-code", str(tmp_path), "--scope", "project", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Scope: project" in result.output
    assert f"Target: {tmp_path / '.mcp.json'}" in result.output


def test_install_client_dry_run_makes_no_filesystem_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    result = CliRunner().invoke(cli, ["install-client", "claude-code", str(repo), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run." in result.output
    assert not (repo / ".mcp.json").exists()


def test_install_client_writes_by_default_non_destructive(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = repo / ".mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"other": {"command": "other"}}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["install-client", "claude-code", str(repo)])

    assert result.exit_code == 0, result.output
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["mcpServers"]["archex"]["args"] == ["mcp"]
    assert payload["mcpServers"]["other"] == {"command": "other"}


def test_install_client_idempotent_on_identical_entry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = repo / ".mcp.json"

    first = CliRunner().invoke(cli, ["install-client", "claude-code", str(repo)])
    after_first = config_path.read_text(encoding="utf-8")
    second = CliRunner().invoke(cli, ["install-client", "claude-code", str(repo)])
    after_second = config_path.read_text(encoding="utf-8")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert after_first == after_second
    assert json.loads(after_second)["mcpServers"]["archex"]["args"] == ["mcp"]


def test_install_client_refuses_conflicting_entry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config_path = repo / ".mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"archex": {"command": "other", "args": []}}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["install-client", "claude-code", str(repo)])

    assert result.exit_code != 0
    assert "already configured" in result.output


def test_install_client_writes_codex_project_toml(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    codex_dir = repo / ".codex"
    codex_dir.mkdir(parents=True)
    config_path = codex_dir / "config.toml"
    config_path.write_text('default_model = "gpt-5"\n', encoding="utf-8")

    result = CliRunner().invoke(cli, ["install-client", "codex", str(repo)])

    assert result.exit_code == 0, result.output
    content = config_path.read_text(encoding="utf-8")
    assert 'default_model = "gpt-5"' in content
    assert "[mcp_servers.archex]" in content
    assert 'args = ["mcp"]' in content

    second = CliRunner().invoke(cli, ["install-client", "codex", str(repo)])
    assert second.exit_code == 0, second.output
    assert config_path.read_text(encoding="utf-8") == content


def test_install_client_refuses_conflicting_codex_entry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    codex_dir = repo / ".codex"
    codex_dir.mkdir(parents=True)
    config_path = codex_dir / "config.toml"
    config_path.write_text(
        '[mcp_servers.archex]\ncommand = "other"\nargs = []\n',
        encoding="utf-8",
    )
    before = config_path.read_text(encoding="utf-8")

    result = CliRunner().invoke(cli, ["install-client", "codex", str(repo)])

    assert result.exit_code != 0
    assert "already configured" in result.output
    assert config_path.read_text(encoding="utf-8") == before


def test_install_client_default_global_write_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".claude.json"

    first = CliRunner().invoke(cli, ["install-client", "claude-code"])
    after_first = target.read_text(encoding="utf-8")
    second = CliRunner().invoke(cli, ["install-client", "claude-code"])
    after_second = target.read_text(encoding="utf-8")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(after_first)["mcpServers"]["archex"]["args"] == ["mcp"]
    assert after_first == after_second


def test_install_client_pi_user_scope_uses_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "pi", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Scope: user" in result.output
    assert f"Target: {tmp_path / '.pi' / 'agent' / 'mcp.json'}" in result.output


def test_install_client_pi_rejects_project_scope(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["install-client", "pi", str(tmp_path), "--scope", "project"],
    )

    assert result.exit_code != 0
    assert "supports only --scope user" in result.output


def test_install_client_omp_user_scope_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "omp", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Scope: user" in result.output
    assert f"Target: {tmp_path / '.omp' / 'agent' / 'mcp.json'}" in result.output
    assert "mcp-schema.json" in result.output


def test_install_client_omp_writes_expected_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".omp" / "agent" / "mcp.json"
    runner = CliRunner()

    preview = runner.invoke(cli, ["install-client", "omp", "--dry-run"])
    written = runner.invoke(cli, ["install-client", "omp"])
    after_first = target.read_text(encoding="utf-8")
    rerun = runner.invoke(cli, ["install-client", "omp"])
    after_second = target.read_text(encoding="utf-8")

    assert preview.exit_code == 0, preview.output
    assert written.exit_code == 0, written.output
    assert rerun.exit_code == 0, rerun.output
    previewed_payload = json.loads(preview.output[preview.output.index("{") :])
    written_payload = json.loads(after_first)
    # The written file must agree with what --dry-run previewed (single source of truth).
    assert written_payload["mcpServers"]["archex"] == previewed_payload["mcpServers"]["archex"]
    assert written_payload["$schema"] == previewed_payload["$schema"]
    # The payload is the standard archex stdio entry plus the oh-my-pi schema.
    assert written_payload["mcpServers"]["archex"] == {"command": "archex", "args": ["mcp"]}
    assert "oh-my-pi" in written_payload["$schema"]
    # Writing is idempotent.
    assert after_first == after_second


def test_install_client_omp_rejects_project_scope(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["install-client", "omp", str(tmp_path), "--scope", "project"],
    )

    assert result.exit_code != 0
    assert "supports only --scope user" in result.output


def test_install_client_opencode_write_injects_schema(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "opencode.json"

    result = CliRunner().invoke(cli, ["install-client", "opencode", str(repo)])
    after_first = target.read_text(encoding="utf-8")
    second = CliRunner().invoke(cli, ["install-client", "opencode", str(repo)])
    after_second = target.read_text(encoding="utf-8")

    assert result.exit_code == 0, result.output
    assert second.exit_code == 0, second.output
    payload = json.loads(after_first)
    assert payload["$schema"] == "https://opencode.ai/config.json"
    assert payload["mcp"]["archex"]["command"] == ["archex", "mcp"]
    assert after_first == after_second


def test_install_client_omp_preserves_existing_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".omp" / "agent" / "mcp.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps({"$schema": "https://example.com/custom.json", "mcpServers": {}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["install-client", "omp"])

    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["$schema"] == "https://example.com/custom.json"
    assert payload["mcpServers"]["archex"] == {"command": "archex", "args": ["mcp"]}


_MCP_TOOLS = ("query_repo", "scout_repo", "analyze_repo", "search_symbols", "get_symbol")
_GUIDANCE_MARKER = "<!-- archex:mcp-guidance start -->"


def test_install_client_agent_file_preview_under_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    agent_file = tmp_path / "AGENTS.md"

    result = CliRunner().invoke(
        cli,
        ["install-client", "claude-code", "--agent-file", str(agent_file), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    for tool in _MCP_TOOLS:
        assert tool in result.output
    assert f"Agent file: {agent_file}" in result.output
    assert not agent_file.exists()
    assert not (tmp_path / ".claude.json").exists()


def test_install_client_agent_file_appends_once_non_destructive(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    agent_file = repo / "CLAUDE.md"
    agent_file.write_text("# Project\n\nExisting guidance.\n", encoding="utf-8")

    first = CliRunner().invoke(
        cli, ["install-client", "claude-code", str(repo), "--agent-file", str(agent_file)]
    )
    after_first = agent_file.read_text(encoding="utf-8")
    second = CliRunner().invoke(
        cli, ["install-client", "claude-code", str(repo), "--agent-file", str(agent_file)]
    )
    after_second = agent_file.read_text(encoding="utf-8")

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert "Existing guidance." in after_first
    assert after_first.count(_GUIDANCE_MARKER) == 1
    assert after_second == after_first
    assert "query_repo" in after_first
    assert "Appended archex MCP guidance" in first.output
    assert "already present" in second.output


def test_install_client_agent_file_global_write_creates_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    agent_file = tmp_path / ".claude" / "CLAUDE.md"

    result = CliRunner().invoke(
        cli, ["install-client", "claude-code", "--agent-file", str(agent_file)]
    )

    assert result.exit_code == 0, result.output
    content = agent_file.read_text(encoding="utf-8")
    assert _GUIDANCE_MARKER in content
    for tool in _MCP_TOOLS:
        assert tool in content


def test_install_client_agent_file_preview_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    agent_file = repo / "AGENTS.md"
    seeded = f"# Repo\n\n{_GUIDANCE_MARKER}\nseeded block\n"
    agent_file.write_text(seeded, encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        ["install-client", "claude-code", str(repo), "--agent-file", str(agent_file), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "already present; no change." in result.output
    assert agent_file.read_text(encoding="utf-8") == seeded


def test_install_client_agent_file_directory_errors_cleanly(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    agent_path = repo / "AGENTS.md"
    agent_path.mkdir()  # a directory where an agent file is expected

    result = CliRunner().invoke(
        cli,
        ["install-client", "claude-code", str(repo), "--agent-file", str(agent_path)],
    )

    assert result.exit_code != 0
    assert "Error" in result.output
    assert "Traceback" not in result.output
