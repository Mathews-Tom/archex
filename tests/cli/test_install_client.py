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


def test_install_client_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from archex.client_setup import discover_clients

    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()

    # None of them exist initially
    discovered = discover_clients(source=repo)
    assert len(discovered) == 10

    assert all(not d.is_installed for d in discovered)
    assert discovered[0].client == "omp"
    assert discovered[0].scope == "user"
    assert "not found" in discovered[0].evidence

    # Create some configs
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text("", encoding="utf-8")
    (repo / ".mcp.json").write_text("", encoding="utf-8")

    discovered_with_some = discover_clients(source=repo)

    codex_user = next(d for d in discovered_with_some if d.client == "codex" and d.scope == "user")
    assert codex_user.is_installed
    assert "exists" in codex_user.evidence

    claude_project = next(
        d for d in discovered_with_some if d.client == "claude-code" and d.scope == "project"
    )
    assert claude_project.is_installed
    assert "exists" in claude_project.evidence

    claude_user = next(
        d for d in discovered_with_some if d.client == "claude-code" and d.scope == "user"
    )
    assert not claude_user.is_installed


def test_install_client_all_detected_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text("", encoding="utf-8")
    (repo / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")

    result = CliRunner().invoke(cli, ["install-client", "--all-detected", "--dry-run", str(repo)])

    assert result.exit_code == 0
    assert "Will write:" in result.output
    assert ".codex/config.toml: add [mcp_servers.archex]" in result.output
    assert ".mcp.json: add mcpServers.archex" in result.output

    # Ensure nothing was written
    assert (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8") == ""


def test_install_client_all_detected_yes_writes_detected_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    codex_config = tmp_path / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text('model = "gpt-5"\n', encoding="utf-8")
    claude_config = repo / ".mcp.json"
    claude_config.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")

    result = CliRunner().invoke(cli, ["install-client", "--all-detected", "--yes", str(repo)])

    assert result.exit_code == 0, result.output
    assert "Wrote codex config" in result.output
    assert "Wrote claude-code config" in result.output
    assert "[mcp_servers.archex]" in codex_config.read_text(encoding="utf-8")
    payload = json.loads(claude_config.read_text(encoding="utf-8"))
    assert payload["mcpServers"]["archex"] == {"command": "archex", "args": ["mcp"]}


def test_install_client_non_tty_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    # Not a TTY, no --yes, no --all-detected
    result = CliRunner().invoke(cli, ["install-client"])
    assert result.exit_code != 0
    assert "install-client is interactive by default, but stdin/stdout are not TTY" in result.output


def test_install_client_interactive_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text("", encoding="utf-8")

    # Simulate TTY and 'yes' response to both prompts
    from unittest.mock import patch

    runner = CliRunner()
    with patch("archex.cli.install_client_cmd._is_interactive", return_value=True):
        result = runner.invoke(cli, ["install-client"], input="y\ny\n")

    assert result.exit_code == 0
    assert "Detected possible clients:" in result.output
    assert "codex" in result.output
    assert "Install archex MCP registration for" in result.output
    assert "Wrote codex config" in result.output

    # Confirm it actually wrote
    content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert "[mcp_servers.archex]" in content


def test_install_client_blocks_missing_mcp_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    from importlib.machinery import ModuleSpec

    def missing_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        return None

    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(importlib.util, "find_spec", missing_spec)

    result = CliRunner().invoke(cli, ["install-client", "codex", str(repo)])

    assert result.exit_code != 0
    assert "Cannot register archex MCP" in result.output
    assert "uv tool install --force 'archex[mcp]'" in result.output
    assert "--allow-missing-mcp" in result.output
    assert (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8") == ""


def test_install_client_allow_missing_mcp_runtime_writes_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util
    from importlib.machinery import ModuleSpec

    def missing_spec(name: str, package: str | None = None) -> ModuleSpec | None:
        return None

    monkeypatch.setenv("HOME", str(tmp_path))
    codex_config = tmp_path / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    codex_config.write_text("", encoding="utf-8")
    monkeypatch.setattr(importlib.util, "find_spec", missing_spec)

    result = CliRunner().invoke(cli, ["install-client", "codex", "--allow-missing-mcp"])

    assert result.exit_code == 0, result.output
    assert "[mcp_servers.archex]" in codex_config.read_text(encoding="utf-8")


def test_install_client_discovers_agent_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text("", encoding="utf-8")
    agent_file = repo / "CLAUDE.md"
    agent_file.write_text("Hello\n", encoding="utf-8")

    runner = CliRunner()
    from unittest.mock import patch

    with patch("archex.cli.install_client_cmd._is_interactive", return_value=True):
        # Answer 'y' to install clients, 'y' to append guidance
        result = runner.invoke(cli, ["install-client", str(repo)], input="y\ny\ny\n")

    assert result.exit_code == 0
    assert "Append archex MCP guidance to detected agent instruction files?" in result.output
    assert str(agent_file) in result.output

    content = agent_file.read_text(encoding="utf-8")
    assert "<!-- archex:mcp-guidance start -->" in content


# ---------------------------------------------------------------------------
# --tool-scope tests (M11)
# ---------------------------------------------------------------------------


def test_install_client_tool_scope_default_omits_tools_arg(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    result = CliRunner().invoke(cli, ["install-client", "claude-code", str(repo), "--dry-run"])
    assert result.exit_code == 0
    assert '"args": [\n        "mcp"\n      ]' in result.output.replace("\r\n", "\n")


def test_install_client_tool_scope_writes_tools_arg(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    result = CliRunner().invoke(
        cli,
        ["install-client", "claude-code", str(repo), "--tool-scope", "core"],
    )
    assert result.exit_code == 0
    payload = json.loads((repo / ".mcp.json").read_text(encoding="utf-8"))
    assert payload["mcpServers"]["archex"]["args"] == ["mcp", "--tools", "core"]


def test_install_client_tool_scope_codex_toml_embeds_tools_arg(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    result = CliRunner().invoke(
        cli,
        ["install-client", "codex", str(repo), "--tool-scope", "graph"],
    )
    assert result.exit_code == 0
    content = (repo / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert 'args = ["mcp", "--tools", "graph"]' in content


def test_install_client_tool_scope_unknown_tool_name_fails_cleanly(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    result = CliRunner().invoke(
        cli,
        ["install-client", "claude-code", str(repo), "--dry-run", "--tool-scope", "not_a_tool"],
    )
    assert result.exit_code != 0
    assert "Unknown MCP tool name" in result.output
    assert not (repo / ".mcp.json").exists()
