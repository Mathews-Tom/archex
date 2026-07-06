from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from click.testing import CliRunner

from archex.cli.main import cli
from archex.client_setup import (
    CodexHookInstallPlan,
    TsHookInstallPlan,
    build_hook_install_plan,
    render_hook_install_preview,
    write_hook_install_plan,
)
from archex.integrations.codex_hook import HOOK_MATCHER as CODEX_HOOK_MATCHER
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


@pytest.mark.parametrize("client", ["cursor"])
def test_build_hook_install_plan_rejects_unsupported_clients(client: ClientName) -> None:
    with pytest.raises(ValueError) as exc_info:
        build_hook_install_plan(client, action="install")

    message = str(exc_info.value)
    assert "claude-code" in message
    assert "M19" in message
    assert "M21" in message
    assert "M22" in message


# --- omp TS hook module (M20) ---


def test_build_hook_install_plan_omp_project_scope_produces_ts_module_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_hook_install_plan("omp", str(repo), action="install")

    assert isinstance(plan, TsHookInstallPlan)
    assert plan.scope == "project"
    assert plan.target_path == repo / ".omp" / "extensions" / "archex-hook.ts"
    assert plan.module_content  # non-empty


def test_build_hook_install_plan_omp_user_scope_produces_ts_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    plan = build_hook_install_plan("omp", action="install")

    assert isinstance(plan, TsHookInstallPlan)
    assert plan.scope == "user"
    assert plan.target_path == tmp_path / ".omp" / "agent" / "extensions" / "archex-hook.ts"


def test_write_hook_install_plan_omp_writes_ts_module_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("omp", str(repo), action="install")

    target = write_hook_install_plan(plan)

    assert isinstance(plan, TsHookInstallPlan)
    assert target == plan.target_path
    assert target.read_text(encoding="utf-8") == plan.module_content


def test_write_hook_install_plan_omp_idempotent_on_reinstall(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = write_hook_install_plan(build_hook_install_plan("omp", str(repo), action="install"))
    after_first = target.read_text(encoding="utf-8")
    mtime_first = target.stat().st_mtime_ns

    write_hook_install_plan(build_hook_install_plan("omp", str(repo), action="install"))

    assert target.read_text(encoding="utf-8") == after_first
    # An identical reinstall is a true no-op: it never rewrites the file.
    assert target.stat().st_mtime_ns == mtime_first


def test_write_hook_install_plan_omp_remove_deletes_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = write_hook_install_plan(build_hook_install_plan("omp", str(repo), action="install"))
    assert target.exists()

    write_hook_install_plan(build_hook_install_plan("omp", str(repo), action="remove"))

    assert not target.exists()


def test_write_hook_install_plan_omp_remove_missing_file_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    result_target = write_hook_install_plan(
        build_hook_install_plan("omp", str(repo), action="remove")
    )

    assert not result_target.exists()


def test_render_hook_install_preview_omp_install_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("omp", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)

    preview = render_hook_install_preview(plan)

    assert "Install" in preview
    assert plan.module_content in preview
    assert not plan.target_path.exists()


def test_render_hook_install_preview_omp_remove_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = write_hook_install_plan(build_hook_install_plan("omp", str(repo), action="install"))
    before = target.read_text(encoding="utf-8")

    preview = render_hook_install_preview(
        build_hook_install_plan("omp", str(repo), action="remove")
    )

    assert "Remove" in preview
    assert target.read_text(encoding="utf-8") == before


def _query_field_keys(module_content: str) -> set[str]:
    """Extract the ``ARCHEX_QUERY_FIELDS`` table's keys from generated TS source.

    This table is the module's *only* tool-name dispatch mechanism (there is no
    if/else chain on ``toolName``): a tool whose name is absent from this table
    is never touched. Asserting its key set is therefore a precise, structural
    way to prove ``read`` is never handled -- stronger than a raw substring
    search, which would false-positive on the module's own prose comments
    describing (in backtick-quoted code snippets) the exact branch that must
    never exist.
    """
    match = re.search(
        r"ARCHEX_QUERY_FIELDS: Readonly<Record<string, ToolQueryMapping>> = \{(.*?)\n\};",
        module_content,
        re.DOTALL,
    )
    assert match is not None, "ARCHEX_QUERY_FIELDS table not found in generated module"
    return set(re.findall(r"^\s*(\w+):\s*\{", match.group(1), re.MULTILINE))


def test_omp_ts_hook_module_query_field_table_excludes_read(tmp_path: Path) -> None:
    """M20 acceptance criterion: the installed hook never registers a handler
    branch for ``read`` -- proven structurally via the dispatch table's keys,
    not by inspection.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("omp", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)

    keys = _query_field_keys(plan.module_content)

    assert "read" not in keys
    assert keys == {"grep", "glob", "find"}


def test_omp_ts_hook_module_bakes_in_active_python_interpreter(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("omp", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)

    assert json.dumps(sys.executable) in plan.module_content
    assert '["-m", "archex.integrations.hook"]' in plan.module_content


def test_omp_ts_hook_module_registers_exactly_one_unconditional_tool_result_handler(
    tmp_path: Path,
) -> None:
    """Subagent-dispatch coverage (M20 risk note): oh-my-pi's ``tool_result``
    event carries no subagent/session discriminator field, and this module
    registers exactly one unconditional handler with no such check -- so a
    subagent-issued grep/glob call is handled identically to a top-level one,
    the same way every other ``tool_result`` event is. Verified structurally
    (no conditional gating the registration or the dispatch) rather than via a
    live nested-subagent session, which is out of reach for this test suite.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("omp", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)
    content = plan.module_content

    assert "subagent" not in content.lower()
    factory_match = re.search(
        r"export default function archexHook\(pi: HookHost\): void \{(.*)\}\s*$",
        content,
        re.DOTALL,
    )
    assert factory_match is not None
    factory_body = factory_match.group(1)
    assert factory_body.count("pi.on(") == 1


# --- pi TS hook module (M20) ---
#
# Pi's extension directories (confirmed against the installed
# @mariozechner/pi-coding-agent 0.68.1) and its `pi.on("tool_result", ...)`
# partial-patch contract (`{ content, details, isError }`) are identical to
# oh-my-pi's -- this reuses the exact same generated module and dispatch
# logic verified above for omp; only the installer's target path differs.


def test_build_hook_install_plan_pi_project_scope_produces_ts_module_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_hook_install_plan("pi", str(repo), action="install")

    assert isinstance(plan, TsHookInstallPlan)
    assert plan.scope == "project"
    assert plan.target_path == repo / ".pi" / "extensions" / "archex-hook.ts"
    assert plan.module_content  # non-empty


def test_build_hook_install_plan_pi_user_scope_produces_ts_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    plan = build_hook_install_plan("pi", action="install")

    assert isinstance(plan, TsHookInstallPlan)
    assert plan.scope == "user"
    assert plan.target_path == tmp_path / ".pi" / "agent" / "extensions" / "archex-hook.ts"


def test_write_hook_install_plan_pi_writes_ts_module_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("pi", str(repo), action="install")

    target = write_hook_install_plan(plan)

    assert isinstance(plan, TsHookInstallPlan)
    assert target == plan.target_path
    assert target.read_text(encoding="utf-8") == plan.module_content


def test_write_hook_install_plan_pi_and_omp_share_identical_module_content(
    tmp_path: Path,
) -> None:
    """PR-2 finding: Pi's `tool_result` contract matches oh-my-pi's exactly
    (same event shape, same partial-patch return contract). The generated
    module already lists both hosts' tool names (`glob` for oh-my-pi, `find`
    for Pi) in one dispatch table, so the *identical* file is reused for Pi
    -- no Pi-specific module variant exists.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    omp_plan = build_hook_install_plan("omp", str(repo), action="install")
    pi_plan = build_hook_install_plan("pi", str(repo), action="install")

    assert isinstance(omp_plan, TsHookInstallPlan)
    assert isinstance(pi_plan, TsHookInstallPlan)
    assert omp_plan.module_content == pi_plan.module_content
    assert omp_plan.target_path != pi_plan.target_path


def test_write_hook_install_plan_pi_remove_deletes_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = write_hook_install_plan(build_hook_install_plan("pi", str(repo), action="install"))
    assert target.exists()

    write_hook_install_plan(build_hook_install_plan("pi", str(repo), action="remove"))

    assert not target.exists()


def test_render_hook_install_preview_pi_install_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("pi", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)

    preview = render_hook_install_preview(plan)

    assert "Install" in preview
    assert not plan.target_path.exists()


# --- OpenCode `tool.execute.after` plugin (M22) ---
#
# Structurally different from the omp/pi `tool_result` module: OpenCode's
# hook contract is `(input, output) => Promise<void>` -- it mutates
# `output.output` in place rather than returning a patch object, and its
# dispatch table (`ARCHEX_AUGMENTED_TOOLS`) is keyed directly on OpenCode's
# own native tool ids (`grep`, `glob`), not a `{claudeToolName, field}`
# translation record, since both tools already carry their query in a field
# named `pattern`. See `_OPENCODE_HOOK_MODULE_TEMPLATE` in `client_setup.py`.


def test_build_hook_install_plan_opencode_project_scope_produces_ts_module_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("opencode", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)
    assert plan.scope == "project"
    assert plan.target_path == repo / ".opencode" / "plugins" / "archex-hook.ts"
    assert plan.module_content  # non-empty


def test_build_hook_install_plan_opencode_user_scope_produces_ts_module_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plan = build_hook_install_plan("opencode", action="install")
    assert isinstance(plan, TsHookInstallPlan)
    assert plan.scope == "user"
    assert plan.target_path == tmp_path / ".config" / "opencode" / "plugins" / "archex-hook.ts"


def test_write_hook_install_plan_opencode_writes_ts_module_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("opencode", str(repo), action="install")

    target = write_hook_install_plan(plan)

    assert isinstance(plan, TsHookInstallPlan)
    assert target == plan.target_path
    assert target.read_text(encoding="utf-8") == plan.module_content


def test_write_hook_install_plan_opencode_idempotent_on_reinstall(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = write_hook_install_plan(
        build_hook_install_plan("opencode", str(repo), action="install")
    )
    mtime_first = target.stat().st_mtime_ns

    write_hook_install_plan(build_hook_install_plan("opencode", str(repo), action="install"))

    assert target.stat().st_mtime_ns == mtime_first


def test_write_hook_install_plan_opencode_remove_deletes_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = write_hook_install_plan(
        build_hook_install_plan("opencode", str(repo), action="install")
    )
    assert target.exists()

    write_hook_install_plan(build_hook_install_plan("opencode", str(repo), action="remove"))

    assert not target.exists()


def test_write_hook_install_plan_opencode_remove_missing_file_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("opencode", str(repo), action="remove")

    result_target = write_hook_install_plan(plan)

    assert not result_target.exists()


def test_render_hook_install_preview_opencode_install_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("opencode", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)

    preview = render_hook_install_preview(plan)

    assert "Install" in preview
    assert "tool.execute.after" in preview
    assert not plan.target_path.exists()


def test_render_hook_install_preview_opencode_remove_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    write_hook_install_plan(build_hook_install_plan("opencode", str(repo), action="install"))
    before = (repo / ".opencode" / "plugins" / "archex-hook.ts").read_text(encoding="utf-8")

    plan = build_hook_install_plan("opencode", str(repo), action="remove")
    render_hook_install_preview(plan)

    assert (repo / ".opencode" / "plugins" / "archex-hook.ts").read_text(encoding="utf-8") == before


def _augmented_tools_keys(module_content: str) -> set[str]:
    """Extract the ``ARCHEX_AUGMENTED_TOOLS`` table's keys from generated
    OpenCode plugin source.

    Same rationale as ``_query_field_keys`` above: this table is the
    plugin's *only* tool-name dispatch (no if/else chain on ``input.tool``),
    so its key set precisely determines which tools are ever touched --
    stronger than a substring search, which would false-positive on the
    module's own prose comments quoting the exact tool names/ids that must
    never match.
    """
    match = re.search(
        r'ARCHEX_AUGMENTED_TOOLS: Readonly<Record<string, "Grep" \| "Glob">> = \{(.*?)\n\};',
        module_content,
        re.DOTALL,
    )
    assert match is not None, "ARCHEX_AUGMENTED_TOOLS table not found in generated module"
    return set(re.findall(r"^\s*(\w+):", match.group(1), re.MULTILINE))


def test_opencode_ts_hook_module_native_vs_mcp_tool_routing(tmp_path: Path) -> None:
    """M22 acceptance criterion (native-vs-MCP routing): the plugin's only
    tool-name dispatch is ``ARCHEX_AUGMENTED_TOOLS``, keyed exactly on
    OpenCode's two native search tool ids. OpenCode registers every MCP tool
    under a mandatory ``{server}_{tool}`` id (confirmed against the
    installed `opencode-ai` 1.14.33's own MCP tool-registration code), so no
    realistic MCP-routed id -- including one from archex's own MCP server --
    can ever collide with this table.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("opencode", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)

    keys = _augmented_tools_keys(plan.module_content)

    assert keys == {"grep", "glob"}
    assert "read" not in keys
    mcp_shaped_ids = {"archex_query_repo", "archex_scout_repo", "github_create_issue"}
    assert keys.isdisjoint(mcp_shaped_ids)


def test_opencode_ts_hook_module_bakes_in_active_python_interpreter(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("opencode", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)

    assert json.dumps(sys.executable) in plan.module_content
    assert '["-m", "archex.integrations.hook"]' in plan.module_content


def test_opencode_ts_hook_module_registers_exactly_one_tool_execute_after_and_never_before(
    tmp_path: Path,
) -> None:
    """The plugin only ever registers a `tool.execute.after` handler -- it
    never wires `tool.execute.before` (the hook OpenCode's own documented
    subagent-bypass bug affects), and registers no session/agent-type
    conditional gating that handler, matching the M20 omp/pi module's
    unconditional-registration precedent.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("opencode", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)
    content = plan.module_content

    assert content.count('"tool.execute.after"') == 1
    assert '"tool.execute.before"' not in content


# --- Codex CLI diagnostics-only hook (M21) ---
#
# Unlike claude-code (a JSON entry merged into settings.json) or omp/pi (a
# standalone .ts module), Codex's hook lives in the same config.toml the MCP
# server registration writes to, as a marker-delimited `[[hooks.PreToolUse]]`
# TOML block. See `archex.integrations.codex_hook` for why this hook is
# diagnostics-only (Codex has no Grep/Glob-equivalent tool-call event) rather
# than augmenting like the claude-code/omp/pi hooks above.


def test_build_hook_install_plan_codex_project_scope_produces_config_toml_path(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_hook_install_plan("codex", str(repo), scope="project", action="install")

    assert isinstance(plan, CodexHookInstallPlan)
    assert plan.target_path == repo / ".codex" / "config.toml"
    assert CODEX_HOOK_MATCHER in plan.block_content
    assert "archex.integrations.codex_hook" in plan.block_content


def test_build_hook_install_plan_codex_user_scope_produces_config_toml_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    plan = build_hook_install_plan("codex", action="install")

    assert isinstance(plan, CodexHookInstallPlan)
    assert plan.target_path == tmp_path / ".codex" / "config.toml"


def test_write_hook_install_plan_codex_writes_toml_block(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("codex", str(repo), scope="project", action="install")

    assert isinstance(plan, CodexHookInstallPlan)
    target = write_hook_install_plan(plan)

    assert target == plan.target_path
    content = target.read_text(encoding="utf-8")
    assert content == plan.block_content
    assert "[[hooks.PreToolUse]]" in content
    assert f'matcher = "{CODEX_HOOK_MATCHER}"' in content


def test_write_hook_install_plan_codex_idempotent_on_reinstall(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("codex", str(repo), scope="project", action="install")
    target = write_hook_install_plan(plan)
    after_first = target.read_text(encoding="utf-8")
    mtime_first = target.stat().st_mtime_ns

    plan2 = build_hook_install_plan("codex", str(repo), scope="project", action="install")
    write_hook_install_plan(plan2)

    assert target.read_text(encoding="utf-8") == after_first
    assert target.stat().st_mtime_ns == mtime_first


def test_write_hook_install_plan_codex_preserves_unrelated_config_toml_content(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target_path = repo / ".codex" / "config.toml"
    target_path.parent.mkdir(parents=True)
    seed = '[mcp_servers.archex]\ncommand = "archex"\nargs = ["mcp"]\n'
    target_path.write_text(seed, encoding="utf-8")

    plan = build_hook_install_plan("codex", str(repo), scope="project", action="install")
    write_hook_install_plan(plan)

    content = target_path.read_text(encoding="utf-8")
    assert "[mcp_servers.archex]" in content
    assert "[[hooks.PreToolUse]]" in content


def test_write_hook_install_plan_codex_remove_restores_original_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target_path = repo / ".codex" / "config.toml"
    target_path.parent.mkdir(parents=True)
    seed = '[mcp_servers.archex]\ncommand = "archex"\nargs = ["mcp"]\n'
    target_path.write_text(seed, encoding="utf-8")
    install_plan = build_hook_install_plan("codex", str(repo), scope="project", action="install")
    write_hook_install_plan(install_plan)

    remove_plan = build_hook_install_plan("codex", str(repo), scope="project", action="remove")
    write_hook_install_plan(remove_plan)

    assert target_path.read_text(encoding="utf-8") == seed


def test_write_hook_install_plan_codex_remove_missing_file_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("codex", str(repo), scope="project", action="remove")

    result_target = write_hook_install_plan(plan)

    assert not result_target.exists()


def test_write_hook_install_plan_codex_remove_without_archex_block_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target_path = repo / ".codex" / "config.toml"
    target_path.parent.mkdir(parents=True)
    before = '[mcp_servers.archex]\ncommand = "archex"\nargs = ["mcp"]\n'
    target_path.write_text(before, encoding="utf-8")

    plan = build_hook_install_plan("codex", str(repo), scope="project", action="remove")
    write_hook_install_plan(plan)

    assert target_path.read_text(encoding="utf-8") == before


def test_render_hook_install_preview_codex_install_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("codex", str(repo), scope="project", action="install")

    preview = render_hook_install_preview(plan)

    assert "Install" in preview
    assert "diagnostics-only" in preview
    assert not plan.target_path.exists()


def test_render_hook_install_preview_codex_remove_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target_path = repo / ".codex" / "config.toml"
    target_path.parent.mkdir(parents=True)
    before = '[mcp_servers.archex]\ncommand = "archex"\nargs = ["mcp"]\n'
    target_path.write_text(before, encoding="utf-8")
    plan = build_hook_install_plan("codex", str(repo), scope="project", action="remove")

    preview = render_hook_install_preview(plan)

    assert "Remove" in preview
    assert target_path.read_text(encoding="utf-8") == before


def test_codex_hook_toml_block_matcher_never_reaches_read(tmp_path: Path) -> None:
    """M21 acceptance criterion: the installed hook config matches the
    Grep/Glob-equivalent tool only, never Read. Codex has no Grep/Glob or
    Read hook at all -- the only tool name this installer ever writes is the
    literal `^Bash$` matcher, asserted structurally here rather than by
    inspection.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("codex", str(repo), scope="project", action="install")

    assert isinstance(plan, CodexHookInstallPlan)
    matches = re.findall(r'matcher = "([^"]+)"', plan.block_content)
    assert matches == ["^Bash$"]
    for matcher in matches:
        assert not re.fullmatch(matcher, "Read")


def test_codex_hook_toml_block_bakes_in_active_python_interpreter(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_hook_install_plan("codex", str(repo), scope="project", action="install")

    assert isinstance(plan, CodexHookInstallPlan)
    assert sys.executable in plan.block_content
    assert "-m archex.integrations.codex_hook" in plan.block_content


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


def test_cli_hooks_rejects_unsupported_client_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "cursor", "--hooks"])

    assert result.exit_code != 0
    assert "claude-code" in result.output
    assert "M19" in result.output
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".cursor").exists()


def test_cli_plain_install_client_still_writes_mcp_config_not_hook_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "claude-code"])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".claude.json").exists()
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_cli_hooks_installs_omp_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "omp", "--hooks"])

    assert result.exit_code == 0, result.output
    assert "Installed" in result.output
    target = tmp_path / ".omp" / "agent" / "extensions" / "archex-hook.ts"
    assert target.exists()
    assert "archexHook" in target.read_text(encoding="utf-8")


def test_cli_hooks_omp_dry_run_previews_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "omp", "--hooks", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run." in result.output
    assert not (tmp_path / ".omp" / "agent" / "extensions" / "archex-hook.ts").exists()


def test_cli_remove_hooks_omp_removes_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CliRunner().invoke(cli, ["install-client", "omp", "--hooks"])
    target = tmp_path / ".omp" / "agent" / "extensions" / "archex-hook.ts"
    assert target.exists()

    result = CliRunner().invoke(cli, ["install-client", "omp", "--remove-hooks"])

    assert result.exit_code == 0, result.output
    assert "Removed" in result.output
    assert not target.exists()


def test_cli_hooks_installs_pi_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "pi", "--hooks"])

    assert result.exit_code == 0, result.output
    assert "Installed" in result.output
    target = tmp_path / ".pi" / "agent" / "extensions" / "archex-hook.ts"
    assert target.exists()
    assert "archexHook" in target.read_text(encoding="utf-8")


def test_cli_hooks_pi_dry_run_previews_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "pi", "--hooks", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run." in result.output
    assert not (tmp_path / ".pi" / "agent" / "extensions" / "archex-hook.ts").exists()


def test_cli_remove_hooks_pi_removes_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CliRunner().invoke(cli, ["install-client", "pi", "--hooks"])
    target = tmp_path / ".pi" / "agent" / "extensions" / "archex-hook.ts"
    assert target.exists()

    result = CliRunner().invoke(cli, ["install-client", "pi", "--remove-hooks"])

    assert result.exit_code == 0, result.output
    assert "Removed" in result.output
    assert not target.exists()


def test_cli_hooks_installs_codex_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "codex", "--hooks"])

    assert result.exit_code == 0, result.output
    assert "Installed" in result.output
    target = tmp_path / ".codex" / "config.toml"
    assert target.exists()
    assert "archex.integrations.codex_hook" in target.read_text(encoding="utf-8")


def test_cli_hooks_codex_dry_run_previews_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "codex", "--hooks", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run." in result.output
    assert not (tmp_path / ".codex" / "config.toml").exists()


def test_cli_remove_hooks_codex_removes_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CliRunner().invoke(cli, ["install-client", "codex", "--hooks"])
    target = tmp_path / ".codex" / "config.toml"
    assert "archex.integrations.codex_hook" in target.read_text(encoding="utf-8")

    result = CliRunner().invoke(cli, ["install-client", "codex", "--remove-hooks"])

    assert result.exit_code == 0, result.output
    assert "Removed" in result.output
    assert target.read_text(encoding="utf-8") == ""
