from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from click.testing import CliRunner

from archex.annotate import HOST_TOOLS
from archex.cli.main import cli
from archex.client_setup import (
    CodexHookInstallPlan,
    CursorHookInstallPlan,
    HookAction,
    TsHookInstallPlan,
    build_hook_install_plan,
    build_post_edit_hook_install_plan,
    build_session_primer_install_plan,
    render_hook_install_preview,
    render_session_primer_install_preview,
    write_hook_install_plan,
    write_post_edit_hook_install_plan,
    write_session_primer_install_plan,
)
from archex.integrations.claude_code_annotate_hook import HOOK_MATCHER
from archex.integrations.codex_hook import HOOK_MATCHER as CODEX_HOOK_MATCHER

if TYPE_CHECKING:
    from typing import Any


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


def _session_start_group_has_archex_entry(group: object) -> bool:
    if not isinstance(group, dict):
        return False
    handlers = cast("dict[str, object]", group).get("hooks")
    if not isinstance(handlers, list):
        return False
    return any(
        isinstance(handler, dict)
        and any(
            isinstance(item, str) and "archex.integrations.session_hook" in item
            for item in cast("list[object]", cast("dict[str, object]", handler).get("args", []))
        )
        for handler in cast("list[object]", handlers)
    )


# --- Claude Code PostToolUse search-annotation hook ---

ANNOTATE_MARKER = "archex.integrations.claude_code_annotate_hook"
LEGACY_MARKER = "archex.integrations.hook"
POST_EDIT_MARKER = "archex.integrations.post_edit_hook"
SESSION_MARKER = "archex.integrations.session_hook"


def _legacy_pre_tool_use_group() -> dict[str, Any]:
    """The entry archex <= 0.33 wrote: a PreToolUse pattern-search hook on Glob|Grep."""
    return {
        "matcher": "Glob|Grep",
        "hooks": [{"type": "command", "command": sys.executable, "args": ["-m", LEGACY_MARKER]}],
    }


def _write_claude(repo: Path, action: HookAction) -> Path:
    return write_hook_install_plan(build_hook_install_plan("claude-code", str(repo), action=action))


def _handlers_with(payload: dict[str, Any], event: str, marker: str) -> list[dict[str, Any]]:
    """Every handler under ``hooks.<event>`` whose args carry ``marker``."""
    found: list[dict[str, Any]] = []
    for group in payload.get("hooks", {}).get(event, []):
        for handler in group.get("hooks", []):
            if any(marker in str(item) for item in handler.get("args", [])):
                found.append(handler)
    return found


def _group_carries(group: dict[str, Any], marker: str) -> bool:
    return bool(_handlers_with({"hooks": {"E": [group]}}, "E", marker))


def _by_matcher(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(groups, key=lambda group: str(group["matcher"]))


def test_claude_hook_matcher_is_the_annotate_cores_claude_search_tools() -> None:
    assert HOOK_MATCHER == "Bash|Grep|Glob"
    assert HOOK_MATCHER.split("|") == list(HOST_TOOLS["claude-code"])


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
            "PostToolUse": [
                {
                    "matcher": "Bash|Grep|Glob",
                    "hooks": [
                        {
                            "type": "command",
                            "command": sys.executable,
                            "args": ["-m", ANNOTATE_MARKER],
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
    group = payload["hooks"]["PostToolUse"][0]
    assert group["matcher"] == HOOK_MATCHER
    assert group["hooks"][0]["args"] == ["-m", ANNOTATE_MARKER]
    assert group["hooks"][0]["command"] == sys.executable


def test_write_hook_install_plan_idempotent_on_reinstall(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_json(repo / ".claude" / "settings.json", _seed_payload())

    target = _write_claude(repo, "install")
    after_first = target.read_text(encoding="utf-8")
    _write_claude(repo, "install")

    assert target.read_text(encoding="utf-8") == after_first
    assert len(_handlers_with(json.loads(after_first), "PostToolUse", ANNOTATE_MARKER)) == 1


def test_install_replaces_the_legacy_pre_tool_use_search_entry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    foreign_in_legacy_group = {"type": "command", "command": "echo foreign-glob-grep"}
    legacy = _legacy_pre_tool_use_group()
    legacy["hooks"].append(foreign_in_legacy_group)
    seed = _seed_payload()
    seed["hooks"]["PreToolUse"].append(legacy)
    _write_json(target, seed)

    _write_claude(repo, "install")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert _handlers_with(payload, "PreToolUse", LEGACY_MARKER) == []
    assert payload["hooks"]["PreToolUse"] == [
        seed["hooks"]["PreToolUse"][0],
        {"matcher": "Glob|Grep", "hooks": [foreign_in_legacy_group]},
    ]
    assert len(_handlers_with(payload, "PostToolUse", ANNOTATE_MARKER)) == 1


def test_install_over_only_a_legacy_entry_leaves_no_pre_tool_use_event(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    _write_json(target, {"hooks": {"PreToolUse": [_legacy_pre_tool_use_group()]}})

    _write_claude(repo, "install")

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload["hooks"]) == {"PostToolUse"}


def test_remove_clears_both_the_annotation_and_the_legacy_entry(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    seed = _seed_payload()
    _write_json(target, seed)
    _write_claude(repo, "install")
    # A legacy entry lands next to the new one, as after an old archex ran install again.
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["hooks"]["PreToolUse"].append(_legacy_pre_tool_use_group())
    _write_json(target, payload)

    _write_claude(repo, "remove")

    assert json.loads(target.read_text(encoding="utf-8")) == seed


def test_write_hook_install_plan_preserves_unrelated_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    _write_json(target, _seed_payload())

    _write_claude(repo, "install")

    payload = json.loads(target.read_text(encoding="utf-8"))
    seed = _seed_payload()
    assert payload["otherTopLevelKey"] == "unrelated-value"
    assert payload["hooks"]["PreToolUse"] == seed["hooks"]["PreToolUse"]
    post_tool_use = payload["hooks"]["PostToolUse"]
    assert seed["hooks"]["PostToolUse"][0] in post_tool_use
    archex_groups = [g for g in post_tool_use if _group_carries(g, ANNOTATE_MARKER)]
    assert len(archex_groups) == 1
    assert archex_groups[0]["matcher"] == HOOK_MATCHER


def test_the_annotation_entry_is_reachable_only_from_its_post_tool_use_group(
    tmp_path: Path,
) -> None:
    """The archex entry sits only under PostToolUse, in a group matched by exactly
    Bash|Grep|Glob, never in a group that matches Read, and never under another event.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    seed = _seed_payload()
    seed["hooks"]["PostToolUse"].append(
        {"matcher": "Read", "hooks": [{"type": "command", "command": "echo read-hook"}]}
    )
    _write_json(target, seed)

    _write_claude(repo, "install")

    hooks_root = json.loads(target.read_text(encoding="utf-8"))["hooks"]
    for event_name, groups in hooks_root.items():
        for group in groups:
            carries = _group_carries(group, ANNOTATE_MARKER)
            if event_name == "PostToolUse" and group["matcher"] == HOOK_MATCHER:
                assert carries
            else:
                assert not carries, f"{event_name}/{group['matcher']} must not carry the entry"
            if "Read" in str(group["matcher"]):
                assert not carries


def test_write_hook_install_plan_remove_reduces_to_empty_object(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = _write_claude(repo, "install")
    assert target.exists()

    _write_claude(repo, "remove")

    assert json.loads(target.read_text(encoding="utf-8")) == {}


def test_write_hook_install_plan_remove_preserves_unrelated_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    _write_json(target, _seed_payload())
    _write_claude(repo, "install")

    _write_claude(repo, "remove")

    assert json.loads(target.read_text(encoding="utf-8")) == _seed_payload()


def test_write_hook_install_plan_remove_missing_file_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"

    result_target = _write_claude(repo, "remove")

    assert result_target == target
    assert not target.exists()
    assert not target.parent.exists()


def test_write_hook_install_plan_remove_without_archex_entry_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    _write_json(target, _seed_payload())
    before = target.read_text(encoding="utf-8")

    _write_claude(repo, "remove")

    assert target.read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    "annotation_first", [True, False], ids=["annotation-first", "others-first"]
)
def test_the_claude_surfaces_survive_each_others_install_reinstall_and_remove(
    tmp_path: Path, annotation_first: bool
) -> None:
    """Foreign handlers, the post-edit PostToolUse entry, and the SessionStart primer
    survive the annotation hook's install, reinstall and remove -- and it survives theirs --
    whichever installs first.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    seed = _seed_payload()
    foreign_session = {
        "matcher": "startup",
        "hooks": [{"type": "command", "command": "echo existing-session-start"}],
    }
    seed["hooks"]["SessionStart"] = [foreign_session]
    _write_json(target, seed)
    seed_with_legacy = json.loads(target.read_text(encoding="utf-8"))
    seed_with_legacy["hooks"]["PreToolUse"].append(_legacy_pre_tool_use_group())
    _write_json(target, seed_with_legacy)

    def install_others() -> None:
        write_post_edit_hook_install_plan(
            build_post_edit_hook_install_plan("claude-code", str(repo), action="install")
        )
        write_session_primer_install_plan(
            build_session_primer_install_plan(str(repo), action="install")
        )

    def assert_others_intact() -> dict[str, Any]:
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["otherTopLevelKey"] == "unrelated-value"
        assert payload["hooks"]["PreToolUse"] == seed["hooks"]["PreToolUse"]
        assert seed["hooks"]["PostToolUse"][0] in payload["hooks"]["PostToolUse"]
        assert foreign_session in payload["hooks"]["SessionStart"]
        assert len(_handlers_with(payload, "PostToolUse", POST_EDIT_MARKER)) == 1
        assert len(_handlers_with(payload, "SessionStart", SESSION_MARKER)) == 1
        return payload

    if annotation_first:
        _write_claude(repo, "install")
        install_others()
    else:
        install_others()
        _write_claude(repo, "install")
    payload = assert_others_intact()
    assert len(_handlers_with(payload, "PostToolUse", ANNOTATE_MARKER)) == 1
    assert _handlers_with(payload, "PreToolUse", LEGACY_MARKER) == []

    # A reinstall converges on the same handlers (group order within an event may move).
    _write_claude(repo, "install")
    reinstalled = json.loads(target.read_text(encoding="utf-8"))
    for event in ("PreToolUse", "PostToolUse", "SessionStart"):
        assert _by_matcher(reinstalled["hooks"][event]) == _by_matcher(payload["hooks"][event])

    _write_claude(repo, "remove")
    payload = assert_others_intact()
    assert _handlers_with(payload, "PostToolUse", ANNOTATE_MARKER) == []

    # The other direction: removing the others leaves a reinstalled annotation hook alone.
    _write_claude(repo, "install")
    write_post_edit_hook_install_plan(
        build_post_edit_hook_install_plan("claude-code", str(repo), action="remove")
    )
    write_session_primer_install_plan(build_session_primer_install_plan(str(repo), action="remove"))
    after = json.loads(target.read_text(encoding="utf-8"))
    assert len(_handlers_with(after, "PostToolUse", ANNOTATE_MARKER)) == 1
    assert _handlers_with(after, "PostToolUse", POST_EDIT_MARKER) == []
    assert after["hooks"]["SessionStart"] == [foreign_session]
    assert seed["hooks"]["PostToolUse"][0] in after["hooks"]["PostToolUse"]


def test_render_hook_install_preview_install_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    plan = build_hook_install_plan("claude-code", str(repo), action="install")

    preview = render_hook_install_preview(plan)

    assert "Install" in preview
    assert "PostToolUse" in preview
    assert "Bash|Grep|Glob" in preview
    assert "retired archex PreToolUse" in preview
    assert "grep/glob-equivalent" not in preview
    assert not target.exists()


def test_render_hook_install_preview_remove_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    _write_claude(repo, "install")
    before = target.read_text(encoding="utf-8")

    preview = render_hook_install_preview(
        build_hook_install_plan("claude-code", str(repo), action="remove")
    )

    assert "Remove" in preview
    assert target.read_text(encoding="utf-8") == before


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


def test_omp_hook_install_reinstall_remove_leaves_foreign_extensions_untouched(
    tmp_path: Path,
) -> None:
    """omp/Pi load every module in the extensions directory, so that directory
    is the "config" other handlers share: install, idempotent reinstall, and
    remove must touch only `archex-hook.ts`.
    """
    repo = tmp_path / "repo"
    extensions = repo / ".omp" / "extensions"
    extensions.mkdir(parents=True)
    foreign = extensions / "other-tool-result.ts"
    foreign_source = (
        'export default function other(pi) { pi.on("tool_result", async () => undefined); }\n'
    )
    foreign.write_text(foreign_source, encoding="utf-8")

    target = write_hook_install_plan(
        build_hook_install_plan("omp", str(repo), scope="project", action="install")
    )
    installed = target.read_text(encoding="utf-8")
    write_hook_install_plan(
        build_hook_install_plan("omp", str(repo), scope="project", action="install")
    )

    assert sorted(p.name for p in extensions.iterdir()) == ["archex-hook.ts", foreign.name]
    assert target.read_text(encoding="utf-8") == installed

    write_hook_install_plan(
        build_hook_install_plan("omp", str(repo), scope="project", action="remove")
    )

    assert [p.name for p in extensions.iterdir()] == [foreign.name]
    assert foreign.read_text(encoding="utf-8") == foreign_source


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


# --- Cursor beforeSubmitPrompt hook (M23, diagnostics-only) ---
#
# Unlike claude-code (a JSON entry merged into settings.json) or omp/pi/
# opencode (a standalone .ts module), Cursor's hook lives in its own
# hooks.json -- a different file from mcp.json. See
# `archex.integrations.cursor_hook` for why this hook is diagnostics-only
# (Cursor's beforeSubmitPrompt output schema has no context-injection field
# at all) rather than injecting context like the claude-code/omp/pi/opencode
# hooks above.


def test_build_hook_install_plan_cursor_project_scope_produces_expected_shape(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    plan = build_hook_install_plan("cursor", str(repo), scope="project", action="install")
    target = write_hook_install_plan(plan)

    assert isinstance(plan, CursorHookInstallPlan)
    assert plan.scope == "project"
    assert target == repo / ".cursor" / "hooks.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == {
        "version": 1,
        "hooks": {
            "beforeSubmitPrompt": [
                {
                    "command": f"{sys.executable} -m archex.integrations.cursor_hook",
                    "timeout": 1,
                }
            ]
        },
    }


def test_build_hook_install_plan_cursor_user_scope_produces_expected_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    plan = build_hook_install_plan("cursor", action="install")
    target = write_hook_install_plan(plan)

    assert isinstance(plan, CursorHookInstallPlan)
    assert plan.scope == "user"
    assert target == tmp_path / ".cursor" / "hooks.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    entry = payload["hooks"]["beforeSubmitPrompt"][0]
    assert entry["command"] == f"{sys.executable} -m archex.integrations.cursor_hook"


def test_write_hook_install_plan_cursor_idempotent_on_reinstall(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("cursor", str(repo), scope="project", action="install")
    write_hook_install_plan(plan)
    after_first = plan.target_path.read_text(encoding="utf-8")

    write_hook_install_plan(plan)
    after_second = plan.target_path.read_text(encoding="utf-8")

    assert after_first == after_second


def test_write_hook_install_plan_cursor_preserves_unrelated_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".cursor" / "hooks.json"
    _write_json(
        target,
        {
            "version": 1,
            "hooks": {
                "afterFileEdit": [{"command": "./hooks/format.sh"}],
                "beforeSubmitPrompt": [{"command": "./hooks/audit.sh"}],
            },
        },
    )
    plan = build_hook_install_plan("cursor", str(repo), scope="project", action="install")

    write_hook_install_plan(plan)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["hooks"]["afterFileEdit"] == [{"command": "./hooks/format.sh"}]
    before_submit = payload["hooks"]["beforeSubmitPrompt"]
    assert {"command": "./hooks/audit.sh"} in before_submit
    assert any("archex.integrations.cursor_hook" in e["command"] for e in before_submit)


def test_write_hook_install_plan_cursor_config_assertion_never_wires_before_read_file(
    tmp_path: Path,
) -> None:
    """M23 acceptance criterion: the installed config never wires anything to
    `beforeReadFile`. Seeded with a pre-existing `beforeReadFile` entry to
    prove install/remove never even reads that key, let alone writes to it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".cursor" / "hooks.json"
    seed = {
        "version": 1,
        "hooks": {"beforeReadFile": [{"command": "./hooks/gate-secrets.sh"}]},
    }
    _write_json(target, seed)
    plan = build_hook_install_plan("cursor", str(repo), scope="project", action="install")

    write_hook_install_plan(plan)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["hooks"]["beforeReadFile"] == [{"command": "./hooks/gate-secrets.sh"}]
    assert "archex.integrations.cursor_hook" not in json.dumps(payload["hooks"]["beforeReadFile"])
    assert any(
        "archex.integrations.cursor_hook" in entry["command"]
        for entry in payload["hooks"]["beforeSubmitPrompt"]
    )

    remove_plan = build_hook_install_plan("cursor", str(repo), scope="project", action="remove")
    write_hook_install_plan(remove_plan)

    after_remove = json.loads(target.read_text(encoding="utf-8"))
    assert after_remove["hooks"]["beforeReadFile"] == [{"command": "./hooks/gate-secrets.sh"}]
    assert "beforeSubmitPrompt" not in after_remove["hooks"]


def test_write_hook_install_plan_cursor_remove_leaves_bare_version_stamp(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    install_plan = build_hook_install_plan("cursor", str(repo), scope="project", action="install")
    target = write_hook_install_plan(install_plan)
    remove_plan = build_hook_install_plan("cursor", str(repo), scope="project", action="remove")

    write_hook_install_plan(remove_plan)

    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}


def test_write_hook_install_plan_cursor_remove_missing_file_is_noop(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("cursor", str(repo), scope="project", action="remove")

    result_target = write_hook_install_plan(plan)

    assert not result_target.exists()


def test_write_hook_install_plan_cursor_remove_without_archex_entry_is_noop(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".cursor" / "hooks.json"
    seed = {"version": 1, "hooks": {"afterFileEdit": [{"command": "./hooks/format.sh"}]}}
    _write_json(target, seed)
    before = target.read_text(encoding="utf-8")
    plan = build_hook_install_plan("cursor", str(repo), scope="project", action="remove")

    write_hook_install_plan(plan)

    assert target.read_text(encoding="utf-8") == before


def test_render_hook_install_preview_cursor_install_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("cursor", str(repo), scope="project", action="install")

    preview = render_hook_install_preview(plan)

    assert "Dry run." in preview
    assert not plan.target_path.exists()


def test_render_hook_install_preview_cursor_remove_does_not_write(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    install_plan = build_hook_install_plan("cursor", str(repo), scope="project", action="install")
    write_hook_install_plan(install_plan)
    before = install_plan.target_path.read_text(encoding="utf-8")
    remove_plan = build_hook_install_plan("cursor", str(repo), scope="project", action="remove")

    render_hook_install_preview(remove_plan)

    assert install_plan.target_path.read_text(encoding="utf-8") == before


# --- CLI wiring: install-client --hooks / --remove-hooks ---


def test_cli_hooks_installs_and_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "claude-code", "--hooks"])

    assert result.exit_code == 0, result.output
    assert "Installed" in result.output
    target = tmp_path / ".claude" / "settings.json"
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["hooks"]["PostToolUse"][0]["matcher"] == HOOK_MATCHER
    assert "PreToolUse" not in payload["hooks"]


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


def test_cli_hooks_migrates_a_legacy_pre_tool_use_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".claude" / "settings.json"
    _write_json(target, {"hooks": {"PreToolUse": [_legacy_pre_tool_use_group()]}})

    result = CliRunner().invoke(cli, ["install-client", "claude-code", "--hooks"])

    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload["hooks"]) == {"PostToolUse"}
    assert _handlers_with(payload, "PostToolUse", ANNOTATE_MARKER)


def test_cli_hooks_and_remove_hooks_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "claude-code", "--hooks", "--remove-hooks"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    assert not (tmp_path / ".claude" / "settings.json").exists()


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


# --- OpenCode CLI wiring: install-client opencode --hooks (M22 PR-2) ---


def test_cli_hooks_installs_opencode_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "opencode", "--hooks"])

    assert result.exit_code == 0, result.output
    assert "Installed" in result.output
    target = tmp_path / ".config" / "opencode" / "plugins" / "archex-hook.ts"
    assert target.exists()
    assert "ArchexHookPlugin" in target.read_text(encoding="utf-8")


def test_cli_hooks_opencode_dry_run_previews_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "opencode", "--hooks", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run." in result.output
    assert not (tmp_path / ".config" / "opencode" / "plugins" / "archex-hook.ts").exists()


def test_cli_remove_hooks_opencode_removes_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CliRunner().invoke(cli, ["install-client", "opencode", "--hooks"])
    target = tmp_path / ".config" / "opencode" / "plugins" / "archex-hook.ts"
    assert target.exists()

    result = CliRunner().invoke(cli, ["install-client", "opencode", "--remove-hooks"])

    assert result.exit_code == 0, result.output
    assert "Removed" in result.output
    assert not target.exists()


def test_cli_hooks_opencode_installed_file_never_targets_read_or_mcp_tool_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M22 acceptance criterion (config assertion): proven against the file
    the CLI actually writes to disk, not just the in-memory plan object PR-1
    exercised (`test_opencode_ts_hook_module_native_vs_mcp_tool_routing`) --
    closing the gap between "the render function is correct" and "the
    installer wrote what the render function produced."
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    CliRunner().invoke(cli, ["install-client", "opencode", "--hooks"])
    target = tmp_path / ".config" / "opencode" / "plugins" / "archex-hook.ts"

    keys = _augmented_tools_keys(target.read_text(encoding="utf-8"))

    assert keys == {"grep", "glob"}
    assert "read" not in keys
    mcp_shaped_ids = {"archex_query_repo", "archex_scout_repo", "github_create_issue"}
    assert keys.isdisjoint(mcp_shaped_ids)


def test_opencode_ts_hook_module_subagent_dispatch_reachability(tmp_path: Path) -> None:
    """M22 subagent-dispatch finding: confirmed **reachable**, via two
    independent checks performed during development (recorded in
    `docs/CLIENT_COMPATIBILITY_MATRIX.md`), not assumed either way:

    1. Source-level: `opencode-ai@1.14.33`'s
       `packages/opencode/src/session/prompt.ts` binds the `TaskPromptOps`
       handed to every tool's context to the exact same `prompt()` closure
       that processes a top-level turn (`ops().prompt = (input) =>
       prompt(input)`). `TaskTool.execute` calls
       `ops.prompt({sessionID: nextSession.id, ...})` for a subagent's own
       turn, which recurses into the identical `resolveTools()`-wrapped
       `tool.execute.after` trigger used for the top-level session -- no
       subagent-specific branch skips it.
    2. Live: a real `opencode run` session instructing a
       `subagent_type: general` Task to call `grep` itself showed the
       archex receipt block appended directly in the *subagent's own child
       session's* exported tool-part output (`opencode export
       <child-session-id>`), not merely relayed/paraphrased by the parent.

    This plugin's own code makes no session/agent-type distinction to rely
    on either way -- asserted here structurally, the same way M20's omp/pi
    module's unconditional-registration was asserted: the handler body
    never references `sessionID`/`callID`, so it cannot special-case a
    subagent's session even if OpenCode's behavior changes in a future
    version.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = build_hook_install_plan("opencode", str(repo), action="install")
    assert isinstance(plan, TsHookInstallPlan)
    content = plan.module_content

    handler_match = re.search(
        r'"tool\.execute\.after": async \(input, output\) => \{(.*?)\n    \},',
        content,
        re.DOTALL,
    )
    assert handler_match is not None, "tool.execute.after handler body not found"
    handler_body = handler_match.group(1)

    assert "sessionID" not in handler_body
    assert "callID" not in handler_body


# --- Cursor CLI wiring: install-client cursor --hooks (M23 PR-2) ---


def test_cli_hooks_installs_cursor_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "cursor", "--hooks"])

    assert result.exit_code == 0, result.output
    assert "Installed" in result.output
    target = tmp_path / ".cursor" / "hooks.json"
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    entry = payload["hooks"]["beforeSubmitPrompt"][0]
    assert "archex.integrations.cursor_hook" in entry["command"]


def test_cli_hooks_cursor_dry_run_previews_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    result = CliRunner().invoke(cli, ["install-client", "cursor", "--hooks", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run." in result.output
    assert not (tmp_path / ".cursor" / "hooks.json").exists()


def test_cli_remove_hooks_cursor_removes_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CliRunner().invoke(cli, ["install-client", "cursor", "--hooks"])
    target = tmp_path / ".cursor" / "hooks.json"
    assert target.exists()

    result = CliRunner().invoke(cli, ["install-client", "cursor", "--remove-hooks"])

    assert result.exit_code == 0, result.output
    assert "Removed" in result.output
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "beforeSubmitPrompt" not in payload.get("hooks", {})


def test_cli_hooks_cursor_installed_file_never_targets_before_read_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M23 acceptance criterion (config assertion), proven against the file
    the CLI actually writes to disk: the installed hooks.json never wires
    anything under `beforeReadFile`, even when one already existed.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    target = tmp_path / ".cursor" / "hooks.json"
    _write_json(
        target,
        {"version": 1, "hooks": {"beforeReadFile": [{"command": "./hooks/gate-secrets.sh"}]}},
    )

    result = CliRunner().invoke(cli, ["install-client", "cursor", "--hooks"])

    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["hooks"]["beforeReadFile"] == [{"command": "./hooks/gate-secrets.sh"}]
    assert "archex.integrations.cursor_hook" not in json.dumps(payload["hooks"]["beforeReadFile"])
    assert any(
        "archex.integrations.cursor_hook" in entry["command"]
        for entry in payload["hooks"]["beforeSubmitPrompt"]
    )


# --- Claude Code SessionStart session-primer wiring ---


def test_session_primer_install_and_remove_preserve_other_claude_hooks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / ".claude" / "settings.json"
    seed = _seed_payload()
    existing_session_start = {
        "matcher": "startup",
        "hooks": [{"type": "command", "command": "echo existing-session-start"}],
    }
    seed["hooks"]["SessionStart"] = [existing_session_start]
    _write_json(target, seed)

    install_plan = build_session_primer_install_plan(str(repo), action="install")
    write_session_primer_install_plan(install_plan)
    installed = json.loads(target.read_text(encoding="utf-8"))

    assert installed["otherTopLevelKey"] == seed["otherTopLevelKey"]
    assert installed["hooks"]["PreToolUse"] == seed["hooks"]["PreToolUse"]
    session_start = installed["hooks"]["SessionStart"]
    assert existing_session_start in session_start
    primer_groups = [
        group for group in session_start if _session_start_group_has_archex_entry(group)
    ]
    assert len(primer_groups) == 1
    assert primer_groups[0]["matcher"] == "resume|startup"

    after_first = target.read_text(encoding="utf-8")
    write_session_primer_install_plan(
        build_session_primer_install_plan(str(repo), action="install")
    )
    assert target.read_text(encoding="utf-8") == after_first

    write_session_primer_install_plan(build_session_primer_install_plan(str(repo), action="remove"))
    removed = json.loads(target.read_text(encoding="utf-8"))
    assert removed["hooks"]["SessionStart"] == [existing_session_start]
    assert removed["hooks"]["PreToolUse"] == seed["hooks"]["PreToolUse"]


def test_session_primer_preview_and_cli_are_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    plan = build_session_primer_install_plan(action="install")

    preview = render_session_primer_install_preview(plan)
    assert "SessionStart" in preview
    assert "Dry run." in preview
    assert not plan.target_path.exists()

    result = CliRunner().invoke(cli, ["install-client", "claude-code", "--session-primer"])
    assert result.exit_code == 0, result.output
    target = tmp_path / ".claude" / "settings.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert _session_start_group_has_archex_entry(payload["hooks"]["SessionStart"][0])

    incompatible = CliRunner().invoke(cli, ["install-client", "omp", "--session-primer"])
    assert incompatible.exit_code != 0
    assert "only supported for claude-code" in incompatible.output
