"""omp/Pi status extension installs, and unsupported-client refusals (R23).

The Claude Code surface is a command the client re-runs; these two are a
module the host already has loaded, which reads the snapshot in-process. The
tests pin what makes that safe to ship: per-client file placement, byte
identity between the two hosts, independence from the other TypeScript
surfaces that share the same directory, and an explicit refusal (with the
upstream reason) for every client that has no persistent status surface.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from archex.cli.main import cli
from archex.client_setup import (
    STATUSLINE_UNSUPPORTED_CLIENTS,
    ClientName,
    TsStatusInstallPlan,
    build_post_edit_hook_install_plan,
    build_statusline_install_plan,
    render_statusline_install_preview,
    render_ts_status_module,
    write_post_edit_hook_install_plan,
    write_statusline_install_plan,
)

_MODULE_NAME = "archex-status.ts"


def _install(client: ClientName, repo: Path, *, scope: str = "project") -> Path:
    plan = build_statusline_install_plan(client, repo, scope=scope, action="install")  # type: ignore[arg-type]
    return write_statusline_install_plan(plan)


def _remove(client: ClientName, repo: Path, *, scope: str = "project") -> Path:
    plan = build_statusline_install_plan(client, repo, scope=scope, action="remove")  # type: ignore[arg-type]
    return write_statusline_install_plan(plan)


def test_omp_module_lands_in_the_omp_extension_directory(tmp_path: Path) -> None:
    target = _install("omp", tmp_path)

    assert target == tmp_path / ".omp" / "extensions" / _MODULE_NAME
    assert target.exists()


def test_pi_module_lands_in_the_pi_extension_directory(tmp_path: Path) -> None:
    target = _install("pi", tmp_path)

    assert target == tmp_path / ".pi" / "extensions" / _MODULE_NAME


def test_both_hosts_receive_byte_identical_modules(tmp_path: Path) -> None:
    omp = _install("omp", tmp_path)
    pi = _install("pi", tmp_path)

    assert omp.read_text(encoding="utf-8") == pi.read_text(encoding="utf-8")


def test_module_bakes_in_the_python_snapshot_contract(tmp_path: Path) -> None:
    from archex.project import PROJECT_DIR_NAME
    from archex.status_snapshot import (
        DEFAULT_STALE_AFTER_SECONDS,
        SNAPSHOT_FILENAME,
        STATUS_SNAPSHOT_VERSION,
        WATCH_OBSERVATION_TTL_SECONDS,
    )

    module = _install("omp", tmp_path).read_text(encoding="utf-8")

    assert "__ARCHEX" not in module, "every placeholder must be substituted"
    assert f"const ARCHEX_SNAPSHOT_VERSION = {STATUS_SNAPSHOT_VERSION};" in module
    assert f"const ARCHEX_STALE_AFTER_SECONDS = {DEFAULT_STALE_AFTER_SECONDS};" in module
    assert f"const ARCHEX_WATCH_TTL_SECONDS = {WATCH_OBSERVATION_TTL_SECONDS};" in module
    assert f'const ARCHEX_PROJECT_DIR = "{PROJECT_DIR_NAME}";' in module
    assert f'const ARCHEX_SNAPSHOT_FILENAME = "{SNAPSHOT_FILENAME}";' in module


def test_module_spawns_nothing_and_imports_no_host_package() -> None:
    module = render_ts_status_module()

    assert "child_process" not in module, "a repaint must not spawn a process"
    assert "spawn(" not in module
    assert "@oh-my-pi" not in module
    assert "@mariozechner" not in module
    # The only filesystem read is the bounded snapshot.
    assert 'from "node:fs"' in module
    assert "readFileSync" in module


def test_reinstall_is_an_idempotent_no_op(tmp_path: Path) -> None:
    target = _install("omp", tmp_path)
    first = target.read_text(encoding="utf-8")
    mtime = target.stat().st_mtime_ns

    _install("omp", tmp_path)

    assert target.read_text(encoding="utf-8") == first
    assert target.stat().st_mtime_ns == mtime, "an unchanged module must not be rewritten"


def test_remove_deletes_only_the_status_module(tmp_path: Path) -> None:
    _install("omp", tmp_path)
    write_post_edit_hook_install_plan(
        build_post_edit_hook_install_plan("omp", tmp_path, scope="project", action="install")
    )
    post_edit = tmp_path / ".omp" / "extensions" / "archex-post-edit-hook.ts"
    assert post_edit.exists()

    _remove("omp", tmp_path)

    assert not (tmp_path / ".omp" / "extensions" / _MODULE_NAME).exists()
    assert post_edit.exists(), "removing the status module must not touch the post-edit hook"


def test_remove_without_an_install_is_a_no_op(tmp_path: Path) -> None:
    target = _remove("pi", tmp_path)

    assert not target.exists()


def test_dry_run_previews_the_module_without_writing(tmp_path: Path) -> None:
    plan = build_statusline_install_plan("omp", tmp_path, scope="project", action="install")

    preview = render_statusline_install_preview(plan)

    assert isinstance(plan, TsStatusInstallPlan)
    assert "Dry run" in preview
    assert "renderArchexStatus" in preview
    assert not plan.target_path.exists()


def test_every_unsupported_client_is_refused_with_its_reason(tmp_path: Path) -> None:
    assert set(STATUSLINE_UNSUPPORTED_CLIENTS) == {"opencode", "codex", "cursor"}
    runner = CliRunner()

    for client, reason in STATUSLINE_UNSUPPORTED_CLIENTS.items():
        result = runner.invoke(
            cli, ["install-client", client, str(tmp_path), "--scope", "project", "--statusline"]
        )

        assert result.exit_code != 0, client
        assert "no persistent status surface" in result.output
        # The reason names the upstream fact, not a generic apology.
        assert reason.split(".")[0][:40] in result.output.replace("\n", " ")
        assert "archex status --cached" in result.output


def test_refusal_writes_nothing(tmp_path: Path) -> None:
    CliRunner().invoke(
        cli, ["install-client", "opencode", str(tmp_path), "--scope", "project", "--statusline"]
    )

    assert not (tmp_path / ".opencode").exists()
    assert list(tmp_path.iterdir()) == []


def test_cli_install_and_remove_report_the_target(tmp_path: Path) -> None:
    runner = CliRunner()

    installed = runner.invoke(
        cli, ["install-client", "omp", str(tmp_path), "--scope", "project", "--statusline"]
    )
    removed = runner.invoke(
        cli, ["install-client", "omp", str(tmp_path), "--scope", "project", "--remove-statusline"]
    )

    assert installed.exit_code == 0, installed.output
    assert "Installed archex status surface for omp" in installed.output
    assert removed.exit_code == 0, removed.output
    assert "Removed archex status surface for omp" in removed.output


def test_user_scope_targets_the_host_agent_directory(tmp_path: Path) -> None:
    plan = build_statusline_install_plan("omp", tmp_path, scope="user", action="install")

    assert isinstance(plan, TsStatusInstallPlan)
    assert plan.target_path == Path.home() / ".omp" / "agent" / "extensions" / _MODULE_NAME
