"""The installed Claude Code status-line renderer, executed for real (R23).

Every test here runs the actual installed script through `/bin/sh` with an
**empty PATH**. That is the repaint-isolation proof: a renderer that shelled
out to `jq`, `python`, `date`, `stat`, or `archex` could not produce correct
output under an empty PATH, and one that opened the index could not run at all
without a Python interpreter. Correct output plus empty stderr means the
repaint touched nothing but the cached snapshot.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pytest

from archex.client_setup import (
    STATUSLINE_SCRIPT_FILENAME,
    build_statusline_install_plan,
    write_statusline_install_plan,
)
from archex.status_snapshot import (
    DEFAULT_STALE_AFTER_SECONDS,
    STATUS_SNAPSHOT_VERSION,
    ReceiptState,
    SnapshotState,
    StatusSnapshot,
    status_snapshot_path,
    utc_now_iso,
)

_NOW = 1_800_000_000

#: Commands a repaint must never launch. `archex` is deliberately absent:
#: the rendered output names it as a remedy ("run: archex index"), and an
#: actual invocation is already caught by the empty-PATH executions.
_FORBIDDEN_COMMANDS = (
    "jq",
    "python",
    "python3",
    "date",
    "stat",
    "cat",
    "grep",
    "sed",
    "awk",
    "env",
    "node",
    "bun",
    "git",
)


def _installed_script(repo: Path) -> Path:
    plan = build_statusline_install_plan("claude-code", repo, scope="project", action="install")
    write_statusline_install_plan(plan)
    return plan.script_path


def _snapshot(**overrides: object) -> StatusSnapshot:
    base: dict[str, object] = {
        "state": SnapshotState.FRESH,
        "written_at": utc_now_iso(_NOW),
        "written_at_epoch": _NOW,
        "index_measured_at": utc_now_iso(_NOW),
        "index_revision": "0123456789abcdef" * 4,
        "generation_id": "f" * 64,
        "indexed_commit": "c" * 40,
        "current_commit": "c" * 40,
        "files_indexed": 1234,
        "chunks_indexed": 5678,
        "receipt_state": ReceiptState.COMPLETE,
    }
    base.update(overrides)
    return StatusSnapshot.model_validate(base)


def _write_snapshot(repo: Path, snapshot: StatusSnapshot) -> Path:
    path = status_snapshot_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _render(
    script: Path,
    *,
    snapshot_path: Path | None = None,
    cwd: Path | None = None,
    stdin: str = "",
    shell: str = "/bin/sh",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment: dict[str, str] = {"PATH": ""}
    if snapshot_path is not None:
        environment["ARCHEX_STATUS_SNAPSHOT"] = str(snapshot_path)
    if env is not None:
        environment.update(env)
    return subprocess.run(
        [shell, str(script)],
        cwd=str(cwd) if cwd is not None else str(script.parent),
        input=stdin,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
        timeout=30,
    )


def test_renderer_names_no_external_command(tmp_path: Path) -> None:
    """Structural tripwire beside the empirical empty-PATH runs below.

    The empty-PATH executions are the real proof that today's renderer forks
    nothing. This guards the branches those runs do not take: a future edit
    that reaches for `date` or `jq` inside a rare code path fails here even
    if no parameterized case exercises it.
    """
    body = _installed_script(tmp_path).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in body.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )

    # `$((...))` is arithmetic expansion, evaluated by the shell itself;
    # `$(...)` is command substitution, which forks.
    assert re.findall(r"\$\((?!\()", code) == [], "command substitution would fork a process"
    assert "`" not in code, "backtick substitution would fork a process on every repaint"
    for command in _FORBIDDEN_COMMANDS:
        found = re.search(rf"(?<![\w./-]){re.escape(command)}(?![\w./-])", code)
        assert found is None, f"{command} would be launched on every repaint"


def test_fresh_snapshot_renders_state_size_and_revision_prefix(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)
    snapshot = _write_snapshot(tmp_path, _snapshot())

    result = _render(script, snapshot_path=snapshot)

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith("archex fresh - 1234 files, 5678 chunks - rev 01234567")


def test_pending_snapshot_names_the_edits_awaiting_synchronization(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)
    snapshot = _write_snapshot(
        tmp_path,
        _snapshot(state=SnapshotState.PENDING, pending_delta_files=3, pending_view_complete=True),
    )

    result = _render(script, snapshot_path=snapshot)

    assert result.stderr == ""
    assert "archex pending - 3 awaiting sync" in result.stdout


def test_incomplete_pending_view_is_marked_as_a_lower_bound(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)
    snapshot = _write_snapshot(
        tmp_path,
        _snapshot(
            state=SnapshotState.PENDING,
            pending_delta_files=200,
            pending_view_complete=False,
            receipt_state=ReceiptState.PARTIAL,
        ),
    )

    result = _render(script, snapshot_path=snapshot)

    assert "archex pending - 200+ awaiting sync" in result.stdout


def test_dirty_snapshot_distinguishes_a_required_reindex(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)

    behind = _write_snapshot(tmp_path, _snapshot(state=SnapshotState.DIRTY))
    assert "archex dirty - index behind tree" in _render(script, snapshot_path=behind).stdout

    flagged = _write_snapshot(tmp_path, _snapshot(state=SnapshotState.DIRTY, reindex_required=True))
    assert "archex dirty - reindex required" in _render(script, snapshot_path=flagged).stdout


def test_missing_snapshot_renders_missing_with_a_remedy(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)

    result = _render(script, snapshot_path=tmp_path / "absent.json")

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.strip() == "archex missing - no status snapshot - run: archex index"


def test_unparsable_snapshot_renders_corrupt(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)
    path = status_snapshot_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json at all\n", encoding="utf-8")

    result = _render(script, snapshot_path=path)

    assert result.stdout.strip() == "archex corrupt - unreadable snapshot - run: archex status"


def test_valid_json_with_an_unknown_state_renders_corrupt(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)
    path = status_snapshot_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": STATUS_SNAPSHOT_VERSION, "state": "sideways"}, indent=2),
        encoding="utf-8",
    )

    result = _render(script, snapshot_path=path)

    assert "archex corrupt" in result.stdout


def test_future_version_renders_unsupported_not_corrupt(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)
    path = status_snapshot_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": STATUS_SNAPSHOT_VERSION + 1, "state": "fresh"}, indent=2),
        encoding="utf-8",
    )

    result = _render(script, snapshot_path=path)

    assert result.stdout.strip() == (
        f"archex unsupported - snapshot v{STATUS_SNAPSHOT_VERSION + 1} - upgrade archex"
    )


def test_session_directory_is_read_from_the_stdin_payload(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script = _installed_script(repo)
    _write_snapshot(repo, _snapshot())
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    payload = json.dumps({"session_id": "s", "cwd": str(repo), "model": {"id": "opus"}})

    result = _render(script, cwd=elsewhere, stdin=f"{payload}\n")

    assert "archex fresh" in result.stdout


def test_snapshot_is_found_from_a_subdirectory_of_the_session(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    script = _installed_script(repo)
    _write_snapshot(repo, _snapshot())

    result = _render(script, cwd=nested, stdin=json.dumps({"cwd": str(nested)}) + "\n")

    assert "archex fresh" in result.stdout


def test_unrelated_directory_renders_missing_rather_than_another_repository(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script = _installed_script(repo)
    _write_snapshot(repo, _snapshot())
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    result = _render(script, cwd=unrelated, stdin=json.dumps({"cwd": str(unrelated)}) + "\n")

    assert "archex missing" in result.stdout


def _clock_shell() -> str | None:
    """A shell where the renderer can read a clock, or None if none exists.

    macOS ships bash 3.2 as both `/bin/sh` and `/bin/bash`, neither of which
    has a builtin clock, so the age and stale segments are only exercisable
    under zsh (whose `zsh/datetime` module the renderer loads through the
    `zmodload` builtin) or bash 5+.
    """
    probes = (
        ("/bin/zsh", "zmodload zsh/datetime 2>/dev/null; echo ${EPOCHSECONDS:-}"),
        ("/opt/homebrew/bin/bash", "echo ${EPOCHSECONDS:-}"),
        ("/usr/local/bin/bash", "echo ${EPOCHSECONDS:-}"),
        ("/bin/bash", "echo ${EPOCHSECONDS:-}"),
    )
    for candidate, script in probes:
        if not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-c", script], capture_output=True, text=True, check=False, timeout=30
        )
        if probe.stdout.strip().isdigit():
            return candidate
    return None


def test_a_shell_with_a_builtin_clock_reports_age_and_stale(tmp_path: Path) -> None:
    shell = _clock_shell()
    if shell is None:
        pytest.skip("no host shell exposes EPOCHSECONDS")
    script = _installed_script(tmp_path)
    now = int(time.time())
    recent = _write_snapshot(
        tmp_path, _snapshot(written_at=utc_now_iso(now - 90), written_at_epoch=now - 90)
    )

    fresh_result = _render(script, snapshot_path=recent, shell=shell)

    assert "archex fresh" in fresh_result.stdout
    assert "1m ago" in fresh_result.stdout

    old_epoch = now - DEFAULT_STALE_AFTER_SECONDS - 60
    old = _write_snapshot(
        tmp_path, _snapshot(written_at=utc_now_iso(old_epoch), written_at_epoch=old_epoch)
    )

    stale_result = _render(script, snapshot_path=old, shell=shell)

    assert stale_result.stdout.startswith("archex stale - unverified since measurement")


def test_a_shell_without_a_builtin_clock_reports_the_measured_state(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)
    ancient = _write_snapshot(tmp_path, _snapshot(written_at=utc_now_iso(1), written_at_epoch=1))

    result = _render(script, snapshot_path=ancient, shell="/bin/sh", env={"EPOCHSECONDS": ""})

    # No clock, so no `stale` claim and no age segment: the renderer reports
    # what was measured instead of inventing a freshness judgement.
    assert result.stdout.strip() == "archex fresh - 1234 files, 5678 chunks - rev 01234567"


def test_recent_watch_observation_is_surfaced(tmp_path: Path) -> None:
    shell = _clock_shell()
    if shell is None:
        pytest.skip("no host shell exposes EPOCHSECONDS")
    script = _installed_script(tmp_path)
    now = int(time.time())
    snapshot = _write_snapshot(
        tmp_path,
        _snapshot(
            written_at=utc_now_iso(now),
            written_at_epoch=now,
            watch_observed_at=utc_now_iso(now),
            watch_observed_epoch=now,
        ),
    )

    result = _render(script, snapshot_path=snapshot, shell=shell)

    assert result.stdout.rstrip().endswith("- watch")


def test_installed_script_name_is_the_ownership_marker(tmp_path: Path) -> None:
    script = _installed_script(tmp_path)

    assert script.name == STATUSLINE_SCRIPT_FILENAME
