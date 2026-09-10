"""The omp/Pi status extension module, executed under Bun (R23).

The module is TypeScript that runs inside the host process, so the only
faithful test is to load it in a real runtime and dispatch real events. These
tests write the rendered module to a temporary directory, drive it from a Bun
driver script, and assert on what it passed to `ctx.ui.setStatus` — the same
call the host's footer renders.

Skipped where `bun` is absent (including CI images without it); the structural
contract of the module is covered separately in
`tests/cli/test_install_client_status_adapters.py`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import TypedDict, cast

import pytest

from archex.client_setup import render_ts_status_module
from archex.status_snapshot import STATUS_SNAPSHOT_VERSION

pytestmark = pytest.mark.skipif(shutil.which("bun") is None, reason="bun is not installed")

_DRIVER = """
import archexStatusExtension, { renderArchexStatus } from "./archex-status.ts";

const calls: Array<[string, string | undefined]> = [];
const ctx = {
  hasUI: true,
  ui: {
    setStatus: (key: string, text: string | undefined) => {
      calls.push([key, text]);
    },
  },
};

const handlers: Record<string, Array<(event: unknown, context: unknown) => void>> = {};
archexStatusExtension({
  on(event: string, handler: (event: unknown, context: unknown) => void) {
    (handlers[event] ??= []).push(handler);
    return undefined;
  },
} as never);

for (const event of Object.keys(handlers)) {
  for (const handler of handlers[event] ?? []) handler({ toolName: "edit" }, ctx);
}

const snapshots: Record<string, string> = JSON.parse(process.env.ARCHEX_TEST_SNAPSHOTS ?? "{}");
const rendered: Record<string, string> = {};
for (const [label, path] of Object.entries(snapshots)) {
  process.env.ARCHEX_STATUS_SNAPSHOT = path;
  rendered[label] = renderArchexStatus(process.cwd());
}

const budgetOverride: Record<string, string> = {};
if (snapshots.recent) {
  process.env.ARCHEX_STATUS_SNAPSHOT = snapshots.recent;
  budgetOverride.default = renderArchexStatus(process.cwd());
  process.env.ARCHEX_STATUS_STALE_AFTER_SECONDS = "5";
  budgetOverride.tightened = renderArchexStatus(process.cwd());
  delete process.env.ARCHEX_STATUS_STALE_AFTER_SECONDS;
}

const withoutUi: Array<[string, string | undefined]> = [];
const headless = {
  hasUI: false,
  ui: {
    setStatus: (key: string, text: string | undefined) => {
      withoutUi.push([key, text]);
    },
  },
};
for (const handler of handlers.turn_end ?? []) handler({}, headless);

const events = Object.keys(handlers).sort();
console.log(JSON.stringify({ events, calls, rendered, withoutUi, budgetOverride }));
"""


class _DriverResult(TypedDict):
    """Shape the Bun driver prints as its single line of stdout."""

    events: list[str]
    calls: list[list[str]]
    rendered: dict[str, str]
    withoutUi: list[list[str]]
    budgetOverride: dict[str, str]


def _drive(tmp_path: Path, snapshots: dict[str, Path]) -> _DriverResult:
    (tmp_path / "archex-status.ts").write_text(render_ts_status_module(), encoding="utf-8")
    driver = tmp_path / "driver.ts"
    driver.write_text(_DRIVER, encoding="utf-8")
    result = subprocess.run(
        ["bun", "run", str(driver)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin:" + str(Path.home() / ".bun" / "bin"),
            "HOME": str(Path.home()),
            "ARCHEX_TEST_SNAPSHOTS": json.dumps({k: str(v) for k, v in snapshots.items()}),
        },
    )
    assert result.returncode == 0, result.stderr
    return cast("_DriverResult", json.loads(result.stdout.strip().splitlines()[-1]))


def _snapshot_file(tmp_path: Path, name: str, document: dict[str, object]) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_module_registers_three_refresh_events_and_sets_status(tmp_path: Path) -> None:
    now = int(time.time())
    # Written where the module discovers it by walking up from the process
    # working directory, so this covers discovery as well as dispatch.
    project = tmp_path / ".archex"
    project.mkdir()
    (project / "status-snapshot.json").write_text(
        json.dumps(
            {
                "version": STATUS_SNAPSHOT_VERSION,
                "state": "fresh",
                "files_indexed": 12,
                "chunks_indexed": 34,
                "index_revision": "0123456789abcdef",
                "written_at_epoch": now,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = _drive(tmp_path, {})

    assert payload["events"] == ["tool_result", "turn_end", "turn_start"]
    calls = payload["calls"]
    assert len(calls) == len(payload["events"]), "every registered event publishes a status"
    assert {call[0] for call in calls} == {"archex"}
    assert all("archex fresh - 12 files, 34 chunks - rev 01234567" in call[1] for call in calls)


def test_a_host_without_ui_receives_no_status_call(tmp_path: Path) -> None:
    payload = _drive(tmp_path, {})

    assert payload["withoutUi"] == [], "print and RPC modes have no status surface"


def test_every_state_renders_distinctly_under_bun(tmp_path: Path) -> None:
    now = int(time.time())
    base = {"version": STATUS_SNAPSHOT_VERSION, "index_revision": "0123456789abcdef"}
    snapshots = {
        "fresh": _snapshot_file(
            tmp_path,
            "fresh",
            {
                **base,
                "state": "fresh",
                "files_indexed": 12,
                "chunks_indexed": 34,
                "written_at_epoch": now,
            },
        ),
        "pending": _snapshot_file(
            tmp_path,
            "pending",
            {
                **base,
                "state": "pending",
                "pending_delta_files": 3,
                "pending_view_complete": True,
                "written_at_epoch": now,
            },
        ),
        "pending_truncated": _snapshot_file(
            tmp_path,
            "pending_truncated",
            {
                **base,
                "state": "pending",
                "pending_delta_files": 200,
                "pending_view_complete": False,
                "written_at_epoch": now,
            },
        ),
        "dirty": _snapshot_file(
            tmp_path, "dirty", {**base, "state": "dirty", "written_at_epoch": now}
        ),
        "reindex": _snapshot_file(
            tmp_path,
            "reindex",
            {**base, "state": "dirty", "reindex_required": True, "written_at_epoch": now},
        ),
        "stale": _snapshot_file(
            tmp_path, "stale", {**base, "state": "fresh", "written_at_epoch": now - 100_000}
        ),
        "unsupported": _snapshot_file(
            tmp_path, "unsupported", {"version": STATUS_SNAPSHOT_VERSION + 1, "state": "fresh"}
        ),
        "unknown_state": _snapshot_file(tmp_path, "unknown_state", {**base, "state": "sideways"}),
        "watching": _snapshot_file(
            tmp_path,
            "watching",
            {**base, "state": "fresh", "written_at_epoch": now, "watch_observed_epoch": now},
        ),
        "missing": tmp_path / "absent.json",
    }
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("not json", encoding="utf-8")
    snapshots["corrupt"] = corrupt

    rendered = _drive(tmp_path, snapshots)["rendered"]

    assert rendered["fresh"].startswith("archex fresh - 12 files, 34 chunks - rev 01234567")
    assert "archex pending - 3 awaiting sync" in rendered["pending"]
    assert "archex pending - 200+ awaiting sync" in rendered["pending_truncated"]
    assert "archex dirty - index behind tree" in rendered["dirty"]
    assert "archex dirty - reindex required" in rendered["reindex"]
    assert rendered["stale"].startswith("archex stale - unverified since measurement")
    assert rendered["missing"] == "archex missing - no status snapshot - run: archex index"
    assert rendered["corrupt"] == "archex corrupt - unreadable snapshot - run: archex status"
    assert rendered["unknown_state"] == "archex corrupt - unreadable snapshot - run: archex status"
    assert rendered["unsupported"] == (
        f"archex unsupported - snapshot v{STATUS_SNAPSHOT_VERSION + 1} - upgrade archex"
    )
    assert rendered["watching"].endswith(" - watch")


def test_the_stale_budget_override_is_honoured(tmp_path: Path) -> None:
    now = int(time.time())
    recent = _snapshot_file(
        tmp_path,
        "recent",
        {"version": STATUS_SNAPSHOT_VERSION, "state": "fresh", "written_at_epoch": now - 60},
    )

    payload = _drive(tmp_path, {"recent": recent})["budgetOverride"]

    # Every renderer reads ARCHEX_STATUS_STALE_AFTER_SECONDS, so a tuned
    # budget cannot make the three surfaces disagree about one snapshot.
    assert payload["default"].startswith("archex fresh")
    assert payload["tightened"].startswith("archex stale")


def test_an_unreadable_snapshot_renders_corrupt_rather_than_being_skipped(tmp_path: Path) -> None:
    project = tmp_path / ".archex"
    project.mkdir()
    unreadable = project / "status-snapshot.json"
    unreadable.write_text("{}", encoding="utf-8")
    unreadable.chmod(0o000)
    try:
        payload = _drive(tmp_path, {})
    finally:
        unreadable.chmod(0o600)

    calls = payload["calls"]
    # A present-but-unreadable document must classify, not be walked past to a
    # parent repository's snapshot.
    assert all("archex corrupt" in call[1] for call in calls)
