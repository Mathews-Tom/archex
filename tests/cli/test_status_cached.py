"""`archex status --cached` contract (R23).

The default mode is authoritative and refreshes the cached snapshot; `--cached`
reads only that snapshot. These tests pin the split: the cached mode never
opens the index, it distinguishes every state the snapshot can be in, and the
authoritative mode keeps the cache honest -- including clearing it when there
is no longer an index to describe.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from click.testing import CliRunner

from archex.cli.main import cli
from archex.project import init_project
from archex.status_snapshot import (
    STATUS_SNAPSHOT_VERSION,
    SnapshotState,
    StatusSnapshot,
    publish_status,
    read_status,
    stale_after_seconds,
    status_snapshot_path,
    utc_now_iso,
)

if TYPE_CHECKING:
    import pytest

_NOW = 1_800_000_000.0


def _publish(repo: Path, *, now: float = _NOW, index_fresh: bool = True) -> None:
    publish_status(
        repo,
        index_fresh=index_fresh,
        index_revision="abc1234",
        generation_id="f" * 64,
        indexed_commit="c" * 40,
        current_commit="c" * 40,
        files_indexed=42,
        chunks_indexed=99,
        now=now,
    )


def _write_raw(repo: Path, payload: object) -> None:
    path = status_snapshot_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_cached_text_reports_the_measurement(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    _publish(python_simple_repo)

    result = CliRunner().invoke(cli, ["status", str(python_simple_repo), "--cached"])

    assert result.exit_code == 0, result.output
    assert "Cached state:       fresh" in result.output
    assert "Files indexed:      42" in result.output
    assert "Receipt:            complete" in result.output


def test_cached_json_carries_the_state_and_the_whole_snapshot(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    _publish(python_simple_repo)

    result = CliRunner().invoke(
        cli, ["status", str(python_simple_repo), "--cached", "--format", "json"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state"] == "fresh"
    assert payload["reader_version"] == STATUS_SNAPSHOT_VERSION
    assert payload["snapshot"]["generation_id"] == "f" * 64
    assert payload["snapshot_path"].endswith("status-snapshot.json")


def test_cached_mode_reports_missing_without_failing(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)

    result = CliRunner().invoke(cli, ["status", str(python_simple_repo), "--cached"])

    assert result.exit_code == 0, result.output
    assert "Cached state:       missing" in result.output
    assert "archex index" in result.output


def test_strict_cached_mode_fails_when_the_state_is_not_fresh(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)

    result = CliRunner().invoke(cli, ["status", str(python_simple_repo), "--cached", "--strict"])

    assert result.exit_code == 1


def test_corrupt_snapshot_fails_the_cached_command(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    path = status_snapshot_path(python_simple_repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    result = CliRunner().invoke(cli, ["status", str(python_simple_repo), "--cached"])

    assert result.exit_code == 1
    assert "Cached state:       corrupt" in result.output


def test_unsupported_version_fails_and_names_the_remedy(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    _write_raw(
        python_simple_repo,
        {"version": STATUS_SNAPSHOT_VERSION + 1, "state": "fresh"},
    )

    result = CliRunner().invoke(cli, ["status", str(python_simple_repo), "--cached"])

    assert result.exit_code == 1
    assert "Cached state:       unsupported" in result.output
    assert str(STATUS_SNAPSHOT_VERSION + 1) in result.output


def test_old_measurement_reports_stale_with_its_age(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    # The command reads the real clock, so the fixture is dated against it.
    stale_epoch = int(time.time()) - stale_after_seconds() - 60
    snapshot = StatusSnapshot(
        state=SnapshotState.FRESH,
        written_at=utc_now_iso(stale_epoch),
        written_at_epoch=stale_epoch,
        index_measured_at=utc_now_iso(stale_epoch),
        files_indexed=42,
    )
    _write_raw(python_simple_repo, snapshot.model_dump(mode="json"))

    result = CliRunner().invoke(cli, ["status", str(python_simple_repo), "--cached"])

    assert "Cached state:       stale" in result.output
    assert "freshness budget" in result.output
    # A stale snapshot still shows its last known measurement, labelled old.
    assert "Files indexed:      42" in result.output


def test_cached_mode_reads_the_snapshot_from_a_subdirectory(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    _publish(python_simple_repo)
    nested = python_simple_repo / "src" / "pkg"
    nested.mkdir(parents=True, exist_ok=True)

    result = CliRunner().invoke(cli, ["status", str(nested), "--cached"])

    # The shell and TypeScript renderers walk up to the nearest snapshot; the
    # CLI reader must answer for the same document rather than reporting
    # `missing` because it was invoked below the root.
    assert result.exit_code == 0, result.output
    assert "Cached state:       fresh" in result.output


def test_cached_mode_outside_any_repository_reports_missing(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["status", str(tmp_path), "--cached"])

    assert "Cached state:       missing" in result.output


def test_cached_mode_opens_no_index(
    python_simple_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_project(python_simple_repo)
    _publish(python_simple_repo)
    from archex.index.store import IndexStore

    def _forbidden(*_args: object, **_kwargs: object) -> object:
        message = "--cached must not open the index"
        raise AssertionError(message)

    monkeypatch.setattr(IndexStore, "__init__", _forbidden)

    result = CliRunner().invoke(cli, ["status", str(python_simple_repo), "--cached"])

    assert result.exit_code == 0, result.output
    assert "Cached state:       fresh" in result.output


def test_authoritative_status_refreshes_the_snapshot(python_simple_repo: Path) -> None:
    init_project(python_simple_repo)
    runner = CliRunner()
    indexed = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])
    assert indexed.exit_code == 0, indexed.output
    status_snapshot_path(python_simple_repo).unlink()

    result = runner.invoke(cli, ["status", str(python_simple_repo), "--format", "json"])

    assert result.exit_code == 0, result.output
    view = read_status(python_simple_repo)
    assert view.state.value == json.loads(result.output)["state"]
    assert view.snapshot is not None
    assert view.snapshot.generation_id != "", "the refresh must carry the real receipt fields"


def test_authoritative_status_clears_a_snapshot_with_no_index_left(
    python_simple_repo: Path,
) -> None:
    init_project(python_simple_repo)
    runner = CliRunner()
    assert runner.invoke(cli, ["index", str(python_simple_repo)]).exit_code == 0
    assert read_status(python_simple_repo).snapshot is not None
    (python_simple_repo / ".archex" / "index.db").unlink()

    runner.invoke(cli, ["status", str(python_simple_repo)])

    assert read_status(python_simple_repo).state.value == "missing"
