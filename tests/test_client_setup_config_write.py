"""``write_config_atomic``: the installer's config writer keeps what an in-place write kept.

``Path.write_text`` rewrites the existing inode, so it already preserves a
symlink and the file's mode; it is only unsafe because a kill between its
truncate and its write leaves the config empty. Each test pins a property a
plain temp-file-and-rename (or ``archex.state_file.write_text_atomic``) loses.
"""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from archex.client_setup import write_config_atomic

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextmanager
def _umask(mask: int) -> Iterator[None]:
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _names(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir())


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o755])
def test_existing_file_keeps_its_mode(tmp_path: Path, mode: int) -> None:
    target = tmp_path / "config.json"
    target.write_text("old", encoding="utf-8")
    target.chmod(mode)

    with _umask(0o022):
        write_config_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert _mode(target) == mode
    assert _names(tmp_path) == ["config.json"]


@pytest.mark.parametrize(("mask", "expected"), [(0o022, 0o644), (0o027, 0o640)])
def test_new_file_gets_the_mode_write_text_gives(tmp_path: Path, mask: int, expected: int) -> None:
    target = tmp_path / ".mcp.json"
    reference = tmp_path / "reference.json"

    with _umask(mask):
        write_config_atomic(target, "new")
        reference.write_text("new", encoding="utf-8")
        # The umask is read by setting it; the writer must have put it back.
        assert os.umask(mask) == mask

    assert target.read_text(encoding="utf-8") == "new"
    assert _mode(target) == expected
    assert _mode(target) == _mode(reference)


def test_symlink_is_kept_and_its_referent_is_replaced(tmp_path: Path) -> None:
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    referent = dotfiles / "claude.json"
    referent.write_text("old", encoding="utf-8")
    referent.chmod(0o600)
    home = tmp_path / "home"
    home.mkdir()
    link = home / ".claude.json"
    link_text = os.path.relpath(referent, home)
    link.symlink_to(link_text)

    with _umask(0o022):
        write_config_atomic(link, "new")

    assert link.is_symlink()
    assert os.readlink(link) == link_text
    assert not referent.is_symlink()
    assert referent.read_text(encoding="utf-8") == "new"
    assert link.read_text(encoding="utf-8") == "new"
    assert _mode(referent) == 0o600
    assert _names(home) == [".claude.json"]
    assert _names(dotfiles) == ["claude.json"]


@pytest.mark.parametrize(
    "failure",
    [OSError("disk gone"), KeyboardInterrupt()],
    ids=["oserror", "keyboard-interrupt"],
)
def test_failed_replace_keeps_the_original_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    target = tmp_path / "config.json"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o600)

    def failing_replace(src: object, dst: object) -> None:
        raise failure

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(type(failure)) as raised:
        write_config_atomic(target, "new")

    assert raised.value is failure
    assert target.read_text(encoding="utf-8") == "old"
    assert _mode(target) == 0o600
    assert _names(tmp_path) == ["config.json"]


@pytest.mark.parametrize("step", ["fdopen", "fchmod", "fsync"])
def test_failure_before_the_rename_closes_the_descriptor_and_removes_the_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    target = tmp_path / "config.json"
    target.write_text("old", encoding="utf-8")
    descriptors: list[int] = []
    real_mkstemp = tempfile.mkstemp

    def spying_mkstemp(*, dir: Path, prefix: str, suffix: str) -> tuple[int, str]:
        fd, name = real_mkstemp(dir=dir, prefix=prefix, suffix=suffix)
        descriptors.append(fd)
        return fd, name

    def failing_step(*args: object, **kwargs: object) -> None:
        raise OSError(f"{step} failed")

    monkeypatch.setattr(tempfile, "mkstemp", spying_mkstemp)
    monkeypatch.setattr(os, step, failing_step)

    with pytest.raises(OSError, match=f"{step} failed"):
        write_config_atomic(target, "new")

    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert target.read_text(encoding="utf-8") == "old"
    assert _names(tmp_path) == ["config.json"]
