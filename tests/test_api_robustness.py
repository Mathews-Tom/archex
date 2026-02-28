"""Robustness tests for api._acquire: cleanup on success and exception paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from archex.api import _acquire  # pyright: ignore[reportPrivateUsage]
from archex.models import RepoSource


def test_acquire_local_path_returns_noop_cleanup(tmp_path: Path) -> None:
    """Local path _acquire returns a no-op cleanup callable."""
    source = RepoSource(local_path=str(tmp_path))
    with patch("archex.api.open_local", return_value=tmp_path):
        repo_path, url, local_path, cleanup = _acquire(source)

    assert repo_path == tmp_path
    assert url is None
    assert local_path == str(tmp_path)
    # No-op cleanup should not raise and not delete anything
    cleanup()
    assert tmp_path.exists()


def test_acquire_url_cleanup_removes_tempdir() -> None:
    """URL _acquire cleanup removes the cloned tempdir."""
    cloned_dir: list[Path] = []

    def fake_clone(url: str, target: str) -> Path:
        p = Path(target)
        p.mkdir(parents=True, exist_ok=True)
        cloned_dir.append(p)
        return p

    source = RepoSource(url="https://example.com/repo.git")
    with patch("archex.api.clone_repo", side_effect=fake_clone):
        _repo_path, _url, _local_path, cleanup = _acquire(source)

    assert len(cloned_dir) == 1
    target = cloned_dir[0]
    assert target.exists()

    cleanup()
    assert not target.exists()


def test_acquire_url_cleanup_called_on_exception() -> None:
    """Cleanup callable, when used in try/finally, executes on exception."""

    def fake_clone(url: str, target: str) -> Path:
        p = Path(target)
        p.mkdir(parents=True, exist_ok=True)
        return p

    source = RepoSource(url="https://example.com/repo.git")
    with patch("archex.api.clone_repo", side_effect=fake_clone):
        _repo_path, _url, _local_path, cleanup = _acquire(source)

    target = _repo_path
    assert target.exists()

    try:
        raise RuntimeError("pipeline failure")
    except RuntimeError:
        pass
    finally:
        cleanup()

    assert not target.exists()


def test_repo_source_requires_url_or_local_path() -> None:
    """RepoSource model validator rejects construction without url or local_path."""
    with pytest.raises(ValueError, match="requires either"):
        RepoSource()


def test_acquire_local_path_cleanup_is_idempotent(tmp_path: Path) -> None:
    """Local path cleanup can be called multiple times without error."""
    source = RepoSource(local_path=str(tmp_path))
    with patch("archex.api.open_local", return_value=tmp_path):
        _repo_path, _url, _local_path, cleanup = _acquire(source)
    cleanup()
    cleanup()  # second call should not raise


def test_acquire_url_cleanup_safe_on_missing_dir() -> None:
    """URL cleanup is safe even if the tempdir was already removed."""

    def fake_clone(url: str, target: str) -> Path:
        return Path(target)

    source = RepoSource(url="https://example.com/repo.git")
    with patch("archex.api.clone_repo", side_effect=fake_clone):
        _repo_path, _url, _local_path, cleanup = _acquire(source)

    # Dir was never actually created; cleanup with ignore_errors=True should not raise
    cleanup()
