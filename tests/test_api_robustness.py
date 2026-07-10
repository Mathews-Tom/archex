"""Robustness tests for api._acquire and ephemeral IndexStore cleanup on failure."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from archex.api import _acquire, _full_index, query  # pyright: ignore[reportPrivateUsage]
from archex.cache import CacheManager
from archex.models import Config, RepoSource


def test_acquire_local_path_returns_noop_cleanup(tmp_path: Path) -> None:
    """Local path _acquire returns a no-op cleanup callable."""
    source = RepoSource(local_path=str(tmp_path))
    with patch("archex.api.open_local", return_value=tmp_path):
        repo_path, url, local_path, cleanup, _head = _acquire(source)

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
        _repo_path, _url, _local_path, cleanup, _head = _acquire(source)

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
        _repo_path, _url, _local_path, cleanup, _head = _acquire(source)

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
        _repo_path, _url, _local_path, cleanup, _head = _acquire(source)
    cleanup()
    cleanup()  # second call should not raise


def test_acquire_url_cleanup_safe_on_missing_dir() -> None:
    """URL cleanup is safe even if the tempdir was already removed."""

    def fake_clone(url: str, target: str) -> Path:
        return Path(target)

    source = RepoSource(url="https://example.com/repo.git")
    with patch("archex.api.clone_repo", side_effect=fake_clone):
        _repo_path, _url, _local_path, cleanup, _head = _acquire(source)

    # Dir was never actually created; cleanup with ignore_errors=True should not raise
    cleanup()


def _tmp_dirs() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).iterdir() if p.is_dir()}


def test_full_index_closes_ephemeral_store_on_mid_pipeline_exception(
    python_simple_repo: Path,
) -> None:
    """A pipeline failure after the ephemeral store opens must not leak its scratch dir.

    _full_index() builds its IndexStore under a fresh mkdtemp() directory with
    delete_dir_on_close=True; that cleanup only fires on close(), so a failure
    anywhere in the pipeline between construction and the normal return must
    still close (and thus clean up) the store rather than leaving it open.
    """
    source = RepoSource(local_path=str(python_simple_repo))
    config = Config(cache=False)
    cache = CacheManager(cache_dir=str(python_simple_repo.parent / "cache"))
    cache_key = cache.cache_key(source)

    before = _tmp_dirs()
    with (
        patch("archex.index.store.IndexStore.insert_chunks", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        _full_index(source, config, cache, cache_key, timing=None)
    after = _tmp_dirs()

    assert after == before, f"leaked scratch dirs: {after - before}"


def test_query_closes_ephemeral_store_on_mid_pipeline_exception(python_simple_repo: Path) -> None:
    """query()'s ephemeral store must not leak its scratch dir on a mid-pipeline failure."""
    source = RepoSource(local_path=str(python_simple_repo))
    config = Config(cache=False)

    before = _tmp_dirs()
    with (
        patch("archex.index.store.IndexStore.insert_chunks", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        query(source, "how does auth work?", config=config)
    after = _tmp_dirs()

    assert after == before, f"leaked scratch dirs: {after - before}"
