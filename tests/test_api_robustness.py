"""Robustness tests for api._acquire and ephemeral IndexStore cleanup on failure."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from archex.api import (
    _acquire,  # pyright: ignore[reportPrivateUsage]
    _DeltaIndexAttempt,  # pyright: ignore[reportPrivateUsage]
    _ensure_index,  # pyright: ignore[reportPrivateUsage]
    _full_index,  # pyright: ignore[reportPrivateUsage]
    _try_delta_index,  # pyright: ignore[reportPrivateUsage]
    index_repository,
    query,
)
from archex.cache import CacheManager
from archex.index.store import IndexStore
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


def test_full_index_closes_ephemeral_store_on_keyboard_interrupt(
    python_simple_repo: Path,
) -> None:
    """A Ctrl-C (KeyboardInterrupt) mid-pipeline must also not leak the scratch dir.

    `except Exception` does not catch KeyboardInterrupt/SystemExit/GeneratorExit
    (they subclass BaseException, not Exception) — a guard using the narrower
    clause would skip cleanup for exactly this case. Distinct from the
    RuntimeError-based test above, which cannot tell the two clauses apart.
    """
    source = RepoSource(local_path=str(python_simple_repo))
    config = Config(cache=False)
    cache = CacheManager(cache_dir=str(python_simple_repo.parent / "cache"))
    cache_key = cache.cache_key(source)

    before = _tmp_dirs()
    with (
        patch("archex.index.store.IndexStore.insert_chunks", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
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


def _close_tracker() -> tuple[list[IndexStore], object]:
    """Wrap IndexStore.close to record every call while still closing for real."""
    close_calls: list[IndexStore] = []
    real_close = IndexStore.close

    def tracking_close(self: IndexStore) -> None:
        close_calls.append(self)
        real_close(self)

    return close_calls, tracking_close


def _instance_tracker() -> tuple[list[IndexStore], list[IndexStore], object, object]:
    """Wrap IndexStore.__init__ and .close to record every instance created and closed.

    _try_delta_index opens several IndexStore instances in sequence before the
    one under test (candidate_store, a manifest-computation store), each
    already correctly closed by pre-existing, unrelated guards. A test that
    only asserts "some store was closed" (via `_close_tracker` above) passes
    vacuously off those unrelated closes even when the store actually under
    test is never closed. Asserting `created[-1] in closed` instead pins the
    check to the store constructed last — the one the test is actually
    exercising.
    """
    created: list[IndexStore] = []
    closed: list[IndexStore] = []
    real_init = IndexStore.__init__
    real_close = IndexStore.close

    def tracking_init(self: IndexStore, *args: object, **kwargs: object) -> None:
        real_init(self, *args, **kwargs)  # type: ignore[arg-type]
        created.append(self)

    def tracking_close(self: IndexStore) -> None:
        closed.append(self)
        real_close(self)

    return created, closed, tracking_init, tracking_close


def _set_metadata_raise_if_indexed_at(key: str, value: str) -> None:
    """set_metadata side effect that only fails for clean_store's own write.

    IndexStore.__init__ -> _migrate_schema() calls set_metadata("schema_version", ...)
    for every store construction, so a blanket "always raise" patch would
    fail before the code path under test is ever reached.
    """
    del value
    if key == "indexed_at":
        raise RuntimeError("boom")


def test_ensure_index_cache_hit_closes_store_on_mid_pipeline_exception(
    python_simple_repo: Path, tmp_path: Path
) -> None:
    """A cache-hit pipeline failure must still close the (non-ephemeral) cached store."""
    source = RepoSource(local_path=str(python_simple_repo))
    config = Config(cache=True, cache_dir=str(tmp_path / "cache"))

    index_repository(source, config=config).close()

    close_calls, tracking_close = _close_tracker()
    with (
        patch.object(IndexStore, "close", tracking_close),
        patch.object(IndexStore, "needs_reindex", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        _ensure_index(source, config=config)

    assert close_calls, "store.close() must run even when a cache-hit pipeline step raises"


def test_try_delta_index_clean_store_closes_on_mid_pipeline_exception(
    python_simple_repo: Path, tmp_path: Path
) -> None:
    """The delta path's 'no working-tree changes' clean_store must close even
    when a step after opening it raises."""
    source = RepoSource(local_path=str(python_simple_repo))
    config = Config(cache=True, cache_dir=str(tmp_path / "cache"))
    cache = CacheManager(cache_dir=str(tmp_path / "cache"))
    cache_key = cache.cache_key(source)

    index_repository(source, config=config).close()

    attempt = _DeltaIndexAttempt(
        source=source,
        config=config,
        cache=cache,
        cache_key=cache_key,
        working_tree_signature=None,
        timing=None,
        index_config=None,
        t_start=0.0,
    )

    created, closed, tracking_init, tracking_close = _instance_tracker()
    with (
        patch.object(IndexStore, "__init__", tracking_init),
        patch.object(IndexStore, "close", tracking_close),
        patch.object(
            IndexStore,
            "set_metadata",
            side_effect=_set_metadata_raise_if_indexed_at,
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        _try_delta_index(attempt)

    assert created, "expected _try_delta_index to construct at least one IndexStore"
    assert created[-1] in closed, (
        "clean_store (the last-constructed store, not an earlier unrelated one) "
        "must close even when a delta-clean step raises"
    )


def test_try_delta_index_main_store_closes_on_mid_pipeline_exception(
    python_simple_repo: Path, tmp_path: Path
) -> None:
    """The delta path's main apply-delta store must close even when a step
    after opening it raises."""
    source = RepoSource(local_path=str(python_simple_repo))
    config = Config(cache=True, cache_dir=str(tmp_path / "cache"))
    cache = CacheManager(cache_dir=str(tmp_path / "cache"))
    cache_key = cache.cache_key(source)

    index_repository(source, config=config).close()

    # A real content change so compute_working_tree_delta reports a non-empty
    # manifest and _try_delta_index reaches the main apply-delta store instead
    # of the "no changes" clean_store branch.
    (python_simple_repo / "utils.py").write_text("def freshly_changed(): return 1\n")

    attempt = _DeltaIndexAttempt(
        source=source,
        config=config,
        cache=cache,
        cache_key=cache_key,
        working_tree_signature=None,
        timing=None,
        index_config=None,
        t_start=0.0,
    )

    created, closed, tracking_init, tracking_close = _instance_tracker()
    with (
        patch.object(IndexStore, "__init__", tracking_init),
        patch.object(IndexStore, "close", tracking_close),
        patch("archex.index.delta.apply_delta", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        _try_delta_index(attempt)

    assert created, "expected _try_delta_index to construct at least one IndexStore"
    assert created[-1] in closed, (
        "the main apply-delta store (the last-constructed store, not an earlier "
        "unrelated one) must close even when apply_delta raises"
    )
