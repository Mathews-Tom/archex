"""Unit tests for archex.serve.runtime.QueryRuntime: generation-keyed warm cache."""

from __future__ import annotations

from pathlib import Path

from archex.api import _ensure_index  # pyright: ignore[reportPrivateUsage]
from archex.models import Config, IndexConfig, RepoSource
from archex.serve.generation import read_generation_id
from archex.serve.runtime import QueryRuntime


def _index_and_generation_id(
    python_simple_repo: Path, cache_dir: Path, index_config: IndexConfig | None = None
) -> tuple[str, str, str]:
    """Index the fixture repo and return (cache_key, db_path, generation_id)."""
    from archex.api import _cache_manager_for_source  # pyright: ignore[reportPrivateUsage]

    source = RepoSource(local_path=str(python_simple_repo))
    config = Config(languages=["python"], cache=True, cache_dir=str(cache_dir))
    store = _ensure_index(source, config=config, index_config=index_config)
    manager = _cache_manager_for_source(source, config)
    cache_key = manager.cache_key(source)
    try:
        generation_id = read_generation_id(store)
        assert generation_id is not None
        db_path = str(manager.db_path(cache_key))
    finally:
        store.close()
    return cache_key, db_path, generation_id


class TestQueryRuntimeGetOrBuild:
    def test_returns_none_for_none_generation_id(self, tmp_path: Path) -> None:
        runtime = QueryRuntime()
        assert (
            runtime.get_or_build("some-key", str(tmp_path / "idx.db"), None, IndexConfig()) is None
        )

    def test_second_call_with_same_generation_reuses_snapshot(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        cache_key, db_path, generation_id = _index_and_generation_id(
            python_simple_repo, tmp_path / "cache"
        )
        runtime = QueryRuntime()
        try:
            first = runtime.get_or_build(cache_key, db_path, generation_id, IndexConfig())
            second = runtime.get_or_build(cache_key, db_path, generation_id, IndexConfig())
            assert first is not None
            assert second is first
        finally:
            runtime.close()

    def test_snapshot_holds_expected_derived_state(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        cache_key, db_path, generation_id = _index_and_generation_id(
            python_simple_repo, tmp_path / "cache"
        )
        runtime = QueryRuntime()
        try:
            snapshot = runtime.get_or_build(cache_key, db_path, generation_id, IndexConfig())
            assert snapshot is not None
            assert snapshot.generation_id == generation_id
            assert len(snapshot.all_chunks) > 0
            assert snapshot.graph.file_count > 0
        finally:
            runtime.close()

    def test_different_generation_id_rebuilds_and_closes_previous_store(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        cache_key, db_path, generation_id = _index_and_generation_id(
            python_simple_repo, tmp_path / "cache"
        )
        runtime = QueryRuntime()
        try:
            first = runtime.get_or_build(cache_key, db_path, generation_id, IndexConfig())
            assert first is not None

            second = runtime.get_or_build(
                cache_key, db_path, "a-different-generation-id", IndexConfig()
            )
            assert second is not None
            assert second is not first
            assert second.generation_id == "a-different-generation-id"
            # The superseded snapshot's store connection must be closed, not leaked.
            import sqlite3

            try:
                first.store.conn.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                pass
            else:
                raise AssertionError("expected the superseded snapshot's store to be closed")
        finally:
            runtime.close()

    def test_different_cache_key_builds_independent_snapshots(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        """The runtime isolates snapshots purely by cache_key string, so two
        distinct keys never collide even against the same underlying store."""
        cache_key, db_path, generation_id = _index_and_generation_id(
            python_simple_repo, tmp_path / "cache"
        )
        runtime = QueryRuntime()
        try:
            snapshot_a = runtime.get_or_build(
                f"{cache_key}-a", db_path, generation_id, IndexConfig()
            )
            snapshot_b = runtime.get_or_build(
                f"{cache_key}-b", db_path, generation_id, IndexConfig()
            )
            assert snapshot_a is not None
            assert snapshot_b is not None
            assert snapshot_a is not snapshot_b
            # Repeat lookups against each key keep returning that key's own snapshot.
            assert (
                runtime.get_or_build(f"{cache_key}-a", db_path, generation_id, IndexConfig())
                is snapshot_a
            )
            assert (
                runtime.get_or_build(f"{cache_key}-b", db_path, generation_id, IndexConfig())
                is snapshot_b
            )
        finally:
            runtime.close()

    def test_close_closes_every_cached_snapshot_store(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        import sqlite3

        cache_key, db_path, generation_id = _index_and_generation_id(
            python_simple_repo, tmp_path / "cache"
        )
        runtime = QueryRuntime()
        snapshot = runtime.get_or_build(cache_key, db_path, generation_id, IndexConfig())
        assert snapshot is not None
        runtime.close()
        try:
            snapshot.store.conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            pass
        else:
            raise AssertionError("expected close() to close the cached store")
