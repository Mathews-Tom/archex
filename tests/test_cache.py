"""Tests for CacheManager."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from archex.cache import CacheManager
from archex.models import RepoSource

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def cache(tmp_path: Path) -> CacheManager:
    return CacheManager(cache_dir=str(tmp_path / "cache"))


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    db = tmp_path / "sample.db"
    db.write_bytes(b"SQLITE")
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_put_get_roundtrip(cache: CacheManager, sample_db: Path) -> None:
    key = "abc123"
    cache.put(key, sample_db)
    result = cache.get(key)
    assert result is not None
    assert result.exists()
    assert result.read_bytes() == b"SQLITE"


def test_get_returns_none_for_missing_key(cache: CacheManager) -> None:
    assert cache.get("nonexistent") is None


def test_invalidate_removes_entry(cache: CacheManager, sample_db: Path) -> None:
    key = "to_delete"
    cache.put(key, sample_db)
    assert cache.get(key) is not None
    cache.invalidate(key)
    assert cache.get(key) is None


def test_invalidate_nonexistent_key_is_safe(cache: CacheManager) -> None:
    # Should not raise
    cache.invalidate("does_not_exist")


def test_clean_removes_old_entries(cache: CacheManager, sample_db: Path) -> None:
    key = "old_entry"
    cache.put(key, sample_db)
    # Backdate the meta file
    meta = cache.meta_path(key)
    meta.write_text(str(time.time() - 48 * 3600))  # 48 hours ago
    removed = cache.clean(max_age_hours=24)
    assert removed == 1
    assert cache.get(key) is None


def test_clean_keeps_recent_entries(cache: CacheManager, sample_db: Path) -> None:
    key = "new_entry"
    cache.put(key, sample_db)
    removed = cache.clean(max_age_hours=24)
    assert removed == 0
    assert cache.get(key) is not None


def test_list_entries_returns_correct_data(cache: CacheManager, sample_db: Path) -> None:
    cache.put("k1", sample_db)
    cache.put("k2", sample_db)
    entries = cache.list_entries()
    assert len(entries) == 2
    keys = {e["key"] for e in entries}
    assert "k1" in keys
    assert "k2" in keys
    for e in entries:
        assert "size_bytes" in e
        assert "path" in e
        assert "created_at" in e


def test_info_returns_summary(cache: CacheManager, sample_db: Path) -> None:
    cache.put("k1", sample_db)
    info = cache.info()
    assert info["total_entries"] == 1
    assert info["total_size_bytes"] > 0
    assert "cache_dir" in info


def test_cache_key_is_stable(cache: CacheManager) -> None:
    source = RepoSource(url="https://github.com/example/repo")
    key1 = cache.cache_key(source)
    key2 = cache.cache_key(source)
    assert key1 == key2
    assert len(key1) == 64  # SHA256 hex


def test_cache_key_differs_by_url(cache: CacheManager) -> None:
    s1 = RepoSource(url="https://github.com/a/repo")
    s2 = RepoSource(url="https://github.com/b/repo")
    assert cache.cache_key(s1) != cache.cache_key(s2)


def test_cache_key_local_path(cache: CacheManager) -> None:
    source = RepoSource(local_path="/home/user/project")
    key = cache.cache_key(source)
    assert len(key) == 64
