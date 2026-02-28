"""Tests for CacheManager."""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

import pytest

from archex.cache import CacheManager
from archex.exceptions import CacheError
from archex.models import RepoSource

if TYPE_CHECKING:
    from pathlib import Path

# Valid 64-char hex keys for use in tests
KEY_A = "a" * 64
KEY_B = "b" * 64
KEY_C = "c" * 64
KEY_DELETE = "d" * 63 + "e"
KEY_OLD = "0" * 63 + "1"
KEY_NEW = "0" * 63 + "2"
KEY_K1 = hashlib.sha256(b"k1").hexdigest()
KEY_K2 = hashlib.sha256(b"k2").hexdigest()
KEY_INFO = hashlib.sha256(b"info").hexdigest()


@pytest.fixture()
def cache(tmp_path: Path) -> CacheManager:
    return CacheManager(cache_dir=str(tmp_path / "cache"))


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    db = tmp_path / "sample.db"
    db.write_bytes(b"SQLITE")
    return db


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_put_get_roundtrip(cache: CacheManager, sample_db: Path) -> None:
    cache.put(KEY_A, sample_db)
    result = cache.get(KEY_A)
    assert result is not None
    assert result.exists()
    assert result.read_bytes() == b"SQLITE"


def test_get_returns_none_for_missing_key(cache: CacheManager) -> None:
    assert cache.get(KEY_B) is None


def test_invalidate_removes_entry(cache: CacheManager, sample_db: Path) -> None:
    cache.put(KEY_DELETE, sample_db)
    assert cache.get(KEY_DELETE) is not None
    cache.invalidate(KEY_DELETE)
    assert cache.get(KEY_DELETE) is None


def test_invalidate_nonexistent_key_is_safe(cache: CacheManager) -> None:
    # Should not raise
    cache.invalidate(KEY_C)


def test_clean_removes_old_entries(cache: CacheManager, sample_db: Path) -> None:
    cache.put(KEY_OLD, sample_db)
    # Backdate the meta file
    meta = cache.meta_path(KEY_OLD)
    meta.write_text(str(time.time() - 48 * 3600))  # 48 hours ago
    removed = cache.clean(max_age_hours=24)
    assert removed == 1
    assert cache.get(KEY_OLD) is None


def test_clean_keeps_recent_entries(cache: CacheManager, sample_db: Path) -> None:
    cache.put(KEY_NEW, sample_db)
    removed = cache.clean(max_age_hours=24)
    assert removed == 0
    assert cache.get(KEY_NEW) is not None


def test_list_entries_returns_correct_data(cache: CacheManager, sample_db: Path) -> None:
    cache.put(KEY_K1, sample_db)
    cache.put(KEY_K2, sample_db)
    entries = cache.list_entries()
    assert len(entries) == 2
    keys = {e["key"] for e in entries}
    assert KEY_K1 in keys
    assert KEY_K2 in keys
    for e in entries:
        assert "size_bytes" in e
        assert "path" in e
        assert "created_at" in e


def test_info_returns_summary(cache: CacheManager, sample_db: Path) -> None:
    cache.put(KEY_INFO, sample_db)
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


# ---------------------------------------------------------------------------
# Key validation — adversarial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    [
        "../../etc/passwd",
        "; rm -rf /",
        "abc123",  # too short
        "a" * 63,  # 63 chars — one short
        "a" * 65,  # 65 chars — one long
        "A" * 64,  # uppercase not allowed
        "z" * 64,  # 'z' is not hex
        "",
        "deadbeef",
        "../" + "a" * 61,
    ],
)
def test_db_path_rejects_invalid_key(cache: CacheManager, bad_key: str) -> None:
    with pytest.raises(CacheError, match="Invalid cache key"):
        cache.db_path(bad_key)


@pytest.mark.parametrize(
    "bad_key",
    [
        "../../etc/passwd",
        "; rm -rf /",
        "abc123",
        "a" * 63,
        "a" * 65,
        "A" * 64,
    ],
)
def test_meta_path_rejects_invalid_key(cache: CacheManager, bad_key: str) -> None:
    with pytest.raises(CacheError, match="Invalid cache key"):
        cache.meta_path(bad_key)


def test_db_path_accepts_valid_sha256_key(cache: CacheManager) -> None:
    key = hashlib.sha256(b"test").hexdigest()
    path = cache.db_path(key)
    assert path.name == f"{key}.db"


def test_meta_path_accepts_valid_sha256_key(cache: CacheManager) -> None:
    key = hashlib.sha256(b"test").hexdigest()
    path = cache.meta_path(key)
    assert path.name == f"{key}.meta"
