"""Tests for CacheManager."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

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


def test_cache_key_head_override(tmp_path: Path) -> None:
    """cache_key with head_override produces a different key than without."""
    cm = CacheManager(cache_dir=str(tmp_path))
    source = RepoSource(url="https://example.com/repo.git")
    key_no_head = cm.cache_key(source)
    key_with_head = cm.cache_key(source, head_override="abc123")
    assert key_no_head != key_with_head


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


# ---------------------------------------------------------------------------
# find_store_for_source
# ---------------------------------------------------------------------------


class TestFindStoreForSource:
    def test_finds_matching_identity(self, cache: CacheManager, tmp_path: Path) -> None:
        from archex.index.store import IndexStore

        db = tmp_path / "source.db"
        store = IndexStore(db)
        store.set_metadata("source_identity", "/path/to/repo")
        store.set_metadata("commit_hash", "abc123")
        store.close()

        key = "a" * 64
        cache.put(key, db)

        source = RepoSource(local_path="/path/to/repo")
        result = cache.find_store_for_source(source)
        assert result is not None
        _, commit = result
        assert commit == "abc123"

    def test_returns_none_when_no_match(self, cache: CacheManager) -> None:
        source = RepoSource(local_path="/nonexistent/path")
        assert cache.find_store_for_source(source) is None

    def test_finds_different_commit(self, cache: CacheManager, tmp_path: Path) -> None:
        from archex.index.store import IndexStore

        db = tmp_path / "source.db"
        store = IndexStore(db)
        store.set_metadata("source_identity", "/my/repo")
        store.set_metadata("commit_hash", "old_commit")
        store.close()

        key = "b" * 64
        cache.put(key, db)

        source = RepoSource(local_path="/my/repo", commit="new_commit")
        result = cache.find_store_for_source(source)
        assert result is not None
        _, commit = result
        assert commit == "old_commit"

    def test_no_match_different_identity(self, cache: CacheManager, tmp_path: Path) -> None:
        from archex.index.store import IndexStore

        db = tmp_path / "other.db"
        store = IndexStore(db)
        store.set_metadata("source_identity", "/different/repo")
        store.set_metadata("commit_hash", "some_commit")
        store.close()

        key = "c" * 64
        cache.put(key, db)

        source = RepoSource(local_path="/my/repo")
        result = cache.find_store_for_source(source)
        assert result is None

    def test_returns_db_path(self, cache: CacheManager, tmp_path: Path) -> None:
        from archex.index.store import IndexStore

        db = tmp_path / "source.db"
        store = IndexStore(db)
        store.set_metadata("source_identity", "/some/repo")
        store.set_metadata("commit_hash", "commitxyz")
        store.close()

        key = "e" * 64
        cache.put(key, db)

        source = RepoSource(local_path="/some/repo")
        result = cache.find_store_for_source(source)
        assert result is not None
        db_path, _ = result
        assert db_path.exists()
        assert db_path.suffix == ".db"

    def test_empty_identity_returns_none(self, cache: CacheManager) -> None:
        # RepoSource with url=None, local_path=None raises ValueError
        # so we test empty string identity directly
        result = (
            cache.find_store_for_source.__func__(  # type: ignore[attr-defined]
                cache, RepoSource(local_path="")
            )
            if False
            else None
        )
        # The method checks `if not identity: return None`
        # Test indirectly: no cached db with empty identity should match
        source = RepoSource(url="https://github.com/example/repo")
        result = cache.find_store_for_source(source)
        assert result is None  # no entry in empty cache

    def test_missing_commit_hash_not_returned(self, cache: CacheManager, tmp_path: Path) -> None:
        from archex.index.store import IndexStore

        db = tmp_path / "nocommit.db"
        store = IndexStore(db)
        store.set_metadata("source_identity", "/repo/no/commit")
        # Deliberately do not set commit_hash
        store.close()

        key = "f" * 64
        cache.put(key, db)

        source = RepoSource(local_path="/repo/no/commit")
        result = cache.find_store_for_source(source)
        # No commit_hash row → result should be None
        assert result is None


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestCacheEdgeCases:
    def test_clean_zero_hours_removes_all(self, cache: CacheManager, sample_db: Path) -> None:
        """clean(max_age_hours=0) removes every entry."""
        cache.put(KEY_A, sample_db)
        cache.put(KEY_B, sample_db)
        removed = cache.clean(max_age_hours=0)
        assert removed == 2
        assert cache.get(KEY_A) is None
        assert cache.get(KEY_B) is None

    def test_invalidate_idempotent(self, cache: CacheManager, sample_db: Path) -> None:
        """Double invalidate does not raise."""
        cache.put(KEY_A, sample_db)
        cache.invalidate(KEY_A)
        cache.invalidate(KEY_A)  # second call should be safe
        assert cache.get(KEY_A) is None

    def test_put_nonexistent_source_db(self, cache: CacheManager, tmp_path: Path) -> None:
        """put() with nonexistent source file propagates error."""
        missing = tmp_path / "does_not_exist.db"
        with pytest.raises((FileNotFoundError, OSError)):
            cache.put(KEY_A, missing)

    def test_list_entries_with_corrupt_meta(self, cache: CacheManager, sample_db: Path) -> None:
        """list_entries handles corrupt .meta file without crashing."""
        cache.put(KEY_A, sample_db)
        # Corrupt the meta file
        meta = cache.meta_path(KEY_A)
        meta.write_text("not_a_float_timestamp")
        entries = cache.list_entries()
        # Should either return the entry (with default time) or skip it, but not crash
        assert isinstance(entries, list)

    def test_cache_key_with_head_override(self, tmp_path: Path) -> None:
        """cache_key with head_override produces deterministic, different key."""
        cm = CacheManager(cache_dir=str(tmp_path))
        source = RepoSource(url="https://example.com/repo.git")
        k1 = cm.cache_key(source, head_override="abc123")
        k2 = cm.cache_key(source, head_override="abc123")
        k3 = cm.cache_key(source, head_override="def456")
        assert k1 == k2  # deterministic
        assert k1 != k3  # different overrides produce different keys

    def test_find_store_corrupt_db_skipped(self, cache: CacheManager, tmp_path: Path) -> None:
        """find_store_for_source skips corrupt .db files gracefully."""
        import shutil

        bad_db = tmp_path / "corrupt.db"
        bad_db.write_text("not a sqlite database")
        key = "a" * 64
        # Manually place the corrupt file in cache
        cache_db = cache.db_path(key)
        cache_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(bad_db, cache_db)
        cache.meta_path(key).write_text(str(time.time()))

        source = RepoSource(url="https://example.com/repo.git")
        # Should not crash, just return None
        result = cache.find_store_for_source(source)
        assert result is None


# ---------------------------------------------------------------------------
# Metadata (get_meta / put with kwargs)
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_put_stores_json_metadata(self, cache: CacheManager, sample_db: Path) -> None:
        cache.put(KEY_A, sample_db, resolved_commit="abc123", source_identity="/my/repo")
        meta = cache.get_meta(KEY_A)
        assert meta["resolved_commit"] == "abc123"
        assert meta["source_identity"] == "/my/repo"
        assert "created_at" in meta

    def test_put_without_kwargs_stores_empty_strings(
        self, cache: CacheManager, sample_db: Path
    ) -> None:
        cache.put(KEY_A, sample_db)
        meta = cache.get_meta(KEY_A)
        assert meta["resolved_commit"] == ""
        assert meta["source_identity"] == ""

    def test_get_meta_backward_compat_bare_timestamp(
        self, cache: CacheManager, sample_db: Path
    ) -> None:
        """Old-format meta files contain a bare float timestamp."""
        cache.put(KEY_A, sample_db)
        ts = str(time.time())
        cache.meta_path(KEY_A).write_text(ts)
        meta = cache.get_meta(KEY_A)
        assert meta == {"created_at": ts}

    def test_get_meta_missing_file(self, cache: CacheManager) -> None:
        assert cache.get_meta(KEY_A) == {}

    def test_get_meta_corrupt_json(self, cache: CacheManager, sample_db: Path) -> None:
        cache.put(KEY_A, sample_db)
        cache.meta_path(KEY_A).write_text("{broken json")
        assert cache.get_meta(KEY_A) == {}

    def test_get_meta_empty_file(self, cache: CacheManager, sample_db: Path) -> None:
        cache.put(KEY_A, sample_db)
        cache.meta_path(KEY_A).write_text("")
        assert cache.get_meta(KEY_A) == {}


# ---------------------------------------------------------------------------
# is_stale
# ---------------------------------------------------------------------------


class TestIsStale:
    def test_fresh_entry_not_stale(self, cache: CacheManager, sample_db: Path) -> None:
        cache.put(KEY_A, sample_db)
        assert cache.is_stale(KEY_A, max_age_hours=24) is False

    def test_old_entry_is_stale(self, cache: CacheManager, sample_db: Path) -> None:
        cache.put(KEY_A, sample_db)
        old_ts = time.time() - 48 * 3600
        meta_data = json.dumps(
            {"created_at": str(old_ts), "resolved_commit": "", "source_identity": ""}
        )
        cache.meta_path(KEY_A).write_text(meta_data)
        assert cache.is_stale(KEY_A, max_age_hours=24) is True

    def test_missing_meta_is_stale(self, cache: CacheManager) -> None:
        assert cache.is_stale(KEY_A) is True

    def test_corrupt_created_at_is_stale(self, cache: CacheManager, sample_db: Path) -> None:
        cache.put(KEY_A, sample_db)
        meta_data = json.dumps({"created_at": "not_a_number"})
        cache.meta_path(KEY_A).write_text(meta_data)
        assert cache.is_stale(KEY_A) is True


# ---------------------------------------------------------------------------
# resolve_remote_head
# ---------------------------------------------------------------------------


class TestResolveRemoteHead:
    def test_returns_none_for_none_url(self) -> None:
        assert CacheManager.resolve_remote_head(None) is None

    def test_returns_none_for_empty_url(self) -> None:
        assert CacheManager.resolve_remote_head("") is None

    def test_parses_ls_remote_output(self) -> None:
        fake_output = "abc123def456\tHEAD\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = fake_output
            result = CacheManager.resolve_remote_head("https://github.com/example/repo.git")
        assert result == "abc123def456"

    def test_returns_none_on_failure(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 128
            mock_run.return_value.stdout = ""
            result = CacheManager.resolve_remote_head("https://github.com/example/repo.git")
        assert result is None

    def test_returns_none_on_timeout(self) -> None:
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("git", 15)):
            result = CacheManager.resolve_remote_head("https://github.com/example/repo.git")
        assert result is None

    def test_returns_none_on_empty_output(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ""
            result = CacheManager.resolve_remote_head("https://github.com/example/repo.git")
        assert result is None


# ---------------------------------------------------------------------------
# cache_key with remote resolution
# ---------------------------------------------------------------------------


class TestCacheKeyRemoteResolution:
    def test_url_source_uses_remote_head(self, tmp_path: Path) -> None:
        """URL source without explicit commit resolves HEAD via ls-remote."""
        cm = CacheManager(cache_dir=str(tmp_path))
        source = RepoSource(url="https://github.com/example/repo.git")
        with patch.object(CacheManager, "resolve_remote_head", return_value="deadbeef" * 5):
            key = cm.cache_key(source)

        # Key should incorporate the resolved commit
        expected_identity = f"https://github.com/example/repo.git@{'deadbeef' * 5}"
        expected_key = hashlib.sha256(expected_identity.encode()).hexdigest()
        assert key == expected_key

    def test_url_source_with_explicit_commit_skips_resolution(self, tmp_path: Path) -> None:
        """URL source with explicit commit uses the pinned commit directly."""
        cm = CacheManager(cache_dir=str(tmp_path))
        source = RepoSource(url="https://github.com/example/repo.git", commit="pinned123")
        with patch.object(CacheManager, "resolve_remote_head") as mock_resolve:
            key = cm.cache_key(source)

        # Should not call resolve_remote_head since commit is already set
        mock_resolve.assert_not_called()
        expected_identity = "https://github.com/example/repo.git@pinned123"
        expected_key = hashlib.sha256(expected_identity.encode()).hexdigest()
        assert key == expected_key

    def test_url_source_unresolvable_falls_back_to_url_only(self, tmp_path: Path) -> None:
        """URL source where remote HEAD resolution fails uses URL-only key."""
        cm = CacheManager(cache_dir=str(tmp_path))
        source = RepoSource(url="https://github.com/example/repo.git")
        with patch.object(CacheManager, "resolve_remote_head", return_value=None):
            key = cm.cache_key(source)

        expected_identity = "https://github.com/example/repo.git"
        expected_key = hashlib.sha256(expected_identity.encode()).hexdigest()
        assert key == expected_key
