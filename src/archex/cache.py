"""Index cache management: read, write, and invalidate cached analysis artifacts."""

from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from archex.exceptions import CacheError

if TYPE_CHECKING:
    from archex.models import RepoSource

_KEY_RE = re.compile(r"^[0-9a-f]{64}$")


class CacheManager:
    """Manage cached SQLite analysis artifacts on disk."""

    def __init__(self, cache_dir: str = "~/.archex/cache") -> None:
        self._cache_dir = Path(cache_dir).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def cache_key(self, source: RepoSource) -> str:
        """Derive a stable SHA256 cache key from the source identity and git HEAD."""
        identity = source.url or source.local_path or ""
        # Include git HEAD commit for local repos to invalidate on new commits
        commit = source.commit or self.git_head(source.local_path)
        if commit:
            identity = f"{identity}@{commit}"
        return hashlib.sha256(identity.encode()).hexdigest()

    @staticmethod
    def git_head(local_path: str | None) -> str | None:
        """Return the HEAD commit hash for a local repo, or None."""
        if local_path is None:
            return None
        git_dir = Path(local_path) / ".git"
        if not git_dir.exists():
            return None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=local_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
        return None

    def _validate_key(self, key: str) -> None:
        if not _KEY_RE.match(key):
            raise CacheError(
                f"Invalid cache key {key!r}: must be exactly 64 lowercase hex characters"
            )

    def db_path(self, key: str) -> Path:
        """Return the database path for a cache key."""
        self._validate_key(key)
        return self._cache_dir / f"{key}.db"

    def meta_path(self, key: str) -> Path:
        """Return the metadata file path for a cache key."""
        self._validate_key(key)
        return self._cache_dir / f"{key}.meta"

    def vector_path(self, key: str) -> Path:
        """Return the vector index file path for a cache key."""
        self._validate_key(key)
        return self._cache_dir / f"{key}.vectors.npz"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, key: str) -> Path | None:
        """Return cached db Path if it exists, else None."""
        db = self.db_path(key)
        if db.exists():
            return db
        return None

    def put(self, key: str, source_db: Path) -> Path:
        """Copy source_db into the cache and record metadata. Return cache path."""
        dest = self.db_path(key)
        shutil.copy2(str(source_db), str(dest))
        meta = self.meta_path(key)
        meta.write_text(str(time.time()))
        return dest

    def invalidate(self, key: str) -> None:
        """Remove the cached entry for key."""
        db = self.db_path(key)
        meta = self.meta_path(key)
        if db.exists():
            db.unlink()
        if meta.exists():
            meta.unlink()

    # ------------------------------------------------------------------
    # Listing & cleanup
    # ------------------------------------------------------------------

    def list_entries(self) -> list[dict[str, str]]:
        """Return a list of cache entries with key, path, size_bytes, created_at."""
        entries: list[dict[str, str]] = []
        for db in sorted(self._cache_dir.glob("*.db")):
            key = db.stem
            meta = self.meta_path(key)
            created_at = meta.read_text().strip() if meta.exists() else "0"
            entries.append(
                {
                    "key": key,
                    "path": str(db),
                    "size_bytes": str(db.stat().st_size),
                    "created_at": created_at,
                }
            )
        return entries

    def clean(self, max_age_hours: int = 24) -> int:
        """Remove entries older than max_age_hours. Return count removed."""
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for db in list(self._cache_dir.glob("*.db")):
            key = db.stem
            meta = self.meta_path(key)
            if meta.exists():
                try:
                    created = float(meta.read_text().strip())
                except ValueError:
                    created = 0.0
            else:
                created = db.stat().st_mtime
            if created < cutoff:
                db.unlink()
                if meta.exists():
                    meta.unlink()
                removed += 1
        return removed

    def find_store_for_source(self, source: RepoSource) -> tuple[Path, str] | None:
        """Find a cached store for the same source identity, regardless of commit.

        Returns (db_path, cached_commit_hash) if found, else None.
        """
        identity = source.url or source.local_path or ""
        if not identity:
            return None

        for db_file in self._cache_dir.glob("*.db"):
            try:
                conn = sqlite3.connect(str(db_file))
                try:
                    cur = conn.execute(
                        "SELECT value FROM metadata WHERE key = ?",
                        ("source_identity",),
                    )
                    row = cur.fetchone()
                    if row and str(row[0]) == identity:
                        cur2 = conn.execute(
                            "SELECT value FROM metadata WHERE key = ?",
                            ("commit_hash",),
                        )
                        commit_row = cur2.fetchone()
                        if commit_row:
                            return (db_file, str(commit_row[0]))
                finally:
                    conn.close()
            except Exception:
                continue

        return None

    def info(self) -> dict[str, Any]:
        """Return summary info about the cache."""
        entries = self.list_entries()
        total_size = sum(int(e["size_bytes"]) for e in entries)
        return {
            "total_entries": len(entries),
            "total_size_bytes": total_size,
            "cache_dir": str(self._cache_dir),
        }
