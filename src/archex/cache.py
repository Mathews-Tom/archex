"""Index cache management: read, write, and invalidate cached analysis artifacts."""

from __future__ import annotations

import hashlib
import re
import shutil
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
        """Derive a stable SHA256 cache key from the source URL or local path."""
        identity = source.url or source.local_path or ""
        return hashlib.sha256(identity.encode()).hexdigest()

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

    def info(self) -> dict[str, Any]:
        """Return summary info about the cache."""
        entries = self.list_entries()
        total_size = sum(int(e["size_bytes"]) for e in entries)
        return {
            "total_entries": len(entries),
            "total_size_bytes": total_size,
            "cache_dir": str(self._cache_dir),
        }
