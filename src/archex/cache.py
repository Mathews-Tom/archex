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

    def cache_key(
        self,
        source: RepoSource,
        *,
        head_override: str | None = None,
        stable_identity: str | None = None,
    ) -> str:
        """Derive a stable SHA256 cache key from the source identity and resolved ref.

        When stable_identity is provided, it overrides the default identity derived
        from source.url or source.local_path. This is used by benchmarks to ensure
        that the same upstream repo always maps to the same cache key, regardless
        of which temp directory it was cloned into.

        For local repos: resolves HEAD via git rev-parse.
        For remote URLs with explicit commit: uses the pinned commit.
        For remote URLs without commit: resolves HEAD via git ls-remote.
        """
        identity = stable_identity or source.url or source.local_path or ""
        commit = (
            source.commit
            or head_override
            or self.git_head(source.local_path)
            or (self.resolve_remote_head(source.url) if source.url else None)
        )
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

    @staticmethod
    def resolve_remote_head(url: str | None, ref: str = "HEAD") -> str | None:
        """Resolve a remote ref to a commit hash via git ls-remote."""
        if not url:
            return None
        try:
            result = subprocess.run(
                ["git", "ls-remote", url, ref],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split()[0]
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

    def vector_path(
        self,
        key: str,
        *,
        vector_mode: str = "raw",
        surrogate_version: str = "v1",
    ) -> Path:
        """Return the representation-specific vector index file path for a cache key."""
        self._validate_key(key)
        if vector_mode == "raw":
            return self._cache_dir / f"{key}.vectors.npz"
        return self._cache_dir / f"{key}.{vector_mode}.{surrogate_version}.vectors.npz"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, key: str) -> Path | None:
        """Return cached db Path if it exists, else None."""
        db = self.db_path(key)
        if db.exists():
            return db
        return None

    def put(
        self,
        key: str,
        source_db: Path,
        *,
        resolved_commit: str | None = None,
        source_identity: str | None = None,
        stable_identity: str | None = None,
    ) -> Path:
        """Copy source_db into the cache and record metadata. Return cache path."""
        import json

        dest = self.db_path(key)
        shutil.copy2(str(source_db), str(dest))
        meta = self.meta_path(key)
        meta_data = {
            "created_at": str(time.time()),
            "resolved_commit": resolved_commit or "",
            "source_identity": source_identity or "",
            "stable_identity": stable_identity or "",
        }
        meta.write_text(json.dumps(meta_data))
        return dest

    def get_meta(self, key: str) -> dict[str, str]:
        """Read cache metadata for a key. Returns empty dict if missing."""
        import json

        meta = self.meta_path(key)
        if not meta.exists():
            return {}
        raw = meta.read_text().strip()
        # Backward compat: old meta files contain bare timestamp
        if raw and not raw.startswith("{"):
            return {"created_at": raw}
        try:
            return json.loads(raw)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, ValueError):
            return {}

    def is_stale(self, key: str, max_age_hours: int = 24) -> bool:
        """Check if a cache entry is older than max_age_hours."""
        meta = self.get_meta(key)
        created_str = meta.get("created_at", "0")
        try:
            created = float(created_str)
        except ValueError:
            return True
        return (time.time() - created) > max_age_hours * 3600

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
            meta_data = self.get_meta(key)
            created_at = meta_data.get("created_at", "0")
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
        removed = 0
        for db in list(self._cache_dir.glob("*.db")):
            key = db.stem
            if self.is_stale(key, max_age_hours):
                db.unlink()
                meta = self.meta_path(key)
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
