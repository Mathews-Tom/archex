"""Portable, compressed index artifact for team-shared bootstrap.

A repo can commit a compacted, compressed copy of its `.archex/index.db`
(the "artifact") so a fresh clone can `archex init --from-artifact` and
delta-sync to HEAD instead of paying a full cold-start reindex.

File layout (NOT a bare `.xz` stream — a small custom framing wraps a raw
LZMA payload so a reader can validate compatibility before paying the cost
of decompressing a potentially large index):

    +----------------------------+
    | MAGIC (12 bytes)           |  b"ARCHEXIDXv1\\n"
    +----------------------------+
    | header_len (4 bytes, u32be)|
    +----------------------------+
    | header (UTF-8 JSON)        |  ArtifactHeader.to_json()
    +----------------------------+
    | payload (LZMA stream)      |  a VACUUM INTO-compacted copy of the
    |                            |  index's SQLite database with derived
    |                            |  FTS5 structures (chunks_fts,
    |                            |  symbols_fts) stripped — rebuilt
    |                            |  locally on import.
    +----------------------------+

The header is read and validated (format version + archex-version compat
range) BEFORE the payload is touched, so an incompatible artifact fails
loudly without ever writing a partial index to disk.
"""

from __future__ import annotations

import json
import logging
import lzma
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from archex import __version__
from archex.exceptions import ArtifactError, ArtifactVersionError
from archex.index.store import IndexStore
from archex.models import IndexConfig

if TYPE_CHECKING:
    from archex.models import Config, DeltaMeta

logger = logging.getLogger(__name__)

ARTIFACT_MAGIC = b"ARCHEXIDXv1\n"

#: Container format version. Bumped only when the framing or payload shape
#: (not the SQLite schema, which is versioned independently via the store's
#: own `schema_version` metadata) changes incompatibly.
ARTIFACT_FORMAT_VERSION = 1

#: archex versions declared able to import an artifact written at
#: ARTIFACT_FORMAT_VERSION. Widen ARTIFACT_MAX_COMPAT_VERSION as later
#: archex releases confirm they still read this format; bump both when a
#: future format break requires a new ARTIFACT_FORMAT_VERSION.
ARTIFACT_MIN_COMPAT_VERSION = "0.15.0"
ARTIFACT_MAX_COMPAT_VERSION = "0.99.99"

#: LZMA preset: balances export latency against compression ratio for
#: SQLite-shaped content (mostly text — source chunks, symbol metadata).
_LZMA_PRESET = 6

#: Chunk size streamed per LZMADecompressor.decompress() call when bounding
#: an artifact payload (see _decompress_artifact_payload). Small enough to
#: check the running total frequently, large enough to keep call overhead
#: negligible for a legitimate multi-megabyte payload.
_DECOMPRESS_CHUNK_BYTES = 4 * 1024 * 1024

#: Hard ceiling on a decompressed artifact payload. `lzma.decompress()`'s
#: `memlimit` bounds only the decoder's internal dictionary/filter-chain
#: memory — which is dictated by the *compression* preset used to write the
#: artifact, not by how many bytes of output it produces — so a payload
#: compressed at a low preset can pass any memlimit while still
#: decompressing to gigabytes. `_decompress_artifact_payload` streams
#: through `LZMADecompressor` instead, checking the running output total,
#: so an oversized payload is rejected before being fully materialized. 1
#: GiB comfortably exceeds any legitimate index this format is meant to
#: carry (a compacted, FTS5-stripped SQLite database of source chunks and
#: symbol metadata).
_MAX_DECOMPRESSED_ARTIFACT_BYTES = 1024**3

#: Hard ceiling on the artifact header itself. Real headers are a small
#: flat JSON dict (well under 1 KiB); this only guards against a crafted
#: artifact declaring an absurd header_len (up to ~4.29 GiB, the u32 max)
#: that forces a multi-gigabyte read before any validation has run.
_MAX_HEADER_BYTES = 64 * 1024

_HEADER_LENGTH_STRUCT = struct.Struct(">I")

# Tables dropped before compression and rebuilt locally on import. Both are
# FTS5 virtual tables fully derivable from `chunks` — shipping them would
# duplicate the corpus text and bake in a tokenizer/build-specific binary
# shape that does not need to travel between machines.
_DERIVED_TABLES = ("chunks_fts", "symbols_fts")


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable tuple of ints.

    Deliberately not a full PEP 440/semver parser — archex versions are
    plain `MAJOR.MINOR.PATCH`, and this only needs to order those correctly
    for a compat-range check.
    """
    parts: list[int] = []
    for segment in version.split("."):
        digits = "".join(ch for ch in segment if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _version_in_range(version: str, min_version: str, max_version: str) -> bool:
    parsed = _parse_version(version)
    return _parse_version(min_version) <= parsed <= _parse_version(max_version)


@dataclass(frozen=True)
class ArtifactHeader:
    """Versioned metadata written ahead of an artifact's compressed payload."""

    format_version: int
    created_by_version: str
    compat_min_version: str
    compat_max_version: str
    index_revision: str
    schema_version: str | None
    chunk_count: int
    file_count: int
    created_at: float

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "created_by_version": self.created_by_version,
            "compat_min_version": self.compat_min_version,
            "compat_max_version": self.compat_max_version,
            "index_revision": self.index_revision,
            "schema_version": self.schema_version,
            "chunk_count": self.chunk_count,
            "file_count": self.file_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ArtifactHeader:
        try:
            return cls(
                format_version=int(data["format_version"]),
                created_by_version=str(data["created_by_version"]),
                compat_min_version=str(data["compat_min_version"]),
                compat_max_version=str(data["compat_max_version"]),
                index_revision=str(data["index_revision"]),
                schema_version=(
                    str(data["schema_version"]) if data.get("schema_version") is not None else None
                ),
                chunk_count=int(data["chunk_count"]),
                file_count=int(data["file_count"]),
                created_at=float(data["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"Malformed artifact header: {exc}") from exc


def _read_header(handle: Any, path: Path) -> ArtifactHeader:
    """Read and parse the header from an open, position-0 binary file handle."""
    magic = handle.read(len(ARTIFACT_MAGIC))
    if magic != ARTIFACT_MAGIC:
        raise ArtifactError(f"Not an archex index artifact (bad magic): {path}")
    length_bytes = handle.read(_HEADER_LENGTH_STRUCT.size)
    if len(length_bytes) != _HEADER_LENGTH_STRUCT.size:
        raise ArtifactError(f"Truncated artifact header: {path}")
    (header_len,) = _HEADER_LENGTH_STRUCT.unpack(length_bytes)
    if header_len > _MAX_HEADER_BYTES:
        raise ArtifactError(
            f"Artifact header for {path} declares {header_len} bytes, "
            f"exceeding the {_MAX_HEADER_BYTES} byte limit"
        )
    header_bytes = handle.read(header_len)
    if len(header_bytes) != header_len:
        raise ArtifactError(f"Truncated artifact header: {path}")
    try:
        raw = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Corrupt artifact header: {path}") from exc
    return ArtifactHeader.from_json(raw)


def read_artifact_header(path: str | Path) -> ArtifactHeader:
    """Read and parse just the header of an artifact, without touching the payload."""
    path = Path(path)
    with path.open("rb") as handle:
        return _read_header(handle, path)


def validate_artifact_compat(header: ArtifactHeader) -> None:
    """Raise `ArtifactVersionError` if this archex build cannot safely import the artifact.

    Checked BEFORE decompression so an incompatible artifact never produces
    a partial import — the failure is loud and immediate.
    """
    if header.format_version != ARTIFACT_FORMAT_VERSION:
        raise ArtifactVersionError(
            f"Artifact format version {header.format_version} is not supported by this "
            f"archex build (supports format version {ARTIFACT_FORMAT_VERSION}). Re-export "
            "the artifact with a compatible archex version."
        )
    if not _version_in_range(__version__, header.compat_min_version, header.compat_max_version):
        raise ArtifactVersionError(
            f"Artifact requires archex {header.compat_min_version}..{header.compat_max_version}, "
            f"but this is archex {__version__}. Upgrade or downgrade archex, or re-export the "
            "artifact from a compatible version."
        )


def _vacuum_into_stripped(store: IndexStore, dest_path: Path) -> None:
    """Write a compacted copy of `store`'s database to `dest_path`, dropping FTS5 tables."""
    if dest_path.exists():
        dest_path.unlink()
    store.conn.execute("VACUUM INTO ?", (str(dest_path),))
    conn = sqlite3.connect(dest_path)
    try:
        for table in _DERIVED_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()


def export_artifact(store: IndexStore, output_path: str | Path) -> ArtifactHeader:
    """Export a compacted, compressed, portable copy of `store` to `output_path`.

    Raises `ArtifactError` if the store has no recorded `commit_hash` (an
    artifact with no revision cannot be delta-synced after import).
    """
    output_path = Path(output_path)
    index_revision = store.get_metadata("commit_hash") or ""
    if not index_revision:
        raise ArtifactError(
            "Cannot export an artifact: the index has no recorded commit_hash. "
            "Index a git-tracked repository first (`archex index` inside a git checkout)."
        )

    schema_version = store.get_metadata("schema_version")
    chunk_count = store.get_chunk_count()
    file_count = store.get_file_count()

    header = ArtifactHeader(
        format_version=ARTIFACT_FORMAT_VERSION,
        created_by_version=__version__,
        compat_min_version=ARTIFACT_MIN_COMPAT_VERSION,
        compat_max_version=ARTIFACT_MAX_COMPAT_VERSION,
        index_revision=index_revision,
        schema_version=schema_version,
        chunk_count=chunk_count,
        file_count=file_count,
        created_at=time.time(),
    )
    header_bytes = json.dumps(header.to_json(), sort_keys=True).encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    compact_path = output_path.with_name(f".{output_path.name}.compact.tmp")
    try:
        _vacuum_into_stripped(store, compact_path)
        compressed = lzma.compress(compact_path.read_bytes(), preset=_LZMA_PRESET)
    finally:
        if compact_path.exists():
            compact_path.unlink()

    tmp_output = output_path.with_name(f".{output_path.name}.tmp")
    with tmp_output.open("wb") as handle:
        handle.write(ARTIFACT_MAGIC)
        handle.write(_HEADER_LENGTH_STRUCT.pack(len(header_bytes)))
        handle.write(header_bytes)
        handle.write(compressed)
    tmp_output.replace(output_path)

    return header


def _rebuild_symbols_fts(store: IndexStore) -> None:
    """Repopulate `symbols_fts` from `chunks` after an artifact import.

    `chunks_fts` has a store-provided rebuild path (`BM25Index.build()`);
    `symbols_fts` (symbol-name search) has no equivalent helper on
    `IndexStore`, so it is rebuilt directly here from the chunks the
    artifact already carries.
    """
    conn = store.conn
    conn.execute("DELETE FROM symbols_fts")
    conn.executemany(
        "INSERT INTO symbols_fts (symbol_id, symbol_name, qualified_name, file_path) "
        "VALUES (?, ?, ?, ?)",
        (
            (chunk.symbol_id, chunk.symbol_name, chunk.qualified_name, chunk.file_path)
            for chunk in store.iter_chunks()
            if chunk.symbol_id is not None
        ),
    )
    conn.commit()


def _decompress_artifact_payload(compressed_payload: bytes, path: Path) -> bytes:
    """Decompress an artifact payload, bounded against a decompression bomb.

    Streams through `LZMADecompressor` in `_DECOMPRESS_CHUNK_BYTES` chunks
    and checks the running output total after every chunk, aborting before
    an oversized payload is ever fully materialized. `lzma.decompress()`'s
    `memlimit` cannot do this: it bounds the decoder's internal dictionary
    memory, which depends on the *compression* preset used to write the
    payload, not on how many bytes of output it produces — a payload
    compressed at a low preset passes any memlimit while still
    decompressing to gigabytes.
    """
    decompressor = lzma.LZMADecompressor()
    remaining = compressed_payload
    chunks: list[bytes] = []
    total = 0
    try:
        while not decompressor.eof:
            out = decompressor.decompress(remaining, max_length=_DECOMPRESS_CHUNK_BYTES)
            remaining = b""
            total += len(out)
            if total > _MAX_DECOMPRESSED_ARTIFACT_BYTES:
                raise ArtifactError(
                    f"Artifact payload for {path} decompresses beyond the "
                    f"{_MAX_DECOMPRESSED_ARTIFACT_BYTES} byte limit"
                )
            chunks.append(out)
            if decompressor.needs_input and not decompressor.eof:
                raise ArtifactError(f"Truncated or corrupt artifact payload: {path}")
    except lzma.LZMAError as exc:
        raise ArtifactError(f"Artifact payload for {path} is corrupt: {exc}") from exc
    return b"".join(chunks)


def import_artifact(path: str | Path, dest_db_path: str | Path) -> ArtifactHeader:
    """Import a portable index artifact, writing a ready-to-use store at `dest_db_path`.

    The artifact's header is read and its format/compat range validated
    BEFORE the payload is decompressed — an out-of-range artifact raises
    `ArtifactVersionError` and never touches `dest_db_path`, so a failed
    import is always loud and never partial.

    The two FTS5 derived tables stripped at export time (`chunks_fts`,
    `symbols_fts`) are rebuilt locally from the imported `chunks` table
    once the database is on disk.
    """
    from archex.index.bm25 import BM25Index

    path = Path(path)
    dest_db_path = Path(dest_db_path)

    with path.open("rb") as handle:
        header = _read_header(handle, path)
        validate_artifact_compat(header)
        compressed_payload = handle.read()

    decompressed = _decompress_artifact_payload(compressed_payload, path)

    dest_db_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("-wal", "-shm"):
        sidecar = dest_db_path.with_name(dest_db_path.name + suffix)
        if sidecar.exists():
            sidecar.unlink()
    dest_db_path.write_bytes(decompressed)

    store = IndexStore(dest_db_path)
    try:
        bm25 = BM25Index(store)
        bm25.build(store.iter_chunks())
        _rebuild_symbols_fts(store)
    finally:
        store.close()

    return header


_STALE_FALLBACK_LOG = (
    "Imported artifact is stale: %.0f%% of files changed since export "
    "(delta_threshold=%.0f%%) — falling back to a full re-index."
)


@dataclass(frozen=True)
class ArtifactSyncResult:
    """Outcome of syncing a freshly-imported artifact store to the working tree."""

    strategy: str
    """One of ``"clean"`` (no drift), ``"delta"``, or ``"full_reindex"``."""

    files_changed: int
    delta_meta: DeltaMeta | None
    sync_time_ms: float


def _full_reindex_in_place(
    repo_root: Path,
    dest_db_path: Path,
    config: Config,
    index_config: IndexConfig,
) -> None:
    """Discard a too-stale imported store and perform an ordinary full re-index in place.

    Builds via the same `_full_index` pipeline `archex index` itself uses,
    then explicitly copies the result onto `dest_db_path` — rather than
    relying on `index_repository()`'s cache-key/project-layout detection to
    land at the right path, which would silently miss `dest_db_path`
    whenever the repo-local project has not been initialized yet (exactly
    the state `archex init --from-artifact` is in when this fallback runs).
    Always builds with caching enabled internally (regardless of the
    caller's own `config.cache`) so `commit_hash`/`source_identity` land in
    the fresh store's metadata, which `sync_imported_artifact` and future
    re-exports depend on.
    """
    import shutil
    import tempfile

    from archex.api import _full_index  # pyright: ignore[reportPrivateUsage]
    from archex.cache import CacheManager
    from archex.models import RepoSource

    source = RepoSource(local_path=str(repo_root))
    fallback_config = config.model_copy(update={"cache": True})
    scratch_dir = Path(tempfile.mkdtemp(prefix="archex-artifact-fallback-"))
    try:
        scratch_cache = CacheManager(cache_dir=str(scratch_dir))
        cache_key = scratch_cache.cache_key(source)
        fresh_store = _full_index(
            source,
            fallback_config,
            scratch_cache,
            cache_key,
            timing=None,
            index_config=index_config,
        )
        try:
            fresh_db_path = fresh_store.db_path
            fresh_store.conn.execute("PRAGMA wal_checkpoint(FULL)")

            dest_db_path.parent.mkdir(parents=True, exist_ok=True)
            for suffix in ("-wal", "-shm"):
                stale_sidecar = dest_db_path.with_name(dest_db_path.name + suffix)
                if stale_sidecar.exists():
                    stale_sidecar.unlink()
            shutil.copy2(fresh_db_path, dest_db_path)
        finally:
            fresh_store.close()
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def sync_imported_artifact(
    repo_root: Path,
    dest_db_path: str | Path,
    config: Config,
    index_config: IndexConfig | None = None,
) -> ArtifactSyncResult:
    """Bring a freshly-imported artifact store up to date with the working tree.

    Applies `compute_working_tree_delta()` + `apply_delta()` — the same
    targeted-update machinery `apply_delta` already provides for ordinary
    delta re-indexing — scoped against the artifact's imported
    `file_states`. When the change set is at or above `config.delta_threshold`
    (the same staleness knob the ordinary delta-indexing path uses), the
    imported store is discarded in favor of an ordinary full re-index: past
    that point a targeted delta costs more than starting fresh, and a loud
    fallback is safer than silently syncing a misleadingly "close enough"
    index.
    """
    from archex.acquire import discover_files
    from archex.cache import CacheManager
    from archex.index.delta import apply_delta, compute_working_tree_delta
    from archex.index.graph import DependencyGraph

    dest_db_path = Path(dest_db_path)
    effective_index_config = index_config or IndexConfig()
    t_start = time.perf_counter()

    store = IndexStore(dest_db_path)
    try:
        manifest = compute_working_tree_delta(repo_root, store, config)
        manifest.base_commit = store.get_metadata("commit_hash") or manifest.base_commit
        current_commit = CacheManager.git_head(str(repo_root))
        if current_commit:
            manifest.current_commit = current_commit

        if not manifest.changes:
            store.set_metadata("indexed_at", str(time.time()))
            store.close()
            return ArtifactSyncResult(
                strategy="clean",
                files_changed=0,
                delta_meta=None,
                sync_time_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        total_files = len(
            discover_files(repo_root, languages=config.languages, max_file_size=config.max_file_size).files
        )
        change_ratio = len(manifest.changes) / total_files if total_files > 0 else 1.0

        if change_ratio >= config.delta_threshold:
            store.close()
            logger.warning(_STALE_FALLBACK_LOG, change_ratio * 100, config.delta_threshold * 100)
            _full_reindex_in_place(repo_root, dest_db_path, config, effective_index_config)
            return ArtifactSyncResult(
                strategy="full_reindex",
                files_changed=len(manifest.changes),
                delta_meta=None,
                sync_time_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        graph = DependencyGraph.from_edges(store.get_edges())
        delta_meta = apply_delta(store, graph, manifest, repo_root, config, effective_index_config)
        store.close()
        return ArtifactSyncResult(
            strategy="delta",
            files_changed=len(manifest.changes),
            delta_meta=delta_meta,
            sync_time_ms=round((time.perf_counter() - t_start) * 1000, 1),
        )
    except BaseException:
        store.close()
        raise


_GITATTRIBUTES_COMMENT = "# archex portable index artifact — never diff/merge-conflict"


def _ensure_local_merge_ours_driver(repo_root: Path) -> None:
    """Best-effort: register the "ours" merge driver in this machine's local git config.

    `.gitattributes`' `merge=ours` names a driver git must be told about via
    `merge.ours.driver` — deliberately local, non-shareable config (a
    security boundary: a malicious repo's committed config could otherwise
    silently make merges no-ops). Every clone must set this once; the
    one-line command is documented in docs/PORTABLE_INDEX_ARTIFACT.md for
    teammates on a fresh clone. Never raises — a failure here (e.g. no git
    binary, not a git repo) is not a reason to fail the export.
    """
    import contextlib
    import subprocess

    with contextlib.suppress(OSError):
        subprocess.run(
            ["git", "config", "merge.ours.driver", "true"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=5,
        )


def ensure_artifact_gitattributes(repo_root: Path, artifact_path: str | Path) -> bool:
    """Ensure a `merge=ours` `.gitattributes` entry exists for an exported artifact.

    Only touches `.gitattributes` when `artifact_path` resolves inside
    `repo_root` — an artifact exported outside the repo (e.g. to `/tmp`) has
    no repo-relative path to attach a merge strategy to, so this is a no-op.
    The edit is written to disk only; it is never staged or committed —
    export is opt-in, and so is deciding to commit its side effects.

    Returns True if `.gitattributes` was created or modified.
    """
    repo_root = repo_root.resolve()
    try:
        relative = Path(artifact_path).resolve().relative_to(repo_root)
    except ValueError:
        return False

    entry = f"{relative.as_posix()} merge=ours -diff"
    gitattributes_path = repo_root / ".gitattributes"
    lines = (
        gitattributes_path.read_text(encoding="utf-8").splitlines()
        if gitattributes_path.exists()
        else []
    )

    _ensure_local_merge_ours_driver(repo_root)

    if entry in {line.strip() for line in lines}:
        return False

    if lines and lines[-1] != "":
        lines.append("")
    lines.append(_GITATTRIBUTES_COMMENT)
    lines.append(entry)
    gitattributes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True
