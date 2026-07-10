"""Tests for the portable index artifact: export format, header, compat validation."""

from __future__ import annotations

import json
import lzma
import shutil
import sqlite3
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from archex.api import index_repository
from archex.cli.main import cli
from archex.exceptions import ArtifactError, ArtifactVersionError
from archex.index.artifact import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_MAGIC,
    ArtifactHeader,
    _decompress_artifact_payload,  # pyright: ignore[reportPrivateUsage]
    ensure_artifact_gitattributes,
    export_artifact,
    import_artifact,
    read_artifact_header,
    sync_imported_artifact,
    validate_artifact_compat,
)
from archex.index.store import IndexStore
from archex.models import Config, IndexConfig, RepoSource
from archex.project import init_project


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _build_store(repo: Path, cache_dir: Path) -> IndexStore:
    source = RepoSource(local_path=str(repo))
    config = Config(languages=["python"], cache=True, cache_dir=str(cache_dir))
    return index_repository(source, config=config, index_config=IndexConfig())


def _decompress_payload(artifact_path: Path) -> bytes:
    with artifact_path.open("rb") as handle:
        handle.read(len(ARTIFACT_MAGIC))
        (header_len,) = struct.unpack(">I", handle.read(4))
        handle.read(header_len)
        payload = handle.read()
    return lzma.decompress(payload)


def _write_raw_artifact(path: Path, header: dict[str, object], payload: bytes) -> None:
    """Hand-frame an artifact file from an arbitrary header, bypassing export_artifact.

    Used to construct headers `export_artifact` would never itself produce
    (an out-of-range format/compat version) so compat validation can be
    tested independently of the export path.
    """
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(ARTIFACT_MAGIC)
        handle.write(struct.pack(">I", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(lzma.compress(payload))


class TestExportArtifact:
    def test_export_header_fields(self, python_simple_repo: Path, tmp_path: Path) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            header = export_artifact(store, artifact_path)

            assert header.format_version == ARTIFACT_FORMAT_VERSION
            assert header.index_revision == store.get_metadata("commit_hash")
            assert header.index_revision
            assert header.chunk_count == store.get_chunk_count()
            assert header.file_count == store.get_file_count()
            assert header.created_by_version
            assert header.created_at > 0
        finally:
            store.close()

    def test_export_writes_readable_header_without_decompressing(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)

            header = read_artifact_header(artifact_path)

            assert header.index_revision == store.get_metadata("commit_hash")
            validate_artifact_compat(header)  # does not raise
        finally:
            store.close()

    def test_export_requires_commit_hash(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "bare.db")
        try:
            with pytest.raises(ArtifactError, match="commit_hash"):
                export_artifact(store, tmp_path / "artifact.xz")
        finally:
            store.close()

    def test_export_strips_derived_fts_tables_and_preserves_chunks(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            assert store.get_chunk_count() > 0
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)

            raw_db = tmp_path / "decompressed.db"
            raw_db.write_bytes(_decompress_payload(artifact_path))

            conn = sqlite3.connect(raw_db)
            try:
                tables = {
                    row[0]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                assert "chunks_fts" not in tables
                assert "symbols_fts" not in tables
                assert "chunks" in tables
                assert "edges" in tables
                assert "file_states" in tables
                chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                assert chunk_count == store.get_chunk_count()
            finally:
                conn.close()
        finally:
            store.close()

    def test_export_compresses_smaller_than_raw_db(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)

            raw_size = store.db_path.stat().st_size
            artifact_size = artifact_path.stat().st_size
            assert artifact_size < raw_size
        finally:
            store.close()

    def test_export_rejects_reading_header_from_bogus_file(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.xz"
        bogus.write_bytes(b"not an archex artifact at all")
        with pytest.raises(ArtifactError, match="bad magic"):
            read_artifact_header(bogus)

    def test_export_creates_parent_directories(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "nested" / "dir" / "artifact.xz"
            export_artifact(store, artifact_path)
            assert artifact_path.exists()
        finally:
            store.close()


class TestArtifactHeaderRoundtrip:
    def _sample(self, **overrides: object) -> ArtifactHeader:
        defaults: dict[str, object] = {
            "format_version": 1,
            "created_by_version": "0.15.2",
            "compat_min_version": "0.15.0",
            "compat_max_version": "0.99.99",
            "index_revision": "abc123",
            "schema_version": "5",
            "chunk_count": 10,
            "file_count": 3,
            "created_at": 1_700_000_000.0,
        }
        defaults.update(overrides)
        return ArtifactHeader(**defaults)  # type: ignore[arg-type]

    def test_to_json_from_json_roundtrip(self) -> None:
        header = self._sample()
        restored = ArtifactHeader.from_json(json.loads(json.dumps(header.to_json())))
        assert restored == header

    def test_from_json_raises_on_missing_field(self) -> None:
        with pytest.raises(ArtifactError, match="Malformed artifact header"):
            ArtifactHeader.from_json({"format_version": 1})

    def test_from_json_tolerates_missing_schema_version(self) -> None:
        header = self._sample(schema_version=None)
        restored = ArtifactHeader.from_json(json.loads(json.dumps(header.to_json())))
        assert restored.schema_version is None


class TestValidateArtifactCompat:
    def _sample(self, **overrides: object) -> ArtifactHeader:
        defaults: dict[str, object] = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "created_by_version": "0.15.2",
            "compat_min_version": "0.15.0",
            "compat_max_version": "0.99.99",
            "index_revision": "abc123",
            "schema_version": "5",
            "chunk_count": 1,
            "file_count": 1,
            "created_at": 0.0,
        }
        defaults.update(overrides)
        return ArtifactHeader(**defaults)  # type: ignore[arg-type]

    def test_accepts_matching_format_and_in_range_version(self) -> None:
        validate_artifact_compat(self._sample())  # does not raise

    def test_rejects_unsupported_format_version(self) -> None:
        header = self._sample(format_version=ARTIFACT_FORMAT_VERSION + 1)
        with pytest.raises(ArtifactVersionError, match="format version"):
            validate_artifact_compat(header)

    def test_rejects_archex_version_below_compat_min(self) -> None:
        header = self._sample(compat_min_version="99.0.0", compat_max_version="99.99.99")
        with pytest.raises(ArtifactVersionError, match="archex"):
            validate_artifact_compat(header)

    def test_rejects_archex_version_above_compat_max(self) -> None:
        header = self._sample(compat_min_version="0.0.0", compat_max_version="0.0.1")
        with pytest.raises(ArtifactVersionError, match="archex"):
            validate_artifact_compat(header)


class TestExportArtifactCli:
    def test_index_export_artifact_flag_writes_file_and_reports_revision(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        init_project(python_simple_repo)
        artifact_path = tmp_path / "artifact.xz"
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "index",
                str(python_simple_repo),
                "--export-artifact",
                str(artifact_path),
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)
        assert artifact_path.exists()
        assert summary["artifact_path"] == str(artifact_path)
        assert summary["artifact_index_revision"]
        assert summary["artifact_size_bytes"] > 0

    def test_index_without_export_artifact_flag_omits_artifact_fields(
        self, python_simple_repo: Path
    ) -> None:
        init_project(python_simple_repo)
        runner = CliRunner()

        result = runner.invoke(cli, ["index", str(python_simple_repo), "--format", "json"])

        assert result.exit_code == 0, result.output
        summary = json.loads(result.output)
        assert "artifact_path" not in summary


_INCOMPATIBLE_FORMAT_HEADER: dict[str, object] = {
    "format_version": ARTIFACT_FORMAT_VERSION + 1,
    "created_by_version": "99.0.0",
    "compat_min_version": "0.0.0",
    "compat_max_version": "99.99.99",
    "index_revision": "deadbeef",
    "schema_version": "5",
    "chunk_count": 1,
    "file_count": 1,
    "created_at": 0.0,
}

_INCOMPATIBLE_VERSION_HEADER: dict[str, object] = {
    "format_version": ARTIFACT_FORMAT_VERSION,
    "created_by_version": "99.0.0",
    "compat_min_version": "99.0.0",
    "compat_max_version": "99.99.99",
    "index_revision": "deadbeef",
    "schema_version": "5",
    "chunk_count": 1,
    "file_count": 1,
    "created_at": 0.0,
}

_COMPATIBLE_HEADER: dict[str, object] = {
    "format_version": ARTIFACT_FORMAT_VERSION,
    "created_by_version": "0.0.0",
    "compat_min_version": "0.0.0",
    "compat_max_version": "99.99.99",
    "index_revision": "deadbeef",
    "schema_version": "5",
    "chunk_count": 1,
    "file_count": 1,
    "created_at": 0.0,
}


class TestImportArtifact:
    def test_import_produces_ready_to_use_store(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)
            original_chunks = {
                (c.file_path, c.symbol_name or "", c.start_line, c.end_line)
                for c in store.get_chunks()
            }
            original_edges = {(e.source, e.target, e.kind.value) for e in store.get_edges()}
        finally:
            store.close()

        dest_db_path = tmp_path / "imported" / "index.db"
        header = import_artifact(artifact_path, dest_db_path)

        assert header.index_revision

        imported = IndexStore(dest_db_path)
        try:
            imported_chunks = {
                (c.file_path, c.symbol_name or "", c.start_line, c.end_line)
                for c in imported.get_chunks()
            }
            imported_edges = {(e.source, e.target, e.kind.value) for e in imported.get_edges()}
            assert imported_chunks == original_chunks
            assert imported_edges == original_edges

            fts_count = imported.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            assert fts_count == imported.get_chunk_count()
            symbols_fts_count = imported.conn.execute(
                "SELECT COUNT(*) FROM symbols_fts"
            ).fetchone()[0]
            assert symbols_fts_count > 0
        finally:
            imported.close()

    def test_import_bm25_search_parity_with_original(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        from archex.index.bm25 import BM25Index

        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)
            original_bm25 = BM25Index(store)
            original_bm25.build(store.get_chunks())
            queries = ["util", "process", "add", "compute"]
            original_results = {
                q: [(c.id, score) for c, score in original_bm25.search(q)] for q in queries
            }
        finally:
            store.close()

        dest_db_path = tmp_path / "imported" / "index.db"
        import_artifact(artifact_path, dest_db_path)

        imported = IndexStore(dest_db_path)
        try:
            imported_bm25 = BM25Index(imported)
            for query in queries:
                imported_results = [(c.id, score) for c, score in imported_bm25.search(query)]
                assert imported_results == original_results[query], f"parity mismatch: {query!r}"
        finally:
            imported.close()

    def test_import_rejects_bogus_file_without_writing_dest(self, tmp_path: Path) -> None:
        bogus = tmp_path / "bogus.xz"
        bogus.write_bytes(b"definitely not an artifact")
        dest = tmp_path / "dest" / "index.db"

        with pytest.raises(ArtifactError, match="bad magic"):
            import_artifact(bogus, dest)

        assert not dest.exists()

    def test_import_rejects_unsupported_format_version_without_writing_dest(
        self, tmp_path: Path
    ) -> None:
        artifact_path = tmp_path / "future.xz"
        _write_raw_artifact(artifact_path, _INCOMPATIBLE_FORMAT_HEADER, b"irrelevant payload")
        dest = tmp_path / "dest" / "index.db"

        with pytest.raises(ArtifactVersionError, match="format version"):
            import_artifact(artifact_path, dest)

        assert not dest.exists()

    def test_import_rejects_out_of_range_archex_version_without_writing_dest(
        self, tmp_path: Path
    ) -> None:
        artifact_path = tmp_path / "incompatible.xz"
        _write_raw_artifact(artifact_path, _INCOMPATIBLE_VERSION_HEADER, b"irrelevant payload")
        dest = tmp_path / "dest" / "index.db"

        with pytest.raises(ArtifactVersionError, match="archex"):
            import_artifact(artifact_path, dest)

        assert not dest.exists()

    def test_import_never_overwrites_existing_dest_on_compat_failure(self, tmp_path: Path) -> None:
        """A stale, incompatible artifact must never clobber an already-good index."""
        dest = tmp_path / "index.db"
        dest.write_bytes(b"pretend this is a valid, already-good index file")
        original_bytes = dest.read_bytes()

        artifact_path = tmp_path / "incompatible.xz"
        _write_raw_artifact(artifact_path, _INCOMPATIBLE_FORMAT_HEADER, b"irrelevant payload")

        with pytest.raises(ArtifactVersionError):
            import_artifact(artifact_path, dest)

        assert dest.read_bytes() == original_bytes

    def test_import_rejects_payload_beyond_decompression_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crafted artifact with a small compressed, huge decompressed payload
        (a decompression bomb) must fail cleanly instead of exhausting memory,
        end to end through import_artifact().

        Compresses at preset=0 (the smallest LZMA dictionary) so the bomb
        payload's compressed size has no bearing on the guard — proving the
        guard tracks actual decompressed output, not a compression-preset-derived
        proxy for it (a real memlimit-based bound would not have caught this).
        """
        import archex.index.artifact as artifact_module

        monkeypatch.setattr(artifact_module, "_MAX_DECOMPRESSED_ARTIFACT_BYTES", 1024)

        bomb_payload = b"\x00" * (10 * 1024 * 1024)
        header_bytes = json.dumps(_COMPATIBLE_HEADER, sort_keys=True).encode("utf-8")
        artifact_path = tmp_path / "bomb.xz"
        with artifact_path.open("wb") as handle:
            handle.write(ARTIFACT_MAGIC)
            handle.write(struct.pack(">I", len(header_bytes)))
            handle.write(header_bytes)
            handle.write(lzma.compress(bomb_payload, preset=0))
        dest = tmp_path / "dest" / "index.db"

        with pytest.raises(ArtifactError, match="decompress"):
            import_artifact(artifact_path, dest)

        assert not dest.exists()


class TestReadHeader:
    """Direct unit test for _read_header()'s header_len ceiling."""

    def test_rejects_oversized_header_length(self, tmp_path: Path) -> None:
        """A crafted artifact declaring an absurd header_len must be rejected
        before any read of that size is attempted — the same length-prefixed-
        read bomb class the payload decompression guard closes, one field
        earlier in the same file.
        """
        artifact_path = tmp_path / "oversized_header.xz"
        with artifact_path.open("wb") as handle:
            handle.write(ARTIFACT_MAGIC)
            handle.write(struct.pack(">I", 2**32 - 1))  # max u32: ~4.29 GiB declared header

        with pytest.raises(ArtifactError, match="exceeding"):
            read_artifact_header(artifact_path)


class TestDecompressArtifactPayload:
    """Direct unit tests for the bomb-bounding decompression helper.

    Complements TestImportArtifact's end-to-end coverage by exercising the
    guard in isolation: a rejection case and an acceptance case under the
    exact same cap, proving it is proportional to actual decompressed
    output size rather than a blanket reject.
    """

    def test_rejects_output_beyond_the_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import archex.index.artifact as artifact_module

        monkeypatch.setattr(artifact_module, "_MAX_DECOMPRESSED_ARTIFACT_BYTES", 1024)
        bomb = lzma.compress(b"\x00" * (10 * 1024 * 1024), preset=0)

        with pytest.raises(ArtifactError, match="decompresses beyond"):
            _decompress_artifact_payload(bomb, Path("bomb.xz"))

    def test_accepts_output_within_the_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import archex.index.artifact as artifact_module

        monkeypatch.setattr(artifact_module, "_MAX_DECOMPRESSED_ARTIFACT_BYTES", 1024)
        payload = b"a small payload well under the cap"
        compressed = lzma.compress(payload)

        result = _decompress_artifact_payload(compressed, Path("small.xz"))

        assert result == payload

    def test_rejects_truncated_input(self) -> None:
        compressed = lzma.compress(b"hello world" * 100)

        with pytest.raises(ArtifactError, match="[Tt]runcated|corrupt"):
            _decompress_artifact_payload(compressed[:20], Path("truncated.xz"))

    def test_rejects_garbage_input(self) -> None:
        with pytest.raises(ArtifactError, match="corrupt"):
            _decompress_artifact_payload(b"not lzma data at all", Path("garbage.xz"))


class TestFromArtifactCli:
    def test_init_from_artifact_bootstraps_project_index(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        init_project(python_simple_repo)
        runner = CliRunner()
        artifact_path = tmp_path / "artifact.xz"

        exported = runner.invoke(
            cli, ["index", str(python_simple_repo), "--export-artifact", str(artifact_path)]
        )
        assert exported.exit_code == 0, exported.output

        fresh_repo = tmp_path / "fresh_clone"
        shutil.copytree(python_simple_repo, fresh_repo)
        shutil.rmtree(fresh_repo / ".archex")

        result = runner.invoke(
            cli, ["init", str(fresh_repo), "--from-artifact", str(artifact_path)]
        )

        assert result.exit_code == 0, result.output
        assert "Imported index artifact" in result.output
        index_path = fresh_repo / ".archex" / "index.db"
        assert index_path.exists()
        store = IndexStore(index_path)
        try:
            assert store.get_chunk_count() > 0
        finally:
            store.close()

    def test_init_from_artifact_fails_loudly_on_incompatible_artifact(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        artifact_path = tmp_path / "incompatible.xz"
        _write_raw_artifact(artifact_path, _INCOMPATIBLE_FORMAT_HEADER, b"irrelevant payload")
        runner = CliRunner()

        result = runner.invoke(
            cli, ["init", str(python_simple_repo), "--from-artifact", str(artifact_path)]
        )

        assert result.exit_code != 0
        assert "format version" in result.output
        index_path = python_simple_repo / ".archex" / "index.db"
        assert not index_path.exists()


class TestSyncImportedArtifact:
    def test_sync_reports_clean_when_working_tree_matches_artifact(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        dest_db_path = tmp_path / "imported" / "index.db"
        import_artifact(artifact_path, dest_db_path)

        config = Config(languages=["python"], cache=False)
        result = sync_imported_artifact(python_simple_repo, dest_db_path, config)

        assert result.strategy == "clean"
        assert result.files_changed == 0
        assert result.delta_meta is None

    def test_sync_applies_targeted_delta_matching_full_reindex(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        from archex.index.bm25 import BM25Index

        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        # Simulate more work landing in the repo after the artifact was exported.
        (python_simple_repo / "utils.py").write_text(
            "def brand_new_util():\n    return 'freshly parsed content'\n"
        )
        _git(python_simple_repo, "add", ".")
        _git(python_simple_repo, "commit", "-m", "advance past the artifact's revision")

        dest_db_path = tmp_path / "imported" / "index.db"
        import_artifact(artifact_path, dest_db_path)

        config = Config(languages=["python"], cache=False)
        result = sync_imported_artifact(python_simple_repo, dest_db_path, config)

        assert result.strategy == "delta"
        assert result.files_changed >= 1
        assert result.delta_meta is not None
        assert result.delta_meta.full_reindex_avoided is True

        synced_store = IndexStore(dest_db_path)
        try:
            synced_chunks = {
                (c.file_path, c.symbol_name or "", c.start_line, c.end_line)
                for c in synced_store.get_chunks()
            }
            synced_bm25 = BM25Index(synced_store)

            baseline_store = _build_store(python_simple_repo, tmp_path / "baseline_cache")
            try:
                baseline_chunks = {
                    (c.file_path, c.symbol_name or "", c.start_line, c.end_line)
                    for c in baseline_store.get_chunks()
                }
                baseline_bm25 = BM25Index(baseline_store)
                baseline_bm25.build(baseline_store.get_chunks())

                assert synced_chunks == baseline_chunks
                # Compare recalled documents, not exact scores: breadcrumbs/summary
                # are never persisted to SQLite, so any FTS rebuild sourced from
                # already-stored chunks (import_artifact's rebuild here, and
                # apply_delta's own `needs_full_bm25_rebuild` path identically)
                # loses them for untouched files while a freshly re-parsed file
                # keeps them — the same asymmetry already accepted by the
                # existing delta-indexing design, not an M16 regression.
                for query in ["util", "process", "brand new", "add"]:
                    synced_ids = {c.id for c, _ in synced_bm25.search(query)}
                    baseline_ids = {c.id for c, _ in baseline_bm25.search(query)}
                    assert synced_ids == baseline_ids, f"recall mismatch: {query!r}"
            finally:
                baseline_store.close()
        finally:
            synced_store.close()

    def test_sync_falls_back_to_full_reindex_when_artifact_is_stale(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        for py_file in sorted(python_simple_repo.rglob("*.py")):
            py_file.write_text(f"def rewritten_{py_file.stem}():\n    return 1\n")
        _git(python_simple_repo, "add", ".")
        _git(python_simple_repo, "commit", "-m", "rewrite everything")

        dest_db_path = tmp_path / "imported" / "index.db"
        import_artifact(artifact_path, dest_db_path)

        config = Config(languages=["python"], cache=True, delta_threshold=0.5)
        result = sync_imported_artifact(python_simple_repo, dest_db_path, config)

        assert result.strategy == "full_reindex"
        assert result.delta_meta is None

        synced_store = IndexStore(dest_db_path)
        try:
            rewritten_chunks = synced_store.get_chunks_for_file("utils.py")
            assert any("rewritten_utils" in c.content for c in rewritten_chunks)
            assert synced_store.get_metadata("commit_hash")
        finally:
            synced_store.close()

    def test_sync_closes_store_on_mid_pipeline_exception(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        """store must close even when a step after opening it raises."""
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        dest_db_path = tmp_path / "imported" / "index.db"
        import_artifact(artifact_path, dest_db_path)
        config = Config(languages=["python"], cache=False)

        close_calls: list[IndexStore] = []
        real_close = IndexStore.close

        def tracking_close(self: IndexStore) -> None:
            close_calls.append(self)
            real_close(self)

        with (
            patch.object(IndexStore, "close", tracking_close),
            patch(
                "archex.index.delta.compute_working_tree_delta",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            sync_imported_artifact(python_simple_repo, dest_db_path, config)

        assert close_calls, "store.close() must run even when a sync step raises"

    def test_full_reindex_fallback_cleans_up_fresh_store_on_exception(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        """A failure during the stale-artifact full-reindex fallback must not
        leak the fresh ephemeral store's scratch directory."""
        import shutil
        import tempfile

        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        for py_file in sorted(python_simple_repo.rglob("*.py")):
            py_file.write_text(f"def rewritten_{py_file.stem}():\n    return 1\n")
        _git(python_simple_repo, "add", ".")
        _git(python_simple_repo, "commit", "-m", "rewrite everything")

        dest_db_path = tmp_path / "imported" / "index.db"
        import_artifact(artifact_path, dest_db_path)
        config = Config(languages=["python"], cache=True, delta_threshold=0.5)

        # _full_index()'s own cache.put() (forced on internally by
        # _full_reindex_in_place's fallback_config) calls shutil.copy2 once
        # before fresh_store is even bound; only the SECOND call is
        # _full_reindex_in_place's own `shutil.copy2(fresh_db_path,
        # dest_db_path)`, the one that actually exercises the guard under
        # test. Failing indiscriminately on every call would trip the first
        # one and never reach fresh_store's own try/finally at all.
        real_copy2 = shutil.copy2
        call_count = 0

        def copy2_fail_on_second_call(*args: object, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("boom")
            return real_copy2(*args, **kwargs)  # type: ignore[arg-type]

        before = {p.name for p in Path(tempfile.gettempdir()).iterdir() if p.is_dir()}
        with (
            patch("shutil.copy2", side_effect=copy2_fail_on_second_call),
            pytest.raises(RuntimeError, match="boom"),
        ):
            sync_imported_artifact(python_simple_repo, dest_db_path, config)
        after = {p.name for p in Path(tempfile.gettempdir()).iterdir() if p.is_dir()}

        assert call_count >= 2, (
            "expected the fault to fire on _full_reindex_in_place's own copy2 call, "
            "not just the earlier cache.put() one"
        )
        assert after == before, f"leaked scratch dirs: {after - before}"


class TestFromArtifactCliSync:
    def test_init_from_artifact_delta_syncs_to_current_working_tree(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        init_project(python_simple_repo)
        runner = CliRunner()
        artifact_path = tmp_path / "artifact.xz"

        exported = runner.invoke(
            cli, ["index", str(python_simple_repo), "--export-artifact", str(artifact_path)]
        )
        assert exported.exit_code == 0, exported.output

        fresh_repo = tmp_path / "fresh_clone"
        shutil.copytree(python_simple_repo, fresh_repo)
        shutil.rmtree(fresh_repo / ".archex")

        # Simulate a teammate's later commit landing before the fresh clone runs init.
        (fresh_repo / "utils.py").write_text("def added_after_export():\n    return 99\n")
        _git(fresh_repo, "add", ".")
        _git(fresh_repo, "commit", "-m", "work that landed after the artifact was exported")

        result = runner.invoke(
            cli, ["init", str(fresh_repo), "--from-artifact", str(artifact_path)]
        )

        assert result.exit_code == 0, result.output
        assert "Delta-sync strategy:     delta" in result.output

        index_path = fresh_repo / ".archex" / "index.db"
        store = IndexStore(index_path)
        try:
            chunks = store.get_chunks_for_file("utils.py")
            assert any("added_after_export" in c.content for c in chunks)
        finally:
            store.close()


class TestEnsureArtifactGitattributes:
    def test_creates_entry_for_artifact_inside_repo(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = python_simple_repo / ".archex-artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        changed = ensure_artifact_gitattributes(python_simple_repo, artifact_path)

        assert changed is True
        gitattributes = (python_simple_repo / ".gitattributes").read_text(encoding="utf-8")
        assert ".archex-artifact.xz merge=ours -diff" in gitattributes

    def test_is_idempotent(self, python_simple_repo: Path, tmp_path: Path) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = python_simple_repo / ".archex-artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        first = ensure_artifact_gitattributes(python_simple_repo, artifact_path)
        content_after_first = (python_simple_repo / ".gitattributes").read_text(encoding="utf-8")
        second = ensure_artifact_gitattributes(python_simple_repo, artifact_path)
        content_after_second = (python_simple_repo / ".gitattributes").read_text(encoding="utf-8")

        assert first is True
        assert second is False
        assert content_after_first == content_after_second

    def test_noop_for_artifact_outside_repo(self, python_simple_repo: Path, tmp_path: Path) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = tmp_path / "outside" / "artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        changed = ensure_artifact_gitattributes(python_simple_repo, artifact_path)

        assert changed is False
        assert not (python_simple_repo / ".gitattributes").exists()

    def test_appends_to_existing_gitattributes(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        (python_simple_repo / ".gitattributes").write_text("*.png binary\n", encoding="utf-8")
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = python_simple_repo / ".archex-artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        ensure_artifact_gitattributes(python_simple_repo, artifact_path)

        content = (python_simple_repo / ".gitattributes").read_text(encoding="utf-8")
        assert "*.png binary" in content
        assert ".archex-artifact.xz merge=ours -diff" in content

    def test_sets_local_merge_ours_driver_config(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        store = _build_store(python_simple_repo, tmp_path / "cache")
        try:
            artifact_path = python_simple_repo / ".archex-artifact.xz"
            export_artifact(store, artifact_path)
        finally:
            store.close()

        ensure_artifact_gitattributes(python_simple_repo, artifact_path)

        driver = _git_output(python_simple_repo, "config", "--get", "merge.ours.driver")
        assert driver == "true"


class TestExportArtifactCliGitattributes:
    def test_index_export_artifact_inside_repo_updates_gitattributes(
        self, python_simple_repo: Path
    ) -> None:
        init_project(python_simple_repo)
        artifact_path = python_simple_repo / ".archex-artifact.xz"
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "index",
                str(python_simple_repo),
                "--export-artifact",
                str(artifact_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Updated .gitattributes" in result.output
        gitattributes = (python_simple_repo / ".gitattributes").read_text(encoding="utf-8")
        assert ".archex-artifact.xz merge=ours -diff" in gitattributes

    def test_index_export_artifact_outside_repo_skips_gitattributes(
        self, python_simple_repo: Path, tmp_path: Path
    ) -> None:
        init_project(python_simple_repo)
        artifact_path = tmp_path / "artifact.xz"
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "index",
                str(python_simple_repo),
                "--export-artifact",
                str(artifact_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Updated .gitattributes" not in result.output
        assert not (python_simple_repo / ".gitattributes").exists()
