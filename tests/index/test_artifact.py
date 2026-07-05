"""Tests for the portable index artifact: export format, header, compat validation."""

from __future__ import annotations

import json
import lzma
import sqlite3
import struct
from pathlib import Path

import pytest
from click.testing import CliRunner

from archex.api import index_repository
from archex.cli.main import cli
from archex.exceptions import ArtifactError, ArtifactVersionError
from archex.index.artifact import (
    ARTIFACT_FORMAT_VERSION,
    ARTIFACT_MAGIC,
    ArtifactHeader,
    export_artifact,
    read_artifact_header,
    validate_artifact_compat,
)
from archex.index.store import IndexStore
from archex.models import Config, IndexConfig, RepoSource
from archex.project import init_project


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
