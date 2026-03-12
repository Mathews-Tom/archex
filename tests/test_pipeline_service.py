"""Tests for archex.pipeline.service: parse_repository and build_chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from archex.models import Config, IndexConfig
from archex.parse.adapters import default_adapter_registry
from archex.pipeline.service import ParseArtifacts, build_chunks, parse_repository


class TestParseRepository:
    def test_returns_parse_artifacts(self, python_simple_repo: Path) -> None:
        config = Config(cache=False)
        adapters = default_adapter_registry.build_all()
        result = parse_repository(python_simple_repo, config, adapters)

        assert isinstance(result, ParseArtifacts)
        assert len(result.files) > 0
        assert len(result.parsed_files) > 0

    def test_discovers_all_python_files(self, python_simple_repo: Path) -> None:
        config = Config(cache=False)
        adapters = default_adapter_registry.build_all()
        result = parse_repository(python_simple_repo, config, adapters)

        paths = {f.path for f in result.files}
        assert "main.py" in paths
        assert "models.py" in paths
        assert "utils.py" in paths

    def test_resolves_imports(self, python_simple_repo: Path) -> None:
        config = Config(cache=False)
        adapters = default_adapter_registry.build_all()
        result = parse_repository(python_simple_repo, config, adapters)

        # main.py imports from models, utils, services.auth
        assert len(result.resolved_imports) > 0
        # At least some files should have resolved imports
        resolved_count = sum(
            1
            for imports in result.resolved_imports.values()
            for imp in imports
            if imp.resolved_path is not None
        )
        assert resolved_count > 0

    def test_language_filter(self, python_simple_repo: Path) -> None:
        config = Config(cache=False, languages=["python"])
        adapters = default_adapter_registry.build_all()
        result = parse_repository(python_simple_repo, config, adapters)

        for f in result.files:
            assert f.language == "python"


class TestBuildChunks:
    def test_produces_chunks_from_parsed_files(self, python_simple_repo: Path) -> None:
        config = Config(cache=False)
        adapters = default_adapter_registry.build_all()
        artifacts = parse_repository(python_simple_repo, config, adapters)

        index_config = IndexConfig()
        chunks = build_chunks(artifacts.files, artifacts.parsed_files, index_config)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.file_path
            assert chunk.content
            assert chunk.language == "python"

    def test_strict_mode_raises_on_missing_file(self, python_simple_repo: Path) -> None:
        config = Config(cache=False)
        adapters = default_adapter_registry.build_all()
        artifacts = parse_repository(python_simple_repo, config, adapters)

        # Corrupt absolute_path on one file to trigger OSError
        artifacts.files[0] = artifacts.files[0].model_copy(
            update={"absolute_path": "/nonexistent/path/to/file.py"}
        )

        index_config = IndexConfig()
        # strict=True should raise ParseError
        from archex.exceptions import ParseError

        with pytest.raises(ParseError, match="Failed to read file"):
            build_chunks(artifacts.files, artifacts.parsed_files, index_config, strict=True)

    def test_non_strict_skips_missing_file(self, python_simple_repo: Path) -> None:
        config = Config(cache=False)
        adapters = default_adapter_registry.build_all()
        artifacts = parse_repository(python_simple_repo, config, adapters)

        # Corrupt absolute_path on one file
        artifacts.files[0] = artifacts.files[0].model_copy(
            update={"absolute_path": "/nonexistent/path/to/file.py"}
        )

        index_config = IndexConfig()
        # strict=False (default) should skip the missing file without raising
        chunks = build_chunks(artifacts.files, artifacts.parsed_files, index_config)
        # Should still produce chunks from the other files
        assert len(chunks) > 0
        # Other files should still produce chunks
        chunk_paths = {c.file_path for c in chunks}
        assert len(chunk_paths) >= 1
