"""Tests for the unified artifact pipeline (produce_artifacts)."""

from __future__ import annotations

from pathlib import Path

from archex.models import Config, EdgeKind, IndexConfig
from archex.parse.adapters import LanguageAdapter, default_adapter_registry
from archex.pipeline.models import ArtifactBundle
from archex.pipeline.service import (
    ParseArtifacts,
    build_chunks,
    parse_repository,
    produce_artifacts,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
PYTHON_SIMPLE = FIXTURES_DIR / "python_simple"


def _adapters() -> dict[str, LanguageAdapter]:
    return default_adapter_registry.build_all()


class TestProduceArtifacts:
    """produce_artifacts returns a well-formed ArtifactBundle."""

    def test_returns_artifact_bundle(self) -> None:
        config = Config(languages=["python"])
        bundle = produce_artifacts(PYTHON_SIMPLE, config, _adapters())
        assert isinstance(bundle, ArtifactBundle)

    def test_files_discovered(self) -> None:
        config = Config(languages=["python"])
        bundle = produce_artifacts(PYTHON_SIMPLE, config, _adapters())
        paths = {f.path for f in bundle.files}
        assert "main.py" in paths
        assert "models.py" in paths
        assert "utils.py" in paths

    def test_parsed_files_nonempty(self) -> None:
        config = Config(languages=["python"])
        bundle = produce_artifacts(PYTHON_SIMPLE, config, _adapters())
        assert len(bundle.parsed_files) > 0
        parsed_paths = {pf.path for pf in bundle.parsed_files}
        assert "main.py" in parsed_paths

    def test_chunks_produced(self) -> None:
        config = Config(languages=["python"])
        bundle = produce_artifacts(PYTHON_SIMPLE, config, _adapters())
        assert len(bundle.chunks) > 0
        chunk_files = {c.file_path for c in bundle.chunks}
        assert "main.py" in chunk_files

    def test_edges_have_imports_kind(self) -> None:
        config = Config(languages=["python"])
        bundle = produce_artifacts(PYTHON_SIMPLE, config, _adapters())
        for edge in bundle.edges:
            assert edge.kind == EdgeKind.IMPORTS

    def test_sources_populated(self) -> None:
        config = Config(languages=["python"])
        bundle = produce_artifacts(PYTHON_SIMPLE, config, _adapters())
        assert len(bundle.sources) > 0
        for path, content in bundle.sources.items():
            assert isinstance(path, str)
            assert isinstance(content, bytes)
            assert len(content) > 0

    def test_resolved_imports_dict(self) -> None:
        config = Config(languages=["python"])
        bundle = produce_artifacts(PYTHON_SIMPLE, config, _adapters())
        assert isinstance(bundle.resolved_imports, dict)

    def test_custom_index_config(self) -> None:
        config = Config(languages=["python"])
        index_config = IndexConfig(chunk_max_tokens=64, chunk_min_tokens=8)
        bundle = produce_artifacts(PYTHON_SIMPLE, config, _adapters(), index_config=index_config)
        for chunk in bundle.chunks:
            # Chunks should respect the smaller token budget (with some tolerance
            # for imports_context prepended during chunking).
            assert chunk.token_count < 200


class TestProduceArtifactsMatchesSeparateCalls:
    """produce_artifacts output matches calling parse_repository + build_chunks separately."""

    def test_same_parsed_files(self) -> None:
        config = Config(languages=["python"])
        adapters = _adapters()
        index_config = IndexConfig()

        bundle = produce_artifacts(PYTHON_SIMPLE, config, adapters, index_config=index_config)
        separate = parse_repository(PYTHON_SIMPLE, config, adapters)

        assert isinstance(separate, ParseArtifacts)
        bundle_paths = sorted(pf.path for pf in bundle.parsed_files)
        separate_paths = sorted(pf.path for pf in separate.parsed_files)
        assert bundle_paths == separate_paths

    def test_same_chunks(self) -> None:
        config = Config(languages=["python"])
        adapters = _adapters()
        index_config = IndexConfig()

        bundle = produce_artifacts(PYTHON_SIMPLE, config, adapters, index_config=index_config)
        separate = parse_repository(PYTHON_SIMPLE, config, adapters)
        separate_chunks = build_chunks(separate.files, separate.parsed_files, index_config)

        bundle_ids = sorted(c.id for c in bundle.chunks)
        separate_ids = sorted(c.id for c in separate_chunks)
        assert bundle_ids == separate_ids

    def test_same_chunk_content(self) -> None:
        config = Config(languages=["python"])
        adapters = _adapters()
        index_config = IndexConfig()

        bundle = produce_artifacts(PYTHON_SIMPLE, config, adapters, index_config=index_config)
        separate = parse_repository(PYTHON_SIMPLE, config, adapters)
        separate_chunks = build_chunks(separate.files, separate.parsed_files, index_config)

        bundle_map = {c.id: c.content for c in bundle.chunks}
        separate_map = {c.id: c.content for c in separate_chunks}
        assert bundle_map == separate_map


class TestMultiLanguage:
    """produce_artifacts works with non-Python fixtures."""

    def test_go_fixture(self) -> None:
        go_fixture = FIXTURES_DIR / "go_simple"
        if not go_fixture.exists():
            return
        config = Config(languages=["go"])
        bundle = produce_artifacts(go_fixture, config, _adapters())
        assert len(bundle.chunks) > 0

    def test_typescript_fixture(self) -> None:
        ts_fixture = FIXTURES_DIR / "typescript_simple"
        if not ts_fixture.exists():
            return
        config = Config(languages=["typescript"])
        bundle = produce_artifacts(ts_fixture, config, _adapters())
        assert len(bundle.chunks) > 0
