"""Tests for the unified artifact pipeline (produce_artifacts)."""

from __future__ import annotations

from pathlib import Path

from archex.models import ChunkRange, Config, EdgeKind, IndexConfig, ParsedFile
from archex.parse.adapters import LanguageAdapter, default_adapter_registry
from archex.pipeline.chunker import ASTChunker
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

    def test_chunk_only_ranges_keep_line_stable_chunks(self) -> None:
        source = b"SELECT * FROM users;\n-- keep the separator\nCREATE TABLE users(id INT);\n"
        parsed = ParsedFile(
            path="schema.sql",
            language="sql",
            chunk_ranges=[
                ChunkRange(start_line=1, end_line=1),
                ChunkRange(start_line=3, end_line=3),
            ],
            lines=3,
        )

        chunks = ASTChunker(IndexConfig()).chunk_file(parsed, source)

        assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(1, 1), (2, 2), (3, 3)]
        assert [chunk.content for chunk in chunks] == [
            "SELECT * FROM users;",
            "-- keep the separator",
            "CREATE TABLE users(id INT);",
        ]

    def test_chunk_only_toml_tables_keep_headers(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[a]\nname = "a"\n\n[b]\nname = "b"\n\n[c]\nname = "c"\n',
        )

        bundle = produce_artifacts(tmp_path, Config(languages=["toml"]), _adapters())

        assert [(chunk.start_line, chunk.end_line) for chunk in bundle.chunks] == [
            (1, 3),
            (4, 6),
            (7, 8),
        ]
        assert [chunk.content.splitlines()[0] for chunk in bundle.chunks] == ["[a]", "[b]", "[c]"]
        assert all(chunk.symbol_name is None for chunk in bundle.chunks)


class TestSurrogateSummaryInclusion:
    """build_chunk_surrogates includes summary in surrogate text."""

    def test_surrogate_includes_summary(self) -> None:
        from archex.models import CodeChunk, SymbolKind
        from archex.pipeline.service import build_chunk_surrogates

        chunk = CodeChunk(
            id="orm.py:Session:1",
            content="class Session: pass",
            file_path="orm.py",
            start_line=1,
            end_line=1,
            symbol_name="Session",
            symbol_kind=SymbolKind.CLASS,
            language="python",
            token_count=10,
            summary="Database session management with scoped_session factory.",
        )
        surrogates = build_chunk_surrogates([chunk])
        assert len(surrogates) == 1
        assert "summary: Database session management" in surrogates[0].surrogate_text

    def test_surrogate_omits_summary_when_none(self) -> None:
        from archex.models import CodeChunk, SymbolKind
        from archex.pipeline.service import build_chunk_surrogates

        chunk = CodeChunk(
            id="util.py:helper:1",
            content="def helper(): pass",
            file_path="util.py",
            start_line=1,
            end_line=1,
            symbol_name="helper",
            symbol_kind=SymbolKind.FUNCTION,
            language="python",
            token_count=5,
        )
        surrogates = build_chunk_surrogates([chunk])
        assert "summary:" not in surrogates[0].surrogate_text
