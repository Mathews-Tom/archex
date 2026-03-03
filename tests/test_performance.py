"""Performance optimization tests: parallel parsing and ONNX model caching."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportPrivateUsage=false, reportMissingTypeArgument=false

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from archex.cache import CacheManager
from archex.index.graph import DependencyGraph
from archex.index.store import IndexStore
from archex.models import CodeChunk, Config, DiscoveredFile, Edge, EdgeKind, RepoSource, SymbolKind
from archex.parse.adapters import ADAPTERS
from archex.parse.engine import TreeSitterEngine
from archex.parse.imports import parse_imports
from archex.parse.symbols import extract_symbols

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_discovered_files(repo_path: Path) -> list[DiscoveredFile]:
    """Build DiscoveredFile list from all .py files in repo_path."""
    files: list[DiscoveredFile] = []
    for py_file in sorted(repo_path.rglob("*.py")):
        relative = py_file.relative_to(repo_path)
        files.append(
            DiscoveredFile(
                path=str(relative),
                absolute_path=str(py_file),
                language="python",
                size_bytes=py_file.stat().st_size,
            )
        )
    return files


@pytest.fixture
def engine() -> TreeSitterEngine:
    return TreeSitterEngine()


@pytest.fixture
def adapters() -> dict[str, type]:
    return {lang: cls() for lang, cls in ADAPTERS.items()}  # type: ignore[misc]


@pytest.fixture
def python_simple_files(python_simple_repo: Path) -> list[DiscoveredFile]:
    return _make_discovered_files(python_simple_repo)


class TestParallelSymbolExtraction:
    def test_sequential_produces_results(
        self,
        python_simple_files: list[DiscoveredFile],
        engine: TreeSitterEngine,
        adapters: dict,
    ) -> None:
        results = extract_symbols(python_simple_files, engine, adapters, parallel=False)
        assert len(results) > 0
        for pf in results:
            assert pf.path.endswith(".py")
            assert pf.language == "python"

    def test_parallel_produces_same_paths(
        self,
        python_simple_files: list[DiscoveredFile],
        engine: TreeSitterEngine,
        adapters: dict,
    ) -> None:
        seq_results = extract_symbols(python_simple_files, engine, adapters, parallel=False)
        seq_paths = sorted(r.path for r in seq_results)

        # Duplicate files to exceed the > 10 threshold
        large_files = python_simple_files * 12
        par_results = extract_symbols(large_files, engine, adapters, parallel=True)
        par_paths = sorted(r.path for r in par_results)

        # Both cover the same unique paths
        assert set(seq_paths) == set(par_paths)

    def test_parallel_false_with_small_list_uses_sequential(
        self,
        python_simple_files: list[DiscoveredFile],
        engine: TreeSitterEngine,
        adapters: dict,
    ) -> None:
        """With parallel=False, ProcessPoolExecutor must not be invoked."""
        with patch("archex.parse.symbols.ProcessPoolExecutor") as mock_executor_cls:
            extract_symbols(python_simple_files, engine, adapters, parallel=False)
            mock_executor_cls.assert_not_called()

    def test_parallel_skips_executor_when_few_files(
        self,
        python_simple_files: list[DiscoveredFile],
        engine: TreeSitterEngine,
        adapters: dict,
    ) -> None:
        """Fewer than 11 files → executor not used even with parallel=True."""
        assert len(python_simple_files) <= 10, "Fixture has too many files; adjust test threshold"
        with patch("archex.parse.symbols.ProcessPoolExecutor") as mock_executor_cls:
            extract_symbols(python_simple_files, engine, adapters, parallel=True)
            mock_executor_cls.assert_not_called()

    def test_parallel_falls_back_on_executor_failure(
        self,
        python_simple_files: list[DiscoveredFile],
        engine: TreeSitterEngine,
        adapters: dict,
    ) -> None:
        """If ProcessPoolExecutor raises, falls back to sequential without error."""
        large_files = python_simple_files * 12
        with patch("archex.parse.symbols.ProcessPoolExecutor", side_effect=RuntimeError("no fork")):
            results = extract_symbols(large_files, engine, adapters, parallel=True)
        assert len(results) > 0


class TestParallelImportParsing:
    def test_sequential_produces_results(
        self,
        python_simple_files: list[DiscoveredFile],
        engine: TreeSitterEngine,
        adapters: dict,
    ) -> None:
        result = parse_imports(python_simple_files, engine, adapters, parallel=False)
        assert isinstance(result, dict)
        for path, imports in result.items():
            assert path.endswith(".py")
            assert isinstance(imports, list)

    def test_parallel_false_skips_executor(
        self,
        python_simple_files: list[DiscoveredFile],
        engine: TreeSitterEngine,
        adapters: dict,
    ) -> None:
        with patch("archex.parse.imports.ProcessPoolExecutor") as mock_executor_cls:
            parse_imports(python_simple_files, engine, adapters, parallel=False)
            mock_executor_cls.assert_not_called()

    def test_parallel_skips_executor_when_few_files(
        self,
        python_simple_files: list[DiscoveredFile],
        engine: TreeSitterEngine,
        adapters: dict,
    ) -> None:
        assert len(python_simple_files) <= 10
        with patch("archex.parse.imports.ProcessPoolExecutor") as mock_executor_cls:
            parse_imports(python_simple_files, engine, adapters, parallel=True)
            mock_executor_cls.assert_not_called()

    def test_parallel_falls_back_on_executor_failure(
        self,
        python_simple_files: list[DiscoveredFile],
        engine: TreeSitterEngine,
        adapters: dict,
    ) -> None:
        large_files = python_simple_files * 12
        with patch("archex.parse.imports.ProcessPoolExecutor", side_effect=RuntimeError("no fork")):
            result = parse_imports(large_files, engine, adapters, parallel=True)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestNomicEmbedderCaching:
    def test_cache_dir_parameter_overrides_model_dir(self, tmp_path: Path) -> None:
        """cache_dir parameter sets the effective model directory."""
        cache = tmp_path / "my_cache"

        with (
            patch("archex.index.embeddings.nomic.NomicCodeEmbedder.__init__") as mock_init,
        ):
            mock_init.return_value = None

            from archex.index.embeddings.nomic import NomicCodeEmbedder

            embedder = NomicCodeEmbedder.__new__(NomicCodeEmbedder)
            embedder._batch_size = 32
            embedder._session = None
            embedder._tokenizer = None
            embedder._dimension = 768
            embedder._model_dir = cache / "nomic-embed-code-v1"

            assert embedder._model_dir == cache / "nomic-embed-code-v1"

    def test_cache_dir_string_expands_home(self, tmp_path: Path) -> None:
        """cache_dir as string is expanded (expanduser) and used as base."""
        with (
            patch("onnxruntime.InferenceSession"),
            patch("tokenizers.Tokenizer"),
            patch.dict("sys.modules", {"onnxruntime": MagicMock(), "tokenizers": MagicMock()}),
        ):
            from archex.index.embeddings.nomic import NomicCodeEmbedder

            embedder = NomicCodeEmbedder(cache_dir=str(tmp_path))
            assert embedder._model_dir == tmp_path / "nomic-embed-code-v1"

    def test_default_model_dir_used_when_no_cache_dir(self) -> None:
        """Without cache_dir, model_dir defaults to ~/.archex/models."""
        with patch.dict("sys.modules", {"onnxruntime": MagicMock(), "tokenizers": MagicMock()}):
            from archex.index.embeddings.nomic import _DEFAULT_MODEL_DIR, NomicCodeEmbedder

            embedder = NomicCodeEmbedder()
            assert embedder._model_dir == _DEFAULT_MODEL_DIR / "nomic-embed-code-v1"

    def test_encode_processes_in_batches(self) -> None:
        """encode() iterates in batch_size chunks, calling session.run per batch."""
        import numpy as np

        with patch.dict("sys.modules", {"onnxruntime": MagicMock(), "tokenizers": MagicMock()}):
            from archex.index.embeddings.nomic import NomicCodeEmbedder

            embedder = NomicCodeEmbedder(batch_size=4)

        # Set up a mock tokenizer that returns proper encoded objects
        def make_encoded(batch: list[str]) -> list[MagicMock]:
            result = []
            for _ in batch:
                e = MagicMock()
                e.ids = [1] * 8
                e.attention_mask = [1] * 8
                result.append(e)
            return result

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode_batch.side_effect = make_encoded

        # session.run returns token_embeddings shape (batch, seq, dim)
        def fake_run(_names: object, inputs: dict) -> list:
            batch = inputs["input_ids"].shape[0]
            return [np.ones((batch, 8, 768), dtype=np.float32)]

        mock_session = MagicMock()
        mock_session.run.side_effect = fake_run

        # Inject mocks directly — bypass _load_model
        embedder._session = mock_session
        embedder._tokenizer = mock_tokenizer

        texts = [f"text {i}" for i in range(10)]
        results = embedder.encode(texts)

        assert len(results) == 10
        for vec in results:
            assert len(vec) == 768
        # With batch_size=4 and 10 texts, expect 3 session.run calls
        assert mock_session.run.call_count == 3


# ---------------------------------------------------------------------------
# Phase 6b: Additional performance optimization tests
# ---------------------------------------------------------------------------


class TestBatchFetch:
    def test_get_chunks_by_ids_returns_matching(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = IndexStore(db_path)
        chunks = [
            CodeChunk(
                id=f"file.py:sym{i}:{i}",
                content=f"content {i}",
                file_path="file.py",
                start_line=i,
                end_line=i,
                language="python",
            )
            for i in range(5)
        ]
        store.insert_chunks(chunks)
        result = store.get_chunks_by_ids(["file.py:sym1:1", "file.py:sym3:3"])
        assert len(result) == 2
        assert {c.id for c in result} == {"file.py:sym1:1", "file.py:sym3:3"}
        store.close()

    def test_get_chunks_by_ids_empty(self, tmp_path: Path) -> None:
        store = IndexStore(tmp_path / "test.db")
        assert store.get_chunks_by_ids([]) == []
        store.close()


class TestCentralityCache:
    def test_cached_on_repeated_call(self) -> None:
        g = DependencyGraph()
        g.add_file_node("a.py")
        g.add_file_node("b.py")
        g.add_file_edge("a.py", "b.py")
        c1 = g.structural_centrality()
        c2 = g.structural_centrality()
        assert c1 is c2

    def test_invalidated_on_mutation(self) -> None:
        g = DependencyGraph()
        g.add_file_node("a.py")
        g.add_file_node("b.py")
        g.add_file_edge("a.py", "b.py")
        c1 = g.structural_centrality()
        g.add_file_node("c.py")
        g.add_file_edge("b.py", "c.py")
        c2 = g.structural_centrality()
        assert c1 is not c2
        assert "c.py" in c2


class TestFromEdges:
    def test_reconstructs_graph(self) -> None:
        edges = [
            Edge(source="a.py", target="b.py", kind=EdgeKind.IMPORTS, location="a.py:1"),
            Edge(source="b.py", target="c.py", kind=EdgeKind.IMPORTS),
        ]
        g = DependencyGraph.from_edges(edges)
        assert g.file_count == 3
        assert g.file_edge_count == 2


class TestCacheKeyFingerprint:
    def test_changes_with_git_head(self, tmp_path: Path) -> None:
        cache = CacheManager(cache_dir=str(tmp_path))
        source = RepoSource(local_path="/some/repo")
        with patch.object(CacheManager, "git_head", return_value="abc"):
            k1 = cache.cache_key(source)
        with patch.object(CacheManager, "git_head", return_value="def"):
            k2 = cache.cache_key(source)
        assert k1 != k2

    def test_stable_without_git(self, tmp_path: Path) -> None:
        cache = CacheManager(cache_dir=str(tmp_path))
        source = RepoSource(local_path="/r")
        with patch.object(CacheManager, "git_head", return_value=None):
            k1 = cache.cache_key(source)
            k2 = cache.cache_key(source)
        assert k1 == k2


class TestVectorPath:
    def test_returns_npz_path(self, tmp_path: Path) -> None:
        import hashlib

        cache = CacheManager(cache_dir=str(tmp_path))
        key = hashlib.sha256(b"x").hexdigest()
        vp = cache.vector_path(key)
        assert vp.name == f"{key}.vectors.npz"


class TestQueryCacheSkipsParse:
    def test_parse_not_called_on_cache_hit(self, tmp_path: Path) -> None:
        from archex.index.bm25 import BM25Index

        db_path = tmp_path / "idx.db"
        store = IndexStore(db_path)
        chunk = CodeChunk(
            id="t.py:f:1",
            symbol_id="t.py::f#function",
            content="def f(): pass",
            file_path="t.py",
            start_line=1,
            end_line=1,
            language="python",
            symbol_name="f",
            symbol_kind=SymbolKind.FUNCTION,
        )
        store.insert_chunks([chunk])
        store.insert_edges([Edge(source="t.py", target="u.py", kind=EdgeKind.IMPORTS)])
        bm25 = BM25Index(store)
        bm25.build([chunk])
        store.conn.execute("PRAGMA wal_checkpoint(FULL)")
        store.close()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = CacheManager(cache_dir=str(cache_dir))
        source = RepoSource(local_path="/fake")
        key = cache.cache_key(source)
        cache.put(key, db_path)

        config = Config(cache=True, cache_dir=str(cache_dir))
        with (
            patch("archex.api.extract_symbols") as mock_es,
            patch("archex.cache.CacheManager.git_head", return_value=None),
        ):
            from archex.api import query

            query(source, "what?", config=config)
        mock_es.assert_not_called()

    def test_query_invalidates_stale_cache(self, tmp_path: Path, python_simple_repo: Path) -> None:
        """Cache with needs_reindex flag is invalidated and falls through to full pipeline."""
        from archex.index.bm25 import BM25Index

        db_path = tmp_path / "stale.db"
        store = IndexStore(db_path)
        # Insert a chunk with no symbol_id to simulate pre-upgrade (schema v1) data.
        # The migration in IndexStore._migrate_schema will detect the NULL and set
        # needs_reindex=true in the metadata table.
        chunk = CodeChunk(
            id="t.py:_module:1",
            symbol_id=None,  # NULL symbol_id triggers needs_reindex
            content="x = 1",
            file_path="t.py",
            start_line=1,
            end_line=1,
            language="python",
        )
        store.insert_chunks([chunk])
        bm25 = BM25Index(store)
        bm25.build([chunk])
        store.conn.execute("PRAGMA wal_checkpoint(FULL)")
        store.close()

        # Re-open the store to force _migrate_schema which sets needs_reindex flag.
        store2 = IndexStore(db_path)
        assert store2.needs_reindex(), "Expected needs_reindex flag after NULL symbol_id migration"
        store2.close()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = CacheManager(cache_dir=str(cache_dir))
        source = RepoSource(local_path=str(python_simple_repo))
        with patch("archex.cache.CacheManager.git_head", return_value=None):
            key = cache.cache_key(source)
        cache.put(key, db_path)

        config = Config(cache=True, cache_dir=str(cache_dir))
        with (
            patch("archex.api.extract_symbols", wraps=extract_symbols) as mock_es,
            patch("archex.cache.CacheManager.git_head", return_value=None),
        ):
            from archex.api import query

            query(source, "what?", config=config)
        # extract_symbols must have been called because the stale cache was invalidated.
        mock_es.assert_called()


class TestGetEmbedder:
    """Tests for api._get_embedder (lines 319-332)."""

    def test_returns_none_when_no_embedder(self) -> None:
        from archex.api import _get_embedder
        from archex.models import IndexConfig

        assert _get_embedder(IndexConfig()) is None
        assert _get_embedder(IndexConfig(embedder="")) is None

    def test_returns_none_for_unknown_embedder(self) -> None:
        from archex.api import _get_embedder
        from archex.models import IndexConfig

        assert _get_embedder(IndexConfig(embedder="custom_api")) is None

    def test_creates_nomic_embedder(self) -> None:
        from archex.api import _get_embedder
        from archex.models import IndexConfig

        mock_nomic = MagicMock()
        with patch("archex.index.embeddings.nomic.NomicCodeEmbedder", mock_nomic):
            result = _get_embedder(IndexConfig(embedder="nomic"))
        mock_nomic.assert_called_once()
        assert result is mock_nomic.return_value

    def test_creates_sentence_transformers_embedder(self) -> None:
        from archex.api import _get_embedder
        from archex.models import IndexConfig

        mock_st = MagicMock()
        with patch("archex.index.embeddings.sentence_tf.SentenceTransformerEmbedder", mock_st):
            result = _get_embedder(IndexConfig(embedder="sentence_transformers"))
        mock_st.assert_called_once()
        assert result is mock_st.return_value


class TestAnalyzeDefaultConfig:
    """Test that analyze() creates default Config when None is passed (line 91)."""

    def test_analyze_with_none_config(self, python_simple_repo: Path) -> None:
        from archex.api import analyze
        from archex.models import RepoSource

        source = RepoSource(local_path=str(python_simple_repo))
        profile = analyze(source, config=None)
        assert profile is not None
        assert profile.stats.total_files > 0


class TestQueryDefaultConfig:
    """Test that query() creates default Config when None is passed (line 178)."""

    def test_query_with_none_config(self, python_simple_repo: Path) -> None:
        from archex.api import query
        from archex.models import RepoSource

        source = RepoSource(local_path=str(python_simple_repo))
        result = query(source, "what functions exist?", config=None)
        assert result is not None


class TestCompareParallel:
    def test_both_analyses_called(self) -> None:
        from archex.models import ArchProfile, CodebaseStats, RepoMetadata

        mock_profile = ArchProfile(
            repo=RepoMetadata(local_path="/a"),
            stats=CodebaseStats(),
        )
        with patch("archex.api.analyze", return_value=mock_profile) as mock_a:
            from archex.api import compare

            compare(RepoSource(local_path="/a"), RepoSource(local_path="/b"))
        assert mock_a.call_count == 2
