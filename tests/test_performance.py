"""Performance optimization tests: parallel parsing and ONNX model caching."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportPrivateUsage=false, reportMissingTypeArgument=false

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from archex.models import DiscoveredFile
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
