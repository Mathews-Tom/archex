"""Tests for embedding providers: protocol conformance and error handling."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import patch

import pytest

from archex.exceptions import ArchexIndexError
from archex.index.embeddings.base import Embedder


class HashEmbedder:
    """Deterministic test embedder that produces vectors from content hashes."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            # Expand hash bytes to fill dimension
            raw: list[float] = []
            for i in range(self._dim):
                byte_val = h[i % len(h)]
                raw.append((byte_val / 255.0) * 2 - 1)
            results.append(raw)
        return results

    @property
    def dimension(self) -> int:
        return self._dim


class TestEmbedderProtocol:
    def test_hash_embedder_satisfies_protocol(self) -> None:
        embedder = HashEmbedder()
        assert isinstance(embedder, Embedder)

    def test_encode_returns_correct_shape(self) -> None:
        embedder = HashEmbedder(dim=128)
        texts = ["hello world", "foo bar"]
        result = embedder.encode(texts)
        assert len(result) == 2
        assert all(len(v) == 128 for v in result)

    def test_encode_deterministic(self) -> None:
        embedder = HashEmbedder()
        a = embedder.encode(["test"])
        b = embedder.encode(["test"])
        assert a == b

    def test_encode_different_texts_different_vectors(self) -> None:
        embedder = HashEmbedder()
        results = embedder.encode(["alpha", "beta"])
        assert results[0] != results[1]

    def test_encode_empty_list(self) -> None:
        embedder = HashEmbedder()
        assert embedder.encode([]) == []

    def test_dimension_property(self) -> None:
        assert HashEmbedder(dim=32).dimension == 32
        assert HashEmbedder(dim=768).dimension == 768


class TestNomicCodeEmbedder:
    def test_import_error_raises_archex_index_error(self) -> None:
        import builtins

        original_import: Any = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> object:
            if name in ("onnxruntime", "tokenizers"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Reload to trigger the import check
            import importlib

            from archex.index.embeddings import nomic

            with pytest.raises(ArchexIndexError, match="onnxruntime"):
                importlib.reload(nomic)
                nomic.NomicCodeEmbedder()

    def test_load_model_missing_onnx_file_raises(self, tmp_path: Any) -> None:
        """_load_model raises ArchexIndexError when model.onnx is missing."""
        from unittest.mock import MagicMock

        mock_ort = MagicMock()
        mock_tok = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort, "tokenizers": mock_tok}):
            from archex.index.embeddings.nomic import NomicCodeEmbedder

            embedder = NomicCodeEmbedder(model_dir=tmp_path)
            with pytest.raises(ArchexIndexError, match="ONNX model not found"):
                embedder._load_model()  # pyright: ignore[reportPrivateUsage]

    def test_load_model_missing_tokenizer_raises(self, tmp_path: Any) -> None:
        """_load_model raises ArchexIndexError when tokenizer.json is missing."""
        from unittest.mock import MagicMock

        model_dir = tmp_path / "nomic-embed-code-v1"
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_text("fake")

        mock_ort = MagicMock()
        mock_tok = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort, "tokenizers": mock_tok}):
            from archex.index.embeddings.nomic import NomicCodeEmbedder

            embedder = NomicCodeEmbedder(model_dir=tmp_path)
            with pytest.raises(ArchexIndexError, match="Tokenizer not found"):
                embedder._load_model()  # pyright: ignore[reportPrivateUsage]

    def test_load_model_success(self, tmp_path: Any) -> None:
        """_load_model succeeds when both model.onnx and tokenizer.json exist."""
        from unittest.mock import MagicMock

        model_dir = tmp_path / "nomic-embed-code-v1"
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_text("fake")
        (model_dir / "tokenizer.json").write_text("fake")

        mock_ort = MagicMock()
        mock_tok_module = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort, "tokenizers": mock_tok_module}):
            from archex.index.embeddings.nomic import NomicCodeEmbedder

            embedder = NomicCodeEmbedder(model_dir=tmp_path)
            embedder._load_model()  # pyright: ignore[reportPrivateUsage]

        assert embedder._session is not None  # pyright: ignore[reportPrivateUsage]
        assert embedder._tokenizer is not None  # pyright: ignore[reportPrivateUsage]

    def test_dimension_property_returns_768(self) -> None:
        """dimension property returns 768."""
        from unittest.mock import MagicMock

        mock_ort = MagicMock()
        mock_tok = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort, "tokenizers": mock_tok}):
            from archex.index.embeddings.nomic import NomicCodeEmbedder

            embedder = NomicCodeEmbedder()
            assert embedder.dimension == 768


class TestSentenceTransformerEmbedder:
    def test_import_error_raises_archex_index_error(self) -> None:
        import builtins

        original_import: Any = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> object:
            if name == "sentence_transformers":
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            import importlib

            from archex.index.embeddings import sentence_tf

            with pytest.raises(ArchexIndexError, match="sentence-transformers"):
                importlib.reload(sentence_tf)
                sentence_tf.SentenceTransformerEmbedder()

    def test_encode_delegates_to_model(self) -> None:
        """encode() lazy-loads the model and delegates to model.encode()."""
        from unittest.mock import MagicMock

        import numpy as np

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.get_sentence_embedding_dimension.return_value = 2
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(model_name="test-model", batch_size=16)
            result = embedder.encode(["hello", "world"])  # pyright: ignore[reportPrivateUsage]

        assert len(result) == 2
        mock_model.encode.assert_called_once()

    def test_dimension_property_loads_model(self) -> None:
        """dimension property triggers lazy model load and returns embedding dim."""
        from unittest.mock import MagicMock

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
            dim = embedder.dimension

        assert dim == 384

    def test_dimension_raises_when_model_returns_none(self) -> None:
        """dimension raises ArchexIndexError if model dimension is None after loading."""
        from unittest.mock import MagicMock

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = None
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
            with pytest.raises(ArchexIndexError, match="dimension unavailable"):
                _ = embedder.dimension

    def test_load_model_is_idempotent(self) -> None:
        """Calling _load_model() twice does not recreate the model."""
        from unittest.mock import MagicMock

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 128
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
            embedder._load_model()  # pyright: ignore[reportPrivateUsage]
            embedder._load_model()  # pyright: ignore[reportPrivateUsage]

        mock_st_module.SentenceTransformer.assert_called_once()


class TestAPIEmbedder:
    def test_empty_api_key_raises(self) -> None:
        from archex.index.embeddings.api import APIEmbedder

        with pytest.raises(ArchexIndexError, match="api_key"):
            APIEmbedder(api_key="")

    def test_construction_with_valid_key(self) -> None:
        from archex.index.embeddings.api import APIEmbedder

        embedder = APIEmbedder(api_key="test-key", model_name="test-model")
        assert embedder.dimension == 1536

    def test_encode_uses_timeout(self) -> None:
        from unittest.mock import MagicMock, patch

        from archex.index.embeddings.api import APIEmbedder

        embedder = APIEmbedder(api_key="test-key")
        with patch("archex.index.embeddings.api.urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"data": [{"index": 0, "embedding": [0.1, 0.2]}]}'
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.__exit__.return_value = None
            mock_urlopen.return_value = mock_resp

            embedder.encode(["test text"])

            # Verify timeout=30 was passed to urlopen
            calls = mock_urlopen.call_args_list
            assert len(calls) > 0
            # Check that the timeout keyword argument is present in the call
            args, kwargs = calls[0]
            assert kwargs.get("timeout") == 30 or (len(args) > 1 and args[1] == 30)
