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
