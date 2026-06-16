"""Tests for EmbedderRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from archex.exceptions import ConfigError
from archex.index.embeddings import (
    JINA_BERT_CODE_REVISION,
    JINA_V2_MAX_SEQ_LENGTH,
    JINA_V2_MODEL_ID,
    JINA_V2_MODEL_REVISION,
    EmbedderRegistry,
    default_embedder_registry,
)
from archex.index.embeddings.base import Embedder
from archex.models import IndexConfig


def _fake_factory(_index_config: IndexConfig) -> Embedder:
    mock = MagicMock(spec=Embedder)
    mock.dimension = 384
    return mock


class TestEmbedderRegistry:
    def test_register_and_create(self) -> None:
        reg = EmbedderRegistry()
        reg.register("test_emb", _fake_factory)
        config = IndexConfig(vector=True, embedder="test_emb")
        emb = reg.create(config)
        assert emb is not None
        assert isinstance(emb, Embedder)

    def test_create_reuses_embedder_instance(self) -> None:
        call_count = 0

        def factory(_index_config: IndexConfig) -> Embedder:
            nonlocal call_count
            call_count += 1
            return _fake_factory(_index_config)

        reg = EmbedderRegistry()
        reg.register("test_emb", factory)
        config = IndexConfig(vector=True, embedder="test_emb")

        first = reg.create(config)
        second = reg.create(config)

        assert first is second
        assert call_count == 1

    def test_register_replaces_cached_embedder_instance(self) -> None:
        reg = EmbedderRegistry()
        reg.register("test_emb", _fake_factory)
        config = IndexConfig(vector=True, embedder="test_emb")
        first = reg.create(config)

        reg.register("test_emb", _fake_factory)
        second = reg.create(config)

        assert first is not second

    def test_create_unknown_raises_config_error(self) -> None:
        reg = EmbedderRegistry()
        config = IndexConfig(vector=True, embedder="unknown")
        with pytest.raises(ConfigError, match="Unknown embedder"):
            reg.create(config)

    def test_create_no_embedder_returns_none(self) -> None:
        reg = EmbedderRegistry()
        config = IndexConfig(vector=False, embedder="")
        assert reg.create(config) is None

    def test_create_supports_zero_arg_embedder_factory(self) -> None:
        def factory() -> Embedder:
            return _fake_factory(IndexConfig(vector=True, embedder="test_emb"))

        reg = EmbedderRegistry()
        reg.register("test_emb", factory)

        assert reg.create(IndexConfig(vector=True, embedder="test_emb")) is not None

    def test_default_registry_has_builtin_embedders(self) -> None:
        assert default_embedder_registry.get("nomic") is not None
        assert default_embedder_registry.get("sentence_transformers") is not None
        assert default_embedder_registry.get("jina-v2") is not None
        assert default_embedder_registry.get("coderank") is not None

    def test_remote_code_embedder_requires_opt_in(self) -> None:
        config = IndexConfig(vector=True, embedder="jina-v2")
        with pytest.raises(ConfigError, match="Remote code is disabled.*jina-embeddings"):
            default_embedder_registry.create(config)

    def test_jina_v2_factory_pins_model_and_code_revisions_when_opted_in(self) -> None:
        from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

        embedder = default_embedder_registry.create(
            IndexConfig(vector=True, embedder="jina-v2", allow_remote_code=True)
        )
        assert isinstance(embedder, SentenceTransformerEmbedder)
        assert embedder._model_name == JINA_V2_MODEL_ID  # pyright: ignore[reportPrivateUsage]
        assert embedder._revision == JINA_V2_MODEL_REVISION  # pyright: ignore[reportPrivateUsage]
        assert embedder._model_kwargs == {  # pyright: ignore[reportPrivateUsage]
            "code_revision": JINA_BERT_CODE_REVISION
        }
        assert embedder._config_kwargs == {  # pyright: ignore[reportPrivateUsage]
            "code_revision": JINA_BERT_CODE_REVISION
        }
        assert embedder._max_seq_length == JINA_V2_MAX_SEQ_LENGTH  # pyright: ignore[reportPrivateUsage]

    def test_load_entry_points(self) -> None:
        reg = EmbedderRegistry()
        mock_ep = MagicMock()
        mock_ep.name = "custom_emb"
        mock_ep.load.return_value = _fake_factory
        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            reg.load_entry_points(group="archex.embedders")
        assert reg.get("custom_emb") is _fake_factory
