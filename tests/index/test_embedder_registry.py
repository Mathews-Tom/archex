"""Tests for EmbedderRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from archex.exceptions import ConfigError
from archex.index.embeddings import EmbedderRegistry, default_embedder_registry
from archex.index.embeddings.base import Embedder
from archex.models import IndexConfig


def _fake_factory() -> Embedder:
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

    def test_create_unknown_raises_config_error(self) -> None:
        reg = EmbedderRegistry()
        config = IndexConfig(vector=True, embedder="unknown")
        with pytest.raises(ConfigError, match="Unknown embedder"):
            reg.create(config)

    def test_create_no_embedder_returns_none(self) -> None:
        reg = EmbedderRegistry()
        config = IndexConfig(vector=False, embedder="")
        assert reg.create(config) is None

    def test_default_registry_has_nomic_and_sentence_tf(self) -> None:
        assert default_embedder_registry.get("nomic") is not None
        assert default_embedder_registry.get("sentence_transformers") is not None

    def test_load_entry_points(self) -> None:
        reg = EmbedderRegistry()
        mock_ep = MagicMock()
        mock_ep.name = "custom_emb"
        mock_ep.load.return_value = _fake_factory
        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            reg.load_entry_points(group="archex.embedders")
        assert reg.get("custom_emb") is _fake_factory
