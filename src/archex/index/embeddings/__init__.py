"""Embeddings sub-package: base protocol, provider implementations, and registry."""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from archex.exceptions import ConfigError
from archex.index.embeddings.base import Embedder

if TYPE_CHECKING:
    from archex.models import IndexConfig

logger = logging.getLogger(__name__)

EmbedderFactory = Callable[[], Embedder]
JINA_V2_MODEL_ID = "jinaai/jina-embeddings-v2-base-code"
JINA_V2_MODEL_REVISION = "516f4baf13dec4ddddda8631e019b5737c8bc250"
JINA_BERT_CODE_REVISION = "3baf9e3ac750e76e8edd3019170176884695fb94"
JINA_V2_MAX_SEQ_LENGTH = 1024


__all__ = [
    "Embedder",
    "EmbedderRegistry",
    "default_embedder_registry",
]


def _fastembed_factory() -> Embedder:
    from archex.index.embeddings.fast import FastEmbedder

    return FastEmbedder()


def _nomic_factory() -> Embedder:
    from archex.index.embeddings.nomic import NomicCodeEmbedder

    return NomicCodeEmbedder()


def _jina_v2_factory() -> Embedder:
    from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        model_name=JINA_V2_MODEL_ID,
        trust_remote_code=True,
        revision=JINA_V2_MODEL_REVISION,
        model_kwargs={"code_revision": JINA_BERT_CODE_REVISION},
        config_kwargs={"code_revision": JINA_BERT_CODE_REVISION},
        max_seq_length=JINA_V2_MAX_SEQ_LENGTH,
    )


def _sentence_tf_factory() -> Embedder:
    from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder()


def _coderank_factory() -> Embedder:
    from archex.index.embeddings.coderank import CodeRankEmbedder

    return CodeRankEmbedder()


class EmbedderRegistry:
    """Registry for embedder factories with entry-point support."""

    def __init__(self) -> None:
        self._factories: dict[str, EmbedderFactory] = {}
        self._entry_points_loaded: bool = False
        self._instances: dict[str, Embedder] = {}
        self._entry_points_strict: bool = False

    def register(self, name: str, factory: EmbedderFactory) -> None:
        """Register an embedder factory by name."""
        self._factories[name] = factory
        self._instances.pop(name, None)

    def get(self, name: str) -> EmbedderFactory | None:
        """Return the factory for an embedder name, or None."""
        return self._factories.get(name)

    def create(self, index_config: IndexConfig) -> Embedder | None:
        """Return a cached embedder for index_config.

        Returns None when no embedder is configured.
        Raises ConfigError for unknown embedder names.
        """
        if not index_config.embedder:
            return None
        cached = self._instances.get(index_config.embedder)
        if cached is not None:
            return cached
        factory = self._factories.get(index_config.embedder)
        if factory is None:
            raise ConfigError(f"Unknown embedder: {index_config.embedder!r}")
        embedder = factory()
        self._instances[index_config.embedder] = embedder
        return embedder

    def load_entry_points(
        self,
        group: str = "archex.embedders",
        strict: bool = False,
    ) -> None:
        """Load embedder factories from installed entry points."""
        if self._entry_points_loaded and (not strict or self._entry_points_strict):
            return
        eps = sorted(importlib.metadata.entry_points(group=group), key=lambda ep: ep.name)
        for ep in eps:
            try:
                factory = ep.load()
                self._factories[ep.name] = factory
                self._instances.pop(ep.name, None)
                logger.info("Loaded embedder %s from entry point", ep.name)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                if strict:
                    raise ConfigError(
                        f"Failed to load embedder entry point {ep.name!r}: {exc}"
                    ) from exc
                logger.warning("Failed to load embedder entry point %s: %s", ep.name, exc)
        self._entry_points_loaded = True
        self._entry_points_strict = strict


default_embedder_registry = EmbedderRegistry()
default_embedder_registry.register("fastembed", _fastembed_factory)
default_embedder_registry.register("nomic", _nomic_factory)
default_embedder_registry.register("sentence_transformers", _sentence_tf_factory)
default_embedder_registry.register("jina-v2", _jina_v2_factory)
default_embedder_registry.register("coderank", _coderank_factory)
