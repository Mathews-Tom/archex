"""Embeddings sub-package: base protocol, provider implementations, and registry."""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
from collections.abc import Callable
from typing import cast

from archex.exceptions import ConfigError
from archex.index.embeddings.base import Embedder
from archex.index.model_policy import (
    JINA_BERT_CODE_REVISION,
    JINA_V2_MAX_SEQ_LENGTH,
    JINA_V2_MODEL_ID,
    JINA_V2_MODEL_REVISION,
    embedder_security_profile,
    remote_code_trust_value,
)
from archex.models import IndexConfig

logger = logging.getLogger(__name__)

EmbedderFactory = Callable[[], Embedder] | Callable[[IndexConfig], Embedder]


__all__ = [
    "Embedder",
    "EmbedderRegistry",
    "default_embedder_registry",
]


def _fastembed_factory(index_config: IndexConfig) -> Embedder:
    del index_config
    from archex.index.embeddings.fast import FastEmbedder

    return FastEmbedder()


def _nomic_factory(index_config: IndexConfig) -> Embedder:
    from archex.index.embeddings.nomic import NomicCodeEmbedder

    profile = embedder_security_profile("nomic")
    return NomicCodeEmbedder(
        allow_remote_code=index_config.allow_remote_code,
        revision=profile.model_revision,
    )


def _jina_v2_factory(index_config: IndexConfig) -> Embedder:
    from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

    profile = embedder_security_profile("jina-v2")
    trust_remote_code = remote_code_trust_value(
        profile,
        allow_remote_code=index_config.allow_remote_code,
    )
    return SentenceTransformerEmbedder(
        model_name=JINA_V2_MODEL_ID,
        trust_remote_code=trust_remote_code,
        allow_remote_code=index_config.allow_remote_code,
        revision=JINA_V2_MODEL_REVISION,
        model_kwargs={"code_revision": JINA_BERT_CODE_REVISION},
        config_kwargs={"code_revision": JINA_BERT_CODE_REVISION},
        max_seq_length=JINA_V2_MAX_SEQ_LENGTH,
    )


def _sentence_tf_factory(index_config: IndexConfig) -> Embedder:
    del index_config
    from archex.index.embeddings.sentence_tf import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder()


def _coderank_factory(index_config: IndexConfig) -> Embedder:
    from archex.index.embeddings.coderank import CodeRankEmbedder

    return CodeRankEmbedder(allow_remote_code=index_config.allow_remote_code)


def _call_factory(factory: EmbedderFactory, index_config: IndexConfig) -> Embedder:
    signature = inspect.signature(factory)
    required_positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
        and parameter.default is inspect.Parameter.empty
    ]
    if not required_positional:
        return cast("Callable[[], Embedder]", factory)()
    return cast("Callable[[IndexConfig], Embedder]", factory)(index_config)


class EmbedderRegistry:
    """Registry for embedder factories with entry-point support."""

    def __init__(self) -> None:
        self._factories: dict[str, EmbedderFactory] = {}
        self._entry_points_loaded: bool = False
        self._instances: dict[tuple[str, bool], Embedder] = {}
        self._entry_points_strict: bool = False

    def register(self, name: str, factory: EmbedderFactory) -> None:
        """Register an embedder factory by name."""
        self._factories[name] = factory
        for key in [key for key in self._instances if key[0] == name]:
            self._instances.pop(key)

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
        cache_key = (index_config.embedder, index_config.allow_remote_code)
        cached = self._instances.get(cache_key)
        if cached is not None:
            return cached
        factory = self._factories.get(index_config.embedder)
        if factory is None:
            raise ConfigError(f"Unknown embedder: {index_config.embedder!r}")
        embedder = _call_factory(factory, index_config)
        self._instances[cache_key] = embedder
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
                for key in [key for key in self._instances if key[0] == ep.name]:
                    self._instances.pop(key)
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
