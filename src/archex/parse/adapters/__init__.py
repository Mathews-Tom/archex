"""Language adapter registry: maps language identifiers to parser adapter implementations."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import TYPE_CHECKING

from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.go import GoAdapter
from archex.parse.adapters.python import PythonAdapter
from archex.parse.adapters.rust import RustAdapter
from archex.parse.adapters.typescript import TypeScriptAdapter

if TYPE_CHECKING:
    from archex.parse.engine import TreeSitterEngine

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry of language adapters, supporting programmatic and entry-point registration."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[LanguageAdapter]] = {}

    def register(self, language: str, adapter_cls: type[LanguageAdapter]) -> None:
        """Register an adapter class for a language identifier."""
        self._adapters[language] = adapter_cls

    def get(self, language: str) -> type[LanguageAdapter] | None:
        """Return the adapter class for a language, or None."""
        return self._adapters.get(language)

    @property
    def languages(self) -> list[str]:
        return list(self._adapters.keys())

    @property
    def adapter_classes(self) -> dict[str, type[LanguageAdapter]]:
        """Return the underlying adapter class mapping."""
        return self._adapters

    def build_all(self) -> dict[str, LanguageAdapter]:
        """Instantiate all registered adapters."""
        return {lang: cls() for lang, cls in self._adapters.items()}

    def load_entry_points(self, group: str = "archex.language_adapters") -> None:
        """Load adapter classes from installed entry points.

        Entry points should map language ID to adapter class, e.g.:
            [project.entry-points."archex.language_adapters"]
            java = "mypackage.adapters:JavaAdapter"
        """
        for ep in importlib.metadata.entry_points(group=group):
            try:
                cls = ep.load()
                self._adapters[ep.name] = cls
                logger.info("Loaded adapter %s from entry point", ep.name)
            except Exception:
                logger.warning("Failed to load adapter entry point %s", ep.name)


# Module-level default registry with built-in adapters
default_adapter_registry = AdapterRegistry()
default_adapter_registry.register("python", PythonAdapter)  # type: ignore[type-abstract]
default_adapter_registry.register("typescript", TypeScriptAdapter)  # type: ignore[type-abstract]
default_adapter_registry.register("go", GoAdapter)  # type: ignore[type-abstract]
default_adapter_registry.register("rust", RustAdapter)  # type: ignore[type-abstract]

# Legacy compat
ADAPTERS: dict[str, type[LanguageAdapter]] = default_adapter_registry.adapter_classes


def get_adapter(language_id: str, engine: TreeSitterEngine) -> LanguageAdapter | None:
    """Return an instantiated LanguageAdapter for language_id, or None if unsupported."""
    adapter_class = default_adapter_registry.get(language_id)
    if adapter_class is None:
        return None
    _ = engine  # reserved for future adapters that need engine at construction
    return adapter_class()


__all__ = [
    "AdapterRegistry",
    "ADAPTERS",
    "LanguageAdapter",
    "default_adapter_registry",
    "get_adapter",
]
