"""Language adapter registry: maps language identifiers to parser adapter implementations."""

from __future__ import annotations

import importlib.metadata
import logging

from archex.exceptions import ConfigError
from archex.parse.adapters.base import LanguageAdapter
from archex.parse.adapters.go import GoAdapter
from archex.parse.adapters.python import PythonAdapter
from archex.parse.adapters.rust import RustAdapter
from archex.parse.adapters.typescript import TypeScriptAdapter

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry of language adapters, supporting programmatic and entry-point registration."""

    def __init__(self) -> None:
        self._adapters: dict[str, type[LanguageAdapter]] = {}
        self._entry_points_loaded: bool = False
        self._entry_points_strict: bool = False

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

    def load_entry_points(
        self,
        group: str = "archex.language_adapters",
        strict: bool = False,
    ) -> None:
        """Load adapter classes from installed entry points.

        Entry points should map language ID to adapter class, e.g.:
            [project.entry-points."archex.language_adapters"]
            java = "mypackage.adapters:JavaAdapter"
        """
        if self._entry_points_loaded and (not strict or self._entry_points_strict):
            return  # already loaded at equal or higher strictness
        eps = sorted(importlib.metadata.entry_points(group=group), key=lambda ep: ep.name)
        for ep in eps:
            try:
                cls = ep.load()
                self._adapters[ep.name] = cls
                logger.info("Loaded adapter %s from entry point", ep.name)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                if strict:
                    raise ConfigError(
                        f"Failed to load adapter entry point {ep.name!r}: {exc}"
                    ) from exc
                logger.warning("Failed to load adapter entry point %s: %s", ep.name, exc)
        self._entry_points_loaded = True
        self._entry_points_strict = strict


# Module-level default registry with built-in adapters
default_adapter_registry = AdapterRegistry()
default_adapter_registry.register("python", PythonAdapter)  # type: ignore[type-abstract]
default_adapter_registry.register("typescript", TypeScriptAdapter)  # type: ignore[type-abstract]
default_adapter_registry.register("javascript", TypeScriptAdapter)  # type: ignore[type-abstract]
default_adapter_registry.register("go", GoAdapter)  # type: ignore[type-abstract]
default_adapter_registry.register("rust", RustAdapter)  # type: ignore[type-abstract]

# Legacy compat
ADAPTERS: dict[str, type[LanguageAdapter]] = default_adapter_registry.adapter_classes


__all__ = [
    "AdapterRegistry",
    "ADAPTERS",
    "LanguageAdapter",
    "default_adapter_registry",
]
