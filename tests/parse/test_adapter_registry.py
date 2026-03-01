from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    import pytest

from archex.parse.adapters import (
    ADAPTERS,
    AdapterRegistry,
    GoAdapter,
    LanguageAdapter,
    PythonAdapter,
    RustAdapter,
    TypeScriptAdapter,
    default_adapter_registry,
)


def _make_ep(name: str, cls: type) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = cls
    return ep


def _make_broken_ep(name: str) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.side_effect = ImportError("missing dep")
    return ep


class _StubAdapter(LanguageAdapter):
    """Minimal concrete adapter for testing."""

    @property
    def language(self) -> str:
        return "stub"

    def parse(self, source: str, path: str) -> object:  # type: ignore[override]  # noqa: ARG002
        return {}


def test_languages_property_returns_registered_ids() -> None:
    registry = AdapterRegistry()
    registry.register("python", PythonAdapter)  # type: ignore[type-abstract]
    registry.register("go", GoAdapter)  # type: ignore[type-abstract]
    assert set(registry.languages) == {"python", "go"}


def test_load_entry_points_success_registers_adapter() -> None:
    registry = AdapterRegistry()
    ep = _make_ep("stub", _StubAdapter)

    with patch("importlib.metadata.entry_points", return_value=[ep]):
        registry.load_entry_points("archex.language_adapters")

    assert registry.get("stub") is _StubAdapter


def test_load_entry_points_failure_logged_no_crash(caplog: pytest.LogCaptureFixture) -> None:
    registry = AdapterRegistry()
    ep = _make_broken_ep("broken")

    with (
        patch("importlib.metadata.entry_points", return_value=[ep]),
        caplog.at_level(logging.WARNING, logger="archex.parse.adapters"),  # type: ignore[no-untyped-call]
    ):
        registry.load_entry_points("archex.language_adapters")

    assert registry.get("broken") is None
    assert any("broken" in r.message for r in caplog.records)  # type: ignore[union-attr]


def test_load_entry_points_mixed_success_and_failure() -> None:
    registry = AdapterRegistry()
    good_ep = _make_ep("stub", _StubAdapter)
    bad_ep = _make_broken_ep("broken")

    with patch("importlib.metadata.entry_points", return_value=[good_ep, bad_ep]):
        registry.load_entry_points("archex.language_adapters")

    assert registry.get("stub") is _StubAdapter
    assert registry.get("broken") is None


def test_default_registry_has_builtin_languages() -> None:
    assert set(default_adapter_registry.languages) >= {"python", "typescript", "go", "rust"}
    assert default_adapter_registry.get("python") is PythonAdapter
    assert default_adapter_registry.get("typescript") is TypeScriptAdapter
    assert default_adapter_registry.get("go") is GoAdapter
    assert default_adapter_registry.get("rust") is RustAdapter
    assert ADAPTERS is default_adapter_registry.adapter_classes
