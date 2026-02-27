"""Tests for LLM provider protocol conformance and error handling."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from archex.exceptions import ProviderError
from archex.providers.base import LLMProvider, get_provider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_mock() -> MagicMock:
    """Build a minimal mock of the anthropic SDK module."""
    sdk = MagicMock()
    client = MagicMock()
    sdk.Anthropic.return_value = client
    text_block = MagicMock()
    text_block.text = "response text"
    msg = MagicMock()
    msg.content = [text_block]
    client.messages.create.return_value = msg
    return sdk


def _make_openai_mock() -> MagicMock:
    """Build a minimal mock of the openai SDK module."""
    sdk = MagicMock()
    client = MagicMock()
    sdk.OpenAI.return_value = client
    choice = MagicMock()
    choice.message.content = "response text"
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return sdk


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


def test_anthropic_provider_satisfies_protocol() -> None:
    anthropic_mock = _make_anthropic_mock()
    with (
        patch.dict(sys.modules, {"anthropic": anthropic_mock}),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
    ):
        from archex.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider()
        assert isinstance(provider, LLMProvider)


def test_openai_provider_satisfies_protocol() -> None:
    openai_mock = _make_openai_mock()
    with (
        patch.dict(sys.modules, {"openai": openai_mock}),
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
    ):
        from archex.providers.openai import OpenAIProvider

        provider = OpenAIProvider()
        assert isinstance(provider, LLMProvider)


def test_openrouter_provider_satisfies_protocol() -> None:
    openai_mock = _make_openai_mock()
    with (
        patch.dict(sys.modules, {"openai": openai_mock}),
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
    ):
        from archex.providers.openrouter import OpenRouterProvider

        provider = OpenRouterProvider()
        assert isinstance(provider, LLMProvider)


# ---------------------------------------------------------------------------
# get_provider() factory tests
# ---------------------------------------------------------------------------


def test_get_provider_anthropic() -> None:
    anthropic_mock = _make_anthropic_mock()
    with (
        patch.dict(sys.modules, {"anthropic": anthropic_mock}),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
    ):
        from archex.providers.anthropic import AnthropicProvider

        provider = get_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)
        assert provider.name == "anthropic"


def test_get_provider_openai() -> None:
    openai_mock = _make_openai_mock()
    with (
        patch.dict(sys.modules, {"openai": openai_mock}),
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
    ):
        from archex.providers.openai import OpenAIProvider

        provider = get_provider("openai")
        assert isinstance(provider, OpenAIProvider)
        assert provider.name == "openai"


def test_get_provider_openrouter() -> None:
    openai_mock = _make_openai_mock()
    with (
        patch.dict(sys.modules, {"openai": openai_mock}),
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
    ):
        from archex.providers.openrouter import OpenRouterProvider

        provider = get_provider("openrouter")
        assert isinstance(provider, OpenRouterProvider)
        assert provider.name == "openrouter"


def test_get_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent_provider")


# ---------------------------------------------------------------------------
# SDK not installed error handling
# ---------------------------------------------------------------------------


def test_anthropic_provider_raises_when_sdk_missing() -> None:
    with patch.dict(sys.modules, {"anthropic": None}):  # type: ignore[dict-item]
        import importlib

        import archex.providers.anthropic as mod

        importlib.reload(mod)
        with pytest.raises(ImportError, match="anthropic"):
            mod.AnthropicProvider(api_key="test-key")


def test_openai_provider_raises_when_sdk_missing() -> None:
    with patch.dict(sys.modules, {"openai": None}):  # type: ignore[dict-item]
        import importlib

        import archex.providers.openai as mod

        importlib.reload(mod)
        with pytest.raises(ImportError, match="openai"):
            mod.OpenAIProvider(api_key="test-key")


def test_openrouter_provider_raises_when_sdk_missing() -> None:
    with patch.dict(sys.modules, {"openai": None}):  # type: ignore[dict-item]
        import importlib

        import archex.providers.openrouter as mod

        importlib.reload(mod)
        with pytest.raises(ImportError, match="openai"):
            mod.OpenRouterProvider(api_key="test-key")


# ---------------------------------------------------------------------------
# Provider name property tests
# ---------------------------------------------------------------------------


def test_anthropic_name() -> None:
    anthropic_mock = _make_anthropic_mock()
    with (
        patch.dict(sys.modules, {"anthropic": anthropic_mock}),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}),
    ):
        from archex.providers.anthropic import AnthropicProvider

        assert AnthropicProvider().name == "anthropic"


def test_openai_name() -> None:
    openai_mock = _make_openai_mock()
    with (
        patch.dict(sys.modules, {"openai": openai_mock}),
        patch.dict("os.environ", {"OPENAI_API_KEY": "k"}),
    ):
        from archex.providers.openai import OpenAIProvider

        assert OpenAIProvider().name == "openai"


def test_openrouter_name() -> None:
    openai_mock = _make_openai_mock()
    with (
        patch.dict(sys.modules, {"openai": openai_mock}),
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}),
    ):
        from archex.providers.openrouter import OpenRouterProvider

        assert OpenRouterProvider().name == "openrouter"


# ---------------------------------------------------------------------------
# Missing API key raises ProviderError
# ---------------------------------------------------------------------------


def test_anthropic_provider_raises_without_api_key() -> None:
    anthropic_mock = _make_anthropic_mock()
    with (
        patch.dict(sys.modules, {"anthropic": anthropic_mock}),
        patch.dict("os.environ", {}, clear=True),
    ):
        import importlib

        import archex.providers.anthropic as mod

        importlib.reload(mod)
        with pytest.raises(ProviderError, match="API key"):
            mod.AnthropicProvider()


def test_openai_provider_raises_without_api_key() -> None:
    openai_mock = _make_openai_mock()
    with patch.dict(sys.modules, {"openai": openai_mock}), patch.dict("os.environ", {}, clear=True):
        import importlib

        import archex.providers.openai as mod

        importlib.reload(mod)
        with pytest.raises(ProviderError, match="API key"):
            mod.OpenAIProvider()
