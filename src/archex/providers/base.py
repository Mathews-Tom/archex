"""Abstract base class for LLM providers: defines the complete/chat interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str:
        """Provider name (e.g., 'anthropic', 'openai')."""
        ...

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        """Send a completion request and return the text response."""
        ...

    def complete_structured(
        self,
        prompt: str,
        schema: dict[str, object],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, object]:
        """Send a structured output request. Returns a dict matching the schema."""
        ...


def get_provider(name: str, config: dict[str, object] | None = None) -> LLMProvider:
    """Get an LLM provider by name. Raises ValueError for unknown providers."""
    if name == "anthropic":
        from archex.providers.anthropic import AnthropicProvider

        return AnthropicProvider(**(config or {}))  # type: ignore[arg-type]
    elif name == "openai":
        from archex.providers.openai import OpenAIProvider

        return OpenAIProvider(**(config or {}))  # type: ignore[arg-type]
    elif name == "openrouter":
        from archex.providers.openrouter import OpenRouterProvider

        return OpenRouterProvider(**(config or {}))  # type: ignore[arg-type]
    raise ValueError(f"Unknown provider: {name}")
