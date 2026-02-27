"""Anthropic provider: Claude API client for architecture enrichment tasks."""

from __future__ import annotations

import os
from typing import Any

from archex.exceptions import ProviderError


class AnthropicProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        try:
            import anthropic as _anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: uv add anthropic"
            ) from exc

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY or pass api_key."
            )
        self._client: Any = _anthropic.Anthropic(api_key=resolved_key)
        self._model = model

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            response = self._client.messages.create(**kwargs)
            return str(response.content[0].text)
        except Exception as exc:
            raise ProviderError(f"Anthropic API error: {exc}") from exc

    def complete_structured(
        self,
        prompt: str,
        schema: dict[str, object],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, object]:
        tool_name = "structured_output"
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": max_tokens,
                "tools": [
                    {
                        "name": tool_name,
                        "description": "Return structured output matching the schema.",
                        "input_schema": schema,
                    }
                ],
                "tool_choice": {"type": "tool", "name": tool_name},
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            response = self._client.messages.create(**kwargs)
            for block in response.content:
                if block.type == "tool_use" and block.name == tool_name:
                    result: dict[str, object] = block.input
                    return result
            raise ProviderError("Anthropic did not return a tool_use block.")
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Anthropic structured API error: {exc}") from exc
