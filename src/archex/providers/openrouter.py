"""OpenRouter provider: multi-model API client via the OpenRouter gateway."""

from __future__ import annotations

import json
import os
from typing import Any

from archex.exceptions import ProviderError

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "anthropic/claude-sonnet-4-20250514",
    ) -> None:
        try:
            import openai as _openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenRouterProvider. "
                "Install it with: uv add openai"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY or pass api_key."
            )
        self._client: Any = _openai.OpenAI(
            api_key=resolved_key,
            base_url=_OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/AetherForge/archex",
                "X-Title": "archex",
            },
        )
        self._model = model

    @property
    def name(self) -> str:
        return "openrouter"

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
        try:
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return str(response.choices[0].message.content)
        except Exception as exc:
            raise ProviderError(f"OpenRouter API error: {exc}") from exc

    def complete_structured(
        self,
        prompt: str,
        schema: dict[str, object],
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, object]:
        try:
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "structured_output", "schema": schema},
                },
            )
            raw = response.choices[0].message.content
            result: dict[str, object] = json.loads(raw)
            return result
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"OpenRouter structured API error: {exc}") from exc
