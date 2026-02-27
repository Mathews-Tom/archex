"""OpenAI provider: GPT API client for architecture enrichment tasks."""

from __future__ import annotations

import json
import os
from typing import Any

from archex.exceptions import ProviderError


class OpenAIProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1",
    ) -> None:
        try:
            import openai as _openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: uv add openai"
            ) from exc

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ProviderError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key.")
        self._client: Any = _openai.OpenAI(api_key=resolved_key)
        self._model = model

    @property
    def name(self) -> str:
        return "openai"

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
            raise ProviderError(f"OpenAI API error: {exc}") from exc

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
            raise ProviderError(f"OpenAI structured API error: {exc}") from exc
