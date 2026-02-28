"""API-backed embedding provider: call a remote OpenAI-compatible embedding endpoint."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from archex.exceptions import ArchexIndexError


class APIEmbedder:
    """Embedding provider using an OpenAI-compatible embedding API.

    Works with OpenAI, Voyage AI, and other compatible endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 100,
        dimension: int = 1536,
    ) -> None:
        if not api_key:
            raise ArchexIndexError("APIEmbedder requires a non-empty api_key")
        self._api_key = api_key
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts by calling the remote embedding API."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            payload = json.dumps(
                {
                    "input": batch,
                    "model": self._model_name,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{self._base_url}/embeddings",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body: dict[str, Any] = json.loads(resp.read())
            except urllib.error.URLError as e:
                raise ArchexIndexError(f"Embedding API request failed: {e}") from e

            data: list[dict[str, Any]] | None = body.get("data")  # type: ignore[assignment]
            if not isinstance(data, list):
                raise ArchexIndexError("Unexpected API response format: missing 'data' list")

            # Sort by index to preserve input order
            sorted_data: list[dict[str, Any]] = sorted(data, key=lambda x: int(x["index"]))
            for item in sorted_data:
                embedding: list[float] = item["embedding"]
                all_embeddings.append(embedding)

        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimension
