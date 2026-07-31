"""Embedding provider abstractions for fact-memory retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx


class EmbeddingProvider(Protocol):
    """Generate dense vectors for retrieval text."""

    model: str

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


@dataclass
class OpenAICompatibleEmbeddingProvider:
    """OpenAI-compatible /v1/embeddings client."""

    api_key: str
    api_base: str
    model: str
    timeout_seconds: float = 30.0

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [str(text or "") for text in texts]
        if not clean_texts:
            return []
        url = self.api_base.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": clean_texts}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        body = response.json()
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise ValueError("Embedding response missing data list.")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0)
        vectors: list[list[float]] = []
        for item in ordered:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise ValueError("Embedding response contains an invalid embedding item.")
            vectors.append([float(value) for value in item["embedding"]])
        if len(vectors) != len(clean_texts):
            raise ValueError(f"Embedding response returned {len(vectors)} vectors for {len(clean_texts)} inputs.")
        return vectors
