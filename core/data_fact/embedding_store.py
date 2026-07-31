"""Persistent embedding cache and similarity helpers for fact memory."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from schemas.data_fact import MemoryCard


@dataclass
class CachedCardEmbedding:
    card: MemoryCard
    text: str
    vector: list[float]
    cache_hit: bool


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def memory_card_embedding_text(card: MemoryCard, fact_request: dict | None = None, guidance: str | None = None) -> str:
    """Build stable semantic text for a memory card embedding."""
    parts = [
        f"kind: {card.kind}",
        f"title: {card.title}",
        f"description: {card.description}",
        "tags: " + ", ".join(card.tags or []),
    ]
    if fact_request:
        parts.append("fact_request: " + json.dumps(fact_request, ensure_ascii=False, sort_keys=True, default=str))
    if guidance:
        parts.append("guidance: " + str(guidance)[:500])
    return "\n".join(part for part in parts if part and part.strip())


def embedding_input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class FactMemoryEmbeddingStore:
    """File-backed card embedding cache."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def load(self, *, database_id: str | None, model: str, card: MemoryCard, text: str) -> CachedCardEmbedding | None:
        path = self._path(database_id=database_id, model=model, card_id=card.id, text=text)
        payload = self._read_json(path)
        if not payload:
            return None
        vector = payload.get("embedding_vector")
        if not isinstance(vector, list):
            return None
        try:
            return CachedCardEmbedding(card=card, text=text, vector=[float(value) for value in vector], cache_hit=True)
        except (TypeError, ValueError):
            return None

    def save(
        self,
        *,
        database_id: str | None,
        model: str,
        card: MemoryCard,
        text: str,
        vector: list[float],
        memory_updated_at: str | None,
    ) -> CachedCardEmbedding:
        path = self._path(database_id=database_id, model=model, card_id=card.id, text=text)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "card_id": card.id,
            "database_id": database_id or "global",
            "embedding_model": model,
            "embedding_input_hash": embedding_input_hash(text),
            "embedding_vector": vector,
            "card_snapshot": card.model_dump(mode="json"),
            "memory_updated_at": memory_updated_at,
            "updated_at": utc_now_iso(),
        }
        if not path.exists():
            payload["created_at"] = payload["updated_at"]
        else:
            previous = self._read_json(path) or {}
            payload["created_at"] = previous.get("created_at") or payload["updated_at"]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return CachedCardEmbedding(card=card, text=text, vector=vector, cache_hit=False)

    def _path(self, *, database_id: str | None, model: str, card_id: str, text: str) -> Path:
        scope = _safe_path_part(database_id or "global")
        model_part = _safe_path_part(model)
        card_part = _safe_path_part(card_id)
        digest = embedding_input_hash(text)[:16]
        return self.root / scope / model_part / f"{card_part}_{digest}.json"

    def _read_json(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None


def top_similar_cards(
    *,
    query_vector: list[float],
    card_embeddings: Iterable[CachedCardEmbedding],
    top_k: int,
    score_threshold: float,
) -> list[tuple[CachedCardEmbedding, float]]:
    scored = [
        (item, cosine_similarity(query_vector, item.vector))
        for item in card_embeddings
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [
        (item, score)
        for item, score in scored[: max(int(top_k), 0)]
        if score >= score_threshold
    ]


def _safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return safe[:120] or "default"
