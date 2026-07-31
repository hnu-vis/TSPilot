"""Fact-memory retrieval for tool input planning."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from core.data_fact.embedding import EmbeddingProvider
from core.data_fact.embedding_store import (
    FactMemoryEmbeddingStore,
    memory_card_embedding_text,
    top_similar_cards,
)
from core.data_fact.memory import memory_cards_view, memory_detail, memory_details
from schemas.data_fact import DataFactRequest, MemoryCard, MemoryDetail
from schemas.state import RequestStateModel


class MemoryHit(BaseModel):
    card_id: str
    reason: str | None = None
    confidence: float | None = None


class MemoryRetrievalResult(BaseModel):
    hits: list[MemoryHit] = Field(default_factory=list)
    fact_requests: list[DataFactRequest] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


@dataclass
class EmbeddingFactMemoryRetriever:
    """Retrieve memory cards with embeddings, then load details by id."""

    embedding_provider: EmbeddingProvider
    embedding_store: FactMemoryEmbeddingStore
    top_k: int = 6
    score_threshold: float = 0.25
    max_cards: int = 64

    async def retrieve_once(self, *, request_state: RequestStateModel) -> MemoryRetrievalResult:
        database_id = _database_id(request_state)
        cards_payload = memory_cards_view(database_id)
        cards = [
            MemoryCard.model_validate(item)
            for item in cards_payload.get("cards", [])[: self.max_cards]
            if isinstance(item, dict)
        ]
        if not cards:
            return MemoryRetrievalResult(diagnostics={"memory_enabled": True, "retriever_type": "embedding", "card_count": 0})

        started = time.perf_counter()
        try:
            card_embeddings, cache_hits, cache_misses = await self._card_embeddings(
                database_id=database_id,
                cards=cards,
                memory_updated_at=cards_payload.get("updated_at"),
            )
            query_text = _request_embedding_text(request_state)
            query_vector = (await self.embedding_provider.embed_texts([query_text]))[0]
            selected = top_similar_cards(
                query_vector=query_vector,
                card_embeddings=card_embeddings,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
            )
            hits = [
                MemoryHit(
                    card_id=item.card.id,
                    reason="embedding_similarity",
                    confidence=round(float(score), 6),
                )
                for item, score in selected
            ]
            details = memory_details(database_id, [hit.card_id for hit in hits])
            fact_requests = _fact_requests_from_details(details)
            return MemoryRetrievalResult(
                hits=hits,
                fact_requests=fact_requests,
                diagnostics={
                    "memory_enabled": True,
                    "retriever_type": "embedding",
                    "embedding_model": self.embedding_provider.model,
                    "card_count": len(cards),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "selected_card_ids": [hit.card_id for hit in hits],
                    "scores": {hit.card_id: hit.confidence for hit in hits},
                    "fact_request_count": len(fact_requests),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            )
        except Exception as exc:
            return MemoryRetrievalResult(
                diagnostics={
                    "memory_enabled": True,
                    "retriever_type": "embedding",
                    "card_count": len(cards),
                    "error_type": "memory_retrieval_failed",
                    "error": str(exc),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                }
            )

    async def _card_embeddings(
        self,
        *,
        database_id: str | None,
        cards: list[MemoryCard],
        memory_updated_at: str | None,
    ):
        cached = []
        missing: list[tuple[MemoryCard, str]] = []
        for card in cards:
            detail = memory_detail(database_id, card.id)
            fact_request = (
                detail.fact_request.model_dump(mode="json", exclude_none=True)
                if detail is not None and detail.fact_request is not None
                else None
            )
            text = memory_card_embedding_text(
                card,
                fact_request=fact_request,
                guidance=detail.guidance if detail is not None else None,
            )
            hit = self.embedding_store.load(
                database_id=database_id,
                model=self.embedding_provider.model,
                card=card,
                text=text,
            )
            if hit is not None:
                cached.append(hit)
            else:
                missing.append((card, text))

        if missing:
            vectors = await self.embedding_provider.embed_texts([text for _card, text in missing])
            for (card, text), vector in zip(missing, vectors):
                cached.append(
                    self.embedding_store.save(
                        database_id=database_id,
                        model=self.embedding_provider.model,
                        card=card,
                        text=text,
                        vector=vector,
                        memory_updated_at=memory_updated_at,
                    )
                )
        return cached, len(cached) - len(missing), len(missing)


@dataclass
class LlmFactMemoryRetriever:
    """Select small memory cards with an LLM, then load details by id."""

    llm: Any | None = None
    max_cards: int = 24
    max_selected: int = 5

    async def retrieve(
        self,
        *,
        request_state: RequestStateModel,
        tool_name: str,
        action_input: dict,
    ) -> MemoryRetrievalResult:
        if self.llm is None:
            return MemoryRetrievalResult(
                diagnostics={"memory_enabled": False, "reason": "No memory retriever LLM is configured."}
            )
        if tool_name not in {"sql_query", "code_interpreter", "forecast", "anomaly"}:
            return MemoryRetrievalResult(diagnostics={"memory_enabled": False, "reason": "Tool is not memory-eligible."})
        database_id = _database_id(request_state)
        cards_payload = memory_cards_view(database_id)
        cards = [
            MemoryCard.model_validate(item)
            for item in cards_payload.get("cards", [])[: self.max_cards]
            if isinstance(item, dict)
        ]
        if not cards:
            return MemoryRetrievalResult(diagnostics={"memory_enabled": True, "card_count": 0})

        started = time.perf_counter()
        try:
            selected = await self._select_cards(
                cards=cards,
                request_state=request_state,
                tool_name=tool_name,
                action_input=action_input,
            )
        except Exception as exc:
            return MemoryRetrievalResult(
                diagnostics={
                    "memory_enabled": True,
                    "card_count": len(cards),
                    "error_type": "memory_retrieval_failed",
                    "error": str(exc),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                }
            )

        selected_ids = [hit.card_id for hit in selected[: self.max_selected] if hit.card_id]
        details = memory_details(database_id, selected_ids)
        fact_requests = _fact_requests_from_details(details)
        return MemoryRetrievalResult(
            hits=selected[: self.max_selected],
            fact_requests=fact_requests,
            diagnostics={
                "memory_enabled": True,
                "card_count": len(cards),
                "selected_card_ids": selected_ids,
                "fact_request_count": len(fact_requests),
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    async def _select_cards(
        self,
        *,
        cards: list[MemoryCard],
        request_state: RequestStateModel,
        tool_name: str,
        action_input: dict,
    ) -> list[MemoryHit]:
        messages = [
            (
                "system",
                (
                    "You select reusable fact-memory cards for the current tool call.\n"
                    "Return exactly one JSON object and no markdown.\n"
                    "Select only cards whose details should be loaded to construct fact_requests for the current tool.\n"
                    "Do not select cards for facts that are unrelated to the user request or not useful for this tool call.\n"
                    "Never return concrete historical fact values; memory only guides what current evidence should produce.\n"
                    "JSON schema: {\"selected\": [{\"card_id\": string, \"reason\": string, \"confidence\": number}]}."
                ),
            ),
            (
                "user",
                "Fact Memory Card Selection JSON:\n"
                + json.dumps(
                    {
                        "tool_name": tool_name,
                        "message": request_state.message,
                        "intent_profile": request_state.intent_profile,
                        "action_input": _bounded_action_input(action_input),
                        "cards": [
                            {
                                "id": card.id,
                                "kind": card.kind,
                                "title": card.title,
                                "description": card.description,
                                "tags": card.tags[:8],
                            }
                            for card in cards
                        ],
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            ),
        ]
        response = await self.llm.ainvoke(messages)
        payload = _parse_json_response(getattr(response, "content", response))
        selected = payload.get("selected") if isinstance(payload, dict) else []
        valid_ids = {card.id for card in cards}
        result: list[MemoryHit] = []
        for item in selected if isinstance(selected, list) else []:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("card_id") or "").strip()
            if card_id not in valid_ids:
                continue
            result.append(
                MemoryHit(
                    card_id=card_id,
                    reason=str(item.get("reason") or "").strip() or None,
                    confidence=_float_or_none(item.get("confidence")),
                )
            )
        return result


# Backward-compatible name for tests or experimental wiring.
FactMemoryRetriever = LlmFactMemoryRetriever


def _fact_requests_from_details(details: list[MemoryDetail]) -> list[DataFactRequest]:
    result: list[DataFactRequest] = []
    seen: set[str] = set()
    for detail in details:
        request = detail.fact_request
        if request is None:
            continue
        key = json.dumps(request.model_dump(mode="json", exclude_none=True), ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        result.append(request)
        seen.add(key)
    return result


def _database_id(request_state: RequestStateModel) -> str | None:
    context = request_state.database_context
    return context.database_id if context is not None else None


def _bounded_action_input(value: dict) -> dict:
    payload = dict(value or {})
    for key in ("database_evidence", "history", "code"):
        if key in payload:
            payload[key] = "<omitted>"
    return payload


def _request_embedding_text(request_state: RequestStateModel) -> str:
    payload = {
        "message": request_state.message,
        "response_language": request_state.response_language,
        "selected_database": request_state.selected_database,
        "selected_database_type": request_state.selected_database_type,
        "intent_profile": request_state.intent_profile,
        "requested_capabilities": request_state.requested_capabilities,
        "time_range": request_state.time_range,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _parse_json_response(content: Any) -> dict:
    if isinstance(content, dict):
        return content
    text = str(content or "").strip()
    if not text:
        raise ValueError("Memory retriever returned empty content.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
