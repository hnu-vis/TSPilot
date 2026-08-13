"""Fact-memory retrieval for tool input planning."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from core.data_fact.embedding import EmbeddingProvider
from core.data_fact.embedding_store import (
    FactMemoryEmbeddingStore,
    memory_card_embedding_text,
    top_similar_cards,
)
from core.data_fact.memory import memory_cards_view, memory_detail, memory_details
from core.data_fact.contracts import fact_request_contract_error
from schemas.data_fact import DataFactRequest, MemoryCard, MemoryDetail, normalize_fact_key
from schemas.state import RequestStateModel


class MemoryHit(BaseModel):
    card_id: str
    reason: str | None = None
    confidence: float | None = None


class MemoryRetrievalResult(BaseModel):
    hits: list[MemoryHit] = Field(default_factory=list)
    fact_requests: list[DataFactRequest] = Field(default_factory=list)
    fact_request_sources: dict[str, list[str]] = Field(default_factory=dict)
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
        cards_payload = memory_cards_view(database_id, max_cards=None)
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
        details_by_id: dict[str, MemoryDetail] | None = None,
    ):
        cached = []
        missing: list[tuple[MemoryCard, str]] = []
        for card in cards:
            detail = details_by_id.get(card.id) if details_by_id is not None else memory_detail(database_id, card.id)
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
class HybridFactMemoryRetriever:
    """Tool-scoped full-recipe retrieval with embedding recall and LLM reranking."""

    llm: Any
    embedding_provider: EmbeddingProvider
    embedding_store: FactMemoryEmbeddingStore
    top_k: int = 6
    score_threshold: float = 0.25
    max_selected: int = 5

    async def retrieve(
        self,
        *,
        request_state: RequestStateModel,
        tool_name: str,
        action_input: dict,
    ) -> MemoryRetrievalResult:
        if tool_name not in {"sql_query", "code_interpreter"}:
            return MemoryRetrievalResult(
                diagnostics={"memory_enabled": False, "reason": "Tool is not memory-eligible."}
            )

        started = time.perf_counter()
        database_id = _database_id(request_state)
        explicit_requests = _validated_fact_requests(action_input.get("fact_requests"))
        cards_payload = memory_cards_view(database_id, max_cards=None)
        all_cards = [
            MemoryCard.model_validate(item)
            for item in cards_payload.get("cards", [])
            if isinstance(item, dict)
        ]
        eligible_details = _eligible_recipe_details(database_id, all_cards, tool_name)
        eligible_cards = [detail.card for detail in eligible_details]
        base_diagnostics = {
            "memory_enabled": True,
            "retriever_type": "hybrid",
            "embedding_model": self.embedding_provider.model,
            "all_card_count": len(all_cards),
            "eligible_recipe_count": len(eligible_details),
        }

        if not eligible_details:
            planned = await self._planner()._plan_tool_fact_requests(
                request_state=request_state,
                tool_name=tool_name,
                action_input=action_input,
            )
            return MemoryRetrievalResult(
                fact_requests=planned,
                diagnostics={
                    **base_diagnostics,
                    "planner_used": True,
                    "planner_reason": "no_eligible_recipe",
                    "planned_fact_request_count": len(planned),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                },
            )

        embedding_retriever = EmbeddingFactMemoryRetriever(
            embedding_provider=self.embedding_provider,
            embedding_store=self.embedding_store,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            max_cards=len(eligible_cards),
        )
        try:
            card_embeddings, cache_hits, cache_misses = await embedding_retriever._card_embeddings(
                database_id=database_id,
                cards=eligible_cards,
                memory_updated_at=cards_payload.get("updated_at"),
                details_by_id={detail.id: detail for detail in eligible_details},
            )
            query_texts = _tool_retrieval_query_texts(request_state, tool_name, action_input)
            query_vectors = await self.embedding_provider.embed_texts(query_texts)
            if not query_vectors:
                raise ValueError("Embedding provider returned no query vector.")
            recalled_by_id = {}
            for query_vector in query_vectors:
                for item, score in top_similar_cards(
                    query_vector=query_vector,
                    card_embeddings=card_embeddings,
                    top_k=self.top_k,
                    score_threshold=self.score_threshold,
                ):
                    current = recalled_by_id.get(item.card.id)
                    if current is None or score > current[1]:
                        recalled_by_id[item.card.id] = (item, score)
            recalled = sorted(recalled_by_id.values(), key=lambda pair: pair[1], reverse=True)
        except Exception as exc:
            return await self._planner_result(
                request_state=request_state,
                tool_name=tool_name,
                action_input=action_input,
                diagnostics={
                    **base_diagnostics,
                    "error_type": "memory_embedding_failed",
                    "error": str(exc),
                    "failure_stage": "embedding_recall",
                },
                started=started,
            )

        recalled_hits = [
            MemoryHit(
                card_id=item.card.id,
                reason="embedding_similarity",
                confidence=round(float(score), 6),
            )
            for item, score in recalled
        ]
        details_by_id = {detail.id: detail for detail in eligible_details}
        anchor_details = _contract_anchor_details(request_state, eligible_details)
        recalled_ids = {hit.card_id for hit in recalled_hits}
        for detail in anchor_details:
            if detail.id in recalled_ids:
                continue
            recalled_hits.append(MemoryHit(card_id=detail.id, reason="task_contract_anchor"))
            recalled_ids.add(detail.id)
        recalled_details = [details_by_id[hit.card_id] for hit in recalled_hits if hit.card_id in details_by_id]
        recalled_details = _recipe_dependency_closure(recalled_details, eligible_details, request_state)
        for detail in recalled_details:
            if detail.id in recalled_ids:
                continue
            recalled_hits.append(MemoryHit(card_id=detail.id, reason="fact_dependency"))
            recalled_ids.add(detail.id)
        recall_diagnostics = {
            **base_diagnostics,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "embedding_query_count": len(query_vectors),
            "embedding_recalled_card_ids": [hit.card_id for hit in recalled_hits],
            "embedding_scores": {
                hit.card_id: hit.confidence for hit in recalled_hits if hit.confidence is not None
            },
            "contract_anchor_card_ids": [detail.id for detail in anchor_details],
            "dependency_expanded_card_ids": [
                detail.id for detail in recalled_details if detail.id not in {item.id for item in anchor_details}
                and detail.id not in {item.card.id for item, _score in recalled}
            ],
        }
        if not recalled_details:
            return await self._planner_result(
                request_state=request_state,
                tool_name=tool_name,
                action_input=action_input,
                diagnostics={
                    **recall_diagnostics,
                    "failure_stage": "embedding_recall",
                    "planner_reason": "no_candidate_above_threshold",
                },
                started=started,
            )

        try:
            selected_hits, needs_planning = await self._rerank(
                request_state=request_state,
                tool_name=tool_name,
                action_input=action_input,
                explicit_requests=explicit_requests,
                recalled_hits=recalled_hits,
                recalled_details=recalled_details,
            )
        except Exception as exc:
            return await self._planner_result(
                request_state=request_state,
                tool_name=tool_name,
                action_input=action_input,
                diagnostics={
                    **recall_diagnostics,
                    "error_type": "memory_rerank_failed",
                    "error": str(exc),
                    "failure_stage": "llm_rerank",
                },
                started=started,
            )

        selected_details = [details_by_id[hit.card_id] for hit in selected_hits if hit.card_id in details_by_id]
        memory_requests = _fact_requests_from_details(selected_details, tool_name=tool_name)
        sources: dict[str, list[str]] = {}
        for detail in selected_details:
            if detail.fact_request is None:
                continue
            sources.setdefault(detail.fact_request.fact_key, []).append(detail.id)
        planned_requests: list[DataFactRequest] = []
        if needs_planning:
            planned_requests = await self._planner()._plan_tool_fact_requests(
                request_state=request_state,
                tool_name=tool_name,
                action_input={
                    **action_input,
                    "retrieved_fact_requests": [
                        item.model_dump(mode="json", exclude_none=True) for item in memory_requests
                    ],
                },
            )
        reconciliation_inputs = [*explicit_requests, *planned_requests]
        if memory_requests and reconciliation_inputs:
            memory_requests = await self._planner()._reconcile_memory_requests(
                request_state=request_state,
                tool_name=tool_name,
                explicit_requests=reconciliation_inputs,
                memory_requests=memory_requests,
            )
            retained_keys = {request.fact_key for request in memory_requests}
            selected_hits = [
                hit
                for hit in selected_hits
                if (details_by_id.get(hit.card_id) is not None)
                and (details_by_id[hit.card_id].fact_request is not None)
                and (details_by_id[hit.card_id].fact_request.fact_key in retained_keys)
            ]
            sources = {key: value for key, value in sources.items() if key in retained_keys}
        combined = _dedupe_request_models([*memory_requests, *planned_requests])
        return MemoryRetrievalResult(
            hits=selected_hits,
            fact_requests=combined,
            fact_request_sources=sources,
            diagnostics={
                **recall_diagnostics,
                "selected_card_ids": [hit.card_id for hit in selected_hits],
                "selected_reasons": {hit.card_id: hit.reason for hit in selected_hits},
                "selected_confidences": {
                    hit.card_id: hit.confidence
                    for hit in selected_hits
                    if hit.confidence is not None
                },
                "memory_fact_request_count": len(memory_requests),
                "planned_fact_request_count": len(planned_requests),
                "planner_used": needs_planning,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    def _planner(self) -> "LlmFactMemoryRetriever":
        return LlmFactMemoryRetriever(llm=self.llm, max_selected=self.max_selected)

    async def _planner_result(
        self,
        *,
        request_state: RequestStateModel,
        tool_name: str,
        action_input: dict,
        diagnostics: dict,
        started: float,
    ) -> MemoryRetrievalResult:
        planned = await self._planner()._plan_tool_fact_requests(
            request_state=request_state,
            tool_name=tool_name,
            action_input=action_input,
        )
        return MemoryRetrievalResult(
            fact_requests=planned,
            diagnostics={
                **diagnostics,
                "planner_used": True,
                "planned_fact_request_count": len(planned),
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    async def _rerank(
        self,
        *,
        request_state: RequestStateModel,
        tool_name: str,
        action_input: dict,
        explicit_requests: list[DataFactRequest],
        recalled_hits: list[MemoryHit],
        recalled_details: list[MemoryDetail],
    ) -> tuple[list[MemoryHit], bool]:
        scores = {hit.card_id: hit.confidence for hit in recalled_hits}
        messages = [
            (
                "system",
                (
                    "You rerank retrieved Data Fact recipes for one tool call. Explicit fact contracts are authoritative. "
                    "Select a recipe when its full contract is required by the user or task contract, or when it is a necessary "
                    "dependency of another selected recipe and is not already verified. It must fit the current "
                    "tool, subject, time range, dimensions, granularity, and calculation semantics, and is not semantically "
                    "equivalent to an explicit or already verified fact. Different names or fact keys do not make facts "
                    "distinct. Treat latest/last/current boundary synonyms and translated or reformatted names as equivalent. "
                    "Interpret SQL structure instead of display names: time_boundary returns only a timestamp, point_value "
                    "returns only a scalar measure, and extreme returns the selected extreme value. Never select a memory "
                    "contract for an output already covered by an equivalent explicit contract. "
                    "Set needs_planning=true only when required facts remain uncovered after explicit and selected "
                    "contracts. Return exactly this JSON object: {\"selected\":[{\"card_id\":string,\"reason\":string,\"confidence\":number}],"
                    "\"needs_planning\":boolean} and no markdown."
                ),
            ),
            (
                "user",
                json.dumps(
                    {
                        "task_context": _retrieval_task_context(request_state),
                        "tool_name": tool_name,
                        "action_input": _bounded_action_input(action_input),
                        "explicit_fact_contracts": [
                            request.model_dump(mode="json", exclude_none=True) for request in explicit_requests
                        ],
                        "verified_fact_dag": _verified_fact_dag(request_state),
                        "candidates": [
                            {
                                "card_id": detail.id,
                                "embedding_score": scores.get(detail.id),
                                "title": detail.card.title,
                                "description": detail.card.description,
                                "tags": detail.card.tags,
                                "preferred_tool": detail.preferred_tool,
                                "fact_request": detail.fact_request.model_dump(mode="json", exclude_none=True)
                                if detail.fact_request is not None else None,
                                "guidance": detail.guidance,
                            }
                            for detail in recalled_details
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
        valid_ids = {detail.id for detail in recalled_details}
        result: list[MemoryHit] = []
        seen: set[str] = set()
        for item in selected if isinstance(selected, list) else []:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("card_id") or "").strip()
            if card_id not in valid_ids or card_id in seen or len(result) >= self.max_selected:
                continue
            result.append(MemoryHit(
                card_id=card_id,
                reason=str(item.get("reason") or "").strip() or None,
                confidence=_float_or_none(item.get("confidence")),
            ))
            seen.add(card_id)
        return result, bool(payload.get("needs_planning"))


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
            explicit_requests = _validated_fact_requests(action_input.get("fact_requests"))
            planned_requests = []
            if not explicit_requests:
                planned_requests = await self._plan_tool_fact_requests(
                    request_state=request_state,
                    tool_name=tool_name,
                    action_input=action_input,
                )
            return MemoryRetrievalResult(
                fact_requests=planned_requests,
                diagnostics={
                    "memory_enabled": True,
                    "card_count": 0,
                    "fact_request_count": len(planned_requests),
                },
            )

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
        fact_requests = _fact_requests_from_details(details, tool_name=tool_name)
        explicit_requests = _validated_fact_requests(action_input.get("fact_requests"))
        if explicit_requests and fact_requests:
            fact_requests = await self._reconcile_memory_requests(
                request_state=request_state,
                tool_name=tool_name,
                explicit_requests=explicit_requests,
                memory_requests=fact_requests,
            )
        elif not explicit_requests and not fact_requests:
            fact_requests = await self._plan_tool_fact_requests(
                request_state=request_state,
                tool_name=tool_name,
                action_input=action_input,
            )
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

    async def _reconcile_memory_requests(
        self,
        *,
        request_state: RequestStateModel,
        tool_name: str,
        explicit_requests: list[DataFactRequest],
        memory_requests: list[DataFactRequest],
    ) -> list[DataFactRequest]:
        """Use semantic reconciliation instead of name-based duplicate rules."""

        messages = [
            (
                "system",
                (
                    "You reconcile explicit and retrieved Data Fact contracts for one tool call. "
                    "Explicit contracts are authoritative and already selected. Select a memory contract only when it fills a "
                    "required semantic output that no explicit contract covers. Keep at most one contract per semantic output. "
                    "Treat latest/last/current boundary synonyms and translated or reformatted names as equivalent when their "
                    "subject, fact type, requirements, dimensions, time scope, and derivation match. Different names or fact_key "
                    "values never make facts distinct. Interpret SQL contract structure rather than an over-broad display name: "
                    "time_boundary returns only a timestamp, point_value returns only a scalar measure, and extreme returns the "
                    "selected extreme value. Therefore an explicit time_boundary already covers an equivalent memory time_boundary, "
                    "even if its name also mentions a value. Do not select a memory duplicate as corroboration, and do not repair or "
                    "replace malformed explicit contracts here. "
                    "Return exactly this JSON object: {\"selected_memory_fact_keys\": [string]} and no markdown."
                ),
            ),
            (
                "user",
                json.dumps(
                    {
                        "task_context": _retrieval_task_context(request_state),
                        "tool_name": tool_name,
                        "explicit_fact_contracts": [
                            request.model_dump(mode="json", exclude_none=True) for request in explicit_requests
                        ],
                        "memory_fact_contracts": [
                            request.model_dump(mode="json", exclude_none=True) for request in memory_requests
                        ],
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            ),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            payload = _parse_json_response(getattr(response, "content", response))
        except Exception:
            return []
        selected = payload.get("selected_memory_fact_keys") if isinstance(payload, dict) else []
        selected_keys = {str(item).strip() for item in selected if str(item).strip()} if isinstance(selected, list) else set()
        return [request for request in memory_requests if request.fact_key in selected_keys]

    async def _plan_tool_fact_requests(
        self,
        *,
        request_state: RequestStateModel,
        tool_name: str,
        action_input: dict,
    ) -> list[DataFactRequest]:
        current_facts = [
            {
                "fact_key": fact.fact_key,
                "name": fact.name,
                "fact_type": fact.fact_type,
                "status": fact.status,
                "derived_from": fact.derived_from,
            }
            for fact in request_state.fact_set.facts[-12:]
        ]
        messages = [
            (
                "system",
                (
                    "You create semantic Data Fact contracts for one tool call only when its intended outputs are numerical, "
                    "statistical, analytical, or interpretive facts needed by the user. Return exactly this JSON object: "
                    "{\"fact_requests\": [object]} and no markdown. Return an empty list when the tool does not produce a useful Fact. "
                    "Each request requires fact_key, name, and fact_type. Set semantic_class for the claim family, derivation for the operation, and result_shape for the payload shape. Use derived_from only when the calculation consumes "
                    "verified parent Facts; a code_interpreter calculation directly grounded in database rows does not need "
                    "synthetic parent Fact keys. Parents are inputs, not duplicate output requests. For sql_query use only point_value or "
                    "time_boundary with requirements.time_position=start|end, extreme with requirements.operator=min|max, or count. "
                    "When query evidence can contain multiple series or groups, include every row-binding dimension in dimensions "
                    "or requirements using the physical result-column name and expected value. "
                    "Use time_boundary for timestamps, point_value only for scalar measure values, and count only when the user asks "
                    "for a row/record count. Return no Fact request for raw tables or detail lists. When the user asks for a "
                    "semantic subset such as recent N observations, Top K, ranked items, pairwise comparisons, or a named "
                    "set that will be cited or visualized, create a collection-valued Fact with result_shape and "
                    "expected_item_count when the requested cardinality is explicit. The raw query Evidence Artifact "
                    "remains the source; the Fact adds semantic selection and validation. For temporal subsets, "
                    "selection must encode the requested grain and distinct key (for example granularity=day, "
                    "distinct_by=date), together with order_by and direction. "
                    "Use code_interpreter for change, ratio, trend, distribution, association, and multi-Fact composition. "
                    "Do not invent parent keys and do not return already verified facts unless this call must replace them."
                ),
            ),
            (
                "user",
                json.dumps(
                    {
                        "task_context": _retrieval_task_context(request_state),
                        "tool_name": tool_name,
                        "action_input": _bounded_action_input(action_input),
                        "current_fact_dag": current_facts,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            ),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            payload = _parse_json_response(getattr(response, "content", response))
        except Exception:
            return []
        requests = _validated_fact_requests(payload.get("fact_requests") if isinstance(payload, dict) else None)
        requests = [request for request in requests[:6] if not fact_request_contract_error(request, tool_name)]
        if tool_name != "code_interpreter":
            return requests

        verified_keys = {
            fact.fact_key
            for fact in request_state.fact_set.facts
            if fact.status == "verified" and fact.fact_key
        }
        has_rows = _latest_evidence_row_count(request_state) > 0
        return [
            request
            for request in requests
            if (
                request.derived_from
                and all(parent_key in verified_keys for parent_key in request.derived_from)
            )
            or (not request.derived_from and has_rows)
        ]

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
                    "Do not select a card when action_input.fact_requests already requests the same semantic fact, even if its name or key differs.\n"
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


def _fact_requests_from_details(details: list[MemoryDetail], tool_name: str | None = None) -> list[DataFactRequest]:
    result: list[DataFactRequest] = []
    seen: set[str] = set()
    for detail in details:
        request = detail.fact_request
        if request is None:
            continue
        if tool_name and detail.preferred_tool and detail.preferred_tool != tool_name:
            continue
        if tool_name and fact_request_contract_error(request, tool_name):
            continue
        key = json.dumps(request.model_dump(mode="json", exclude_none=True), ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        result.append(request)
        seen.add(key)
    return result


def _eligible_recipe_details(
    database_id: str | None,
    cards: list[MemoryCard],
    tool_name: str,
) -> list[MemoryDetail]:
    recipe_cards = [card for card in cards if card.kind == "fact_recipe"]
    details = memory_details(database_id, [card.id for card in recipe_cards])
    return [
        detail
        for detail in details
        if detail.fact_request is not None
        and (not detail.preferred_tool or detail.preferred_tool == tool_name)
        and not fact_request_contract_error(detail.fact_request, tool_name)
    ]


def _contract_anchor_details(
    request_state: RequestStateModel,
    details: list[MemoryDetail],
) -> list[MemoryDetail]:
    contract = request_state.task_contract
    if contract is None:
        return []
    output_keys: set[str] = set()
    for output in contract.required_outputs:
        if not output.required:
            continue
        for value in [output.id, *output.measures, *output.dimensions]:
            if str(value or "").strip():
                output_keys.add(normalize_fact_key(str(value)))
    return [
        detail
        for detail in details
        if detail.fact_request is not None
        and (
            detail.fact_request.fact_key in output_keys
            or normalize_fact_key(detail.fact_request.name) in output_keys
        )
    ]


def _recipe_dependency_closure(
    selected: list[MemoryDetail],
    eligible: list[MemoryDetail],
    request_state: RequestStateModel,
) -> list[MemoryDetail]:
    by_fact_key: dict[str, list[MemoryDetail]] = {}
    for detail in eligible:
        if detail.fact_request is not None:
            by_fact_key.setdefault(detail.fact_request.fact_key, []).append(detail)
    verified = {
        fact.fact_key
        for fact in request_state.fact_set.facts
        if fact.status == "verified" and fact.fact_key
    }
    result = {detail.id: detail for detail in selected}
    pending = list(selected)
    while pending:
        detail = pending.pop()
        request = detail.fact_request
        if request is None:
            continue
        for dependency in request.derived_from:
            dependency_key = normalize_fact_key(dependency)
            if dependency_key in verified:
                continue
            for parent in by_fact_key.get(dependency_key, []):
                if parent.id in result:
                    continue
                result[parent.id] = parent
                pending.append(parent)
    return list(result.values())


def _dedupe_request_models(requests: list[DataFactRequest]) -> list[DataFactRequest]:
    result: list[DataFactRequest] = []
    seen: set[str] = set()
    for request in requests:
        payload = request.model_dump(mode="json", exclude_none=True)
        requirements = dict(payload.get("requirements") or {})
        for key in ("source", "memory_card_ids", "retrieval_reason", "retrieval_confidence"):
            requirements.pop(key, None)
        payload["requirements"] = requirements
        identity = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        if identity in seen:
            continue
        result.append(request)
        seen.add(identity)
    return result


def _retrieval_task_context(request_state: RequestStateModel) -> dict:
    contract = request_state.task_contract
    active_todo = next(
        (
            item
            for item in request_state.todo_list
            if isinstance(item, dict) and item.get("status") in {"pending", "in_progress"}
        ),
        None,
    )
    return {
        "message": request_state.message,
        "focus": request_state.focus,
        "time_range": request_state.time_range,
        "intent_profile": request_state.intent_profile,
        "requested_capabilities": request_state.requested_capabilities,
        "task_contract": contract.model_dump(mode="json", exclude_none=True) if contract is not None else None,
        "active_todo": {
            key: active_todo.get(key)
            for key in ("content", "task_type", "acceptance_criteria")
            if active_todo.get(key) not in (None, "", [], {})
        } if active_todo else None,
    }


def _verified_fact_dag(request_state: RequestStateModel) -> list[dict]:
    return [
        {
            "fact_key": fact.fact_key,
            "name": fact.name,
            "fact_type": fact.fact_type,
            "subject": fact.subject,
            "dimensions": fact.dimensions,
            "time_range": fact.time_range,
            "derived_from": fact.derived_from,
        }
        for fact in request_state.fact_set.facts[-20:]
        if fact.status == "verified"
    ]


def _tool_retrieval_query_texts(
    request_state: RequestStateModel,
    tool_name: str,
    action_input: dict,
) -> list[str]:
    payload = {
        "task": _retrieval_task_context(request_state),
        "tool_name": tool_name,
        "action_input": _bounded_action_input(action_input),
        "explicit_fact_contracts": [
            request.model_dump(mode="json", exclude_none=True)
            for request in _validated_fact_requests(action_input.get("fact_requests"))
        ],
        "verified_fact_dag": _verified_fact_dag(request_state),
    }
    texts = [json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)]
    contract = request_state.task_contract
    if contract is not None:
        for output in contract.required_outputs:
            if not output.required:
                continue
            texts.append(json.dumps(
                {
                    "task_message": request_state.message,
                    "tool_name": tool_name,
                    "required_output": output.model_dump(mode="json", exclude_none=True),
                    "explicit_fact_contracts": payload["explicit_fact_contracts"],
                    "verified_fact_dag": payload["verified_fact_dag"],
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ))
    return list(dict.fromkeys(texts))


def _validated_fact_requests(value: Any) -> list[DataFactRequest]:
    if not isinstance(value, list):
        return []
    result: list[DataFactRequest] = []
    for item in value:
        try:
            result.append(item if isinstance(item, DataFactRequest) else DataFactRequest.model_validate(item))
        except Exception:
            continue
    return result


def _latest_evidence_row_count(request_state: RequestStateModel) -> int:
    evidence = getattr(request_state, "latest_database_evidence", None)
    data = getattr(evidence, "data", None) if evidence is not None else None
    if not isinstance(data, dict):
        return 0
    rows = data.get("rows")
    if isinstance(rows, list):
        return len(rows)
    points = data.get("points")
    return len(points) if isinstance(points, list) else 0


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
