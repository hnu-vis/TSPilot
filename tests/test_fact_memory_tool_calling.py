from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from core.data_fact.embedding_store import FactMemoryEmbeddingStore, memory_card_embedding_text
from core.data_fact.retriever import EmbeddingFactMemoryRetriever, FactMemoryRetriever, MemoryHit, MemoryRetrievalResult
from runtime.tool_executor import ToolExecutor
from schemas.data_fact import DataFact, DataFactRequest, MemoryCard
from schemas.state import ConversationStateModel, RequestStateModel
from tools.base import BaseTool
from tools.registry import ToolRegistry, ToolSpec


class EchoInput(BaseModel):
    message: str
    constraints: dict = Field(default_factory=dict)
    fact_requests: list[DataFactRequest] = Field(default_factory=list)


class EchoOutput(BaseModel):
    summary: str
    fact_requests: list[dict] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)


class EchoTool(BaseTool):
    async def execute(self, validated_input: EchoInput, **kwargs) -> dict:
        return {
            "summary": "ok",
            "fact_requests": [item.model_dump(mode="json") for item in validated_input.fact_requests],
            "constraints": validated_input.constraints,
        }


class BrokenLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("memory model unavailable")


class Message:
    def __init__(self, content: str):
        self.content = content


class SelectingLLM:
    async def ainvoke(self, messages):
        return Message(json.dumps({"selected": [{"card_id": "recipe.sql_query.extreme.max_value", "confidence": 0.9}]}))


class ReconcilingLLM:
    async def ainvoke(self, messages):
        prompt = str(messages[-1][1])
        if "explicit_fact_contracts" in prompt:
            return Message(json.dumps({"selected_memory_fact_keys": []}))
        return Message(json.dumps({"selected": [{"card_id": "recipe.sql_query.extreme.max_value", "confidence": 0.9}]}))


class PlanningLLM:
    async def ainvoke(self, messages):
        return Message(
            json.dumps(
                {
                    "fact_requests": [
                        {
                            "fact_key": "price.change",
                            "name": "price_change",
                            "fact_type": "difference",
                            "derived_from": ["price.start", "price.end"],
                        },
                        {
                            "fact_key": "price.fabricated",
                            "name": "fabricated_price",
                            "fact_type": "point_value",
                        },
                    ]
                }
            )
        )


class ToolScopedRetriever:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def retrieve(self, *, request_state, tool_name: str, action_input: dict):
        self.calls.append((tool_name, action_input))
        return MemoryRetrievalResult(
            hits=[MemoryHit(card_id="recipe.sql_query.extreme.max_value", confidence=0.9)],
            fact_requests=[
                DataFactRequest(
                    name="max_value",
                    fact_type="extreme",
                    requirements={"operator": "max"},
                )
            ],
            diagnostics={"retriever_type": "llm"},
        )


class FakeEmbeddingProvider:
    model = "fake-embedding"

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "max_value" in lowered or "最大值" in text or "最高" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "min_value" in lowered or "最小值" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def _registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                tool_name="sql_query",
                description="echo sql",
                input_model=EchoInput,
                output_model=EchoOutput,
                tool=EchoTool(),
                prompt_visible=True,
                runtime_access="none",
                result_target="evidence",
                produces_terminal_payload=False,
                supports_streaming=False,
            )
        ]
    )


def _request_state(message: str = "bitcoin 的 usd 最大值是多少") -> RequestStateModel:
    return RequestStateModel(
        request_id="req_memory",
        conversation_id="conv_memory",
        message=message,
        status="running",
    )


@pytest.mark.asyncio
async def test_tool_executor_injects_memory_fact_requests_into_tool_call():
    retriever = ToolScopedRetriever()
    executor = ToolExecutor(_registry(), memory_retriever=retriever)
    request_state = _request_state()
    request_state.iteration = 1
    result = await executor.execute(
        "sql_query",
        {"message": request_state.message},
        request_state,
        ConversationStateModel(conversation_id="conv_memory"),
    )

    fact_requests = result.full_payload["fact_requests"]
    assert fact_requests[0]["name"] == "max_value"
    assert fact_requests[0]["fact_type"] == "extreme"
    assert fact_requests[0]["requirements"]["source"] == "memory"
    assert request_state.tool_history[-1].tool_input["fact_requests"] == fact_requests
    assert request_state.tool_history[-1].tool_input["constraints"]["memory_diagnostics"]["selected_card_ids"] == [
        "recipe.sql_query.extreme.max_value"
    ]
    assert retriever.calls[0][0] == "sql_query"
    assert request_state.completion_state["memory_context"]["tool_calls"][0]["source"] == "tool_scoped_memory_retrieval"


@pytest.mark.asyncio
async def test_memory_retriever_llm_failure_returns_empty_without_keyword_fallback():
    retriever = FactMemoryRetriever(llm=BrokenLLM())
    result = await retriever.retrieve(
        request_state=_request_state("最大值是多少"),
        tool_name="sql_query",
        action_input={"message": "最大值是多少"},
    )

    assert result.fact_requests == []
    assert result.diagnostics["error_type"] == "memory_retrieval_failed"


@pytest.mark.asyncio
async def test_memory_retriever_selects_card_then_loads_detail():
    retriever = FactMemoryRetriever(llm=SelectingLLM())
    result = await retriever.retrieve(
        request_state=_request_state(),
        tool_name="sql_query",
        action_input={"message": "bitcoin 的 usd 最大值是多少"},
    )

    assert [hit.card_id for hit in result.hits] == ["recipe.sql_query.extreme.max_value"]
    assert any(request.name == "max_value" and request.fact_type == "extreme" for request in result.fact_requests)


@pytest.mark.asyncio
async def test_memory_retriever_uses_llm_to_remove_semantic_duplicate_of_explicit_fact():
    retriever = FactMemoryRetriever(llm=ReconcilingLLM())
    result = await retriever.retrieve(
        request_state=_request_state(),
        tool_name="sql_query",
        action_input={
            "message": "bitcoin 的 usd 最大值是多少",
            "fact_requests": [
                {
                    "fact_key": "bitcoin.maximum_price",
                    "name": "Bitcoin maximum price",
                    "fact_type": "extreme",
                    "requirements": {"operator": "max"},
                }
            ],
        },
    )

    assert result.fact_requests == []


@pytest.mark.asyncio
async def test_code_fact_planner_keeps_only_contracts_grounded_in_verified_parents():
    state = _request_state("calculate the price change")
    state.fact_set.facts = [
        DataFact(
            fact_id="fact_start",
            fact_key="price.start",
            name="start_price",
            fact_type="point_value",
            statement="Start is 10.",
            value=10,
            method="sql_query",
        ),
        DataFact(
            fact_id="fact_end",
            fact_key="price.end",
            name="end_price",
            fact_type="point_value",
            statement="End is 12.",
            value=12,
            method="sql_query",
        ),
    ]
    retriever = FactMemoryRetriever(llm=PlanningLLM())

    requests = await retriever._plan_tool_fact_requests(
        request_state=state,
        tool_name="code_interpreter",
        action_input={"analysis_goal": "calculate the price change"},
    )

    assert [request.fact_key for request in requests] == ["price.change"]


@pytest.mark.asyncio
async def test_embedding_memory_retriever_selects_card_and_uses_local_cache(tmp_path):
    provider = FakeEmbeddingProvider()
    retriever = EmbeddingFactMemoryRetriever(
        embedding_provider=provider,
        embedding_store=FactMemoryEmbeddingStore(tmp_path),
        top_k=3,
        score_threshold=0.2,
    )

    result = await retriever.retrieve_once(request_state=_request_state("bitcoin 的 usd 最大值是多少"))

    assert any(hit.card_id == "recipe.sql_query.extreme.max_value" for hit in result.hits)
    assert any(request.name == "max_value" for request in result.fact_requests)
    assert result.diagnostics["retriever_type"] == "embedding"
    assert result.diagnostics["cache_misses"] > 0

    second = await retriever.retrieve_once(request_state=_request_state("bitcoin 的 usd 最大值是多少"))
    assert second.diagnostics["cache_hits"] > 0
    assert len(provider.calls) == 3  # first cards + first query + second query


def test_embedding_store_invalidates_when_card_text_changes(tmp_path):
    store = FactMemoryEmbeddingStore(tmp_path)
    card = MemoryCard(
        id="recipe.sql_query.extreme.max_value",
        kind="fact_recipe",
        title="max_value",
        description="Generate max value.",
        tags=["extreme"],
    )
    text = memory_card_embedding_text(card)
    saved = store.save(
        database_id=None,
        model="fake",
        card=card,
        text=text,
        vector=[1.0, 0.0],
        memory_updated_at=None,
    )
    assert saved.cache_hit is False
    assert store.load(database_id=None, model="fake", card=card, text=text).cache_hit is True

    changed = card.model_copy(update={"description": "Generate max value with time."})
    changed_text = memory_card_embedding_text(changed)
    assert store.load(database_id=None, model="fake", card=changed, text=changed_text) is None
