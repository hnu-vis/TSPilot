from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from core.data_fact.embedding_store import FactMemoryEmbeddingStore, memory_card_embedding_text
from core.data_fact.retriever import EmbeddingFactMemoryRetriever, FactMemoryRetriever
from runtime.tool_executor import ToolExecutor
from schemas.data_fact import DataFactRequest, MemoryCard
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
        return Message(json.dumps({"selected": [{"card_id": "recipe.extreme.max_value", "confidence": 0.9}]}))


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
    executor = ToolExecutor(_registry())
    request_state = _request_state()
    request_state.iteration = 1
    request_state.completion_state["memory_context"] = {
        "fact_requests": [
            {
                "name": "max_value",
                "fact_type": "extreme",
                "requirements": {"operator": "max"},
            }
        ],
        "selected_card_ids": ["recipe.extreme.max_value"],
        "diagnostics": {"retriever_type": "embedding"},
        "used_by_tools": {},
    }
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
        "recipe.extreme.max_value"
    ]
    assert request_state.completion_state["memory_context"]["used_by_tools"]["sql_query"] == 1


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

    assert [hit.card_id for hit in result.hits] == ["recipe.extreme.max_value"]
    assert any(request.name == "max_value" and request.fact_type == "extreme" for request in result.fact_requests)


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

    assert any(hit.card_id == "recipe.extreme.max_value" for hit in result.hits)
    assert any(request.name == "max_value" for request in result.fact_requests)
    assert result.diagnostics["retriever_type"] == "embedding"
    assert result.diagnostics["cache_misses"] > 0

    second = await retriever.retrieve_once(request_state=_request_state("bitcoin 的 usd 最大值是多少"))
    assert second.diagnostics["cache_hits"] > 0
    assert len(provider.calls) == 3  # first cards + first query + second query


def test_embedding_store_invalidates_when_card_text_changes(tmp_path):
    store = FactMemoryEmbeddingStore(tmp_path)
    card = MemoryCard(
        id="recipe.extreme.max_value",
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
