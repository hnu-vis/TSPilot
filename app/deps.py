"""Dependency factories."""
from __future__ import annotations

from functools import lru_cache

from agents.data_agent import DataAgent
from langchain_openai import ChatOpenAI
from prompts.data_agent import DataAgentPromptBuilder
from core.data_fact.embedding import OpenAICompatibleEmbeddingProvider
from core.data_fact.embedding_store import FactMemoryEmbeddingStore
from core.data_fact.learning import FactLearningOutbox, FactMemoryLearner, FactMemoryLearningWorker
from core.data_fact.retriever import HybridFactMemoryRetriever
from runtime.plain_chat import PlainChatService
from runtime.react_loop import ReActLoop
from runtime.tool_executor import ToolExecutor
from app.settings import get_settings
from tools.registry import build_tool_registry


@lru_cache(maxsize=1)
def get_prompt_builder() -> DataAgentPromptBuilder:
    return DataAgentPromptBuilder()


@lru_cache(maxsize=1)
def get_llm():
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the LLM-based data_agent.")

    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        streaming=False,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


@lru_cache(maxsize=1)
def get_data_agent_llm():
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the LLM-based data_agent.")

    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        streaming=False,
    )


@lru_cache(maxsize=1)
def get_tool_registry():
    return build_tool_registry(get_settings(), llm=get_llm())


@lru_cache(maxsize=1)
def get_tool_executor() -> ToolExecutor:
    return ToolExecutor(get_tool_registry(), memory_retriever=get_fact_memory_retriever())


@lru_cache(maxsize=1)
def get_fact_memory_retriever() -> HybridFactMemoryRetriever:
    settings = get_settings()
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        api_key=settings.resolved_embedding_api_key or "",
        api_base=settings.resolved_embedding_api_base,
        model=settings.embedding_model,
    )
    return HybridFactMemoryRetriever(
        llm=get_llm(),
        embedding_provider=embedding_provider,
        embedding_store=FactMemoryEmbeddingStore(settings.resolved_memory_embedding_cache_dir),
        top_k=settings.memory_embedding_top_k,
        score_threshold=settings.memory_embedding_score_threshold,
    )


@lru_cache(maxsize=1)
def get_fact_learning_outbox() -> FactLearningOutbox:
    return FactLearningOutbox(get_settings().resolved_fact_memory_learning_dir)


@lru_cache(maxsize=1)
def get_fact_memory_learning_worker() -> FactMemoryLearningWorker:
    settings = get_settings()
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        api_key=settings.resolved_embedding_api_key or "",
        api_base=settings.resolved_embedding_api_base,
        model=settings.embedding_model,
    )
    learner = FactMemoryLearner(
        llm=get_llm(),
        embedding_provider=embedding_provider,
        embedding_store=FactMemoryEmbeddingStore(settings.resolved_memory_embedding_cache_dir),
        outbox=get_fact_learning_outbox(),
        neighbor_top_k=settings.memory_embedding_top_k,
        neighbor_threshold=settings.memory_embedding_score_threshold,
        max_attempts=settings.fact_memory_learning_max_attempts,
    )
    return FactMemoryLearningWorker(
        learner,
        batch_size=settings.fact_memory_learning_batch_size,
        max_wait_seconds=settings.fact_memory_learning_max_wait_seconds,
        poll_seconds=settings.fact_memory_learning_poll_seconds,
        lease_seconds=settings.fact_memory_learning_lease_seconds,
        llm_chunk_size=settings.fact_memory_learning_llm_chunk_size,
    )
@lru_cache(maxsize=1)
def get_data_agent() -> DataAgent:
    return DataAgent(prompt_builder=get_prompt_builder(), llm=get_data_agent_llm())


def get_react_loop() -> ReActLoop:
    settings = get_settings()
    return ReActLoop(
        data_agent=get_data_agent(),
        tool_executor=get_tool_executor(),
        settings=settings,
        fact_learning_outbox=get_fact_learning_outbox() if settings.fact_memory_learning_enabled else None,
    )


def get_plain_chat_service() -> PlainChatService:
    return PlainChatService(llm=get_llm())
