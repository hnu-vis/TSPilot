"""Dependency factories."""
from __future__ import annotations

from functools import lru_cache

from agents.data_agent import DataAgent
from langchain_openai import ChatOpenAI
from prompts.data_agent import DataAgentPromptBuilder
from core.key_insight.embedding import OpenAICompatibleEmbeddingProvider
from core.key_insight.embedding_store import InsightMemoryEmbeddingStore
from core.key_insight.learning import (
    InsightLearningOutbox,
    InsightLearningScheduleStore,
    InsightMemoryLearner,
    InsightMemoryLearningWorker,
)
from core.key_insight.retriever import HybridInsightMemoryRetriever
from runtime.plain_chat import PlainChatService
from runtime.react_loop import ReActLoop
from runtime.tool_executor import ToolExecutor
from runtime.timeout_policy import TimeoutPolicy, load_timeout_policy
from app.settings import get_settings
from app.model_config import get_model_config_store
from tools.registry import build_tool_registry
from core.visualization import VisualizationArtifactStore


@lru_cache(maxsize=1)
def get_prompt_builder() -> DataAgentPromptBuilder:
    return DataAgentPromptBuilder()


@lru_cache(maxsize=1)
def get_timeout_policy() -> TimeoutPolicy:
    return load_timeout_policy(get_settings().resolved_timeout_config_path)


@lru_cache(maxsize=1)
def get_llm():
    model_settings = get_model_config_store().effective_ai()
    if not model_settings.api_key:
        raise RuntimeError("The active LLM connection has no API key. Configure it in Model Management.")

    return ChatOpenAI(
        api_key=model_settings.api_key,
        base_url=model_settings.api_base,
        model=model_settings.model,
        temperature=model_settings.temperature,
        streaming=False,
        timeout=get_timeout_policy().services.llm_transport_seconds,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


@lru_cache(maxsize=1)
def get_data_agent_llm():
    model_settings = get_model_config_store().effective_ai()
    if not model_settings.api_key:
        raise RuntimeError("The active LLM connection has no API key. Configure it in Model Management.")

    return ChatOpenAI(
        api_key=model_settings.api_key,
        base_url=model_settings.api_base,
        model=model_settings.model,
        temperature=model_settings.temperature,
        streaming=False,
        timeout=get_timeout_policy().services.llm_transport_seconds,
    )


@lru_cache(maxsize=1)
def get_tool_registry():
    return build_tool_registry(
        get_settings(),
        llm=get_llm(),
        visualization_artifact_store=get_visualization_artifact_store(),
        timeout_policy=get_timeout_policy(),
    )


@lru_cache(maxsize=1)
def get_visualization_artifact_store() -> VisualizationArtifactStore:
    return VisualizationArtifactStore(get_settings().resolved_visualization_artifact_dir)


@lru_cache(maxsize=1)
def get_tool_executor() -> ToolExecutor:
    return ToolExecutor(
        get_tool_registry(),
        memory_retriever=get_insight_memory_retriever(),
        timeout_policy=get_timeout_policy(),
    )


@lru_cache(maxsize=1)
def get_insight_memory_retriever() -> HybridInsightMemoryRetriever:
    settings = get_settings()
    model_settings = get_model_config_store().effective_ai()
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        api_key=model_settings.embedding_api_key or "",
        api_base=model_settings.embedding_api_base,
        model=model_settings.embedding_model,
        timeout_seconds=get_timeout_policy().services.embedding_request_seconds,
    )
    return HybridInsightMemoryRetriever(
        llm=get_llm(),
        embedding_provider=embedding_provider,
        embedding_store=InsightMemoryEmbeddingStore(settings.resolved_memory_embedding_cache_dir),
        top_k=settings.memory_embedding_top_k,
        score_threshold=settings.memory_embedding_score_threshold,
    )


@lru_cache(maxsize=1)
def get_insight_learning_outbox() -> InsightLearningOutbox:
    return InsightLearningOutbox(get_settings().resolved_insight_memory_learning_dir)


@lru_cache(maxsize=1)
def get_insight_learning_schedule_store() -> InsightLearningScheduleStore:
    settings = get_settings()
    return InsightLearningScheduleStore(
        settings.resolved_insight_memory_learning_dir,
        default_max_wait_seconds=settings.insight_memory_learning_max_wait_seconds,
    )


@lru_cache(maxsize=1)
def get_insight_memory_learning_worker() -> InsightMemoryLearningWorker:
    settings = get_settings()
    model_settings = get_model_config_store().effective_ai()
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        api_key=model_settings.embedding_api_key or "",
        api_base=model_settings.embedding_api_base,
        model=model_settings.embedding_model,
        timeout_seconds=get_timeout_policy().services.embedding_request_seconds,
    )
    learner = InsightMemoryLearner(
        llm=get_llm(),
        embedding_provider=embedding_provider,
        embedding_store=InsightMemoryEmbeddingStore(settings.resolved_memory_embedding_cache_dir),
        outbox=get_insight_learning_outbox(),
        neighbor_top_k=settings.memory_embedding_top_k,
        neighbor_threshold=settings.memory_embedding_score_threshold,
        max_attempts=settings.insight_memory_learning_max_attempts,
    )
    return InsightMemoryLearningWorker(
        learner,
        batch_size=settings.insight_memory_learning_batch_size,
        max_wait_seconds=settings.insight_memory_learning_max_wait_seconds,
        poll_seconds=settings.insight_memory_learning_poll_seconds,
        lease_seconds=settings.insight_memory_learning_lease_seconds,
        llm_chunk_size=settings.insight_memory_learning_llm_chunk_size,
        schedule_store=get_insight_learning_schedule_store(),
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
        insight_learning_outbox=get_insight_learning_outbox() if settings.insight_memory_learning_enabled else None,
        timeout_policy=get_timeout_policy(),
    )


def get_plain_chat_service() -> PlainChatService:
    return PlainChatService(llm=get_llm())


def get_react_loop_for_model(model_connection_id: str | None = None) -> ReActLoop:
    """Build an isolated runtime when a conversation selects a non-default LLM."""
    if not model_connection_id:
        return get_react_loop()
    settings = get_settings()
    model_settings = get_model_config_store().effective_ai(model_connection_id)
    structured_llm = _create_chat_llm(model_settings, structured=True)
    agent_llm = _create_chat_llm(model_settings, structured=False)
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        api_key=model_settings.embedding_api_key or "",
        api_base=model_settings.embedding_api_base,
        model=model_settings.embedding_model,
        timeout_seconds=get_timeout_policy().services.embedding_request_seconds,
    )
    retriever = HybridInsightMemoryRetriever(
        llm=structured_llm,
        embedding_provider=embedding_provider,
        embedding_store=InsightMemoryEmbeddingStore(settings.resolved_memory_embedding_cache_dir),
        top_k=settings.memory_embedding_top_k,
        score_threshold=settings.memory_embedding_score_threshold,
    )
    tool_registry = build_tool_registry(
        settings,
        llm=structured_llm,
        visualization_artifact_store=get_visualization_artifact_store(),
        timeout_policy=get_timeout_policy(),
    )
    return ReActLoop(
        data_agent=DataAgent(prompt_builder=get_prompt_builder(), llm=agent_llm),
        tool_executor=ToolExecutor(
            tool_registry,
            memory_retriever=retriever,
            timeout_policy=get_timeout_policy(),
        ),
        settings=settings,
        insight_learning_outbox=get_insight_learning_outbox() if settings.insight_memory_learning_enabled else None,
        timeout_policy=get_timeout_policy(),
    )


def get_plain_chat_service_for_model(model_connection_id: str | None = None) -> PlainChatService:
    if not model_connection_id:
        return get_plain_chat_service()
    model_settings = get_model_config_store().effective_ai(model_connection_id)
    return PlainChatService(llm=_create_chat_llm(model_settings, structured=True))


def _create_chat_llm(model_settings, *, structured: bool):
    if not model_settings.api_key:
        raise RuntimeError(
            "The selected LLM connection has no API key. Configure its API key in Model Management."
        )
    kwargs = {
        "api_key": model_settings.api_key,
        "base_url": model_settings.api_base,
        "model": model_settings.model,
        "temperature": model_settings.temperature,
        "streaming": False,
        "timeout": get_timeout_policy().services.llm_transport_seconds,
    }
    if structured:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)


async def clear_runtime_model_dependencies() -> None:
    """Recreate model-backed dependencies for subsequent requests after an update."""
    if get_tool_registry.cache_info().currsize:
        await get_tool_registry().close()
    for factory in (
        get_data_agent,
        get_tool_executor,
        get_tool_registry,
        get_insight_memory_retriever,
        get_data_agent_llm,
        get_llm,
    ):
        cache_clear = getattr(factory, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
