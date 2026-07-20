"""Dependency factories."""
from __future__ import annotations

from functools import lru_cache

from agents.data_agent import DataAgent
from langchain_openai import ChatOpenAI
from prompts.data_agent import DataAgentPromptBuilder
from core.runtime_evaluator import RuntimeLLMEvaluator
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
def get_tool_registry():
    return build_tool_registry(get_settings(), llm=get_llm())


@lru_cache(maxsize=1)
def get_tool_executor() -> ToolExecutor:
    return ToolExecutor(get_tool_registry())


@lru_cache(maxsize=1)
def get_data_agent() -> DataAgent:
    return DataAgent(prompt_builder=get_prompt_builder(), llm=get_llm())


@lru_cache(maxsize=1)
def get_runtime_evaluator() -> RuntimeLLMEvaluator:
    return RuntimeLLMEvaluator(get_llm())


def get_react_loop() -> ReActLoop:
    return ReActLoop(
        data_agent=get_data_agent(),
        tool_executor=get_tool_executor(),
        settings=get_settings(),
        runtime_evaluator=get_runtime_evaluator(),
    )
