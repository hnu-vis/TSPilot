from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from app.settings import get_settings
from runtime.timeout_policy import ToolTimeouts, load_timeout_policy
from runtime.tool_executor import ToolExecutor
from schemas.state import ConversationStateModel, RequestStateModel
from tools.base import BaseTool, StructuredToolError
from tools.registry import ToolRegistry, ToolSpec, build_tool_registry


class _Input(BaseModel):
    query: str


class _Output(BaseModel):
    summary: str


class _SlowTool(BaseTool):
    async def execute(self, validated_input, **kwargs) -> dict:
        await asyncio.sleep(1)
        return {"summary": validated_input.query}


def test_timeout_policy_is_the_single_runtime_and_tool_budget_source():
    policy = load_timeout_policy()

    assert policy.runtime.agent_turn_seconds == 45
    assert policy.runtime.request_deadline_seconds == 600
    assert policy.tool("code_interpreter").execution_seconds == 180
    assert policy.tool("code_interpreter").stage_seconds("sandbox_seconds") == 120
    assert not hasattr(get_settings(), "agent_turn_timeout_seconds")
    assert not hasattr(get_settings(), "request_deadline_seconds")


def test_registry_propagates_timeout_policy_to_tool_specs_and_stages(tmp_path):
    settings = get_settings().model_copy(update={"visualization_artifact_dir": str(tmp_path)})
    registry = build_tool_registry(settings, llm=None)

    code_spec = registry.resolve("code_interpreter")
    assert code_spec.execution_timeout_seconds == 180
    assert code_spec.tool._sandbox_timeout_seconds == 120
    assert registry.resolve("visualization").execution_timeout_seconds == 300


def test_stage_budget_cannot_exceed_owning_tool_budget():
    with pytest.raises(ValueError, match="cannot exceed"):
        ToolTimeouts(
            execution_seconds=10,
            stages={"llm_call_seconds": 11},
        )


@pytest.mark.asyncio
async def test_tool_executor_enforces_generic_tool_budget():
    registry = ToolRegistry([
        ToolSpec(
            tool_name="rag",
            description="slow test tool",
            input_model=_Input,
            output_model=_Output,
            tool=_SlowTool(),
            prompt_visible=True,
            runtime_access="none",
            result_target="analysis",
            produces_terminal_payload=False,
            supports_streaming=False,
            execution_timeout_seconds=0.01,
        )
    ])
    executor = ToolExecutor(registry)
    request_state = RequestStateModel(
        request_id="req_timeout",
        message="wait",
        status="running",
    )
    conversation_state = ConversationStateModel(conversation_id="conv_timeout")

    with pytest.raises(StructuredToolError) as error:
        await executor.execute(
            "rag",
            {"query": "slow"},
            request_state,
            conversation_state,
        )

    assert error.value.error_type == "tool_execution_timeout"
    assert error.value.diagnostics["tool"] == "rag"
    assert error.value.diagnostics["timeout_seconds"] <= 0.01
