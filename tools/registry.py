"""Tool registry."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.settings import Settings
from schemas.analysis import AnalysisResult
from schemas.database import DatabaseEvidence
from schemas.insight import InsightResult
from schemas.output import FinalAnswer
from schemas.timeseries import AnomalyResult, ForecastResult
from tools.anomaly import AnomalyInput, AnomalyTool
from tools.base import BaseTool
from tools.forecast import ForecastInput, ForecastTool
from tools.format_answer import FormatAnswerInput, FormatAnswerTool
from tools.insight import InsightInput, InsightTool
from tools.rag import RagInput, RagTool
from tools.skill import SkillInput, SkillTool
from tools.sql_query import SqlQueryInput, SqlQueryTool
from tools.terminate import TerminateInput, TerminateTool
from tools.todowrite import TodoWriteInput, TodoWriteResult, TodoWriteTool


@dataclass
class ToolSpec:
    tool_name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    tool: BaseTool
    prompt_visible: bool
    runtime_access: str
    result_target: str
    produces_terminal_payload: bool
    supports_streaming: bool


class DictResult(BaseModel):
    summary: str
    results: list[dict] = Field(default_factory=list)


class SkillResult(BaseModel):
    summary: str
    skill_name: str | None = None
    results: list[dict] = Field(default_factory=list)


class ToolRegistry:
    """Resolve tools by action name."""

    def __init__(self, specs: list[ToolSpec]):
        self._specs = {spec.tool_name: spec for spec in specs}

    def resolve(self, action_name: str) -> ToolSpec:
        if action_name not in self._specs:
            raise KeyError(f"Unknown tool '{action_name}'.")
        return self._specs[action_name]


def build_tool_registry(settings: Settings, llm=None) -> ToolRegistry:
    specs = [
        ToolSpec(
            tool_name="todowrite",
            description="Update or create a todo list.",
            input_model=TodoWriteInput,
            output_model=TodoWriteResult,
            tool=TodoWriteTool(),
            prompt_visible=True,
            runtime_access="none",
            result_target="todo",
            produces_terminal_payload=False,
            supports_streaming=False,
        ),
        ToolSpec(
            tool_name="sql_query",
            description="Unified read-only database query tool. Use natural-language message for automatic planning or explicit query for SQL/Flux/PromQL analysis.",
            input_model=SqlQueryInput,
            output_model=DatabaseEvidence,
            tool=SqlQueryTool(settings, llm=llm),
            prompt_visible=True,
            runtime_access="request_and_conversation_read",
            result_target="evidence",
            produces_terminal_payload=False,
            supports_streaming=False,
        ),
        ToolSpec(
            tool_name="insight",
            description="Execute generated Python analysis code over full evidence artifacts.",
            input_model=InsightInput,
            output_model=AnalysisResult,
            tool=InsightTool(),
            prompt_visible=True,
            runtime_access="request_state_read",
            result_target="analysis",
            produces_terminal_payload=False,
            supports_streaming=False,
        ),
        ToolSpec(
            tool_name="forecast",
            description="Run forecasting on time-series evidence.",
            input_model=ForecastInput,
            output_model=ForecastResult,
            tool=ForecastTool(),
            prompt_visible=True,
            runtime_access="request_state_read",
            result_target="analysis",
            produces_terminal_payload=False,
            supports_streaming=False,
        ),
        ToolSpec(
            tool_name="anomaly",
            description="Run anomaly detection on time-series evidence.",
            input_model=AnomalyInput,
            output_model=AnomalyResult,
            tool=AnomalyTool(),
            prompt_visible=True,
            runtime_access="request_state_read",
            result_target="analysis",
            produces_terminal_payload=False,
            supports_streaming=False,
        ),
        ToolSpec(
            tool_name="rag",
            description="Extension knowledge retrieval.",
            input_model=RagInput,
            output_model=DictResult,
            tool=RagTool(),
            prompt_visible=True,
            runtime_access="none",
            result_target="analysis",
            produces_terminal_payload=False,
            supports_streaming=False,
        ),
        ToolSpec(
            tool_name="skill",
            description="Invoke a packaged workflow.",
            input_model=SkillInput,
            output_model=SkillResult,
            tool=SkillTool(),
            prompt_visible=True,
            runtime_access="request_state_read",
            result_target="analysis",
            produces_terminal_payload=False,
            supports_streaming=False,
        ),
        ToolSpec(
            tool_name="format_answer",
            description="Assemble the final answer from verified outputs.",
            input_model=FormatAnswerInput,
            output_model=FinalAnswer,
            tool=FormatAnswerTool(),
            prompt_visible=False,
            runtime_access="request_state_read",
            result_target="presentation",
            produces_terminal_payload=True,
            supports_streaming=False,
        ),
        ToolSpec(
            tool_name="terminate",
            description="Terminate the ReAct loop and assemble the final answer from verified outputs.",
            input_model=TerminateInput,
            output_model=FinalAnswer,
            tool=TerminateTool(),
            prompt_visible=True,
            runtime_access="request_state_read",
            result_target="presentation",
            produces_terminal_payload=True,
            supports_streaming=False,
        ),
    ]
    return ToolRegistry(specs)
