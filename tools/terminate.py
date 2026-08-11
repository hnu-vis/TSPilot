"""Terminal ReAct action backed by model-free final answer assembly."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from schemas.output import FinalResponsePlan
from schemas.state import RequestStateModel
from tools.base import BaseTool
from tools.format_answer import FormatAnswerInput, FormatAnswerTool


class TerminateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response_plan: FinalResponsePlan
    unavailable_outputs: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None


class TerminateTool(BaseTool):
    """The final model action carries prose and semantic visualization intents."""

    def __init__(self, formatter: FormatAnswerTool | None = None, *, llm=None):
        self._formatter = formatter or FormatAnswerTool(llm=llm)

    async def execute(
        self,
        validated_input: TerminateInput,
        *,
        request_state: RequestStateModel,
        **kwargs,
    ) -> dict:
        return await self._formatter.execute(
            FormatAnswerInput(response_plan=validated_input.response_plan),
            request_state=request_state,
            **kwargs,
        )
