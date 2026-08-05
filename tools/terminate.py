"""Terminal ReAct action backed by final answer assembly."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from schemas.state import RequestStateModel
from tools.base import BaseTool
from tools.format_answer import FormatAnswerInput, FormatAnswerTool


class TerminateInput(BaseModel):
    result: str | None = None
    summary_goal: str | None = None
    direct_answer: str | None = None
    include_analysis_ids: list[str] = Field(default_factory=list)
    include_fact_ids: list[str] = Field(default_factory=list)
    include_visualization_ids: list[str] = Field(default_factory=list)
    section_plan: list[str] = Field(default_factory=list)
    unavailable_outputs: list[str] = Field(default_factory=list)
    unavailable_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data):
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for key in ("result", "direct_answer", "summary_goal"):
            value = normalized.get(key)
            if isinstance(value, (dict, list, tuple)):
                raise ValueError(
                    f"terminate.{key} must be a natural-language string, not a structured object. "
                    "Use evidence IDs for structure and write the user-visible answer as prose."
                )
        result = normalized.get("result")
        if not normalized.get("direct_answer") and result:
            normalized["direct_answer"] = result
        if not normalized.get("summary_goal"):
            normalized["summary_goal"] = normalized.get("message") or result or "Assemble the final answer."
        for key in ("include_analysis_ids", "include_fact_ids", "include_visualization_ids", "section_plan", "unavailable_outputs"):
            normalized[key] = cls._normalize_listish(normalized.get(key))
        return normalized

    @staticmethod
    def _normalize_listish(value):
        if value in (None, "", False, True):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
        return []


class TerminateTool(BaseTool):
    """Runtime-visible terminal action.

    The ReAct policy sees termination as the final control action, while answer
    formatting remains an internal assembler for the stable FinalAnswer schema.
    """

    def __init__(self, formatter: FormatAnswerTool | None = None):
        self._formatter = formatter or FormatAnswerTool()

    async def execute(
        self,
        validated_input: TerminateInput,
        *,
        request_state: RequestStateModel,
        **kwargs,
    ) -> dict:
        include_fact_ids = validated_input.include_fact_ids or self._process_fact_ids(request_state)
        formatter_input = FormatAnswerInput(
            summary_goal=validated_input.summary_goal,
            direct_answer=validated_input.direct_answer,
            include_analysis_ids=validated_input.include_analysis_ids,
            include_fact_ids=include_fact_ids,
            include_visualization_ids=validated_input.include_visualization_ids,
            section_plan=validated_input.section_plan,
        )
        return await self._formatter.execute(formatter_input, request_state=request_state, **kwargs)

    def _process_fact_ids(self, request_state: RequestStateModel) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for event in request_state.fact_events:
            for fact_id in [*event.produced_fact_ids, *event.unavailable_fact_ids]:
                if fact_id not in seen:
                    selected.append(fact_id)
                    seen.add(fact_id)
        return selected
