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
        include_fact_ids = (
            self._resolve_fact_references(validated_input.include_fact_ids, request_state)
            if validated_input.include_fact_ids
            else self._process_fact_ids(request_state)
        )
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
        available = {fact.fact_id for fact in request_state.fact_set.facts}
        selected: list[str] = []
        for event in request_state.fact_events:
            for fact_id in [*event.produced_fact_ids, *event.unavailable_fact_ids]:
                if fact_id in available and fact_id not in selected:
                    selected.append(fact_id)
        return selected

    def _resolve_fact_references(self, references: list[str], request_state: RequestStateModel) -> list[str]:
        lookup: dict[str, set[str]] = {}
        for fact in request_state.fact_set.facts:
            keys = {fact.fact_id, f"fact:{fact.fact_id}", fact.name}
            if fact.fact_key:
                keys.update({fact.fact_key, f"fact:{fact.fact_key}"})
            for key in keys:
                lookup.setdefault(key, set()).add(fact.fact_id)
        resolved: list[str] = []
        unresolved: list[str] = []
        ambiguous: list[str] = []
        for reference in references:
            value = str(reference or "").strip()
            matches = lookup.get(value, set())
            if not matches:
                unresolved.append(value)
            elif len(matches) > 1:
                ambiguous.append(value)
            else:
                fact_id = next(iter(matches))
                if fact_id not in resolved:
                    resolved.append(fact_id)
        if ambiguous:
            raise ValueError(
                "terminate.include_fact_ids contains ambiguous Fact names; use fact_key or fact_id: "
                + ", ".join(ambiguous)
            )
        if unresolved:
            available = [fact.fact_key or fact.fact_id for fact in request_state.fact_set.facts]
            raise ValueError(
                "terminate.include_fact_ids contains unresolved Fact references: "
                + ", ".join(unresolved)
                + ". Available Fact references: "
                + ", ".join(available)
            )
        return resolved
