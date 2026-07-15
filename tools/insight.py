"""Insight tool."""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.insight import build_insight_result
from schemas.database import DatabaseEvidence
from schemas.insight import InsightResult
from tools.base import BaseTool


class InsightInput(BaseModel):
    database_evidence: DatabaseEvidence | None = None
    requested_fact_types: list[str] = Field(default_factory=list)
    focus: str | None = None
    constraints: dict | None = None


class InsightTool(BaseTool):
    async def execute(self, validated_input: InsightInput, **kwargs) -> dict:
        request_state = kwargs.get("request_state")
        database_evidence = validated_input.database_evidence
        requested_fact_types = list(validated_input.requested_fact_types)
        focus = validated_input.focus
        if request_state is not None:
            database_evidence = _resolve_database_evidence(database_evidence, request_state)
            if not requested_fact_types:
                requested_fact_types = list(getattr(request_state, "requested_fact_types", []) or [])
            if focus is None:
                focus = getattr(request_state, "focus", None)
        if database_evidence is None:
            raise ValueError("Insight requires database_evidence or a latest_database_evidence in request state.")
        result = build_insight_result(
            database_evidence,
            requested_fact_types,
            focus,
        )
        return result.model_dump(mode="json")


def _resolve_database_evidence(database_evidence, request_state):
    if database_evidence is None:
        latest = request_state.latest_database_evidence
        if latest is None:
            return None
        return request_state.database_evidence_artifacts.get(latest.evidence_id, latest)
    return request_state.database_evidence_artifacts.get(database_evidence.evidence_id, database_evidence)
