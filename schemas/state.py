"""Request and conversation state models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schemas.analysis import AnalysisResult
from schemas.api import Message
from schemas.database import DatabaseEvidence
from schemas.database_context import DatabaseContext
from schemas.insight import InsightResult, RejectedFact, VerifiedFact
from schemas.output import FinalAnswer
from schemas.timeseries import AnomalyResult, ForecastResult
from schemas.tool import ToolCall, ToolObservation
from schemas.visualization import VisualizationPayload


class RequestStateModel(BaseModel):
    request_id: str
    conversation_id: str | None = None
    message: str
    database_context: DatabaseContext | None = None
    selected_database: str | None = None
    selected_database_type: str | None = None
    time_range: dict | None = None
    constraints: dict = Field(default_factory=dict)
    history: list[Message] = Field(default_factory=list)

    status: Literal["running", "completed", "failed"]
    current_intent: str | None = None
    intent_profile: dict = Field(default_factory=dict)
    requested_fact_types: list[str] = Field(default_factory=list)
    answer_requirements: list[str] = Field(default_factory=list)
    answer_coverage: dict[str, bool] = Field(default_factory=dict)
    focus: str | None = None
    todo_list: list[dict] = Field(default_factory=list)
    plan_current_step: int = 0
    planning_complete: bool = False
    iteration: int = 0
    max_iterations: int = 4
    context_budget: dict = Field(default_factory=dict)
    context_status: Literal["ok", "summarized", "truncated", "overflowed"] = "ok"
    context_overflow_reason: str | None = None

    latest_database_evidence: DatabaseEvidence | None = None
    database_evidence_artifacts: dict[str, DatabaseEvidence] = Field(default_factory=dict)
    latest_insight: InsightResult | None = None
    insight_artifacts: dict[str, InsightResult] = Field(default_factory=dict)
    latest_analysis_id: str | None = None
    analysis_artifacts: dict[str, AnalysisResult] = Field(default_factory=dict)
    latest_forecast: ForecastResult | None = None
    forecast_artifacts: dict[str, ForecastResult] = Field(default_factory=dict)
    latest_anomaly: AnomalyResult | None = None
    anomaly_artifacts: dict[str, AnomalyResult] = Field(default_factory=dict)
    latest_rag: dict | None = None
    latest_skill: dict | None = None
    verified_facts: list[VerifiedFact] = Field(default_factory=list)
    rejected_facts: list[RejectedFact] = Field(default_factory=list)

    final_answer_draft: FinalAnswer | None = None
    visualizations: list[VisualizationPayload] = Field(default_factory=list)

    tool_history: list[ToolCall] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    prompt_context_summary: str | None = None


class ConversationStateModel(BaseModel):
    conversation_id: str
    database_context: DatabaseContext | None = None
    recent_messages: list[Message] = Field(default_factory=list)
    session_summary: str | None = None
    intent_profile: dict = Field(default_factory=dict)
    todo_list: list[dict] = Field(default_factory=list)
    plan_current_step: int = 0
    planning_complete: bool = False
    recent_todo_summary: str | None = None
    latest_database_evidence: DatabaseEvidence | None = None
    database_evidence_artifacts: dict[str, DatabaseEvidence] = Field(default_factory=dict)
    latest_insight: InsightResult | None = None
    insight_artifacts: dict[str, InsightResult] = Field(default_factory=dict)
    latest_analysis_id: str | None = None
    analysis_artifacts: dict[str, AnalysisResult] = Field(default_factory=dict)
    latest_forecast: ForecastResult | None = None
    forecast_artifacts: dict[str, ForecastResult] = Field(default_factory=dict)
    latest_anomaly: AnomalyResult | None = None
    anomaly_artifacts: dict[str, AnomalyResult] = Field(default_factory=dict)
    latest_rag: dict | None = None
    latest_skill: dict | None = None
    recent_visualizations: list[VisualizationPayload] = Field(default_factory=list)
    updated_at: str | None = None
    context_budget: dict | None = None
