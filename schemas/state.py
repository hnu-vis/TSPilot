"""Request and conversation state models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from schemas.action_output import ActionOutput
from schemas.analysis import AnalysisResult
from schemas.api import Message
from schemas.database import DatabaseEvidence
from schemas.database_context import DatabaseContext
from schemas.data_fact import DataFact, FactCoverage, FactEvent, FactSet
from schemas.output import FinalAnswer
from schemas.timeseries import AnomalyResult, ForecastResult
from schemas.task_contract import TaskContract
from schemas.tool import ReActTranscriptStep, ToolCall, ToolObservation
from schemas.visualization import VisualizationPayload


class RequestStateModel(BaseModel):
    request_id: str
    conversation_id: str | None = None
    conversation_run_dir: str | None = None
    request_log_dir: str | None = None
    message: str
    response_language: Literal["zh", "en"] = "en"
    database_context: DatabaseContext | None = None
    selected_database: str | None = None
    selected_database_type: str | None = None
    time_range: dict | None = None
    constraints: dict = Field(default_factory=dict)
    history: list[Message] = Field(default_factory=list)

    status: Literal["running", "completed", "failed"]
    current_intent: str | None = None
    intent_profile: dict = Field(default_factory=dict)
    requested_capabilities: list[str] = Field(default_factory=list)
    focus: str | None = None
    todo_list: list[dict] = Field(default_factory=list)
    plan_current_step: int = 0
    planning_complete: bool = False
    iteration: int = 0
    max_iterations: int = 4
    context_budget: dict = Field(default_factory=dict)
    context_status: Literal["ok", "summarized", "truncated", "overflowed"] = "ok"
    context_overflow_reason: str | None = None
    completion_state: dict = Field(default_factory=dict)
    task_contract: TaskContract | None = None

    latest_database_evidence: DatabaseEvidence | None = None
    database_evidence_artifacts: dict[str, DatabaseEvidence] = Field(default_factory=dict)
    latest_analysis_id: str | None = None
    analysis_artifacts: dict[str, AnalysisResult] = Field(default_factory=dict)
    latest_forecast: ForecastResult | None = None
    forecast_artifacts: dict[str, ForecastResult] = Field(default_factory=dict)
    latest_anomaly: AnomalyResult | None = None
    anomaly_artifacts: dict[str, AnomalyResult] = Field(default_factory=dict)
    latest_rag: dict | None = None
    latest_skill: dict | None = None
    fact_set: FactSet = Field(default_factory=FactSet)
    fact_coverage: FactCoverage = Field(default_factory=FactCoverage)
    fact_events: list[FactEvent] = Field(default_factory=list)
    final_answer_draft: FinalAnswer | None = None
    visualizations: list[VisualizationPayload] = Field(default_factory=list)

    tool_history: list[ToolCall] = Field(default_factory=list)
    observations: list[ToolObservation] = Field(default_factory=list)
    react_transcript: list[ReActTranscriptStep] = Field(default_factory=list)
    action_outputs: list[ActionOutput] = Field(default_factory=list)
    latest_action_output: ActionOutput | None = None
    memory_fragments: list[dict] = Field(default_factory=list)
    resource_index: dict = Field(default_factory=dict)
    errors: list[dict] = Field(default_factory=list)
    prompt_context_summary: str | None = None


class ConversationStateModel(BaseModel):
    conversation_id: str
    database_context: DatabaseContext | None = None
    recent_messages: list[Message] = Field(default_factory=list)
    session_summary: str | None = None
    intent_profile: dict = Field(default_factory=dict)
    requested_capabilities: list[str] = Field(default_factory=list)
    todo_list: list[dict] = Field(default_factory=list)
    plan_current_step: int = 0
    planning_complete: bool = False
    recent_todo_summary: str | None = None
    latest_database_evidence: DatabaseEvidence | None = None
    database_evidence_artifacts: dict[str, DatabaseEvidence] = Field(default_factory=dict)
    latest_analysis_id: str | None = None
    analysis_artifacts: dict[str, AnalysisResult] = Field(default_factory=dict)
    latest_forecast: ForecastResult | None = None
    forecast_artifacts: dict[str, ForecastResult] = Field(default_factory=dict)
    latest_anomaly: AnomalyResult | None = None
    anomaly_artifacts: dict[str, AnomalyResult] = Field(default_factory=dict)
    latest_rag: dict | None = None
    latest_skill: dict | None = None
    recent_fact_memory: list[DataFact] = Field(default_factory=list)
    fact_memory_summary: str | None = None
    recent_visualizations: list[VisualizationPayload] = Field(default_factory=list)
    updated_at: str | None = None
    context_budget: dict | None = None
    recent_react_transcript: list[ReActTranscriptStep] = Field(default_factory=list)
    task_contract: TaskContract | None = None
