"""Conversation-state helpers."""
from __future__ import annotations

from datetime import datetime, timezone

from schemas.state import ConversationStateModel, RequestStateModel


def sync_from_request(
    request_state: RequestStateModel,
    conversation_state: ConversationStateModel,
) -> None:
    conversation_state.database_context = request_state.database_context
    conversation_state.recent_messages = request_state.history[-10:]
    conversation_state.todo_list = request_state.todo_list
    conversation_state.plan_current_step = request_state.plan_current_step
    conversation_state.planning_complete = request_state.planning_complete
    conversation_state.recent_todo_summary = request_state.todo_list[0]["content"] if request_state.todo_list else None
    conversation_state.latest_database_evidence = request_state.latest_database_evidence
    conversation_state.database_evidence_artifacts = request_state.database_evidence_artifacts
    conversation_state.latest_insight = request_state.latest_insight
    conversation_state.insight_artifacts = request_state.insight_artifacts
    conversation_state.latest_forecast = request_state.latest_forecast
    conversation_state.forecast_artifacts = request_state.forecast_artifacts
    conversation_state.latest_anomaly = request_state.latest_anomaly
    conversation_state.anomaly_artifacts = request_state.anomaly_artifacts
    conversation_state.latest_rag = request_state.latest_rag
    conversation_state.latest_skill = request_state.latest_skill
    conversation_state.recent_visualizations = request_state.visualizations[-6:]
    conversation_state.updated_at = datetime.now(timezone.utc).isoformat()
    conversation_state.context_budget = request_state.context_budget
