"""Prompt/context builder for the outer agent."""
from __future__ import annotations

import json

from schemas.state import ConversationStateModel, RequestStateModel


class DataAgentPromptBuilder:
    """Build the bounded model-visible context."""

    def build_system_prompt(self) -> str:
        return (
            "You are the single outer data_agent for TSPilot v0.2.\n"
            "Your job is to choose exactly one next action from the current state.\n"
            "Do not pre-commit a full workflow. Do not explain multiple future steps.\n"
            "Observation is runtime-owned and must never be emitted.\n"
            "Respond with exactly one JSON object and nothing else.\n"
            "Output schema: "
            "{\"thought\": str, \"action\": str, \"action_input\": object}.\n"
            "Allowed actions: todowrite, query_database, insight, forecast, anomaly, rag, skill, format_answer.\n"
            "Choose only the next best action for the current state.\n"
            "Decide from the current evidence gap, not from an imagined workflow.\n"
            "Do not emit any non-tool action or follow-up-question action.\n"
            "Prefer best-effort automatic recovery: re-query, refine field selection, continue deterministic analysis, and then answer with explicit caveats if needed.\n"
            "Use todowrite when no useful plan exists yet and the request is multi-step, asks for the execution process, or needs non-trivial analysis such as seasonality.\n"
            "Do not repeat todowrite when a plan already exists and a concrete data or analysis action is available.\n"
            "When a todo plan exists, choose the next action that advances the current in_progress todo step.\n"
            "Do not expand the workflow beyond the current plan unless the runtime observation explicitly shows the plan is no longer workable.\n"
            "Tool contracts:\n"
            "- todowrite: create or update a plan. Use only when planning is missing or genuinely needs restructuring.\n"
            "- query_database: retrieve evidence from the selected datasource. It returns evidence only, not conclusions.\n"
            "- insight: convert existing evidence into verified facts.\n"
            "- anomaly: detect anomalies from existing time-series evidence only.\n"
            "- forecast: generate a short-term forecast from existing time-series evidence only.\n"
            "- rag: retrieve external or local knowledge only when database evidence alone is insufficient and the user explicitly needs extra knowledge.\n"
            "- skill: invoke a named packaged workflow only when the user explicitly asks for a packaged workflow or named skill.\n"
            "- format_answer: assemble the final answer from verified outputs already available in state.\n"
            "Choose tools only from their current-state preconditions. Do not call a tool just because it appears in the user request.\n"
            "Action Input must be valid JSON.\n"
            "Use the exact action-input field names defined here.\n"
            "For todowrite, use: {\"message\": str, \"current_intent\": str|null, "
            "\"requested_fact_types\": list[str], \"focus\": str|null, \"todos\": list[object], "
            "\"evidence_summary\": str|object|null}.\n"
            "Each todo item should use {\"content\": str, \"task_type\": \"plan|query|insight|anomaly|forecast|answer|rag|skill|generic\", \"status\": \"pending|in_progress|completed\", \"priority\": int}.\n"
            "The todo output should represent a compact analysis plan with one current in_progress step and planning_complete=false until the plan is finished.\n"
            "Task types must stay narrow to the user's actual request. Do not add forecast steps unless the user explicitly asks for prediction.\n"
            "For query_database, always include message, database_context, time_range, and constraints when available.\n"
            "Do not output markdown fences."
        )

    def build_context(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> dict:
        return {
            "message": request_state.message,
            "database_context": (
                request_state.database_context.model_dump(mode="json")
                if request_state.database_context
                else None
            ),
            "selected_database": request_state.selected_database,
            "selected_database_type": request_state.selected_database_type,
            "time_range": request_state.time_range,
            "constraints": request_state.constraints,
            "history": [message.model_dump(mode="json") for message in request_state.history],
            "todo_list": request_state.todo_list,
            "plan_current_step": request_state.plan_current_step,
            "planning_complete": request_state.planning_complete,
            "requested_fact_types": request_state.requested_fact_types,
            "focus": request_state.focus,
            "latest_database_evidence": (
                self._summarize_database_evidence(request_state.latest_database_evidence)
                if request_state.latest_database_evidence
                else None
            ),
            "latest_insight": (
                self._summarize_insight(request_state.latest_insight)
                if request_state.latest_insight
                else None
            ),
            "latest_forecast": (
                self._summarize_forecast(request_state.latest_forecast)
                if request_state.latest_forecast
                else None
            ),
            "latest_anomaly": (
                self._summarize_anomaly(request_state.latest_anomaly)
                if request_state.latest_anomaly
                else None
            ),
            "latest_rag": request_state.latest_rag,
            "latest_skill": request_state.latest_skill,
            "verified_facts": [
                fact.model_dump(mode="json")
                for fact in request_state.verified_facts
            ],
            "visualizations": [
                self._summarize_visualization(visualization)
                for visualization in request_state.visualizations
            ],
            "latest_observation_summaries": [
                {
                    "tool_name": observation.tool_name,
                    "success": observation.success,
                    "summary": observation.summary,
                }
                for observation in request_state.observations[-4:]
            ],
            "recent_messages": [message.model_dump(mode="json") for message in conversation_state.recent_messages],
            "recent_todo_summary": conversation_state.recent_todo_summary,
            "prompt_context_summary": request_state.prompt_context_summary,
        }

    def build_user_prompt(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> str:
        context = self.build_context(request_state, conversation_state)
        return "Context JSON:\n" + json.dumps(context, ensure_ascii=False, indent=2)

    def _summarize_database_evidence(self, evidence) -> dict:
        payload = evidence.model_dump(mode="json")
        data = dict(payload.get("data") or {})
        diagnostics = dict(payload.get("diagnostics") or {})
        summary_stats = diagnostics.get("summary_stats") or {}
        if isinstance(data.get("points"), list):
            data["points"] = data["points"][:8]
        if isinstance(data.get("rows"), list):
            data["rows"] = data["rows"][:4]
        if isinstance(data.get("series"), list):
            series_preview = []
            for item in data["series"][:3]:
                item_copy = dict(item)
                if isinstance(item_copy.get("points"), list):
                    item_copy["points"] = item_copy["points"][:4]
                series_preview.append(item_copy)
            data["series"] = series_preview
        payload["data"] = data
        payload["diagnostics"] = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "summary_stats", "query_trace", "series_count"}
        }
        if summary_stats:
            payload["summary_stats"] = summary_stats
        return payload

    def _summarize_insight(self, insight) -> dict:
        payload = insight.model_dump(mode="json")
        payload["fact_candidates"] = payload.get("fact_candidates", [])[:8]
        payload["completed_facts"] = [
            {
                "fact_id": fact.get("fact_id"),
                "fact_type": fact.get("fact_type"),
                "statement": fact.get("statement"),
                "focus": fact.get("focus"),
            }
            for fact in payload.get("completed_facts", [])[:6]
        ]
        payload["verified_facts"] = [
            {
                "fact_id": fact.get("fact_id"),
                "fact_type": fact.get("fact_type"),
                "statement": fact.get("statement"),
                "confidence": fact.get("confidence"),
            }
            for fact in payload.get("verified_facts", [])[:6]
        ]
        payload["rejected_facts"] = [
            {
                "fact_id": fact.get("fact_id"),
                "fact_type": fact.get("fact_type"),
                "reason": fact.get("reason"),
            }
            for fact in payload.get("rejected_facts", [])[:6]
        ]
        payload["visualizations"] = [
            self._summarize_visualization_from_dict(item)
            for item in payload.get("visualizations", [])[:4]
        ]
        diagnostics = dict(payload.get("diagnostics") or {})
        payload["diagnostics"] = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "snapshot_ref"}
        }
        return payload

    def _summarize_forecast(self, forecast) -> dict:
        payload = forecast.model_dump(mode="json")
        payload["forecast_points"] = payload.get("forecast_points", [])[:6]
        payload["visualizations"] = [
            self._summarize_visualization_from_dict(item)
            for item in payload.get("visualizations", [])[:3]
        ]
        diagnostics = dict(payload.get("diagnostics") or {})
        payload["diagnostics"] = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "snapshot_ref", "series_name"}
        }
        return payload

    def _summarize_anomaly(self, anomaly) -> dict:
        payload = anomaly.model_dump(mode="json")
        payload["anomaly_points"] = payload.get("anomaly_points", [])[:8]
        payload["scores"] = payload.get("scores", [])[:8]
        payload["visualizations"] = [
            self._summarize_visualization_from_dict(item)
            for item in payload.get("visualizations", [])[:3]
        ]
        diagnostics = dict(payload.get("diagnostics") or {})
        payload["diagnostics"] = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "snapshot_ref", "threshold"}
        }
        return payload

    def _summarize_visualization(self, visualization) -> dict:
        return self._summarize_visualization_from_dict(visualization.model_dump(mode="json"))

    def _summarize_visualization_from_dict(self, payload: dict) -> dict:
        return {
            "visualization_id": payload.get("visualization_id"),
            "visualization_type": payload.get("visualization_type"),
            "visualization_kind": payload.get("visualization_kind"),
            "renderer": payload.get("renderer"),
            "title": payload.get("title"),
            "summary": payload.get("summary"),
            "binding_fact_ids": payload.get("binding_fact_ids", [])[:6],
            "binding_evidence_ids": payload.get("binding_evidence_ids", [])[:6],
            "requested_fact_types": payload.get("requested_fact_types", [])[:6],
            "time_column": payload.get("time_column"),
            "primary_measure": payload.get("primary_measure"),
            "display_priority": payload.get("display_priority"),
            "row_count": payload.get("row_count"),
            "columns": payload.get("columns", [])[:8],
            "annotations_count": len(payload.get("annotations", []) or []),
            "rows_count": len(payload.get("rows", []) or []),
            "display_rows_count": len(payload.get("display_rows", []) or []),
            "chart_summary": self._summarize_chart(payload.get("chart")),
        }

    def _summarize_chart(self, chart: dict | None) -> dict | None:
        if not isinstance(chart, dict):
            return None
        if "x_axis_count" in chart and "x_axis_data" not in chart:
            return chart
        x_axis_data = list(chart.get("x_axis_data") or [])
        series_data = list(chart.get("series_data") or [])
        return {
            "x_axis_count": len(x_axis_data),
            "x_axis_preview": [*x_axis_data[:3], *x_axis_data[-3:]] if len(x_axis_data) > 6 else x_axis_data,
            "series": [
                {
                    "name": item.get("name"),
                    "points_count": len(item.get("data") or []),
                }
                for item in series_data[:4]
                if isinstance(item, dict)
            ],
        }
