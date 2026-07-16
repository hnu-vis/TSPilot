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
            "Never emit a second JSON object. Never emit a bare tool input. Never emit markdown, prose, or trailing text.\n"
            "Output schema: "
            "{\"thought\": str, \"action\": str, \"action_input\": object}.\n"
            "Allowed actions: todowrite, query_database, insight, forecast, anomaly, rag, skill, format_answer.\n"
            "Choose only the next best action for the current state.\n"
            "Decide from the current evidence gap, not from an imagined workflow.\n"
            "Do not emit any non-tool action or follow-up-question action.\n"
            "Prefer best-effort automatic recovery: re-query, refine field selection, continue deterministic analysis, and then answer with explicit caveats if needed.\n"
            "Use todowrite when no useful plan exists yet and the request is multi-step, asks for the execution process, or needs non-trivial analysis such as seasonality.\n"
            "Use todowrite again whenever you need to update todo statuses after a successful step, move the in_progress step, or revise a stale plan.\n"
            "When a todo plan exists and a non-todowrite tool succeeds, prefer an explicit todowrite update before continuing if the todo status is now stale.\n"
            "Todo is model-maintained process state for visibility, not a runtime scheduler. You may revise or skip stale todo items when evidence shows the plan should change.\n"
            "Tool contracts:\n"
            "- todowrite: create or update the full todo plan and statuses. Use for initial planning and explicit progress updates.\n"
            "- query_database: retrieve evidence from the selected datasource. It returns evidence only, not conclusions.\n"
            "- insight: convert existing evidence into verified facts.\n"
            "- anomaly: detect anomalies from existing time-series evidence only.\n"
            "- forecast: generate a short-term forecast from existing time-series evidence only.\n"
            "- rag: retrieve external or local knowledge only when database evidence alone is insufficient and the user explicitly needs extra knowledge.\n"
            "- skill: invoke a named packaged workflow only when the user explicitly asks for a packaged workflow or named skill.\n"
            "- format_answer: assemble the final answer from verified outputs already available in state. "
            "When no datasource is selected and the user asks a greeting, capability question, clarification, or other non-data question, use format_answer directly and provide a concise direct_answer. "
            "When the user asks a data-analysis question without a datasource, use format_answer to explain that a database/context is needed and suggest selecting one.\n"
            "Choose tools only from their current-state preconditions. Do not call a tool just because it appears in the user request.\n"
            "Action Input must be valid JSON.\n"
            "Use the exact action-input field names defined here.\n"
            "For todowrite, use: {\"message\": str, \"current_intent\": str|null, "
            "\"requested_fact_types\": list[str], \"focus\": str|null, \"todos\": list[object], "
            "\"evidence_summary\": str|object|null}.\n"
            "Each todo item should use {\"content\": str, \"task_type\": \"plan|query|insight|anomaly|forecast|answer|rag|skill|generic\", \"status\": \"pending|in_progress|completed\", \"priority\": int}.\n"
            "The todo output should include the complete latest todo list, not only a delta. Keep at most one in_progress step unless all steps are completed.\n"
            "Task types must stay narrow to the user's actual request. Do not add forecast steps unless the user explicitly asks for prediction.\n"
            "For query_database, always include message, database_context, time_range, and constraints when available.\n"
            "For format_answer, use: {\"summary_goal\": str, \"direct_answer\": str|null, "
            "\"include_fact_ids\": list[str], \"include_visualization_ids\": list[str], \"section_plan\": list[str]}.\n"
            "Do not output markdown fences."
        )

    def build_context(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> dict:
        return {
            "message": request_state.message,
            "available_actions": self._available_actions(),
            "execution_state": self._execution_state(request_state),
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
                    "error": observation.error,
                    "payload": self._summarize_observation_payload(observation.payload),
                }
                for observation in request_state.observations[-4:]
            ],
            "recent_messages": [message.model_dump(mode="json") for message in conversation_state.recent_messages],
            "recent_todo_summary": conversation_state.recent_todo_summary,
            "prompt_context_summary": request_state.prompt_context_summary,
        }

    def _available_actions(self) -> list[dict]:
        return [
            {
                "action": "todowrite",
                "use_when": "Create the plan, update completed/in_progress/pending statuses, or revise stale plan state.",
                "input": "message, current_intent, requested_fact_types, focus, todos, evidence_summary",
            },
            {
                "action": "query_database",
                "use_when": "Database evidence is missing or the existing evidence does not cover the user's data need.",
                "input": "message, database_context, time_range, constraints",
            },
            {
                "action": "insight",
                "use_when": "Evidence is available and the user needs grounded facts such as trend, seasonality, extrema, distribution, or outliers.",
                "input": "database_evidence, requested_fact_types, focus, constraints",
            },
            {
                "action": "anomaly",
                "use_when": "Time-series evidence is available and the user specifically needs anomaly/spike/outlier detection.",
                "input": "database_evidence, constraints",
            },
            {
                "action": "forecast",
                "use_when": "Time-series evidence is available and the user specifically asks for prediction or forecast.",
                "input": "database_evidence, horizon, constraints",
            },
            {
                "action": "rag",
                "use_when": "The user explicitly needs external or knowledge-base retrieval beyond database evidence.",
                "input": "query, filters",
            },
            {
                "action": "skill",
                "use_when": "The user explicitly asks for a named packaged workflow or skill.",
                "input": "skill_name, parameters",
            },
            {
                "action": "format_answer",
                "use_when": "Enough evidence-backed outputs are available, or the request is conversational / cannot proceed without more context.",
                "input": "summary_goal, direct_answer, include_fact_ids, include_visualization_ids, section_plan",
            },
        ]

    def _execution_state(self, request_state: RequestStateModel) -> dict:
        last_success = next((item for item in reversed(request_state.observations) if item.success), None)
        last_failure = next((item for item in reversed(request_state.observations) if not item.success), None)
        last_tool = request_state.tool_history[-1].tool_name if request_state.tool_history else None
        return {
            "iteration": request_state.iteration,
            "max_iterations": request_state.max_iterations,
            "tool_sequence": [call.tool_name for call in request_state.tool_history],
            "last_tool": last_tool,
            "last_successful_tool": last_success.tool_name if last_success else None,
            "last_failure": (
                {
                    "tool_name": last_failure.tool_name,
                    "summary": last_failure.summary,
                    "error": last_failure.error,
                }
                if last_failure
                else None
            ),
            "artifacts": {
                "has_database_evidence": request_state.latest_database_evidence is not None,
                "has_insight": request_state.latest_insight is not None,
                "has_forecast": request_state.latest_forecast is not None,
                "has_anomaly": request_state.latest_anomaly is not None,
                "has_final_answer": request_state.final_answer_draft is not None,
                "verified_fact_count": len(request_state.verified_facts),
                "visualization_count": len(request_state.visualizations),
            },
            "todo_update_suggested": bool(
                request_state.todo_list
                and last_success is not None
                and last_success.tool_name != "todowrite"
            ),
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

    def _summarize_observation_payload(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}
        summarized = {}
        for key in (
            "recovery_hint",
            "error",
            "evidence_id",
            "insight_id",
            "forecast_id",
            "anomaly_id",
            "summary",
            "current_step",
            "planning_complete",
            "completed_count",
            "pending_count",
        ):
            if key in payload:
                summarized[key] = payload[key]
        if isinstance(payload.get("todos"), list):
            summarized["todos"] = payload["todos"][:8]
        if isinstance(payload.get("verified_facts"), list):
            summarized["verified_facts"] = [
                {
                    "fact_id": fact.get("fact_id"),
                    "fact_type": fact.get("fact_type"),
                    "statement": fact.get("statement"),
                }
                for fact in payload["verified_facts"][:6]
                if isinstance(fact, dict)
            ]
        if isinstance(payload.get("valid_actions"), list):
            summarized["valid_actions"] = payload["valid_actions"]
        return summarized

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
