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
            "{\"thought\": str, \"action_intention\": str|null, \"action_reason\": str|null, \"action\": str, \"action_input\": object}.\n"
            "action_intention must name only this step's concrete purpose, <= 18 Chinese characters or <= 8 English words. "
            "action_reason must briefly explain why this step is needed now, <= 30 Chinese characters or <= 12 English words.\n"
            "Allowed actions: todowrite, sql_query, insight, forecast, anomaly, rag, skill, terminate.\n"
            "Choose only the next best action for the current state.\n"
            "Decide from the current evidence gap, not from an imagined workflow.\n"
            "For tasks with an explicit numbered or bulleted deliverable list, verify each user-visible deliverable against observations before terminate. "
            "If a deliverable asks for its own query text, row count, raw records, extrema, bounds, or validation result and that item is not present in observations, call the appropriate tool instead of answering with a caveat.\n"
            "The context is grouped as task, state, evidence, outputs, recent_observations, and available_actions. "
            "Use the user's message, current plan, and observations as the source of task intent.\n"
            "Context is budgeted: evidence previews and recent observations may be sampled or summarized. "
            "Use diagnostics.prompt_sampling, summary_stats, data_completeness, artifact_ref, and query text to decide whether the visible preview is complete. "
            "Use diagnostics.task_coverage from sql_query observations to decide whether the current query already covers the user request; if missing_or_uncertain is non-empty, prefer another focused sql_query before terminate. "
            "When a task needs facts not present in the prompt preview, call sql_query or insight over the full artifact instead of guessing from the preview.\n"
            "Do not emit any non-tool action or follow-up-question action.\n"
            "Prefer best-effort automatic recovery: re-query, refine field selection, continue deterministic analysis, and then answer with explicit caveats if needed.\n"
            "Use todowrite when no useful plan exists yet and the request is multi-step, asks for the execution process, or needs non-trivial analysis such as seasonality.\n"
            "Do not call todowrite when a todo plan already exists. Runtime owns plan progress and advances todo statuses after successful actions.\n"
            "Todo is initial visible process state, not a model-maintained scheduler or tool contract. After each action, judge whether the current task has enough evidence to answer; if not, choose the next non-todowrite action.\n"
            "When a todo step is in_progress, treat its task_type as a progress label only. Choose the next action from the ReAct state, observations, and evidence gaps, not by mechanically matching the label.\n"
            "Tool contracts:\n"
            "- todowrite: create the initial full todo plan only when no plan exists. Plan user-visible deliverables, not internal tool stages. Do not create todo items for field confirmation, query generation, or query planning; those are internal to sql_query.\n"
            "- sql_query: query the selected datasource and return database evidence, including the actual backend query text when available. "
            "Use automatic message-based input for normal database requests. Use explicit query/query_language only when repairing a failed generated query, or when the user supplied an exact query. "
            "It is the primary database-analysis action for raw pulls, exact aggregates, grouping, ranking, bucketing, period checks, and validation queries. "
            "It may be called repeatedly when the last observation reveals missing filters, suspicious outliers, or another grounded SQL check is needed.\n"
            "- insight: execute generated Python analysis code over existing full evidence artifacts. Use it only when SQL cannot express the needed computation well, or when structured fact extraction/visualization is needed after enough database queries. It returns structured analysis results, not raw rows.\n"
            "- anomaly: detect anomalies from existing time-series evidence only.\n"
            "- forecast: generate a short-term forecast from existing time-series evidence only.\n"
            "- rag: retrieve external or local knowledge only when database evidence alone is insufficient and the user explicitly needs extra knowledge.\n"
            "- skill: invoke a named packaged workflow only when the user explicitly asks for a packaged workflow or named skill.\n"
            "- terminate: end the ReAct loop when verified outputs already answer the task, and provide the final response payload. "
            "Do not terminate with phrases like 'if needed, continue' for an explicitly requested deliverable; perform the needed next action first. "
            "When no datasource is selected and the user asks a greeting, capability question, clarification, or other non-data question, use terminate directly and provide a concise result/direct_answer. "
            "When the user asks a data-analysis question without a datasource, use terminate to explain that a database/context is needed and suggest selecting one.\n"
            "Choose tools only from their current-state preconditions. Do not call a tool just because it appears in the user request.\n"
            "Action Input must be valid JSON.\n"
            "Use the exact action-input field names defined here.\n"
            "For todowrite, use: {\"message\": str, \"current_intent\": str|null, "
            "\"requested_fact_types\": list[str], \"focus\": str|null, \"todos\": list[object], "
            "\"evidence_summary\": str|object|null}.\n"
            "Each todo item should use {\"content\": str, \"task_type\": \"query|insight|anomaly|forecast|answer|rag|skill|generic\", \"status\": \"pending|in_progress|completed\", \"priority\": int, \"acceptance_criteria\": str|null}.\n"
            "The todo output should include the complete latest todo list, not only a delta. Keep at most one in_progress step unless all steps are completed.\n"
            "Task types must stay narrow to the user's actual request. Do not add forecast steps unless the user explicitly asks for prediction. For database plans, split by requested result: count, earliest/latest rows, grouped results, time bounds, comparisons, and final answer. Do not split by internal preparation stages.\n"
            "For sql_query automatic planning, use: {\"message\": str, \"database_context\": object, \"time_range\": object|null, \"constraints\": object}. This is the normal path for database queries.\n"
            "For sql_query explicit analysis, use: {\"database_context\": object, \"query\": str, \"query_language\": str|null, \"purpose\": str|null, \"constraints\": object}. Only write read-only SELECT/WITH SQL, Flux without output/write functions, or read-only backend query language.\n"
            "After a sql_query observation, inspect its query, columns, counts, and sample rows/points. If the sample shows wrong entity filters, mixed units/categories, suspicious extreme values, or insufficient aggregation, issue another explicit sql_query that corrects or validates the data. "
            "If diagnostics.task_coverage.requires_followup is true or missing_or_uncertain lists requested facts, continue with a focused sql_query for those facts instead of answering. "
            "For data questions requiring exact aggregation, grouping, ranking, median/quantile, period checks, validation of anomalies, threshold proportions, or comparison across categories, prefer an explicit sql_query when the database can compute it directly. "
            "Use insight after the SQL evidence is sufficiently grounded, or when the computation requires Python over full artifacts. Do not calculate final facts from prompt previews alone.\n"
            "For insight, use: {\"database_evidence\": str|object|null, \"analysis_goal\": str, \"code_type\": \"python_rows_v1\", \"analysis_code\": str, \"expected_result_schema\": object|null, \"constraints\": object}. The generated code may use rows, points, columns, metadata, diagnostics, math, and statistics. It must assign result={\"summary\": str, \"metrics\": object, \"details\": object}.\n"
            "For terminate, use: {\"result\": str|null, \"summary_goal\": str|null, \"direct_answer\": str|null, "
            "\"include_analysis_ids\": list[str], \"include_fact_ids\": list[str], \"include_visualization_ids\": list[str], \"section_plan\": list[str]}.\n"
            "Do not output markdown fences."
        )

    def build_context(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> dict:
        latest_evidence = (
            self._summarize_database_evidence(request_state.latest_database_evidence)
            if request_state.latest_database_evidence
            else None
        )
        latest_evidence_id = latest_evidence.get("evidence_id") if isinstance(latest_evidence, dict) else None
        prior_queries = [
            item
            for item in self._summarize_query_history(request_state)
            if item.get("evidence_id") != latest_evidence_id
        ]
        return {
            "task": {
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
                "history": [message.model_dump(mode="json") for message in request_state.history[-4:]],
            },
            "state": {
                "execution": self._execution_state(request_state),
                "decision_frame": self._decision_frame(request_state),
                "todo_list": request_state.todo_list,
                "plan_current_step": request_state.plan_current_step,
                "planning_complete": request_state.planning_complete,
                "requested_fact_types": request_state.requested_fact_types,
                "focus": request_state.focus,
                "recent_todo_summary": conversation_state.recent_todo_summary,
                "prompt_context_summary": request_state.prompt_context_summary,
            },
            "evidence": {
                "latest": latest_evidence,
                "prior_queries": prior_queries[-5:],
            },
            "outputs": {
                "latest_insight": (
                    self._summarize_insight_status(request_state.latest_insight)
                    if request_state.latest_insight
                    else None
                ),
                "analysis_workspace": self._analysis_workspace(request_state),
                "latest_forecast": (
                    self._summarize_forecast_status(request_state.latest_forecast)
                    if request_state.latest_forecast
                    else None
                ),
                "latest_anomaly": (
                    self._summarize_anomaly_status(request_state.latest_anomaly)
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
            },
            "recent_observations": [
                {
                    "tool_name": observation.tool_name,
                    "success": observation.success,
                    "summary": observation.summary,
                    "error": observation.error,
                    "payload": self._summarize_observation_payload(observation.payload),
                }
                for observation in request_state.observations[-4:]
            ],
            "available_actions": self._available_actions(),
        }

    def _available_actions(self) -> list[dict]:
        return [
            {
                "action": "todowrite",
                "use_when": "Create the initial plan only when todo_list is empty and the request needs visible multi-step analysis.",
                "input": "message, current_intent, requested_fact_types, focus, todos, evidence_summary; todos may include acceptance_criteria",
            },
            {
                "action": "sql_query",
                "use_when": "Database evidence is missing, or a read-only follow-up query can compute the exact aggregation, grouping, ranking, filter validation, or diagnostic needed to answer correctly.",
                "input": "prefer message, database_context, time_range, constraints for automatic database querying; use query/query_language only for repair or for a user-supplied exact query",
            },
            {
                "action": "insight",
                "use_when": "Evidence is available and the user needs grounded facts such as trend, seasonality, extrema, distribution, or outliers.",
                "input": "database_evidence, analysis_goal, code_type, analysis_code, expected_result_schema, constraints",
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
                "action": "terminate",
                "use_when": "Enough evidence-backed outputs are available, or the request is conversational / cannot proceed without more context.",
                "input": "summary_goal, direct_answer, include_analysis_ids, include_fact_ids, include_visualization_ids, section_plan",
            },
        ]

    def _execution_state(self, request_state: RequestStateModel) -> dict:
        last_success = next((item for item in reversed(request_state.observations) if item.success), None)
        last_failure = next((item for item in reversed(request_state.observations) if not item.success), None)
        last_call = request_state.tool_history[-1] if request_state.tool_history else None
        last_tool = last_call.tool_name if last_call else None
        return {
            "iteration": request_state.iteration,
            "max_iterations": request_state.max_iterations,
            "tool_sequence": [call.tool_name for call in request_state.tool_history],
            "last_tool": last_tool,
            "last_action_reason": last_call.reason if last_call else None,
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
                "has_analysis": bool(request_state.analysis_artifacts),
                "has_forecast": request_state.latest_forecast is not None,
                "has_anomaly": request_state.latest_anomaly is not None,
                "has_final_answer": request_state.final_answer_draft is not None,
                "verified_fact_count": len(request_state.verified_facts),
                "visualization_count": len(request_state.visualizations),
                "analysis_count": len(request_state.analysis_artifacts),
            },
            "plan_progress_owner": "runtime" if request_state.todo_list else "none",
        }

    def _decision_frame(self, request_state: RequestStateModel) -> dict:
        current_todo = next(
            (todo for todo in request_state.todo_list if todo.get("status") == "in_progress"),
            None,
        )
        latest_evidence = request_state.latest_database_evidence
        latest_query_summary = None
        if latest_evidence is not None:
            diagnostics = latest_evidence.diagnostics or {}
            summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
            latest_query_summary = {
                "evidence_id": latest_evidence.evidence_id,
                "result_type": latest_evidence.result_type,
                "query_language": latest_evidence.query_language,
                "columns": latest_evidence.columns[:20],
                "rows_count": summary_stats.get("rows_count"),
                "points_count": summary_stats.get("points_count"),
                "series_count": summary_stats.get("series_count"),
                "sample_rows_visible": isinstance((latest_evidence.data or {}).get("rows"), list),
                "sample_points_visible": isinstance((latest_evidence.data or {}).get("points"), list),
            }

        next_action_guidance = []
        active_progress = self._active_progress_hint(current_todo)
        if request_state.database_context is None:
            next_action_guidance.append("terminate: explain that a datasource/context is required for data analysis.")
        elif latest_evidence is None:
            next_action_guidance.append("sql_query: obtain database evidence before analysis.")
        else:
            next_action_guidance.extend(
                [
                    "sql_query: continue with explicit read-only queries for additional aggregation, filtering, bucketing, or period validation.",
                    "insight: use Python over full evidence only when SQL is insufficient or structured facts/visualizations are now needed.",
                    "terminate: use only when enough database-backed evidence or derived outputs answer the user request.",
                ]
            )
        if request_state.todo_list:
            next_action_guidance.insert(0, "todowrite is locked after initial planning; runtime advances plan status.")

        return {
            "current_todo": current_todo,
            "completion_state": request_state.completion_state,
            "latest_query_summary": latest_query_summary,
            "active_progress_hint": active_progress,
            "recommended_next_action_types": next_action_guidance,
            "sql_loop_rule": (
                "A prior sql_query does not force insight. If the last SQL sample/counts reveal a missing filter, outlier, "
                "or another database-checkable gap, call sql_query again with a focused read-only query."
            ),
            "context_budget_rule": (
                "Prompt context contains bounded previews only. Use prompt_sampling/data_completeness and artifact refs to distinguish complete results from previews; "
                "run a tool when full-data computation is needed."
            ),
        }

    def _active_progress_hint(self, current_todo: dict | None) -> dict:
        if current_todo is None:
            return {
                "task_type": None,
                "suggested_action": None,
            }
        task_type = str(current_todo.get("task_type") or "").strip().lower()
        suggested_action = {
            "query": "sql_query",
            "insight": "insight",
            "anomaly": "anomaly",
            "forecast": "forecast",
            "answer": "terminate",
            "rag": "rag",
            "skill": "skill",
        }.get(task_type)
        return {
            "task_type": task_type or None,
            "suggested_action": suggested_action,
        }

    def build_user_prompt(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> str:
        context = self.build_context(request_state, conversation_state)
        return "\n".join(
            [
                "User Task:",
                json.dumps(context["task"], ensure_ascii=False, indent=2),
                "",
                "Available Tools:",
                json.dumps(context["available_actions"], ensure_ascii=False, indent=2),
                "",
                "Previous Thought/Action/Observation:",
                self._react_transcript(request_state),
                "",
                "Runtime State JSON:",
                json.dumps(context, ensure_ascii=False, indent=2),
            ]
        )

    def _react_transcript(self, request_state: RequestStateModel) -> str:
        if not request_state.tool_history and not request_state.observations:
            return "(none)"
        lines = []
        calls_by_iteration = {call.iteration: call for call in request_state.tool_history}
        observations_by_iteration: dict[int, list] = {}
        for observation in request_state.observations:
            iteration = self._observation_iteration(observation, calls_by_iteration)
            observations_by_iteration.setdefault(iteration, []).append(observation)
        for iteration in sorted(set(calls_by_iteration) | set(observations_by_iteration)):
            call = calls_by_iteration.get(iteration)
            if call is not None:
                if call.reason:
                    lines.append(f"Thought: {self._truncate_text(call.reason, 800)}")
                lines.append(f"Action: {call.tool_name}")
                lines.append(
                    "Action Input: "
                    + self._truncate_text(json.dumps(call.tool_input, ensure_ascii=False), 2000)
                )
            for observation in observations_by_iteration.get(iteration, []):
                lines.append(
                    "Observation: "
                    + self._truncate_text(
                        json.dumps(
                            {
                                "tool_name": observation.tool_name,
                                "success": observation.success,
                                "summary": observation.summary,
                                "error": observation.error,
                                "payload": self._summarize_observation_payload(observation.payload),
                            },
                            ensure_ascii=False,
                        ),
                        3000,
                    )
                )
        return "\n".join(lines) if lines else "(none)"

    def _observation_iteration(self, observation, calls_by_iteration: dict[int, object]) -> int:
        for iteration in sorted(calls_by_iteration, reverse=True):
            if calls_by_iteration[iteration].tool_name == observation.tool_name:
                return iteration
        return max(calls_by_iteration.keys(), default=0)

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
        payload["query"] = self._truncate_text(payload.get("query"), 4000)
        payload["summary"] = self._truncate_text(payload.get("summary"), 1200)
        payload["metadata"] = self._bounded_value(payload.get("metadata") or {}, max_string_chars=600, max_list_items=8, max_dict_items=16)
        visible_diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "summary_stats", "prompt_sampling", "query_trace", "series_count", "task_coverage"}
        }
        visible_diagnostics["prompt_sampling"] = self._prompt_sampling(
            full_counts=summary_stats,
            data=data,
            fallback=visible_diagnostics.get("prompt_sampling") if isinstance(visible_diagnostics.get("prompt_sampling"), dict) else None,
        )
        if "query_trace" in visible_diagnostics:
            visible_diagnostics["query_trace"] = self._bounded_value(
                visible_diagnostics["query_trace"],
                max_string_chars=1200,
                max_list_items=6,
                max_dict_items=16,
            )
        payload["diagnostics"] = visible_diagnostics
        if summary_stats:
            payload["summary_stats"] = summary_stats
        return payload

    def _summarize_query_history(self, request_state: RequestStateModel) -> list[dict]:
        """Expose prior database queries as stable model-visible context."""

        history = []
        for evidence in request_state.database_evidence_artifacts.values():
            item = self._summarize_database_evidence(evidence)
            data = item.get("data") if isinstance(item.get("data"), dict) else {}
            diagnostics = item.get("diagnostics") if isinstance(item.get("diagnostics"), dict) else {}
            summary_stats = diagnostics.get("summary_stats") or item.get("summary_stats") or {}
            row_count = summary_stats.get("rows_count")
            point_count = summary_stats.get("points_count")
            series_count = summary_stats.get("series_count")
            if row_count is None and isinstance(data.get("rows"), list):
                row_count = len(evidence.data.get("rows", [])) if isinstance(evidence.data, dict) else len(data["rows"])
            if point_count is None and isinstance(data.get("points"), list):
                point_count = len(evidence.data.get("points", [])) if isinstance(evidence.data, dict) else len(data["points"])
            if series_count is None and isinstance(data.get("series"), list):
                series_count = len(evidence.data.get("series", [])) if isinstance(evidence.data, dict) else len(data["series"])
            preview = {}
            if isinstance(data.get("rows"), list):
                preview["rows"] = data["rows"][:3]
            if isinstance(data.get("points"), list):
                preview["points"] = data["points"][:4]
            if isinstance(data.get("series"), list):
                preview["series"] = data["series"][:2]
            history.append(
                {
                    "evidence_id": item.get("evidence_id"),
                    "database": item.get("database"),
                    "query_language": item.get("query_language"),
                    "query": self._truncate_text(item.get("query"), 2500),
                    "summary": self._truncate_text(item.get("summary"), 800),
                    "result_type": item.get("result_type"),
                    "columns": (item.get("columns") or [])[:20],
                    "row_count": row_count,
                    "point_count": point_count,
                    "series_count": series_count,
                    "task_coverage": self._bounded_value(
                        diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else {},
                        max_string_chars=800,
                        max_list_items=8,
                        max_dict_items=12,
                    ),
                    "metadata": self._bounded_value(
                        item.get("metadata") or {},
                        max_string_chars=400,
                        max_list_items=6,
                        max_dict_items=12,
                    ),
                    "preview": preview,
                }
            )
        return history[-6:]

    def _analysis_workspace(self, request_state: RequestStateModel) -> dict:
        analyses = []
        for analysis in request_state.analysis_artifacts.values():
            payload = analysis.model_dump(mode="json")
            analyses.append(
                {
                    "analysis_id": payload.get("analysis_id"),
                    "goal": payload.get("analysis_goal"),
                    "summary": payload.get("summary"),
                    "status": payload.get("status"),
                    "code_type": payload.get("code_type"),
                    "code_hash": payload.get("code_hash"),
                    "input_ref": f"evidence:{payload.get('input_evidence_id')}",
                    "input_row_count": payload.get("input_row_count"),
                    "result_preview": self._bounded_value(
                        payload.get("result") or {},
                        max_string_chars=1000,
                        max_list_items=8,
                        max_dict_items=16,
                    ),
                }
            )
        return {
            "latest_analysis_id": request_state.latest_analysis_id,
            "analysis_count": len(analyses),
            "analyses": analyses[-8:],
        }

    def _summarize_insight_status(self, insight) -> dict:
        payload = insight.model_dump(mode="json")
        return {
            "insight_id": payload.get("insight_id"),
            "summary": self._truncate_text(payload.get("summary"), 800),
            "verified_fact_count": len(payload.get("verified_facts", []) or []),
            "completed_fact_count": len(payload.get("completed_facts", []) or []),
            "visualization_count": len(payload.get("visualizations", []) or []),
        }

    def _summarize_forecast_status(self, forecast) -> dict:
        payload = forecast.model_dump(mode="json")
        return {
            "forecast_id": payload.get("forecast_id"),
            "summary": self._truncate_text(payload.get("summary"), 800),
            "forecast_point_count": len(payload.get("forecast_points", []) or []),
            "visualization_count": len(payload.get("visualizations", []) or []),
        }

    def _summarize_anomaly_status(self, anomaly) -> dict:
        payload = anomaly.model_dump(mode="json")
        return {
            "anomaly_id": payload.get("anomaly_id"),
            "summary": self._truncate_text(payload.get("summary"), 800),
            "anomaly_point_count": len(payload.get("anomaly_points", []) or []),
            "visualization_count": len(payload.get("visualizations", []) or []),
        }

    def _summarize_observation_payload(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            return {}
        summarized = {}
        preview = {}
        for key in (
            "recovery_hint",
            "error",
            "evidence_id",
            "analysis_id",
            "analysis_goal",
            "code_type",
            "code_hash",
            "input_evidence_id",
            "input_row_count",
            "insight_id",
            "forecast_id",
            "anomaly_id",
            "summary",
            "current_step",
            "planning_complete",
            "completed_count",
            "pending_count",
            "query",
            "query_language",
            "columns",
            "metadata",
        ):
            if key in payload:
                summarized[key] = self._bounded_value(payload[key], max_string_chars=1200, max_list_items=12, max_dict_items=16)
        if isinstance(payload.get("data"), dict):
            data = dict(payload["data"])
            if isinstance(data.get("rows"), list):
                preview["rows"] = data["rows"][:3]
            if isinstance(data.get("points"), list):
                preview["points"] = data["points"][:4]
            if isinstance(data.get("series"), list):
                preview["series"] = data["series"][:2]
            if preview:
                summarized["data_preview"] = preview
        if isinstance(payload.get("result"), dict):
            summarized["result_preview"] = self._bounded_value(
                payload["result"],
                max_string_chars=1000,
                max_list_items=8,
                max_dict_items=16,
            )
        if isinstance(payload.get("diagnostics"), dict):
            diagnostics = dict(payload["diagnostics"])
            summarized["diagnostics"] = {
                key: self._bounded_value(value, max_string_chars=1000, max_list_items=8, max_dict_items=16)
                for key, value in diagnostics.items()
                if key in {"summary_stats", "query_trace", "sql_query", "task_coverage", "artifact_ref", "snapshot_ref", "sandbox", "runtime_ms"}
                or key == "prompt_sampling"
            }
            summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
            summarized["diagnostics"]["prompt_sampling"] = self._prompt_sampling(
                full_counts=summary_stats,
                data=preview,
                fallback=diagnostics.get("prompt_sampling") if isinstance(diagnostics.get("prompt_sampling"), dict) else None,
            )
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

    def _prompt_sampling(self, *, full_counts: dict, data: dict, fallback: dict | None = None) -> dict:
        visible_counts = {
            "points_count": len(data.get("points", [])) if isinstance(data.get("points"), list) else None,
            "rows_count": len(data.get("rows", [])) if isinstance(data.get("rows"), list) else None,
            "series_count": len(data.get("series", [])) if isinstance(data.get("series"), list) else None,
        }
        counts = {key: value for key, value in (full_counts or {}).items() if value is not None}
        if not counts and fallback:
            counts = dict(fallback.get("full_counts") or {})
        return {
            "policy": "head_tail_edges",
            "sampled_for_prompt": any(
                isinstance(counts.get(key), int)
                and isinstance(visible_counts.get(key), int)
                and visible_counts[key] < counts[key]
                for key in ("points_count", "rows_count", "series_count")
            ),
            "full_counts": counts,
            "visible_counts": {key: value for key, value in visible_counts.items() if value is not None},
            "full_artifact_ref": (fallback or {}).get("full_artifact_ref"),
        }

    def _truncate_text(self, value, max_chars: int):
        if not isinstance(value, str):
            return value
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"

    def _bounded_value(
        self,
        value,
        *,
        max_string_chars: int = 800,
        max_list_items: int = 8,
        max_dict_items: int = 12,
    ):
        if isinstance(value, str):
            return self._truncate_text(value, max_string_chars)
        if isinstance(value, list):
            items = [
                self._bounded_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_list_items=max_list_items,
                    max_dict_items=max_dict_items,
                )
                for item in value[:max_list_items]
            ]
            if len(value) > max_list_items:
                items.append({"truncated_items": len(value) - max_list_items})
            return items
        if isinstance(value, dict):
            bounded = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= max_dict_items:
                    bounded["truncated_keys"] = len(value) - max_dict_items
                    break
                bounded[key] = self._bounded_value(
                    item,
                    max_string_chars=max_string_chars,
                    max_list_items=max_list_items,
                    max_dict_items=max_dict_items,
                )
            return bounded
        return value

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
