"""Prompt/context builder for the outer agent."""
from __future__ import annotations

import json

from core.data_fact import data_fact_prompt_view, prompt_fact_memory_view
from runtime.action_policy import runtime_action_constraints
from schemas.state import ConversationStateModel, RequestStateModel


class DataAgentPromptBuilder:
    """Build the bounded model-visible context."""

    def build_system_prompt(self) -> str:
        return self._compact_system_prompt()

    def _compact_system_prompt(self) -> str:
        return (
            "You are the outer ReAct data_agent for TSPilot v0.2.\n"
            "Choose exactly one next action from the current state. Do not execute tools yourself.\n"
            "Respond with exactly one JSON object and no markdown/prose/trailing text.\n"
            "Output schema: {\"thought\": str, \"task_contract\": object|null, "
            "\"previous_observation_assessment\": object|null, \"action_intention\": str|null, "
            "\"action_reason\": str|null, \"action\": str, \"action_input\": object}.\n"
            "Allowed actions: todowrite, sql_query, code_interpreter, forecast, anomaly, rag, skill, terminate.\n"
            "Language policy: task.response_language controls all natural-language fields; keep JSON keys, action names, query code, identifiers, and database values unchanged.\n"
            "Core ReAct rule: Thought_n selects Action_n; runtime provides Observation_n; Thought_n+1 must assess Observation_n before selecting the next action.\n"
            "Use previous_observation_assessment only after a meaningful observation. It should state covered, missing, can_answer, next_action_reason, and fact coverage when relevant.\n"
            "Task contract: create or update a compact user-visible output contract only when it helps verify multiple required outputs. Do not put internal tool stages in the contract.\n"
            "Evidence rule: exact numeric claims in terminate must be grounded by produced DataFacts or verified database/analysis/forecast/anomaly artifacts. Never do mental arithmetic in terminate.\n"
            "Use the smallest next action that fills the current evidence gap. Do not repeat an equivalent failed action unless action_input materially addresses the latest structured diagnostics.\n"
            "The context field state.next_action_constraints is authoritative. If it lists required_actions, choose one of them; if it lists prohibited_actions, do not choose those actions.\n"
            "When the user explicitly asks to use code_interpreter or the requested answer needs derived arithmetic over database values, first use sql_query to get simple grounded evidence, then use code_interpreter for the computation; do not force sql_query to encode fragile multi-step calculations.\n"
            "If sql_query task_coverage or query_task_contract says downstream_action=code_interpreter, call code_interpreter on that evidence next instead of repeating sql_query. "
            "If terminate is blocked by a stale gap assessment but forecast/anomaly/analysis artifacts exist, reassess the latest artifact coverage and either terminate with those refs or name the precise missing output.\n"
            "Action parameter cards:\n"
            "- sql_query: Query the selected datasource for grounded database evidence. Parameters: message, database_context, time_range?, constraints?, fact_requests?, or explicit query/query_language/purpose for user-supplied exact queries or focused repair. Normal database questions should use message mode; sql_query internally performs schema linking, query generation, validation, execution, and DataFact extraction. For later anomaly or forecast, ask sql_query for raw time-series evidence over the relevant series, not max/min/count aggregates.\n"
            "- code_interpreter: Analyze existing full evidence artifacts for derived metrics or custom analysis. Preferred parameters: database_evidence, analysis_goal, analysis_request, required_outputs?, expected_result_schema?, constraints?, fact_requests?. Omit code unless the user explicitly requested custom code; the tool can run a canonical analysis_request template. If code is provided, it must use injected rows/points/columns/database_evidence variables and assign result={\"summary\": non-empty string, \"metrics\": object, \"details\": object}; do not return a bare expression.\n"
            "- forecast: Forecast from existing raw time-series evidence when the user asks for prediction. If no raw time-series evidence exists, call sql_query first. Parameters: database_evidence, horizon, model_name?, series_name?, constraints?, fact_requests?. Omit model_name unless the user named a model.\n"
            "- anomaly: Detect anomalies from existing raw time-series evidence when the user asks for anomaly/spike/outlier detection. If no raw time-series evidence exists, call sql_query first. Parameters: database_evidence, detector_name?, series_name?, constraints?, fact_requests?. Omit detector_name unless the user named a detector.\n"
            "- todowrite: Create a visible plan only for complex tasks with 3+ independently verifiable user-visible outputs. Parameters: message, current_intent?, focus?, task_contract?, todos, evidence_summary?.\n"
            "- rag: Retrieve non-database knowledge only when explicitly needed. Parameters: query, filters?.\n"
            "- skill: Invoke a named packaged workflow only when explicitly requested. Parameters: skill_name, parameters.\n"
            "- terminate: End when evidence covers the user request or the task cannot proceed. Parameters: result?, summary_goal?, direct_answer?, include_analysis_ids, include_fact_ids, include_visualization_ids, section_plan, unavailable_outputs, unavailable_reason?.\n"
            "Tool-internal rules live inside tools. Do not recreate schema linking, forecasting, anomaly detection, or code execution logic in the outer prompt.\n"
            "If a tool returns structured failure diagnostics, use them as evidence for the next action.\n"
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
                "response_language": request_state.response_language,
                "database_context": (
                    request_state.database_context.model_dump(mode="json")
                    if request_state.database_context
                    else None
                ),
                "selected_database": request_state.selected_database,
                "selected_database_type": request_state.selected_database_type,
                "time_range": request_state.time_range,
                "constraints": request_state.constraints,
                "current_intent": request_state.current_intent,
                "requested_capabilities": request_state.requested_capabilities,
                "history": [message.model_dump(mode="json") for message in request_state.history[-4:]],
            },
            "state": {
                "execution": self._execution_state(request_state),
                "next_action_constraints": runtime_action_constraints(request_state),
                "todo_list": request_state.todo_list,
                "plan_current_step": request_state.plan_current_step,
                "planning_complete": request_state.planning_complete,
                "focus": request_state.focus,
                "recent_todo_summary": conversation_state.recent_todo_summary,
                "prompt_context_summary": request_state.prompt_context_summary,
                "task_contract": (
                    request_state.task_contract.model_dump(mode="json")
                    if request_state.task_contract
                    else None
                ),
                "data_fact_context": data_fact_prompt_view(request_state),
                "long_term_fact_memory": prompt_fact_memory_view(self._database_id_for_fact_memory(request_state)),
                "conversation_fact_memory": self._conversation_fact_memory(conversation_state),
            },
            "evidence": {
                "latest": latest_evidence,
                "prior_queries": prior_queries[-5:],
            },
            "outputs": {
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
                "data_facts": data_fact_prompt_view(request_state),
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
            "recent_react_transcript": [
                self._react_step_context(step)
                for step in request_state.react_transcript[-6:]
            ],
            "available_actions": self._available_actions(),
        }

    def _available_actions(self) -> list[dict]:
        return [
            {
                "action": "todowrite",
                "use_when": "Complex task needs 3+ user-visible deliverables and no plan exists.",
                "parameters": ["message", "current_intent?", "focus?", "task_contract?", "todos", "evidence_summary?"],
            },
            {
                "action": "sql_query",
                "use_when": "Need grounded database evidence, exact aggregates, grouping, ranking, or validation.",
                "parameters": ["message|query", "database_context", "time_range?", "constraints?", "fact_requests?", "query_language?", "purpose?"],
            },
            {
                "action": "code_interpreter",
                "use_when": "Existing evidence needs derived metrics, statistics, ratios, windows, or custom computation.",
                "parameters": ["database_evidence", "analysis_goal", "analysis_request?", "required_outputs?", "code?", "expected_result_schema?", "constraints?", "fact_requests?"],
            },
            {
                "action": "anomaly",
                "use_when": "User asks for anomaly/spike/outlier detection on time-series evidence.",
                "parameters": ["database_evidence", "detector_name?", "series_name?", "constraints?", "fact_requests?"],
            },
            {
                "action": "forecast",
                "use_when": "User asks for prediction/forecast on time-series evidence.",
                "parameters": ["database_evidence", "horizon", "model_name?", "series_name?", "constraints?", "fact_requests?"],
            },
            {
                "action": "rag",
                "use_when": "External/local knowledge is explicitly needed beyond database evidence.",
                "parameters": ["query", "filters?"],
            },
            {
                "action": "skill",
                "use_when": "User explicitly asks for a named packaged workflow or skill.",
                "parameters": ["skill_name", "parameters"],
            },
            {
                "action": "terminate",
                "use_when": "Evidence covers the request, or task cannot proceed with available context.",
                "parameters": ["result?", "summary_goal?", "direct_answer?", "include_analysis_ids", "include_fact_ids", "include_visualization_ids", "section_plan", "unavailable_outputs", "unavailable_reason?"],
            },
        ]

    def _conversation_fact_memory(self, conversation_state: ConversationStateModel) -> dict:
        facts = list(conversation_state.recent_fact_memory or [])[-12:]
        return {
            "summary": conversation_state.fact_memory_summary,
            "recent": [
                {
                    "fact_id": fact.fact_id,
                    "name": fact.name,
                    "fact_type": fact.fact_type,
                    "status": fact.status,
                    "statement": fact.statement,
                }
                for fact in facts
            ],
        }

    def _database_id_for_fact_memory(self, request_state: RequestStateModel) -> str | None:
        if request_state.selected_database:
            return request_state.selected_database
        if request_state.database_context is not None:
            return request_state.database_context.database_id
        return None

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
        if request_state.react_transcript:
            lines = []
            for step in request_state.react_transcript[-6:]:
                if step.question:
                    lines.append(f"Question: {self._truncate_text(step.question, 800)}")
                if step.thought:
                    lines.append(f"Thought: {self._truncate_text(step.thought, 800)}")
                if step.phase:
                    lines.append(f"Phase: {self._truncate_text(step.phase, 300)}")
                if step.action_intention:
                    lines.append(f"Action Intention: {self._truncate_text(step.action_intention, 300)}")
                if step.action_reason:
                    lines.append(f"Action Reason: {self._truncate_text(step.action_reason, 500)}")
                lines.append(f"Action: {step.action}")
                lines.append(
                    "Action Input: "
                    + self._truncate_text(json.dumps(step.action_input, ensure_ascii=False), 2000)
                )
                if step.observation is not None:
                    lines.append(
                        "Observation: "
                        + self._truncate_text(
                            json.dumps(
                                self._observation_context(step.observation),
                                ensure_ascii=False,
                            ),
                            3000,
                        )
                    )
            return "\n".join(lines) if lines else "(none)"

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

    def _react_step_context(self, step) -> dict:
        return {
            "iteration": step.iteration,
            "question": self._truncate_text(step.question, 800),
            "thought": self._truncate_text(step.thought, 800),
            "phase": self._truncate_text(step.phase, 300),
            "action_intention": self._truncate_text(step.action_intention, 300),
            "action_reason": self._truncate_text(step.action_reason, 500),
            "action": step.action,
            "action_input": self._bounded_value(
                step.action_input,
                max_string_chars=1200,
                max_list_items=8,
                max_dict_items=16,
            ),
            "observation": (
                self._observation_context(step.observation)
                if step.observation is not None
                else None
            ),
        }

    def _observation_context(self, observation) -> dict:
        context = {
            "tool_name": observation.tool_name,
            "success": observation.success,
            "summary": observation.summary,
            "payload": self._summarize_observation_payload(observation.payload),
        }
        if observation.error:
            context["error"] = observation.error
        if observation.payload_truncated:
            context["payload_truncated"] = True
        if observation.payload_ref:
            context["payload_ref"] = observation.payload_ref
        return context

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
                if isinstance(item, dict):
                    series_preview.append(self._summarize_series_preview(item, point_limit=4))
            data["series"] = series_preview
        payload["data"] = data
        payload["query"] = self._truncate_text(payload.get("query"), 4000)
        payload["summary"] = self._truncate_text(payload.get("summary"), 1200)
        payload["metadata"] = self._bounded_value(payload.get("metadata") or {}, max_string_chars=600, max_list_items=8, max_dict_items=16)
        visible_diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "summary_stats", "prompt_sampling", "query_trace", "series_count"}
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
                preview["series"] = [
                    self._summarize_series_preview(series, point_limit=4)
                    for series in data["series"][:2]
                    if isinstance(series, dict)
                ]
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

    def _summarize_forecast_status(self, forecast) -> dict:
        payload = forecast.model_dump(mode="json")
        forecast_points = payload.get("forecast_points", []) or []
        return {
            "forecast_id": payload.get("forecast_id"),
            "status": payload.get("status"),
            "horizon": payload.get("horizon"),
            "forecast_plan": payload.get("forecast_plan"),
            "summary": self._truncate_text(payload.get("summary"), 800),
            "forecast_point_count": len(forecast_points),
            "forecast_points_preview": forecast_points[:6],
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
            "error_type",
            "retryable",
            "recommended_next_action",
            "recommended_strategy",
            "blocked_strategy",
            "failure_signature",
            "repeated_failure_count",
            "evidence_id",
            "result_type",
            "analysis_id",
            "analysis_goal",
            "input_evidence_id",
            "input_row_count",
            "forecast_id",
            "anomaly_id",
            "current_step",
            "planning_complete",
            "query",
            "query_language",
            "columns",
            "task_contract",
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
                preview["series"] = [
                    self._summarize_series_preview(series, point_limit=4)
                    for series in data["series"][:2]
                    if isinstance(series, dict)
                ]
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
            summarized_diagnostics = {}
            for key in (
                "summary_stats",
                "artifact_ref",
                "prompt_sampling",
                "query_shape_issues",
                "query_task_contract",
                "recommended_downstream_action",
                "strategy_hint",
                "classification",
                "coverage",
            ):
                if key in diagnostics:
                    summarized_diagnostics[key] = self._bounded_value(
                        diagnostics[key],
                        max_string_chars=1000,
                        max_list_items=8,
                        max_dict_items=16,
                    )
            if payload.get("error") and isinstance(diagnostics.get("query_trace"), dict):
                summarized_diagnostics["query_trace"] = self._bounded_value(
                    diagnostics["query_trace"],
                    max_string_chars=1000,
                    max_list_items=8,
                    max_dict_items=16,
                )
            summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
            summarized_diagnostics["prompt_sampling"] = self._prompt_sampling(
                full_counts=summary_stats,
                data=preview,
                fallback=diagnostics.get("prompt_sampling") if isinstance(diagnostics.get("prompt_sampling"), dict) else None,
            )
            summarized["diagnostics"] = summarized_diagnostics
        if isinstance(payload.get("todos"), list):
            summarized["todos"] = [
                {
                    key: todo.get(key)
                    for key in ("content", "task_type", "status", "priority", "acceptance_criteria")
                    if todo.get(key) not in (None, "", [], {})
                }
                for todo in payload["todos"][:8]
                if isinstance(todo, dict)
            ]
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

    def _summarize_series_preview(self, series: dict, *, point_limit: int) -> dict:
        item = {
            key: self._bounded_value(value, max_string_chars=400, max_list_items=6, max_dict_items=8)
            for key, value in series.items()
            if key not in {"points", "rows"}
        }
        points = series.get("points")
        if isinstance(points, list):
            item["points_count"] = series.get("points_count") or len(points)
            item["points"] = self._sample_edges(points, limit=point_limit)
        rows = series.get("rows")
        if isinstance(rows, list):
            item["rows_count"] = series.get("rows_count") or len(rows)
            item["rows"] = self._sample_edges(rows, limit=point_limit)
        return item

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

    def _sample_edges(self, items: list, *, limit: int) -> list:
        if len(items) <= limit:
            return items
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        return [*items[:head], *items[-tail:]]

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
