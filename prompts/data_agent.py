"""Prompt/context builder for the outer agent."""
from __future__ import annotations

import json

from core.harness import default_capability_registry
from core.harness.observation_view import model_observation_view
from core.key_insight import key_insight_prompt_view
from runtime.action_policy import runtime_action_constraints
from runtime.prompt_locale import prompt_locale_instruction
from schemas.state import ConversationStateModel, RequestStateModel


class DataAgentPromptBuilder:
    """Build the bounded model-visible context."""

    def build_system_prompt(self, response_language: str = "en") -> str:
        return prompt_locale_instruction(response_language) + self._compact_system_prompt()

    def _compact_system_prompt(self) -> str:
        return (
            "You are the outer ReAct tool-calling data_agent for TSPilot v0.2.\n"
            "Your only job is to choose exactly one next tool action from the current state. Do not execute tools yourself.\n"
            "Respond with exactly one JSON object and no markdown/prose/trailing text.\n"
            "Output schema: {\"thought\": str, \"task_contract\": object|null, "
            "\"previous_observation_assessment\": object|null, \"action_intention\": str|null, "
            "\"action_reason\": str|null, \"action\": str, \"action_input\": object}.\n"
            "Allowed actions: todowrite, sql_query, code_interpreter, forecast, anomaly, visualization, rag, skill, terminate.\n"
            "Language policy: task.response_language controls all natural-language fields; keep JSON keys, action names, identifiers, and database values unchanged.\n"
            "Core ReAct rule: Thought_n selects Action_n; runtime provides Observation_n; Thought_n+1 must use Observation_n and progress_summary before choosing the next action.\n"
            "Do not repeat actions listed as completed in progress_summary unless last_observation says the existing artifact is insufficient.\n"
            "The context field state.next_action_constraints is authoritative. If it lists required_actions, choose one of them; if it lists prohibited_actions, do not choose those actions.\n"
            "Use the smallest next action that fills the current missing capability. When all requested capabilities are covered, call terminate.\n"
            "Exact numeric claims in terminate must be grounded by artifact refs and verified insight state in context. Never do mental arithmetic in terminate. "
            "Use visualization after evidence/analysis is ready whenever the user requests a chart or a visual pattern. Its action_input contains message, optional source_refs, and optional constraints; never author marks, layers, renderer options, or data arrays in the outer action. "
            "The visualization tool owns semantic planning and requires full-range time-series data. If it reports incomplete data, follow its sql_query repair contract before retrying visualization. "
            "For terminate, action_input must contain response_plan with title, summary, sections, and visualization_ids. Each section has section_type, heading, content, and source_refs. "
            "Use only visualization_ids returned by successful visualization observations. In section source_refs, cite evidence/insight/view refs; a visualization section may also cite a selected visualization as visualization:<id> or its returned bare id. The final formatter only assembles existing artifacts and never creates charts or queries data. "
            "Terminate schema: {\"response_plan\":{\"title\":str|null,\"summary\":str,\"sections\":[{\"section_type\":str,\"heading\":str|null,\"content\":str,\"source_refs\":[str]}],\"visualization_ids\":[str]},\"unavailable_outputs\":[str],\"unavailable_reason\":str|null}. "
            "SQL boundary: the outer ReAct agent must not write SQL, Flux, PromQL, database query code, schema-linking logic, dialect logic, or repair code. "
            "For sql_query, provide only natural-language message and optional purpose describing the evidence needed.\n"
            "Key Insight contract: use insight_requests to name the key insights a tool must produce. Give every request a stable semantic insight_key. "
            "For an insight computed from earlier insights, list their insight_key values in derived_from. Reuse insight keys from state.insight_state; "
            "do not put Evidence IDs, artifact refs, or metric labels in derived_from. An analytical Key Insight computed directly from the "
            "selected database Evidence may leave derived_from empty because the analysis artifact records that evidence dependency. "
            "SQL should produce evidence-backed atomic key insights; code_interpreter should produce derived or analytical key insights.\n"
            "SQL Key Insight contracts support point_value or time_boundary with requirements.time_position=start|end, extreme with "
            "requirements.operator=min|max, and count. Use time_boundary for timestamps and point_value only for scalar measure values. "
            "Use count only when the requested insight is a row/record count. Tables, detail lists, and complete time series are query Evidence "
            "Artifacts, not scalar Key Insights, so leave insight_requests empty for those outputs. Do not request change, ratio, trend, or other derived Key Insight types from sql_query.\n"
            "Code interpreter boundary: use code_interpreter only to calculate derived or analytical Key Insights from grounded Evidence or verified parent Insights. "
            "Every call must include the exact non-empty insight_requests it should calculate; every request object must contain insight_key, name, and insight_type, plus optional requirements or derived_from. "
            "Do not use type as an alias for insight_type, and do not request unrelated supporting metrics. Python code is optional because the tool can generate it internally.\n"
            "Code interpreter sandbox contract: generated Python code receives canonical variables df, time, value, "
            "time_col, value_col, series, analysis_context, plus compatibility variables data, rows, points, columns, "
            "metadata, and diagnostics. Prefer value/time/df/series; do not guess business field names or index raw "
            "rows/points unless canonical inputs are unavailable. data supports both original rows/points and column arrays. "
            "Use pandas-compatible frequency aliases such as 'h' for hourly grouping. "
            "The tool returns computation-only candidates and binds them to formal Key Insights internally. Do not ask it to produce summaries, metrics/details containers, Data Views, chart roles, or final-answer prose. "
            "When verification or reuse genuinely requires a complete calculated table or series, the tool publishes it as independent derived Evidence; visualization consumes that Evidence plus formal Insight refs. "
            "When the user requires anomaly exclusion and no rule is supplied, call anomaly before code_interpreter. The Anomaly Artifact is authoritative; code_interpreter and visualization must use that exact anomaly set rather than detecting a second set. Do not author deterministic fallback branches for a missing artifact or failed business calculation; return a structured unavailable output after the allowed repair path instead. "
            "The computation must preserve each requested insight_key and provide a calculation trace; the independent LLM Insight Binder owns statements and semantic binding.\n"
            "Tool-internal rules live inside tools. Do not recreate schema linking, query generation, validation, forecasting, anomaly detection, or code execution logic in the outer prompt.\n"
            "If a tool returns structured failure diagnostics, choose the recommended next action or a materially different action that addresses the diagnostics.\n"
            "Do not output markdown fences."
        )

    def build_context(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> dict:
        return self._outer_react_view(request_state, conversation_state)

    def _outer_react_view(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> dict:
        action_space = runtime_action_constraints(request_state)
        return {
            "task": self._task_context(request_state),
            "tools": self._available_actions(),
            "state": {
                "execution": self._execution_state(request_state),
                "next_action_constraints": action_space,
                "progress_summary": self._progress_summary(request_state, conversation_state),
                "todo_progress": self._todo_progress_context(request_state, conversation_state),
                "task_contract": self._task_contract_context(request_state),
                "artifact_inventory": self._artifact_inventory(request_state),
                "insight_state": key_insight_prompt_view(request_state),
            },
            "artifacts": {"refs": self._artifact_ref_index(request_state)},
            "last_observation": (
                self._action_output_observation_context(request_state.latest_action_output)
                if request_state.latest_action_output
                else None
            ),
            "recent_trajectory": self._recent_memory_context(request_state),
        }

    def _available_actions(self) -> list[dict]:
        cards = []
        for card in default_capability_registry().action_cards():
            item = {
                "action": card.get("action"),
                "use_when": card.get("use_when"),
                "parameters": self._minimal_parameters(card.get("action"), card.get("parameters") or []),
            }
            cards.append({key: value for key, value in item.items() if value not in (None, "", [], {})})
        return cards

    def _minimal_parameters(self, action: str | None, parameters: list[str]) -> list[str]:
        if action == "sql_query":
            return ["message", "purpose?", "insight_requests?"]
        if action == "todowrite":
            return ["message", "todos", "task_contract?"]
        if action == "code_interpreter":
            return ["database_evidence", "analysis_goal", "insight_requests", "code?", "constraints?"]
        if action == "forecast":
            return ["database_evidence", "horizon", "constraints?"]
        if action == "anomaly":
            return ["database_evidence", "constraints?"]
        if action == "terminate":
            return ["response_plan", "unavailable_outputs?", "unavailable_reason?"]
        if action == "visualization":
            return ["message", "source_refs?", "constraints?"]
        return list(parameters)[:4]

    def _task_context(self, request_state: RequestStateModel) -> dict:
        payload = {
            "message": request_state.message,
            "response_language": request_state.response_language,
            "database_context": self._outer_database_context(request_state),
            "time_range": request_state.time_range,
            "constraints": request_state.constraints,
            "history": [message.model_dump(mode="json") for message in request_state.history[-4:]],
        }
        focus = str(request_state.focus or "").strip()
        if focus and focus != str(request_state.message or "").strip():
            payload["focus"] = focus
        capabilities = self._requested_capabilities_context(request_state)
        if capabilities:
            payload["requested_capabilities"] = capabilities
        return {
            key: value
            for key, value in payload.items()
            if value not in (None, "", [], {})
        }

    def _outer_database_context(self, request_state: RequestStateModel) -> dict | None:
        context = request_state.database_context
        if context is None:
            return None
        payload = context.model_dump(mode="json")
        return {
            "database_id": payload.get("database_id"),
            "database_type": payload.get("database_type"),
            "display_name": payload.get("display_name"),
        }

    def _requested_capabilities_context(self, request_state: RequestStateModel) -> list[str]:
        capabilities = [
            str(item).strip()
            for item in (request_state.requested_capabilities or [])
            if str(item).strip()
        ]
        if not capabilities or capabilities == ["query"]:
            return []
        return capabilities

    def _execution_state(self, request_state: RequestStateModel) -> dict:
        last_success = next((item for item in reversed(request_state.observations) if item.success), None)
        last_failure = next((item for item in reversed(request_state.observations) if not item.success), None)
        last_call = request_state.tool_history[-1] if request_state.tool_history else None
        last_tool = last_call.tool_name if last_call else None
        return {
            "iteration": request_state.iteration,
            "max_iterations": request_state.max_iterations,
            "tool_sequence": [call.tool_name for call in request_state.tool_history[-8:]],
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
        }

    def _progress_summary(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> list[dict]:
        progress = []
        outputs = [output for output in request_state.action_outputs if output.tool_name != "terminate"]
        for output in outputs[-8:]:
            if not output.success:
                continue
            iteration = output.meta.get("iteration") if isinstance(output.meta, dict) else None
            progress.append(
                {
                    "status": "completed",
                    "step": iteration,
                    "action": output.tool_name,
                    "resource_ref": output.resource_ref,
                    "covers": self._covered_capability(output.tool_name),
                    "summary": self._truncate_text(output.content, 240),
                }
            )
        if conversation_state.recent_todo_summary:
            progress.append({"status": "context", "summary": conversation_state.recent_todo_summary})
        return [
            {key: value for key, value in item.items() if value not in (None, "", [], {})}
            for item in progress
        ]

    def _covered_capability(self, tool_name: str) -> str | None:
        return {
            "todowrite": "todo_plan",
            "sql_query": "query",
            "code_interpreter": "analysis",
            "forecast": "forecast",
            "anomaly": "anomaly",
            "rag": "external_knowledge",
            "skill": "skill",
        }.get(str(tool_name or "").strip())

    def _todo_progress_context(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> dict:
        todos = [todo for todo in request_state.todo_list if isinstance(todo, dict)]
        by_status: dict[str, int] = {}
        for todo in todos:
            status = str(todo.get("status") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
        current = next((todo for todo in todos if todo.get("status") == "in_progress"), None)
        pending = [todo for todo in todos if todo.get("status") not in {"completed", "in_progress"}]
        return {
            "planning_complete": request_state.planning_complete,
            "current_step": request_state.plan_current_step,
            "total": len(todos),
            "by_status": by_status,
            "current": self._todo_item_context(current),
            "pending_preview": [self._todo_item_context(todo) for todo in pending[:6]],
            "recent_summary": conversation_state.recent_todo_summary,
        }

    def _todo_item_context(self, todo: dict | None) -> dict | None:
        if not isinstance(todo, dict):
            return None
        return {
            key: self._bounded_value(todo.get(key), max_string_chars=500, max_list_items=4, max_dict_items=6)
            for key in ("content", "task_type", "status", "priority", "acceptance_criteria")
            if todo.get(key) not in (None, "", [], {})
        }

    def _task_contract_context(self, request_state: RequestStateModel) -> dict | None:
        if request_state.task_contract is None:
            return None
        payload = request_state.task_contract.model_dump(mode="json")
        return self._bounded_value(payload, max_string_chars=700, max_list_items=8, max_dict_items=14)

    def _final_answer_context(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> dict:
        memory = []
        for insight in list(conversation_state.recent_insight_memory or [])[-6:]:
            memory.append(
                {
                    "insight_id": insight.insight_id,
                    "name": insight.name,
                    "insight_type": insight.insight_type,
                    "status": insight.status,
                    "statement": self._truncate_text(insight.statement, 500),
                }
            )
        return {"artifact_refs": self._artifact_ref_index(request_state)}

    def _insights_from_action_outputs(self, request_state: RequestStateModel) -> list[dict]:
        insights: list[dict] = []
        for action_output in request_state.action_outputs[-6:]:
            observation = action_output.observations
            if not isinstance(observation, dict):
                continue
            for key in ("insights", "produced_insights_preview"):
                raw_insights = observation.get(key)
                if not isinstance(raw_insights, list):
                    continue
                for insight in raw_insights:
                    if not isinstance(insight, dict):
                        continue
                    insights.append(
                        {
                            item_key: self._bounded_value(insight.get(item_key), max_string_chars=500, max_list_items=4, max_dict_items=6)
                            for item_key in ("insight_id", "name", "insight_type", "statement", "value", "status")
                            if insight.get(item_key) not in (None, "", [], {})
                        }
                    )
        return insights[-12:]

    def _artifact_inventory(self, request_state: RequestStateModel) -> dict:
        return {
            "database_evidence_count": len(request_state.database_evidence_artifacts),
            "analysis_count": len(request_state.analysis_artifacts),
            "derived_evidence_count": len(request_state.derived_evidence_artifacts),
            "has_forecast": request_state.latest_forecast is not None,
            "has_anomaly": request_state.latest_anomaly is not None,
            "verified_insight_count": sum(
                1 for insight in request_state.insight_set.insights if insight.status == "verified"
            ),
            "visualization_count": len(request_state.visualizations),
        }

    def _resource_index_context(self, request_state: RequestStateModel) -> dict:
        resources = (request_state.resource_index or {}).get("resources")
        if not isinstance(resources, dict):
            resources = {}
        items = []
        for ref, payload in list(resources.items())[-12:]:
            item = {"resource_ref": ref}
            if isinstance(payload, dict):
                item.update(
                    {
                        "tool_name": payload.get("tool_name"),
                        "resource_type": payload.get("resource_type"),
                        "iteration": payload.get("iteration"),
                        "status": payload.get("status"),
                    }
                )
            items.append({key: value for key, value in item.items() if value not in (None, "", [], {})})
        return {"resources": items}

    def _action_output_observation_context(self, action_output) -> dict | str | None:
        if action_output is None:
            return None
        observation = action_output.observations
        return self._bounded_value(observation, max_string_chars=900, max_list_items=8, max_dict_items=24)

    def _recent_memory_context(self, request_state: RequestStateModel) -> list[dict]:
        fragments = request_state.memory_fragments
        latest_iteration = None
        if request_state.latest_action_output is not None and isinstance(request_state.latest_action_output.meta, dict):
            latest_iteration = request_state.latest_action_output.meta.get("iteration")
        if latest_iteration is not None:
            fragments = [
                fragment for fragment in fragments
                if not isinstance(fragment, dict) or fragment.get("iteration") != latest_iteration
            ]
        fragments = fragments[-3:]
        if fragments:
            return [
                self._trajectory_receipt(fragment)
                for fragment in fragments
                if isinstance(fragment, dict)
            ]
        outputs = request_state.action_outputs
        if latest_iteration is not None:
            outputs = [
                output for output in outputs
                if not isinstance(output.meta, dict) or output.meta.get("iteration") != latest_iteration
            ]
        return [
            self._trajectory_receipt({
                "iteration": output.meta.get("iteration") if isinstance(output.meta, dict) else None,
                "action": output.tool_name,
                "observation": output.observations,
                "resource_ref": output.resource_ref,
                "status": "succeeded" if output.success else "failed",
            })
            for output in outputs[-3:]
        ]

    def _trajectory_receipt(self, fragment: dict) -> dict:
        observation = fragment.get("observation") if isinstance(fragment.get("observation"), dict) else {}
        status = fragment.get("status")
        receipt = {
            "iteration": fragment.get("iteration"),
            "action": fragment.get("action"),
            "status": status,
            "resource_ref": fragment.get("resource_ref") or observation.get("resource_ref"),
            "summary": self._truncate_text(observation.get("summary"), 400),
            "coverage_delta": observation.get("coverage_delta"),
        }
        if status == "failed" or observation.get("success") is False:
            receipt["failure"] = self._bounded_value(
                observation,
                max_string_chars=900,
                max_list_items=8,
                max_dict_items=20,
            )
        return {
            key: value
            for key, value in receipt.items()
            if value not in (None, "", [], {})
        }

    def _artifact_ref_index(self, request_state: RequestStateModel) -> dict:
        evidence_refs = [f"evidence:{item}" for item in list(request_state.database_evidence_artifacts.keys())[-8:]]
        analysis_refs = [f"analysis:{item}" for item in list(request_state.analysis_artifacts.keys())[-8:]]
        return {
            "database_evidence": evidence_refs,
            "analysis": analysis_refs,
            "latest_analysis": f"analysis:{request_state.latest_analysis_id}" if request_state.latest_analysis_id else None,
            "derived_evidence": [
                f"derived_evidence:{item}" for item in list(request_state.derived_evidence_artifacts.keys())[-8:]
            ],
            "forecast": [f"forecast:{item}" for item in list(request_state.forecast_artifacts.keys())[-8:]],
            "anomaly": [f"anomaly:{item}" for item in list(request_state.anomaly_artifacts.keys())[-8:]],
            "visualization": [
                f"visualization:{item.visualization_id}"
                for item in request_state.visualizations[-8:]
            ],
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
                json.dumps(context["tools"], ensure_ascii=False, indent=2),
                "",
                "Outer ReAct State:",
                json.dumps(
                    {
                        "state": context["state"],
                        "artifacts": context["artifacts"],
                        "last_observation": context["last_observation"],
                        "recent_trajectory": context["recent_trajectory"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
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
                    + self._truncate_text(json.dumps(self._action_input_context(step.action, step.action_input), ensure_ascii=False), 2000)
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
                    + self._truncate_text(json.dumps(self._action_input_context(call.tool_name, call.tool_input), ensure_ascii=False), 2000)
                )
            for observation in observations_by_iteration.get(iteration, []):
                lines.append(
                    "Observation: "
                    + self._truncate_text(
                        json.dumps(
                            {
                                "tool_name": observation.tool_name,
                                "success": observation.success,
                    "summary": self._observation_summary_context(observation),
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
                self._action_input_context(step.action, step.action_input),
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
        return model_observation_view(observation) or {}

    def _action_input_context(self, action_name: str, action_input: dict | None) -> dict:
        if not isinstance(action_input, dict):
            return {}
        if action_name != "sql_query":
            return action_input
        sanitized = {
            key: value
            for key, value in action_input.items()
            if key not in {"query", "message|query", "query_language"}
        }
        if any(key in action_input for key in ("query", "message|query", "query_language")):
            sanitized["omitted_internal_query_fields"] = True
            sanitized["sql_boundary_hint"] = "Outer agent must call sql_query with natural-language message/purpose only."
        return sanitized

    def _observation_summary_context(self, observation) -> str | None:
        summary = getattr(observation, "summary", None)
        if getattr(observation, "tool_name", None) == "sql_query":
            return self._strip_query_code(summary)
        return summary

    def _strip_query_code(self, value):
        if not isinstance(value, str):
            return value
        text = value
        for marker in (" for query '", " for query `", " Query statement:", "\nQuery statement:"):
            if marker in text:
                text = text.split(marker, 1)[0]
        return text

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
        query_was_available = bool(payload.get("query"))
        if isinstance(data.get("points"), list):
            data["points"] = data["points"][:8]
        if isinstance(data.get("rows"), list):
            data["rows"] = [self._outer_row_preview(row) for row in data["rows"][:4]]
        if isinstance(data.get("series"), list):
            series_preview = []
            for item in data["series"][:3]:
                if isinstance(item, dict):
                    series_preview.append(self._summarize_series_preview(item, point_limit=4))
            data["series"] = series_preview
        payload["data"] = data
        if query_was_available:
            payload["query_available_in_artifact"] = True
        payload.pop("query", None)
        payload.pop("query_language", None)
        payload["summary"] = self._truncate_text(self._strip_query_code(payload.get("summary")), 1200)
        payload["metadata"] = self._outer_evidence_metadata(payload.get("metadata") or {})
        if isinstance(payload.get("columns"), list):
            payload["columns"] = [self._outer_column_name(column) for column in payload["columns"]]
        visible_diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if key in {"artifact_kind", "artifact_ref", "summary_stats", "prompt_sampling", "series_count"}
        }
        visible_diagnostics["prompt_sampling"] = self._prompt_sampling(
            full_counts=summary_stats,
            data=data,
            fallback=visible_diagnostics.get("prompt_sampling") if isinstance(visible_diagnostics.get("prompt_sampling"), dict) else None,
        )
        payload["diagnostics"] = visible_diagnostics
        if summary_stats:
            payload["summary_stats"] = summary_stats
        return payload

    def _outer_evidence_metadata(self, metadata: dict) -> dict:
        if not isinstance(metadata, dict):
            return {}
        visible = {}
        for key in ("unit", "units", "currency", "symbol", "source", "time_range", "aggregation", "granularity"):
            if metadata.get(key) not in (None, "", [], {}):
                visible[key] = metadata.get(key)
        return self._bounded_value(visible, max_string_chars=400, max_list_items=4, max_dict_items=8)

    def _outer_column_name(self, column) -> str:
        text = str(column or "").strip()
        if text.startswith("_"):
            text = text.lstrip("_")
        return text or "column"

    def _outer_row_preview(self, row):
        if not isinstance(row, dict):
            return self._bounded_value(row, max_string_chars=400, max_list_items=6, max_dict_items=8)
        normalized = {}
        for key, value in row.items():
            name = self._outer_column_name(key)
            if name in normalized:
                suffix = 2
                candidate = f"{name}_{suffix}"
                while candidate in normalized:
                    suffix += 1
                    candidate = f"{name}_{suffix}"
                name = candidate
            normalized[name] = self._bounded_value(value, max_string_chars=400, max_list_items=6, max_dict_items=8)
        return normalized

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
                    "query_available_in_artifact": bool(item.get("query_available_in_artifact")),
                    "summary": self._truncate_text(self._strip_query_code(item.get("summary")), 800),
                    "result_type": item.get("result_type"),
                    "columns": [self._outer_column_name(column) for column in (item.get("columns") or [])[:20]],
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
                    "computed_insights": self._bounded_value(
                        payload.get("computed_insights") or [],
                        max_string_chars=1000,
                        max_list_items=8,
                        max_dict_items=16,
                    ),
                    "derived_evidence_refs": [
                        f"derived_evidence:{item.get('evidence_id')}"
                        for item in payload.get("derived_evidence", [])
                        if isinstance(item, dict) and item.get("evidence_id")
                    ],
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
            "columns",
            "task_contract",
        ):
            if key in payload:
                value = payload[key]
                if key == "columns" and isinstance(value, list):
                    value = [self._outer_column_name(column) for column in value]
                summarized[key] = self._bounded_value(value, max_string_chars=1200, max_list_items=12, max_dict_items=16)
        if isinstance(payload.get("data"), dict):
            data = dict(payload["data"])
            if isinstance(data.get("rows"), list):
                preview["rows"] = [self._outer_row_preview(row) for row in data["rows"][:3]]
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
                "recommended_downstream_action",
                "strategy_hint",
                "coverage",
            ):
                if key in diagnostics:
                    summarized_diagnostics[key] = self._bounded_value(
                        diagnostics[key],
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
        if isinstance(payload.get("valid_actions"), list):
            summarized["valid_actions"] = payload["valid_actions"]
        return summarized

    def _summarize_series_preview(self, series: dict, *, point_limit: int) -> dict:
        item = {}
        for key, value in series.items():
            if key in {"points", "rows"}:
                continue
            item[self._outer_column_name(key)] = self._bounded_value(value, max_string_chars=400, max_list_items=6, max_dict_items=8)
        points = series.get("points")
        if isinstance(points, list):
            item["points_count"] = series.get("points_count") or len(points)
            item["points"] = [self._outer_row_preview(point) for point in self._sample_edges(points, limit=point_limit)]
        rows = series.get("rows")
        if isinstance(rows, list):
            item["rows_count"] = series.get("rows_count") or len(rows)
            item["rows"] = [self._outer_row_preview(row) for row in self._sample_edges(rows, limit=point_limit)]
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
            "binding_insight_ids": payload.get("binding_insight_ids", [])[:6],
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
