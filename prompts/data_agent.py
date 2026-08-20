"""Prompt/context builder for the outer agent."""
from __future__ import annotations

import json

from core.harness import default_capability_registry
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
            "You are the outer ReAct tool-calling data_agent for TSPilot.\n"
            "Your only job is to choose exactly one next tool action from the current state. Do not execute tools yourself.\n"
            "Respond with exactly one JSON object and no markdown/prose/trailing text.\n"
            "Output schema: {\"thought\": str, \"task_contract\": object|null, "
            "\"previous_observation_assessment\": object|null, \"action\": str, \"action_input\": object}.\n"
            "Allowed actions: todowrite, sql_query, code_interpreter, forecast, anomaly, visualization, rag, skill, terminate.\n"
            "Language policy: task.response_language controls all natural-language fields; keep JSON keys, action names, identifiers, and database values unchanged.\n"
            "Core ReAct rule: Thought_n selects Action_n; runtime provides Observation_n; Thought_n+1 must use the latest Observation, recent_trajectory, and current artifact/Insight refs before choosing the next action.\n"
            "Thought is one concise decision record: state the accepted facts from the latest Observation, the exact remaining output or Insight gap, and why the selected Action is the smallest action that closes it. Do not repeat the user request, narrate policy constraints, or add separate intention/reason fields.\n"
            "A successful tool observation proves artifact production only. Complete the active todo only when previous_observation_assessment confirms its acceptance criteria from the returned artifact.\n"
            "In previous_observation_assessment, set completed_active_todo=true only when the immediately previous observation satisfies the current in_progress Todo. Runtime owns Todo transitions; do not restate completed Todo lists or choose the next Todo in this receipt. Do not claim a future Todo complete from a prerequisite or from code that substitutes for its owning specialized tool.\n"
            "Do not repeat a successful action shown in recent_trajectory unless the latest Observation identifies a concrete insufficiency.\n"
            "The context field state.next_action_constraints is authoritative. If it lists required_actions, choose one of them; if it lists prohibited_actions, do not choose those actions.\n"
            "Use the smallest next action that fills the current missing capability. When all requested capabilities are covered, call terminate.\n"
            "Task-contract coverage is semantic, not a list of available tools. Give every independently requested user-facing "
            "answer its own required output. A forecast artifact covers the future series, but it does not by itself cover requested "
            "derived conclusions such as direction, endpoint change, percentage change, threshold crossing, ranking, or comparison. "
            "Represent each such conclusion as output_type=analysis and evidence_kind=calculated, with success_criteria naming the "
            "required values; satisfy it through code_interpreter grounded in the owning forecast/anomaly/evidence source. Do not let "
            "a chart or a terminate narrative substitute for that calculation.\n"
            "Visual verification is a core completion requirement. When authoring or refining task_contract, add a required output with output_type=visualization for analytical conclusions that have a natural visual form, including time-series extrema, anomalies, trends, forecasts, intervals, and timestamped decision points. Pure metadata or explanatory answers need no forced chart. "
            "The visual verification output succeeds only when the conclusion is a grounded layer over contextual data covering the complete user analysis interval at the granularity needed to inspect it.\n"
            "Exact numeric claims in terminate must be copied from verified insight_state values or immutable artifacts.facts; never do mental arithmetic in terminate. Artifact records are bounded presentation receipts, not inputs for new calculations. "
            "Use visualization after evidence/analysis is ready whenever task_contract requires visual verification, even if the user did not explicitly ask for a chart. Its action_input contains message, optional source_refs, and optional constraints; never author marks, layers, renderer options, or data arrays in the outer action. "
            "Before choosing visualization, use Thought to concisely explain why the selected verified conclusion can be inspected against its grounded context. Put only visually inspectable conclusion Insights in source_refs as insight:<exact insight_id or insight_key>; method, provenance, and input-count receipts remain answer text and are not visualization targets. Do not repeat a target's Evidence, Analysis, Derived Evidence, Forecast, or Anomaly refs in the Action: visualization resolves its lineage and loads the related complete data itself. If the conclusion or its visual context is not grounded, choose the source-owning action instead. Never justify visualization with required, forced, policy, or runtime language. "
            "The visualization tool owns semantic planning and verifies complete contextual coverage. If it reports missing evidence, follow the returned sql_query, anomaly, forecast, or code_interpreter repair contract before retrying visualization. A SQL query made only to add visual context does not invalidate or require rerunning unrelated existing analysis artifacts. "
            "When visualization returns status=needs_sources, it has not created a visualization. Keep the visualization todo active, call required_data_request.required_action with its exact input_source_refs and insight_requests, then retry visualization. "
            "A successful source-owner action does not itself satisfy a visualization dependency; the runtime will require a new visualization turn to re-evaluate the updated Insight and artifact state. "
            "When visualization returns status=unavailable, do not retry it or invent a chart. Call terminate with visualization in unavailable_outputs, preserve the grounded text answer, and explain the returned unavailable_reason. "
            "Presentation point budgets are runtime-owned and do not limit persisted raw evidence needed for anomaly detection, forecasting, exact extrema, optimization, or other calculations; those tools require the complete analysis interval at the necessary granularity. "
            "For terminate, action_input must contain response_plan with title, summary, sections, and visualization_ids. Each section has section_type, heading, content, and source_refs. Write its content from the exact verified Insight values and artifact facts in context; refs without their facts are citations, not evidence for invented prose. "
            "Use only visualization_ids returned by successful visualization observations. In section source_refs, cite evidence/insight/view refs; a visualization section may also cite a selected visualization as visualization:<id> or its returned bare id. The final formatter only assembles existing artifacts and never creates charts or queries data. "
            "When a visualization section cites a successful visualization:<id>, do not manually repeat its internal view or layer source refs; the visualization artifact already preserves that lineage, and copying internal refs creates avoidable citation errors. "
            "For final section citations, prefer insight:<exact insight_key> copied from state.insight_state. Use an opaque insight_id only when copying it verbatim from state; never reconstruct or abbreviate an opaque identifier. "
            "Terminate schema: {\"response_plan\":{\"title\":str|null,\"summary\":str,\"sections\":[{\"section_type\":str,\"heading\":str|null,\"content\":str,\"source_refs\":[str]}],\"visualization_ids\":[str]},\"unavailable_outputs\":[str],\"unavailable_reason\":str|null}. "
            "SQL boundary: the outer ReAct agent must not write SQL, Flux, PromQL, database query code, schema-linking logic, dialect logic, or repair code. "
            "For sql_query, provide only natural-language message and optional purpose describing the evidence needed.\n"
            "Key Insight contract: use insight_requests to name the key insights a tool must produce. Give every request a stable semantic insight_key. "
            "For an insight computed from earlier insights, list their insight_key values in derived_from. Reuse insight keys from state.insight_state; "
            "do not put Evidence IDs, artifact refs, or metric labels in derived_from. An analytical Key Insight computed directly from the "
            "selected database Evidence may leave derived_from empty because the analysis artifact records that evidence dependency. "
            "SQL should produce evidence-backed atomic key insights; code_interpreter should produce derived or analytical key insights.\n"
            "A parent belongs in derived_from only when its value participates in computing the child value. A parent used only for a consistency check, display context, annotation, or later visualization is not a derivation dependency. Keep each Insight atomic: do not copy already verified extrema, timestamps, or other facts into a new calculated Insight merely to carry them forward; cite their existing Insight refs separately. Keep final wording semantically equal to calculation_trace: an endpoint difference supports an endpoint-change claim, not an unqualified global trend claim.\n"
            "SQL Key Insight contracts support point_value or time_boundary with requirements.time_position=start|end, extreme with "
            "requirements.operator=min|max, and count. Use time_boundary for timestamps and point_value only for scalar measure values. "
            "A SQL extreme Insight preserves its located row as one item containing both numeric value and timestamp. Request one extreme "
            "for a located maximum/minimum; do not request a redundant extreme_time solely to reconstruct the same visual point. "
            "SQL time_boundary means only the first or last timestamp of the queried dataset. Never use it for optimal buy/sell time, intervention time, anomaly time, forecast event time, or any other calculated/selected decision point; those belong to code_interpreter or their owning analysis artifact. "
            "Use count only when the requested insight is a row/record count. Tables, detail lists, and complete time series are query Evidence "
            "Artifacts, not scalar Key Insights, so leave insight_requests empty for those outputs. Do not request change, ratio, trend, or other derived Key Insight types from sql_query.\n"
            "Code interpreter boundary: use code_interpreter only to calculate derived or analytical Key Insights from grounded Evidence or verified parent Insights. "
            "Use source_refs when the calculation depends on forecast, anomaly, derived_evidence, insight, or multiple artifacts; preserve the exact refs returned by a visualization source request. "
            "Code interpreter must not replace forecast or anomaly: call the owning specialized tool first, then use code_interpreter only to derive requested conclusions from that artifact. "
            "Every call must include the exact non-empty insight_requests it should calculate; every request object must contain insight_key, name, and insight_type, plus optional requirements or derived_from. "
            "Do not use type as an alias for insight_type, and do not request unrelated supporting metrics. Do not create separate method, basis, provenance, input-count, or display-context Insights: calculation_trace, artifact facts, and evidence_refs preserve those details on the one user-facing conclusion Insight. Reuse already verified Insight refs instead of requesting code_interpreter to recompute the same semantic keys. State required semantic scope and outputs, but never invent a calculation method the user did not prescribe; the code generator and semantic Binder choose and audit a defensible method. Python code is optional because the tool can generate it internally.\n"
            "When a calculated event or decision must be located on a visualization, request a collection/series Insight whose items each contain the semantic role label, timestamp, and numeric value, or separate point Insights that each contain both timestamp and numeric value. Do not split one located point into unrelated scalar time and scalar value Insights, do not model a calculated decision time as time_boundary, and do not request raw visualization context as an evidence Insight from code_interpreter.\n"
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
                "todo_progress": self._todo_progress_context(request_state, conversation_state),
                "task_contract": self._task_contract_context(request_state),
                "insight_state": key_insight_prompt_view(request_state),
            },
            "artifacts": {
                "refs": self._artifact_ref_index(request_state),
                "facts": self._artifact_fact_index(request_state),
            },
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
            return ["source_refs?", "database_evidence?", "analysis_goal", "insight_requests", "code?", "constraints?"]
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
            "constraints": self._outer_task_constraints(request_state.constraints),
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

    def _outer_task_constraints(self, constraints: dict | None) -> dict:
        """Expose semantic constraints, leaving presentation budgets to runtime."""

        return {
            key: value
            for key, value in (constraints or {}).items()
            if key != "max_points" and value not in (None, "", [], {})
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
        return {
            "iteration": request_state.iteration,
            "max_iterations": request_state.max_iterations,
        }

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

    def _artifact_fact_index(self, request_state: RequestStateModel) -> list[dict]:
        """Expose only immutable facts needed to compose a grounded final answer."""

        facts: list[dict] = []
        for evidence_id, evidence in list(request_state.database_evidence_artifacts.items())[-8:]:
            data = evidence.data if isinstance(evidence.data, dict) else {}
            records = data.get("rows") if isinstance(data.get("rows"), list) else data.get("points")
            records = [dict(item) for item in records or [] if isinstance(item, dict)]
            diagnostics = evidence.diagnostics if isinstance(evidence.diagnostics, dict) else {}
            metadata = evidence.metadata if isinstance(evidence.metadata, dict) else {}
            row_count = diagnostics.get("row_count_total")
            if not isinstance(row_count, int):
                row_count = len(records)
            facts.append(self._drop_empty({
                "source_ref": f"evidence:{evidence_id}",
                "kind": "database_evidence",
                "purpose": metadata.get("purpose"),
                "summary": evidence.summary,
                "result_type": evidence.result_type,
                "query_language": evidence.query_language,
                "query": self._truncate_text(str(evidence.query or ""), 1600) or None,
                "columns": list(evidence.columns or [])[:20],
                "row_count": row_count,
                "records": self._sample_records(records, limit=12),
                "materialization_complete": diagnostics.get("is_full_fidelity"),
            }))
        for evidence_id, evidence in list(request_state.derived_evidence_artifacts.items())[-8:]:
            records = [dict(item) for item in evidence.rows if isinstance(item, dict)]
            facts.append(self._drop_empty({
                "source_ref": f"derived_evidence:{evidence_id}",
                "kind": "derived_evidence",
                "name": evidence.name,
                "shape": evidence.shape,
                "row_count": len(records) if records else int(evidence.scalar is not None),
                "records": self._sample_records(records, limit=12),
                "scalar": self._bounded_value(evidence.scalar, max_list_items=8, max_dict_items=16),
            }))
        for forecast_id, forecast in list(request_state.forecast_artifacts.items())[-8:]:
            points = [item.model_dump(mode="json") for item in forecast.forecast_points]
            intervals = [item.model_dump(mode="json") for item in forecast.confidence_interval]
            facts.append(self._drop_empty({
                "source_ref": f"forecast:{forecast_id}",
                "kind": "forecast",
                "status": forecast.status,
                "model_name": forecast.model_name,
                "horizon": forecast.horizon,
                "forecast_point_count": len(points),
                "forecast_points": self._sample_records(points, limit=12),
                "confidence_interval": self._sample_records(intervals, limit=12),
            }))
        for anomaly_id, anomaly in list(request_state.anomaly_artifacts.items())[-8:]:
            points = [dict(item) for item in anomaly.anomaly_points if isinstance(item, dict)]
            facts.append(self._drop_empty({
                "source_ref": f"anomaly:{anomaly_id}",
                "kind": "anomaly",
                "detector_name": anomaly.detector_name,
                "anomaly_count": len(points),
                "anomaly_points": self._sample_records(points, limit=12),
            }))
        for visualization in request_state.visualizations[-8:]:
            verification = visualization.verification
            facts.append(self._drop_empty({
                "source_ref": f"visualization:{visualization.visualization_id}",
                "kind": "visualization",
                "title": visualization.title,
                "summary": visualization.summary,
                "verification": verification.model_dump(mode="json") if verification else None,
            }))
        return facts

    def _sample_records(self, records: list[dict], *, limit: int) -> list[dict]:
        if len(records) <= limit:
            selected = records
        else:
            edge = max(1, limit // 2)
            selected = [*records[:edge], *records[-edge:]]
        return [
            self._bounded_value(item, max_string_chars=500, max_list_items=6, max_dict_items=20)
            for item in selected
        ]

    @staticmethod
    def _drop_empty(payload: dict) -> dict:
        return {key: value for key, value in payload.items() if value not in (None, "", [], {})}

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
