"""Single outer ReAct loop."""
from __future__ import annotations

from typing import AsyncIterator

from app.settings import Settings
from runtime.action_policy import build_policy_observation, validate_action
from runtime.conversation_state import sync_from_request
from runtime.conversation_log import ConversationTraceLogger
from runtime.request_state import apply_observation_async, append_trace, build_final_response
from runtime.trace import TraceEventModel
from runtime.token_usage import token_usage_summary
from schemas.api import ChatResponse
from schemas.output import FinalAnswer
from schemas.state import ConversationStateModel, RequestStateModel
from runtime.tool_executor import ToolExecutor
from agents.data_agent import DataAgent
from schemas.tool import ToolObservation


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


class ReActLoop:
    """Execute the strict outer loop."""

    def __init__(self, data_agent: DataAgent, tool_executor: ToolExecutor, settings: Settings, goal_verifier=None):
        self._data_agent = data_agent
        self._tool_executor = tool_executor
        self._settings = settings
        self._goal_verifier = goal_verifier
        self._trace_logger = ConversationTraceLogger(settings)

    async def run(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> ChatResponse:
        trace_events = [event async for event in self._iterate(request_state, conversation_state)]
        response = build_final_response(request_state, trace_events)
        self._trace_logger.persist(
            request_state=request_state,
            response=response,
            internal_trace=trace_events,
            mode="json",
        )
        return response

    async def run_sse(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> AsyncIterator[TraceEventModel]:
        internal_trace: list[TraceEventModel] = []
        public_trace: list[TraceEventModel] = []
        logged = False
        try:
            conversation_event = append_trace(
                request_state,
                "conversation_id",
                {
                    "conversation_id": request_state.conversation_id,
                    "request_id": request_state.request_id,
                },
            )
            public_trace.append(conversation_event)
            yield conversation_event
            placeholder_event = append_trace(
                request_state,
                "agent_step",
                {
                    "agent": "data_agent",
                    "status": "running",
                    "phase": "reasoning",
                    "message": "正在判断下一步。",
                    "iteration": 1,
                    "placeholder": True,
                },
            )
            public_trace.append(placeholder_event)
            yield placeholder_event
            async for event in self._iterate(request_state, conversation_state):
                internal_trace.append(event)
                async for mapped in self._map_trace_to_sse(request_state, event):
                    public_trace.append(mapped)
                    yield mapped
            response = build_final_response(request_state, internal_trace)
            self._trace_logger.persist(
                request_state=request_state,
                response=response,
                internal_trace=internal_trace,
                public_trace=public_trace,
                mode="sse",
            )
            logged = True
        finally:
            if not logged and internal_trace:
                response = build_final_response(request_state, internal_trace)
                self._trace_logger.persist(
                    request_state=request_state,
                    response=response,
                    internal_trace=internal_trace,
                    public_trace=public_trace,
                    mode="sse",
                    interrupted=True,
                )

    async def _iterate(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> AsyncIterator[TraceEventModel]:
        while request_state.iteration < request_state.max_iterations:
            request_state.iteration += 1
            try:
                turn = await self._data_agent.next_turn(request_state, conversation_state)
            except Exception as exc:
                request_state.status = "failed"
                request_state.errors.append({"stage": "agent", "message": str(exc)})
                yield append_trace(
                    request_state,
                    "error",
                    {"message": f"Failed to obtain a valid model turn: {exc}"},
                )
                return
            yield append_trace(
                request_state,
                "thought",
                {
                    "iteration": request_state.iteration,
                    "thought": turn.thought,
                    "action_intention": turn.action_intention,
                    "action_reason": turn.action_reason,
                },
            )
            yield append_trace(
                request_state,
                "action",
                {
                    "iteration": request_state.iteration,
                    "thought": turn.thought,
                    "action": turn.action,
                    "action_input": turn.action_input,
                    "action_intention": turn.action_intention,
                    "action_reason": turn.action_reason,
                },
            )

            allowed, reason = validate_action(request_state, turn.action)
            if not allowed:
                observation = build_policy_observation(request_state, turn.action, reason or "Invalid action.")
                request_state.observations.append(observation)
                yield append_trace(
                    request_state,
                    "observation",
                    observation.model_dump(mode="json"),
                )
                sync_from_request(request_state, conversation_state)
                continue

            try:
                execution_result = await self._tool_executor.execute(
                    turn.action,
                    turn.action_input,
                    request_state,
                    conversation_state,
                    action_reason=turn.action_reason or turn.thought,
                )
            except Exception as exc:
                error_detail = _truncate_text(str(exc), 2000)
                message = f"Tool '{turn.action}' failed: {error_detail}"
                request_state.errors.append({"stage": turn.action, "message": error_detail})
                observation = ToolObservation(
                    tool_name=turn.action,
                    success=False,
                    summary=message,
                    payload={
                        "error": error_detail,
                        "recovery_hint": (
                            "Use the current context and this failure observation to choose the next best action. "
                            "You may correct the tool input, call a prerequisite tool, update todos, or answer with caveats."
                        ),
                    },
                    error=message,
                    payload_truncated=False,
                    payload_ref=None,
                )
                request_state.observations.append(observation)
                yield append_trace(
                    request_state,
                    "observation",
                    observation.model_dump(mode="json"),
                )
                sync_from_request(request_state, conversation_state)
                continue
            if execution_result.tool_spec.produces_terminal_payload:
                verifier_observation = await self._verify_terminal_candidate(
                    request_state,
                    execution_result.full_payload,
                )
                if verifier_observation is not None:
                    request_state.observations.append(verifier_observation)
                    yield append_trace(
                        request_state,
                        "observation",
                        verifier_observation.model_dump(mode="json"),
                    )
                    sync_from_request(request_state, conversation_state)
                    continue

            execution_result.observation = await apply_observation_async(
                request_state,
                execution_result.observation,
                execution_result.full_payload,
                execution_result.tool_spec,
                thought=turn.thought,
                action_reason=turn.action_reason,
            )
            yield append_trace(
                request_state,
                "observation",
                execution_result.observation.model_dump(mode="json"),
            )
            sync_from_request(request_state, conversation_state)

            if execution_result.tool_spec.produces_terminal_payload:
                request_state.status = "completed"
                yield append_trace(
                    request_state,
                    "final_answer",
                    request_state.final_answer_draft.model_dump(mode="json") if request_state.final_answer_draft else {},
                )
                yield append_trace(
                    request_state,
                    "terminate",
                    {"request_id": request_state.request_id, "status": request_state.status},
                )
                return

        request_state.status = "failed"
        yield append_trace(
            request_state,
            "error",
            {"message": "Maximum iterations reached before a terminal payload was produced."},
        )

    async def _verify_terminal_candidate(
        self,
        request_state: RequestStateModel,
        full_payload: dict,
    ) -> ToolObservation | None:
        if self._goal_verifier is None or request_state.database_context is None:
            return None
        try:
            candidate = FinalAnswer.model_validate(full_payload)
        except Exception as exc:
            return ToolObservation(
                tool_name="goal_verifier",
                success=False,
                summary=f"Final answer candidate was invalid: {exc}",
                payload={
                    "can_answer": False,
                    "missing_items": ["valid_final_answer"],
                    "unsupported_claims": [],
                    "next_action_hint": "Call terminate again with a valid final answer payload.",
                    "reason": str(exc),
                },
                error=str(exc),
                payload_truncated=False,
                payload_ref=None,
            )
        try:
            verdict = await self._goal_verifier.verify_final_answer(
                request_state=request_state,
                candidate_answer=candidate,
            )
        except Exception as exc:
            request_state.completion_state["goal_verifier_error"] = str(exc)
            return ToolObservation(
                tool_name="goal_verifier",
                success=False,
                summary=f"Final answer verification failed: {exc}",
                payload={
                    "can_answer": False,
                    "missing_items": ["goal_verification"],
                    "unsupported_claims": [],
                    "next_action_hint": "Continue with grounded evidence or retry final assembly after resolving the verifier error.",
                    "reason": str(exc),
                },
                error=str(exc),
                payload_truncated=False,
                payload_ref=None,
            )

        verdict_payload = verdict.model_dump(mode="json")
        request_state.completion_state["latest_goal_verification"] = verdict_payload
        if verdict.can_answer:
            request_state.completion_state["goal_verifier_failures"] = 0
            return None

        previous_failures = int(request_state.completion_state.get("goal_verifier_failures") or 0)
        max_rejections = max(0, int(getattr(self._settings, "goal_verifier_max_rejections", 1)))
        if previous_failures >= max_rejections:
            request_state.completion_state["goal_verifier_bypassed"] = {
                **verdict_payload,
                "candidate_summary": candidate.summary,
                "failure_count": previous_failures,
                "reason": verdict.reason or "Final answer did not satisfy the verifier.",
            }
            return None

        failures = previous_failures + 1
        request_state.completion_state["goal_verifier_failures"] = failures
        request_state.completion_state["semantic_repair_directive"] = {
            "source": "goal_verifier",
            "reason": verdict.reason or "Final answer did not satisfy the user task.",
            "missing_items": verdict.missing_items,
            "unsupported_claims": verdict.unsupported_claims,
            "next_action_hint": verdict.next_action_hint,
            "candidate_summary": candidate.summary,
            "failure_count": failures,
        }
        return ToolObservation(
            tool_name="goal_verifier",
            success=False,
            summary="Final answer rejected by verifier: " + (verdict.reason or "missing required coverage"),
            payload={
                **verdict_payload,
                "candidate_summary": candidate.summary,
                "recovery_hint": verdict.next_action_hint
                or "Use the missing_items and unsupported_claims to choose the next sql_query, insight, or corrected terminate action.",
                "failure_count": failures,
            },
            error=verdict.reason or "Final answer did not satisfy the user task.",
            payload_truncated=False,
            payload_ref=None,
        )

    async def _map_trace_to_sse(
        self,
        request_state: RequestStateModel,
        event: TraceEventModel,
    ) -> AsyncIterator[TraceEventModel]:
        event_type = event.event_type
        payload = event.payload

        if event_type == "thought":
            yield append_trace(
                request_state,
                "thought",
                {
                    "agent": "data_agent",
                    "status": "running",
                    "phase": "reasoning",
                    "message": payload.get("thought") or "正在判断下一步。",
                    "iteration": payload.get("iteration"),
                    "thought": payload.get("thought"),
                    "intention": payload.get("action_intention"),
                    "reason": payload.get("action_reason"),
                },
            )
            return

        if event_type == "action":
            action_name = str(payload.get("action", ""))
            yield append_trace(
                request_state,
                "agent_step",
                {
                    "agent": "data_agent",
                    "status": "running",
                    "phase": self._phase_for_action(action_name),
                    "message": self._message_for_action(action_name),
                    "iteration": payload.get("iteration"),
                },
            )
            yield append_trace(
                request_state,
                "tool_call",
                {
                    "tool": action_name,
                    "summary": self._message_for_action(action_name),
                    "iteration": payload.get("iteration"),
                    "thought": payload.get("thought"),
                    "intention": payload.get("action_intention"),
                    "reason": payload.get("action_reason"),
                    "action_input": payload.get("action_input", {}),
                    "input_preview": self._input_preview(action_name, payload.get("action_input", {})),
                },
            )
            return

        if event_type == "observation":
            yield append_trace(
                request_state,
                "tool_result",
                {
                    "tool": payload.get("tool_name"),
                    "success": payload.get("success", False),
                    "summary": payload.get("summary"),
                    "iteration": self._iteration_from_payload_ref(payload.get("payload_ref")) or request_state.iteration,
                    "payload_preview": self._payload_preview(payload, request_state),
                    "observation": payload,
                    "payload_ref": payload.get("payload_ref"),
                },
            )
            return

        if event_type == "final_answer":
            yield append_trace(
                request_state,
                "agent_step",
                {
                    "agent": "data_agent",
                    "status": "complete",
                    "phase": "answer_assembly",
                    "message": "最终答案已组装完成。",
                    "iteration": request_state.iteration,
                },
            )
            yield append_trace(
                request_state,
                "final_answer",
                {
                    "conversation_id": request_state.conversation_id,
                    "request_id": request_state.request_id,
                    "answer": payload,
                    "token_usage": token_usage_summary(request_state),
                },
            )
            return

        if event_type in {"terminate", "error"}:
            yield append_trace(request_state, event_type, payload)

    def _phase_for_action(self, action_name: str) -> str:
        mapping = {
            "sql_query": "tool_selection",
            "insight": "analysis",
            "format_answer": "answer_assembly",
            "terminate": "answer_assembly",
            "forecast": "analysis",
            "anomaly": "analysis",
            "rag": "analysis",
            "skill": "analysis",
            "todowrite": "intent",
        }
        return mapping.get(action_name, "intent")

    def _message_for_action(self, action_name: str) -> str:
        mapping = {
            "sql_query": "正在查询数据源。",
            "insight": "正在将证据转换为已验证事实。",
            "format_answer": "正在组装最终回答。",
            "terminate": "正在结束并组装最终回答。",
            "forecast": "正在执行趋势预测。",
            "anomaly": "正在检测异常。",
            "rag": "正在补充外部知识。",
            "skill": "正在执行预定义技能。",
            "todowrite": "正在整理任务步骤。",
        }
        return mapping.get(action_name, "正在处理请求。")

    def _input_preview(self, action_name: str, action_input: dict) -> dict:
        if action_name == "sql_query" and not action_input.get("query"):
            database_context = action_input.get("database_context") or {}
            return {
                "database_id": database_context.get("database_id"),
                "database_type": database_context.get("database_type"),
                "time_range": action_input.get("time_range"),
            }
        if action_name == "sql_query" and action_input.get("query"):
            database_context = action_input.get("database_context") or {}
            return {
                "database_id": database_context.get("database_id"),
                "database_type": database_context.get("database_type"),
                "query_language": action_input.get("query_language"),
                "purpose": action_input.get("purpose"),
            }
        if action_name == "insight":
            evidence = action_input.get("database_evidence") or {}
            evidence_id = evidence.get("evidence_id") if isinstance(evidence, dict) else evidence
            return {
                "evidence_id": evidence_id,
                "analysis_goal": action_input.get("analysis_goal") or action_input.get("focus"),
                "code_type": action_input.get("code_type"),
                "analysis_code_chars": len(str(action_input.get("analysis_code") or "")),
            }
        if action_name in {"format_answer", "terminate"}:
            return {
                "has_result": bool(action_input.get("result") or action_input.get("direct_answer")),
                "include_analysis_count": len(action_input.get("include_analysis_ids", [])),
                "include_fact_count": len(action_input.get("include_fact_ids", [])),
                "include_visualization_count": len(action_input.get("include_visualization_ids", [])),
                "section_plan": action_input.get("section_plan", []),
            }
        if action_name == "todowrite":
            return {
                "todo_count": len(action_input.get("todos", [])),
                "focus": action_input.get("focus"),
            }
        if action_name == "rag":
            return {"query": action_input.get("query"), "filters": action_input.get("filters", {})}
        if action_name == "skill":
            return {"skill_name": action_input.get("skill_name")}
        return {}

    def _payload_preview(self, payload: dict, request_state: RequestStateModel | None = None) -> dict:
        preview = {
            "summary": payload.get("summary"),
            "payload_truncated": payload.get("payload_truncated", False),
        }
        visible_payload = payload.get("payload") or {}
        for key in ("evidence_id", "analysis_id", "analysis_goal", "code_type", "code_hash", "input_row_count", "insight_id", "forecast_id", "anomaly_id", "title", "summary"):
            if key in visible_payload:
                preview[key] = visible_payload.get(key)
        if isinstance(visible_payload.get("result"), dict):
            preview["result_preview"] = visible_payload["result"]
        if "visualizations" in visible_payload:
            preview["visualization_count"] = len(visible_payload.get("visualizations", []))
        if "verified_facts" in visible_payload:
            preview["verified_fact_count"] = len(visible_payload.get("verified_facts", []))
        if "todos" in visible_payload:
            todos = visible_payload.get("todos", [])
            preview["todo_total"] = len(todos)
            preview["todos"] = [
                {
                    "content": todo.get("content"),
                    "task_type": todo.get("task_type"),
                    "status": todo.get("status"),
                    "priority": todo.get("priority"),
                }
                for todo in todos[:12]
                if isinstance(todo, dict)
            ]
            preview["in_progress"] = next(
                (todo.get("content") for todo in todos if todo.get("status") == "in_progress"),
                None,
            )
            preview["completed_count"] = visible_payload.get("completed_count")
            preview["pending_count"] = visible_payload.get("pending_count")
        if "results" in visible_payload:
            preview["result_count"] = len(visible_payload.get("results", []))
        if payload.get("tool_name") in {"sql_query", "query_database"}:
            preview.update(self._sql_payload_preview(visible_payload))
        if request_state is not None:
            preview.update(self._completion_payload_preview(request_state))
        return preview

    def _completion_payload_preview(self, request_state: RequestStateModel) -> dict:
        latest_step = request_state.completion_state.get("latest_step")
        todo_total = len(request_state.todo_list)
        completed = len([todo for todo in request_state.todo_list if todo.get("status") == "completed"])
        in_progress = next((todo for todo in request_state.todo_list if todo.get("status") == "in_progress"), None)
        preview = {
            "todo_progress": {
                "total": todo_total,
                "completed": completed,
                "in_progress": in_progress.get("content") if isinstance(in_progress, dict) else None,
            },
            "todos": [
                {
                    "content": todo.get("content"),
                    "task_type": todo.get("task_type"),
                    "status": todo.get("status"),
                    "priority": todo.get("priority"),
                }
                for todo in request_state.todo_list[:12]
                if isinstance(todo, dict)
            ],
            "todo_total": todo_total,
            "completed_count": completed,
            "pending_count": len([todo for todo in request_state.todo_list if todo.get("status") == "pending"]),
            "in_progress": in_progress.get("content") if isinstance(in_progress, dict) else None,
        }
        if isinstance(latest_step, dict):
            preview["last_progress_update"] = {
                "completed": latest_step.get("completed"),
                "reason": latest_step.get("reason"),
                "todo_index": latest_step.get("todo_index"),
            }
        return preview

    def _sql_payload_preview(self, visible_payload: dict) -> dict:
        data = visible_payload.get("data") if isinstance(visible_payload.get("data"), dict) else {}
        diagnostics = visible_payload.get("diagnostics") if isinstance(visible_payload.get("diagnostics"), dict) else {}
        summary_stats = diagnostics.get("summary_stats") if isinstance(diagnostics.get("summary_stats"), dict) else {}
        rows = data.get("rows") if isinstance(data.get("rows"), list) else []
        points = data.get("points") if isinstance(data.get("points"), list) else []
        columns = visible_payload.get("columns") if isinstance(visible_payload.get("columns"), list) else []
        row_count = summary_stats.get("rows_count")
        point_count = summary_stats.get("points_count")
        if row_count is None and isinstance(visible_payload.get("row_count"), int):
            row_count = visible_payload.get("row_count")
        if point_count is None and isinstance(visible_payload.get("point_count"), int):
            point_count = visible_payload.get("point_count")

        return {
            "query_language": visible_payload.get("query_language"),
            "query": self._truncate_preview_text(visible_payload.get("query"), 5000),
            "columns": columns[:40],
            "row_count": row_count if row_count is not None else len(rows),
            "point_count": point_count if point_count is not None else len(points),
            "sample_rows": rows[:5],
            "sample_points": points[:5],
            "sampling": diagnostics.get("prompt_sampling"),
            "schema_linking": self._schema_linking_payload_preview(diagnostics),
            "task_coverage": diagnostics.get("task_coverage") if isinstance(diagnostics.get("task_coverage"), dict) else None,
            "truncated": bool(
                visible_payload.get("payload_truncated")
                or diagnostics.get("truncated")
                or diagnostics.get("artifact_ref")
                or payload_truncated_marker(visible_payload)
            ),
        }

    def _schema_linking_payload_preview(self, diagnostics: dict) -> dict | None:
        query_trace = diagnostics.get("query_trace") if isinstance(diagnostics.get("query_trace"), dict) else {}
        logical_plan = query_trace.get("logical_plan") if isinstance(query_trace.get("logical_plan"), dict) else {}
        linking = logical_plan.get("schema_linking") if isinstance(logical_plan.get("schema_linking"), dict) else None
        if not linking:
            return None
        filters = logical_plan.get("filters") if isinstance(logical_plan.get("filters"), list) else []
        required_filters = [
            {
                "source": item.get("source"),
                "column": item.get("column"),
                "operator": item.get("operator"),
                "value": item.get("value"),
            }
            for item in filters
            if isinstance(item, dict)
            and item.get("column") not in {"_measurement", "_field", "_time", "time", "timestamp", "_start", "_stop", "result", "table"}
        ]
        field_mappings = query_trace.get("field_mappings") if isinstance(query_trace.get("field_mappings"), list) else []
        return {
            "confidence": linking.get("confidence"),
            "sources": linking.get("sources", [])[:6] if isinstance(linking.get("sources"), list) else [],
            "time_columns": linking.get("time_columns", [])[:8] if isinstance(linking.get("time_columns"), list) else [],
            "value_columns": linking.get("value_columns", [])[:12] if isinstance(linking.get("value_columns"), list) else [],
            "ambiguous_terms": linking.get("ambiguous_terms") if isinstance(linking.get("ambiguous_terms"), dict) else {},
            "evidence": linking.get("evidence", [])[:8] if isinstance(linking.get("evidence"), list) else [],
            "field_mappings": field_mappings[:12],
            "required_filters": required_filters[:12],
        }

    def _truncate_preview_text(self, value, max_chars: int):
        if not isinstance(value, str):
            return value
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"

    def _iteration_from_payload_ref(self, payload_ref: str | None) -> int | None:
        if not payload_ref:
            return None
        parts = payload_ref.split(":")
        if len(parts) >= 4 and parts[2].isdigit():
            return int(parts[2])
        return None


def payload_truncated_marker(value) -> bool:
    if isinstance(value, dict):
        return any(key in value for key in ("truncated_items", "truncated_keys"))
    if isinstance(value, list):
        return any(payload_truncated_marker(item) for item in value)
    return False
