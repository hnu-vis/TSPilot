"""Single outer ReAct loop."""
from __future__ import annotations

from typing import AsyncIterator

from app.settings import Settings
from runtime.action_policy import build_policy_observation, validate_action
from runtime.conversation_state import sync_from_request
from runtime.request_state import apply_observation, append_trace, build_final_response, enrich_observation_payload
from runtime.trace import TraceEventModel
from schemas.api import ChatResponse
from schemas.state import ConversationStateModel, RequestStateModel
from runtime.tool_executor import ToolExecutor
from agents.data_agent import DataAgent


class ReActLoop:
    """Execute the strict outer loop."""

    def __init__(self, data_agent: DataAgent, tool_executor: ToolExecutor, settings: Settings):
        self._data_agent = data_agent
        self._tool_executor = tool_executor
        self._settings = settings

    async def run(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> ChatResponse:
        trace_events = [event async for event in self._iterate(request_state, conversation_state)]
        return build_final_response(request_state, trace_events)

    async def run_sse(
        self,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> AsyncIterator[TraceEventModel]:
        yield append_trace(
            request_state,
            "conversation_id",
            {
                "conversation_id": request_state.conversation_id,
                "request_id": request_state.request_id,
            },
        )
        async for event in self._iterate(request_state, conversation_state):
            async for mapped in self._map_trace_to_sse(request_state, event):
                yield mapped

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
                {"iteration": request_state.iteration, "thought": turn.thought},
            )
            yield append_trace(
                request_state,
                "action",
                {
                    "iteration": request_state.iteration,
                    "action": turn.action,
                    "action_input": turn.action_input,
                },
            )

            allowed, reason, policy = validate_action(request_state, turn.action)
            if not allowed:
                observation = build_policy_observation(request_state, turn.action, policy, reason or "Invalid action.")
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
                )
            except Exception as exc:
                request_state.status = "failed"
                request_state.errors.append({"stage": turn.action, "message": str(exc)})
                yield append_trace(
                    request_state,
                    "error",
                    {"message": f"Tool '{turn.action}' failed: {exc}"},
                )
                return
            apply_observation(
                request_state,
                execution_result.observation,
                execution_result.full_payload,
                execution_result.tool_spec,
            )
            execution_result.observation = enrich_observation_payload(
                request_state,
                execution_result.observation,
                execution_result.full_payload,
                execution_result.tool_spec,
            )
            yield append_trace(
                request_state,
                "observation",
                execution_result.observation.model_dump(mode="json"),
            )
            sync_from_request(request_state, conversation_state)

            if turn.action == "format_answer":
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

    async def _map_trace_to_sse(
        self,
        request_state: RequestStateModel,
        event: TraceEventModel,
    ) -> AsyncIterator[TraceEventModel]:
        event_type = event.event_type
        payload = event.payload

        if event_type == "thought":
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
                    "payload_preview": self._payload_preview(payload),
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
                },
            )
            return

        if event_type in {"terminate", "error"}:
            yield append_trace(request_state, event_type, payload)

    def _phase_for_action(self, action_name: str) -> str:
        mapping = {
            "query_database": "tool_selection",
            "insight": "analysis",
            "format_answer": "answer_assembly",
            "forecast": "analysis",
            "anomaly": "analysis",
            "rag": "analysis",
            "skill": "analysis",
            "todowrite": "intent",
        }
        return mapping.get(action_name, "intent")

    def _message_for_action(self, action_name: str) -> str:
        mapping = {
            "query_database": "正在准备查询数据源。",
            "insight": "正在将证据转换为已验证事实。",
            "format_answer": "正在组装最终回答。",
            "forecast": "正在执行趋势预测。",
            "anomaly": "正在检测异常。",
            "rag": "正在补充外部知识。",
            "skill": "正在执行预定义技能。",
            "todowrite": "正在整理任务步骤。",
        }
        return mapping.get(action_name, "正在处理请求。")

    def _input_preview(self, action_name: str, action_input: dict) -> dict:
        if action_name == "query_database":
            database_context = action_input.get("database_context") or {}
            return {
                "database_id": database_context.get("database_id"),
                "database_type": database_context.get("database_type"),
                "time_range": action_input.get("time_range"),
            }
        if action_name == "insight":
            evidence = action_input.get("database_evidence") or {}
            return {
                "evidence_id": evidence.get("evidence_id"),
                "result_type": evidence.get("result_type"),
                "requested_fact_types": action_input.get("requested_fact_types", []),
            }
        if action_name == "format_answer":
            return {
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

    def _payload_preview(self, payload: dict) -> dict:
        preview = {
            "summary": payload.get("summary"),
            "payload_truncated": payload.get("payload_truncated", False),
        }
        visible_payload = payload.get("payload") or {}
        for key in ("evidence_id", "insight_id", "forecast_id", "anomaly_id", "title", "summary"):
            if key in visible_payload:
                preview[key] = visible_payload.get(key)
        if "visualizations" in visible_payload:
            preview["visualization_count"] = len(visible_payload.get("visualizations", []))
        if "verified_facts" in visible_payload:
            preview["verified_fact_count"] = len(visible_payload.get("verified_facts", []))
        if "todos" in visible_payload:
            todos = visible_payload.get("todos", [])
            preview["todo_total"] = len(todos)
            preview["in_progress"] = next(
                (todo.get("content") for todo in todos if todo.get("status") == "in_progress"),
                None,
            )
            preview["completed_count"] = visible_payload.get("completed_count")
            preview["pending_count"] = visible_payload.get("pending_count")
        if "results" in visible_payload:
            preview["result_count"] = len(visible_payload.get("results", []))
        return preview

    def _iteration_from_payload_ref(self, payload_ref: str | None) -> int | None:
        if not payload_ref:
            return None
        parts = payload_ref.split(":")
        if len(parts) >= 4 and parts[2].isdigit():
            return int(parts[2])
        return None
