"""Single outer ReAct loop."""
from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import AsyncIterator

from app.settings import Settings
from core.data_fact.retriever import EmbeddingFactMemoryRetriever, MemoryRetrievalResult
from core.database.dialects import dialect_for_database
from core.harness import ActionOutputBuilder, StateTransitionEngine
from core.harness.observation_view import public_observation_view
from runtime.action_policy import build_policy_observation, validate_action
from runtime.conversation_state import sync_from_request
from runtime.conversation_log import ConversationTraceLogger
from runtime.request_state import (
    append_react_transcript_step,
    apply_task_contract,
    append_trace,
    build_final_response,
    public_final_answer_payload,
)
from core.completion import apply_previous_observation_assessment
from runtime.trace import TraceEventModel
from runtime.token_usage import token_usage_summary
from schemas.api import ChatResponse
from schemas.state import ConversationStateModel, RequestStateModel
from runtime.tool_executor import ToolExecutor
from agents.data_agent import DataAgent
from schemas.tool import ToolObservation
from tools.base import StructuredToolError

HEARTBEAT_INTERVAL_SECONDS = 1.0


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


class ReActLoop:
    """Execute the strict outer loop."""

    def __init__(
        self,
        data_agent: DataAgent,
        tool_executor: ToolExecutor,
        settings: Settings,
        memory_retriever: EmbeddingFactMemoryRetriever | None = None,
    ):
        self._data_agent = data_agent
        self._tool_executor = tool_executor
        self._settings = settings
        self._memory_retriever = memory_retriever
        self._trace_logger = ConversationTraceLogger(settings)
        self._transition_engine = StateTransitionEngine()
        self._action_output_builder = ActionOutputBuilder()

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
                    "message": "正在理解问题并选择下一步工具。",
                    "iteration": 1,
                    "placeholder": True,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": 0.0,
                },
            )
            public_trace.append(placeholder_event)
            yield placeholder_event
            async for event in self._iterate(request_state, conversation_state, emit_heartbeats=True):
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
        *,
        emit_heartbeats: bool = False,
    ) -> AsyncIterator[TraceEventModel]:
        await self._initialize_memory_context(request_state)
        while request_state.iteration < request_state.max_iterations:
            request_state.iteration += 1
            try:
                turn_task = asyncio.create_task(self._data_agent.next_turn(request_state, conversation_state))
                async for heartbeat in self._heartbeat_until_done(
                    request_state,
                    turn_task,
                    emit_heartbeats=emit_heartbeats,
                    phase="reasoning",
                    message="正在选择下一步工具。",
                    iteration=request_state.iteration,
                ):
                    yield heartbeat
                turn = await turn_task
            except asyncio.CancelledError:
                if "turn_task" in locals() and not turn_task.done():
                    turn_task.cancel()
                    try:
                        await turn_task
                    except asyncio.CancelledError:
                        pass
                raise
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
                    "task_contract": (
                        turn.task_contract.model_dump(mode="json")
                        if turn.task_contract
                        else None
                    ),
                    "action_intention": turn.action_intention,
                    "action_reason": turn.action_reason,
                },
            )
            try:
                contract = apply_task_contract(request_state, turn.task_contract)
            except Exception as exc:
                observation = build_policy_observation(
                    request_state,
                    turn.action,
                    f"Task contract was invalid: {exc}",
                )
                request_state.observations.append(observation)
                action_output = self._store_observation_action_output(
                    request_state,
                    observation,
                    action_input=turn.action_input,
                    result_target="policy",
                )
                self._attach_action_output_timing(action_output, self._instant_tool_timing())
                append_react_transcript_step(
                    request_state,
                    iteration=request_state.iteration,
                    thought=turn.thought,
                    action=turn.action,
                    action_input=turn.action_input,
                    observation=observation,
                    action_intention=turn.action_intention,
                    action_reason=turn.action_reason,
                )
                yield append_trace(
                    request_state,
                    "action_output",
                    action_output.model_dump(mode="json"),
                )
                sync_from_request(request_state, conversation_state)
                continue

            yield append_trace(
                request_state,
                "action",
                {
                    "iteration": request_state.iteration,
                    "thought": turn.thought,
                    "previous_observation_assessment": (
                        turn.previous_observation_assessment.model_dump(mode="json")
                        if turn.previous_observation_assessment
                        else None
                    ),
                    "task_contract": (
                        contract.model_dump(mode="json")
                        if contract is not None
                        else None
                    ),
                    "action": turn.action,
                    "action_input": turn.action_input,
                    "action_intention": turn.action_intention,
                    "action_reason": turn.action_reason,
                },
            )

            if turn.previous_observation_assessment is not None:
                had_active_todo = any(
                    todo.get("status") == "in_progress"
                    for todo in request_state.todo_list
                    if isinstance(todo, dict)
                )
                assessment = apply_previous_observation_assessment(
                    request_state,
                    turn.previous_observation_assessment,
                )
                yield append_trace(
                    request_state,
                    "todo_assessment",
                    {
                        **assessment.model_dump(),
                        "iteration": request_state.iteration,
                        "assessment": turn.previous_observation_assessment.model_dump(mode="json"),
                    },
                )
                if had_active_todo and turn.previous_observation_assessment.completed_active_todo and not assessment.completed:
                    observation = ToolObservation(
                        tool_name="todo_assessment",
                        success=False,
                        summary=assessment.reason,
                        payload={
                            "completion_state": request_state.completion_state,
                            "recovery_hint": assessment.next_action_hint
                            or "Use the latest observation to choose the next valid action.",
                        },
                        error=assessment.reason,
                        payload_truncated=False,
                        payload_ref=None,
                    )
                    request_state.observations.append(observation)
                    action_output = self._store_observation_action_output(
                        request_state,
                        observation,
                        action_input=turn.action_input,
                        result_target="policy",
                    )
                    self._attach_action_output_timing(action_output, self._instant_tool_timing())
                    append_react_transcript_step(
                        request_state,
                        iteration=request_state.iteration,
                        thought=turn.thought,
                        action=turn.action,
                        action_input=turn.action_input,
                        observation=observation,
                        action_intention=turn.action_intention,
                        action_reason=turn.action_reason,
                    )
                    yield append_trace(
                        request_state,
                        "action_output",
                        action_output.model_dump(mode="json"),
                    )
                    sync_from_request(request_state, conversation_state)
                    continue

            allowed, reason = validate_action(
                request_state,
                turn.action,
                turn.action_input,
                action_reason=turn.action_reason,
            )
            if not allowed:
                observation = build_policy_observation(request_state, turn.action, reason or "Invalid action.")
                request_state.observations.append(observation)
                action_output = self._store_observation_action_output(
                    request_state,
                    observation,
                    action_input=turn.action_input,
                    result_target="policy",
                )
                self._attach_action_output_timing(action_output, self._instant_tool_timing())
                append_react_transcript_step(
                    request_state,
                    iteration=request_state.iteration,
                    thought=turn.thought,
                    action=turn.action,
                    action_input=turn.action_input,
                    observation=observation,
                    action_intention=turn.action_intention,
                    action_reason=turn.action_reason,
                )
                yield append_trace(
                    request_state,
                    "action_output",
                    action_output.model_dump(mode="json"),
                )
                sync_from_request(request_state, conversation_state)
                continue

            try:
                tool_started_at = datetime.now(timezone.utc).isoformat()
                tool_started_monotonic = time.monotonic()
                tool_task = asyncio.create_task(self._tool_executor.execute(
                    turn.action,
                    turn.action_input,
                    request_state,
                    conversation_state,
                    action_reason=turn.action_reason or turn.thought,
                ))
                async for heartbeat in self._heartbeat_until_done(
                    request_state,
                    tool_task,
                    emit_heartbeats=emit_heartbeats,
                    phase=self._phase_for_action(turn.action),
                    message=f"正在执行 {turn.action}",
                    iteration=request_state.iteration,
                    tool=turn.action,
                ):
                    yield heartbeat
                execution_result = await tool_task
                tool_timing = self._tool_timing(tool_started_monotonic, tool_started_at)
            except asyncio.CancelledError:
                if "tool_task" in locals() and not tool_task.done():
                    tool_task.cancel()
                    try:
                        await tool_task
                    except asyncio.CancelledError:
                        pass
                raise
            except Exception as exc:
                if "tool_started_monotonic" in locals() and "tool_started_at" in locals():
                    tool_timing = self._tool_timing(tool_started_monotonic, tool_started_at)
                else:
                    tool_timing = self._instant_tool_timing()
                error_detail = _truncate_text(str(exc), 2000)
                message = f"Tool '{turn.action}' failed: {error_detail}"
                request_state.errors.append({"stage": turn.action, "message": error_detail})
                if isinstance(exc, StructuredToolError):
                    payload = {
                        **exc.to_observation_payload(),
                        "recovery_hint": (
                            "Use this structured failure observation to choose the next best ReAct action. "
                            "Do not repeat an equivalent failing action unless the action_input materially addresses the diagnostics."
                        ),
                    }
                else:
                    payload = {
                        "error": error_detail,
                        "recovery_hint": (
                            "Use the current context and this failure observation to choose the next best action. "
                            "You may correct the tool input, call a prerequisite tool, update todos, or answer with caveats."
                        ),
                    }
                payload.update(self._failure_strategy_payload(request_state, turn.action, payload))
                observation = ToolObservation(
                    tool_name=turn.action,
                    success=False,
                    summary=message,
                    payload=payload,
                    error=message,
                    payload_truncated=False,
                    payload_ref=None,
                )
                request_state.observations.append(observation)
                action_output = self._store_observation_action_output(
                    request_state,
                    observation,
                    action_input=turn.action_input,
                    result_target="tool_error",
                )
                self._attach_action_output_timing(action_output, tool_timing)
                append_react_transcript_step(
                    request_state,
                    iteration=request_state.iteration,
                    thought=turn.thought,
                    action=turn.action,
                    action_input=turn.action_input,
                    observation=observation,
                    action_intention=turn.action_intention,
                    action_reason=turn.action_reason,
                )
                yield append_trace(
                    request_state,
                    "action_output",
                    action_output.model_dump(mode="json"),
                )
                sync_from_request(request_state, conversation_state)
                continue
            transition_result = await self._transition_engine.apply_async(
                request_state,
                execution_result.observation,
                execution_result.full_payload,
                execution_result.tool_spec,
            )
            execution_result.observation = transition_result.observation
            self._attach_action_output_timing(execution_result.action_output, tool_timing)
            self._attach_todo_snapshot(execution_result.action_output, request_state)
            self._store_action_output(request_state, execution_result.action_output)
            append_react_transcript_step(
                request_state,
                iteration=request_state.iteration,
                thought=turn.thought,
                action=turn.action,
                action_input=turn.action_input,
                observation=execution_result.observation,
                action_intention=turn.action_intention,
                action_reason=turn.action_reason,
            )
            yield append_trace(
                request_state,
                "action_output",
                execution_result.action_output.model_dump(mode="json"),
            )
            artifact_payload = self._artifact_registered_payload(execution_result.action_output)
            if artifact_payload:
                yield append_trace(request_state, "artifact_registered", artifact_payload)
            coverage_payload = self._coverage_updated_payload(request_state, execution_result.action_output)
            if coverage_payload:
                yield append_trace(request_state, "coverage_updated", coverage_payload)
            todo_payload = self._todo_snapshot(request_state)
            if todo_payload:
                yield append_trace(request_state, "todo_updated", todo_payload)
            sync_from_request(request_state, conversation_state)

            if execution_result.tool_spec.produces_terminal_payload:
                request_state.status = "completed"
                yield append_trace(
                    request_state,
                    "final_answer",
                    public_final_answer_payload(request_state.final_answer_draft.model_dump(mode="json"))
                    if request_state.final_answer_draft
                    else {},
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

    async def _initialize_memory_context(self, request_state: RequestStateModel) -> None:
        if "memory_context" in request_state.completion_state:
            return
        if self._memory_retriever is None:
            request_state.completion_state["memory_context"] = {
                "hits": [],
                "fact_requests": [],
                "selected_card_ids": [],
                "diagnostics": {"memory_enabled": False, "reason": "No memory retriever configured."},
                "used_by_tools": {},
            }
            return
        result = await self._memory_retriever.retrieve_once(request_state=request_state)
        if not isinstance(result, MemoryRetrievalResult):
            result = MemoryRetrievalResult.model_validate(result)
        request_state.completion_state["memory_context"] = {
            "hits": [hit.model_dump(mode="json") for hit in result.hits],
            "fact_requests": [item.model_dump(mode="json", exclude_none=True) for item in result.fact_requests],
            "selected_card_ids": [hit.card_id for hit in result.hits],
            "diagnostics": result.diagnostics,
            "used_by_tools": {},
        }

    def _store_observation_action_output(
        self,
        request_state: RequestStateModel,
        observation: ToolObservation,
        *,
        action_input: dict | None,
        result_target: str,
    ):
        action_output = self._action_output_builder.from_observation(
            observation,
            result_target=result_target,
            action_input=action_input or {},
            iteration=request_state.iteration,
            request_id=request_state.request_id,
        )
        self._store_action_output(request_state, action_output)
        return action_output

    def _store_action_output(self, request_state: RequestStateModel, action_output):
        request_state.action_outputs.append(action_output)
        request_state.latest_action_output = action_output
        if isinstance(action_output.memory_fragment, dict):
            request_state.memory_fragments.append(action_output.memory_fragment)
        elif isinstance(action_output.memory_fragment, str) and action_output.memory_fragment.strip():
            request_state.memory_fragments.append(
                {
                    "iteration": (action_output.meta or {}).get("iteration"),
                    "action": action_output.tool_name,
                    "observation": action_output.memory_fragment,
                    "resource_ref": action_output.resource_ref,
                    "status": "succeeded" if action_output.success else "failed",
                }
            )
        if action_output.resource_ref:
            resources = request_state.resource_index.setdefault("resources", {})
            resources[action_output.resource_ref] = {
                "tool_name": action_output.tool_name,
                "resource_type": action_output.resource_type,
                "iteration": (action_output.meta or {}).get("iteration"),
                "status": "succeeded" if action_output.success else "failed",
            }

    def _instant_tool_timing(self) -> dict:
        started_at = datetime.now(timezone.utc).isoformat()
        return {
            "started_at": started_at,
            "completed_at": started_at,
            "duration_ms": 0,
            "elapsed_seconds": 0.0,
        }

    def _tool_timing(self, started_monotonic: float, started_at: str) -> dict:
        elapsed = max(0.0, time.monotonic() - started_monotonic)
        return {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int(round(elapsed * 1000)),
            "elapsed_seconds": round(elapsed, 1),
        }

    def _attach_action_output_timing(self, action_output, timing: dict | None) -> None:
        if not isinstance(timing, dict):
            return
        meta = dict(action_output.meta or {})
        for key in ("started_at", "completed_at", "duration_ms", "elapsed_seconds"):
            value = timing.get(key)
            if value is not None:
                meta[key] = value
        action_output.meta = meta

    def _attach_todo_snapshot(self, action_output, request_state: RequestStateModel) -> None:
        snapshot = self._todo_snapshot(request_state)
        if not snapshot:
            return
        view = dict(action_output.view or {})
        payload = dict(view.get("payload") or view)
        payload.update(snapshot)
        if isinstance(action_output.view, dict) and "payload" in action_output.view:
            view["payload"] = payload
        else:
            view = payload
        action_output.view = view

    def _todo_snapshot(self, request_state: RequestStateModel) -> dict:
        if not request_state.todo_list:
            return {}
        todos = [
            {
                key: todo.get(key)
                for key in ("content", "task_type", "status", "priority", "acceptance_criteria", "result_ref", "completion_reason")
                if isinstance(todo, dict) and todo.get(key) not in (None, "", [], {})
            }
            for todo in request_state.todo_list[:12]
            if isinstance(todo, dict)
        ]
        completed = sum(1 for todo in request_state.todo_list if isinstance(todo, dict) and todo.get("status") == "completed")
        in_progress = next(
            (
                str(todo.get("content") or "")
                for todo in request_state.todo_list
                if isinstance(todo, dict) and todo.get("status") == "in_progress"
            ),
            None,
        )
        latest_step = request_state.completion_state.get("latest_step")
        return {
            "current_step": request_state.plan_current_step,
            "planning_complete": request_state.planning_complete,
            "todo_total": len(request_state.todo_list),
            "completed_count": completed,
            "pending_count": len(request_state.todo_list) - completed,
            "todos": todos,
            "todo_progress": {
                "total": len(request_state.todo_list),
                "completed": completed,
                "in_progress": in_progress,
            },
            "latest_step": latest_step if isinstance(latest_step, dict) else None,
        }

    def _artifact_registered_payload(self, action_output) -> dict:
        if not action_output.resource_ref:
            return {}
        return {
            "tool": action_output.tool_name,
            "resource_ref": action_output.resource_ref,
            "resource_type": action_output.resource_type,
            "success": action_output.success,
            "iteration": (action_output.meta or {}).get("iteration"),
        }

    def _coverage_updated_payload(self, request_state: RequestStateModel, action_output) -> dict:
        coverage = getattr(request_state, "fact_coverage", None)
        latest_goal = request_state.completion_state.get("latest_goal")
        latest_step = request_state.completion_state.get("latest_step")
        if coverage is None and not isinstance(latest_goal, dict) and not isinstance(latest_step, dict):
            return {}
        payload = {
            "tool": action_output.tool_name,
            "iteration": (action_output.meta or {}).get("iteration"),
            "resource_ref": action_output.resource_ref,
        }
        if coverage is not None:
            payload["fact_coverage"] = coverage.model_dump(mode="json") if hasattr(coverage, "model_dump") else coverage
        if isinstance(latest_goal, dict):
            payload["goal_coverage"] = latest_goal
        if isinstance(latest_step, dict):
            payload["step_coverage"] = latest_step
        return payload

    async def _heartbeat_until_done(
        self,
        request_state: RequestStateModel,
        task: asyncio.Task,
        *,
        emit_heartbeats: bool,
        phase: str,
        message: str,
        iteration: int,
        tool: str | None = None,
    ) -> AsyncIterator[TraceEventModel]:
        if not emit_heartbeats:
            return
        started_at = time.monotonic()
        started_at_iso = datetime.now(timezone.utc).isoformat()
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=HEARTBEAT_INTERVAL_SECONDS)
            except TimeoutError:
                elapsed = round(time.monotonic() - started_at, 1)
                payload = {
                    "agent": "data_agent",
                    "status": "running",
                    "phase": phase,
                    "message": message,
                    "iteration": iteration,
                    "heartbeat": True,
                    "started_at": started_at_iso,
                    "elapsed_seconds": elapsed,
                }
                if tool:
                    payload["tool"] = tool
                yield append_trace(request_state, "agent_step", payload)
            else:
                return

    async def _map_trace_to_sse(
        self,
        request_state: RequestStateModel,
        event: TraceEventModel,
    ) -> AsyncIterator[TraceEventModel]:
        event_type = event.event_type
        payload = event.payload

        if event_type == "agent_step":
            yield event
            return

        if event_type == "action_output":
            iteration = payload.get("meta", {}).get("iteration") if isinstance(payload.get("meta"), dict) else request_state.iteration
            view = payload.get("view") if isinstance(payload.get("view"), dict) else {}
            timing = self._timing_from_action_output_payload(payload)
            result_target = payload.get("meta", {}).get("result_target") if isinstance(payload.get("meta"), dict) else None
            if result_target == "policy":
                yield append_trace(
                    request_state,
                    "policy_decision",
                    {
                        "tool": payload.get("tool_name"),
                        "accepted": bool(payload.get("success", False)),
                        "summary": payload.get("content"),
                        "iteration": iteration,
                        "payload_preview": view.get("payload") if isinstance(view.get("payload"), dict) else view,
                        **timing,
                    },
                )
                return
            yield append_trace(
                request_state,
                "tool_result",
                {
                    "tool": payload.get("tool_name"),
                    "success": payload.get("success", False),
                    "summary": payload.get("content"),
                    "iteration": iteration,
                    "payload_preview": view.get("payload") if isinstance(view.get("payload"), dict) else view,
                    "resource_ref": payload.get("resource_ref"),
                    **timing,
                },
            )
            yield append_trace(
                request_state,
                "step.done",
                {
                    "type": "step.done",
                    "step": iteration,
                    "id": self._step_event_id(iteration),
                    "status": "done" if payload.get("success", False) else "failed",
                    "observation": payload.get("observations"),
                    "resource_ref": payload.get("resource_ref"),
                    **timing,
                },
            )
            return

        if event_type in {"artifact_registered", "coverage_updated", "todo_updated", "policy_decision"}:
            yield event
            return

        if event_type == "thought":
            iteration = payload.get("iteration")
            step_id = self._step_event_id(iteration)
            yield append_trace(
                request_state,
                "step.start",
                {
                    "type": "step.start",
                    "step": iteration,
                    "id": step_id,
                    "title": f"react round {iteration}",
                    "detail": "正在判断下一步。",
                },
            )
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
            yield append_trace(
                request_state,
                "step.chunk",
                {
                    "type": "step.chunk",
                    "id": step_id,
                    "step": iteration,
                    "output_type": "thought",
                    "content": payload.get("thought") or "",
                },
            )
            return

        if event_type == "action":
            action_name = str(payload.get("action", ""))
            iteration = payload.get("iteration")
            step_id = self._step_event_id(iteration)
            yield append_trace(
                request_state,
                "step.meta",
                {
                    "type": "step.meta",
                    "step": iteration,
                    "id": step_id,
                    "thought": payload.get("thought"),
                    "action": action_name,
                    "action_input": payload.get("action_input", {}),
                    "task_contract": payload.get("task_contract"),
                    "action_intention": payload.get("action_intention"),
                    "action_reason": payload.get("action_reason"),
                    "previous_observation_assessment": payload.get("previous_observation_assessment"),
                },
            )
            yield append_trace(
                request_state,
                "agent_step",
                {
                    "agent": "data_agent",
                    "status": "running",
                    "phase": self._phase_for_action(action_name),
                    "message": self._message_for_action(action_name),
                    "iteration": iteration,
                },
            )
            yield append_trace(
                request_state,
                "tool_call",
                {
                    "tool": action_name,
                    "summary": self._message_for_action(action_name),
                    "iteration": iteration,
                    "thought": payload.get("thought"),
                    "intention": payload.get("action_intention"),
                    "reason": payload.get("action_reason"),
                    "action_input": payload.get("action_input", {}),
                    "input_preview": self._input_preview(action_name, payload.get("action_input", {})),
                },
            )
            return

        if event_type == "observation":
            iteration = self._iteration_from_payload_ref(payload.get("payload_ref")) or request_state.iteration
            public_view = public_observation_view(payload) or {}
            yield append_trace(
                request_state,
                "tool_result",
                {
                    "tool": public_view.get("tool_name") or payload.get("tool_name"),
                    "success": public_view.get("success", False),
                    "summary": public_view.get("summary"),
                    "iteration": iteration,
                    "payload_preview": public_view.get("payload") or {},
                    "artifact_ref": public_view.get("artifact_ref"),
                    "payload_ref": public_view.get("payload_ref") or payload.get("payload_ref"),
                },
            )
            yield append_trace(
                request_state,
                "step.done",
                {
                    "type": "step.done",
                    "step": iteration,
                    "id": self._step_event_id(iteration),
                    "status": "done" if public_view.get("success", False) else "failed",
                    "observation": public_view,
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

    def _step_event_id(self, iteration) -> str:
        try:
            numeric = int(iteration)
        except (TypeError, ValueError):
            numeric = 0
        return f"iteration-{numeric}"

    def _timing_from_action_output_payload(self, payload: dict) -> dict:
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        return {
            key: meta[key]
            for key in ("started_at", "completed_at", "duration_ms", "elapsed_seconds")
            if key in meta and meta[key] is not None
        }

    def _phase_for_action(self, action_name: str) -> str:
        mapping = {
            "sql_query": "tool_selection",
            "code_interpreter": "analysis",
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
            "code_interpreter": "正在执行数据分析。",
            "terminate": "正在结束并组装最终回答。",
            "forecast": "正在执行趋势预测。",
            "anomaly": "正在检测异常。",
            "rag": "正在补充外部知识。",
            "skill": "正在执行预定义技能。",
            "todowrite": "正在整理任务步骤。",
        }
        return mapping.get(action_name, "正在处理请求。")

    def _input_preview(self, action_name: str, action_input: dict) -> dict:
        if not isinstance(action_input, dict):
            action_input = {}
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
        if action_name == "code_interpreter":
            evidence = action_input.get("database_evidence") or {}
            evidence_id = evidence.get("evidence_id") if isinstance(evidence, dict) else evidence
            code = str(action_input.get("analysis_code") or action_input.get("code") or "")
            return {
                "evidence_id": evidence_id,
                "analysis_goal": action_input.get("analysis_goal") or action_input.get("focus"),
                "code_type": action_input.get("code_type"),
                "analysis_code_chars": len(code),
                "code_preview": self._truncate_preview_text(code, 8000),
            }
        if action_name == "terminate":
            include_analysis_ids = action_input.get("include_analysis_ids")
            include_fact_ids = action_input.get("include_fact_ids")
            include_visualization_ids = action_input.get("include_visualization_ids")
            section_plan = action_input.get("section_plan")
            return {
                "has_result": bool(action_input.get("result") or action_input.get("direct_answer")),
                "include_analysis_count": len(include_analysis_ids) if isinstance(include_analysis_ids, list) else 0,
                "include_fact_count": len(include_fact_ids) if isinstance(include_fact_ids, list) else 0,
                "include_visualization_count": len(include_visualization_ids) if isinstance(include_visualization_ids, list) else 0,
                "section_plan": section_plan if isinstance(section_plan, list) else [],
            }
        if action_name == "todowrite":
            todos = action_input.get("todos")
            return {
                "todo_count": len(todos) if isinstance(todos, list) else 0,
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
        for key in (
            "error_type",
            "retryable",
            "recommended_next_action",
            "recommended_strategy",
            "blocked_strategy",
            "failure_signature",
            "repeated_failure_count",
            "missing_requirements",
            "evidence_id",
            "analysis_id",
            "analysis_goal",
            "code_type",
            "code_hash",
            "input_row_count",
            "forecast_id",
            "anomaly_id",
            "title",
            "summary",
            "current_step",
            "planning_complete",
            "task_contract",
            "validation_failure",
        ):
            if key in visible_payload:
                preview[key] = visible_payload.get(key)
        validation_failure = visible_payload.get("validation_failure")
        if isinstance(validation_failure, dict):
            repair_contract = validation_failure.get("repair_contract")
            retry_policy = validation_failure.get("retry_policy")
            if isinstance(repair_contract, dict):
                preview["repair_contract"] = repair_contract
            if isinstance(retry_policy, dict):
                preview["retry_policy"] = retry_policy
        if isinstance(visible_payload.get("diagnostics"), dict):
            diagnostics_payload = visible_payload["diagnostics"]
            if "missing_required_filters" in diagnostics_payload:
                preview["missing_required_filters"] = diagnostics_payload.get("missing_required_filters")
            for diagnostic_key in (
                "query_shape_issues",
                "query_task_contract",
                "recommended_downstream_action",
                "strategy_hint",
                "classification",
                "next_action_hint",
            ):
                if diagnostic_key in diagnostics_payload:
                    preview[diagnostic_key] = diagnostics_payload.get(diagnostic_key)
        if isinstance(visible_payload.get("result"), dict):
            preview["result_preview"] = visible_payload["result"]
        if "visualizations" in visible_payload:
            preview["visualization_count"] = len(visible_payload.get("visualizations", []))
        if "produced_facts" in visible_payload:
            produced_facts = visible_payload.get("produced_facts", [])
            preview["produced_facts"] = produced_facts[:8] if isinstance(produced_facts, list) else []
            preview["verified_fact_count"] = len(preview["produced_facts"])
        if "fact_coverage" in visible_payload:
            preview["fact_coverage"] = visible_payload.get("fact_coverage")
        if "data_fact_context" in visible_payload:
            preview["data_fact_context"] = visible_payload.get("data_fact_context")
        if "todos" in visible_payload:
            todos = visible_payload.get("todos", [])
            preview["todo_total"] = len(todos)
            preview["todos"] = [
                {
                    "content": todo.get("content"),
                    "task_type": todo.get("task_type"),
                    "status": todo.get("status"),
                    "priority": todo.get("priority"),
                    "acceptance_criteria": todo.get("acceptance_criteria"),
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
        if payload.get("tool_name") == "code_interpreter":
            preview.update(self._code_interpreter_payload_preview(visible_payload))
        if payload.get("tool_name") == "forecast":
            preview.update(self._forecast_payload_preview(visible_payload))
        if payload.get("tool_name") == "anomaly":
            preview.update(self._anomaly_payload_preview(visible_payload))
        if request_state is not None:
            preview.update(self._completion_payload_preview(request_state))
        return preview

    def _failure_strategy_payload(self, request_state: RequestStateModel, tool_name: str, payload: dict) -> dict:
        signature = self._failure_signature(tool_name, payload)
        recent = []
        for observation in reversed(request_state.observations[-8:]):
            if observation.success:
                break
            obs_payload = observation.payload if isinstance(observation.payload, dict) else {}
            if self._failure_signature(observation.tool_name, obs_payload) == signature:
                recent.append(observation)
            else:
                break
        repeated_count = len(recent) + 1
        strategy = {
            "failure_signature": signature,
            "repeated_failure_count": repeated_count,
        }
        if repeated_count >= 2:
            strategy["blocked_strategy"] = (
                "Equivalent failures are repeating. Change the evidence strategy or choose the recommended downstream tool; "
                "do not repeat the same action_input shape."
            )
            diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
            recommended = diagnostics.get("recommended_downstream_action") or payload.get("recommended_next_action")
            if tool_name == "sql_query" and recommended == "code_interpreter":
                strategy["recommended_next_action"] = "sql_query"
                strategy["recommended_strategy"] = "Ask sql_query for raw/simple evidence only, then use code_interpreter for derived computation."
            elif tool_name == "terminate":
                strategy["recommended_next_action"] = diagnostics.get("next_action_hint") or "inspect_missing_outputs"
            elif recommended:
                strategy["recommended_next_action"] = recommended
            if tool_name == "sql_query":
                shape_issues = diagnostics.get("query_shape_issues") if isinstance(diagnostics.get("query_shape_issues"), list) else []
                classification = diagnostics.get("classification") if isinstance(diagnostics.get("classification"), dict) else {}
                strategy["repair_mode"] = "schema_grounded_query_repair"
                strategy["recommended_strategy"] = (
                    "Use the structured schema_linking/physical model from the latest observation to change the query shape; "
                    "do not retry the same logical-to-physical field mapping."
                )
                if shape_issues:
                    strategy["query_shape_issues"] = shape_issues[:5]
                if classification:
                    strategy["error_classification"] = classification
        return strategy

    def _failure_signature(self, tool_name: str, payload: dict) -> str:
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
        parts = [
            str(tool_name),
            str(payload.get("error_type") or ""),
            str(diagnostics.get("query_language") or ""),
            str(diagnostics.get("recommended_downstream_action") or ""),
            self._normalize_error_text(str(payload.get("error") or "")),
        ]
        return "|".join(parts)

    def _normalize_error_text(self, text: str) -> str:
        text = re.sub(r"\d{4}-\d{2}-\d{2}[T ][^\\s'\",)]+", "<timestamp>", text)
        text = re.sub(r"\b\d+\b", "<n>", text)
        return text[:500]

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
                    "acceptance_criteria": todo.get("acceptance_criteria"),
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
            "task_coverage": self._task_coverage_payload_preview(diagnostics),
            "truncated": bool(
                visible_payload.get("payload_truncated")
                or diagnostics.get("truncated")
                or diagnostics.get("artifact_ref")
                or payload_truncated_marker(visible_payload)
            ),
        }

    def _task_coverage_payload_preview(self, diagnostics: dict) -> dict | None:
        coverage = diagnostics.get("task_coverage")
        if not isinstance(coverage, dict):
            return None
        preview = dict(coverage)
        if not isinstance(preview.get("missing"), list):
            legacy_missing = preview.get("missing_or_uncertain")
            preview["missing"] = legacy_missing if isinstance(legacy_missing, list) else []
        if not isinstance(preview.get("runtime_missing"), list):
            legacy_runtime_missing = preview.get("runtime_missing_or_uncertain")
            preview["runtime_missing"] = legacy_runtime_missing if isinstance(legacy_runtime_missing, list) else []
        preview.pop("missing_or_uncertain", None)
        preview.pop("runtime_missing_or_uncertain", None)
        return preview

    def _code_interpreter_payload_preview(self, visible_payload: dict) -> dict:
        result = visible_payload.get("result") if isinstance(visible_payload.get("result"), dict) else {}
        diagnostics = visible_payload.get("diagnostics") if isinstance(visible_payload.get("diagnostics"), dict) else {}
        return {
            "analysis_goal": visible_payload.get("analysis_goal"),
            "analysis_status": visible_payload.get("status"),
            "analysis_summary": visible_payload.get("summary"),
            "analysis_result": result,
            "analysis_metrics": result.get("metrics") if isinstance(result.get("metrics"), dict) else {},
            "analysis_details": result.get("details") if isinstance(result.get("details"), dict) else {},
            "input_row_count": visible_payload.get("input_row_count"),
            "input_evidence_id": visible_payload.get("input_evidence_id"),
            "code_hash": visible_payload.get("code_hash"),
            "code_type": visible_payload.get("code_type"),
            "runtime_ms": diagnostics.get("runtime_ms"),
            "input_columns": diagnostics.get("input_columns") if isinstance(diagnostics.get("input_columns"), list) else [],
            "code_preview": (
                diagnostics.get("executed_code_preview", {}).get("preview")
                if isinstance(diagnostics.get("executed_code_preview"), dict)
                else diagnostics.get("generated_code_preview", {}).get("preview")
                if isinstance(diagnostics.get("generated_code_preview"), dict)
                else None
            ),
            "analysis_code_chars": (
                diagnostics.get("executed_code_preview", {}).get("char_count")
                if isinstance(diagnostics.get("executed_code_preview"), dict)
                else diagnostics.get("generated_code_preview", {}).get("char_count")
                if isinstance(diagnostics.get("generated_code_preview"), dict)
                else None
            ),
            "internal_repair_attempts": diagnostics.get("internal_repair_attempts")
            if isinstance(diagnostics.get("internal_repair_attempts"), list)
            else [],
            "canonical_inputs": diagnostics.get("canonical_inputs") if isinstance(diagnostics.get("canonical_inputs"), dict) else None,
        }

    def _forecast_payload_preview(self, visible_payload: dict) -> dict:
        diagnostics = visible_payload.get("diagnostics") if isinstance(visible_payload.get("diagnostics"), dict) else {}
        points = visible_payload.get("forecast_points") if isinstance(visible_payload.get("forecast_points"), list) else []
        plan = visible_payload.get("forecast_plan")
        if not isinstance(plan, dict):
            plan = diagnostics.get("forecast_plan") if isinstance(diagnostics.get("forecast_plan"), dict) else None
        return {
            "forecast_status": visible_payload.get("status"),
            "forecast_plan": plan,
            "forecast_points": points[:12],
            "forecast_point_count": len(points),
            "model_name": visible_payload.get("model_name"),
            "horizon": visible_payload.get("horizon"),
        }

    def _anomaly_payload_preview(self, visible_payload: dict) -> dict:
        anomaly_points = visible_payload.get("anomaly_points") if isinstance(visible_payload.get("anomaly_points"), list) else []
        scores = visible_payload.get("scores") if isinstance(visible_payload.get("scores"), list) else []
        return {
            "detector_name": visible_payload.get("detector_name"),
            "anomaly_points": anomaly_points[:12],
            "anomaly_scores": scores[:12],
            "anomaly_point_count": len(anomaly_points),
            "anomaly_span_count": len(visible_payload.get("anomaly_spans", [])) if isinstance(visible_payload.get("anomaly_spans"), list) else 0,
        }

    def _schema_linking_payload_preview(self, diagnostics: dict) -> dict | None:
        query_trace = diagnostics.get("query_trace") if isinstance(diagnostics.get("query_trace"), dict) else {}
        logical_plan = query_trace.get("logical_plan") if isinstance(query_trace.get("logical_plan"), dict) else {}
        linking = logical_plan.get("schema_linking") if isinstance(logical_plan.get("schema_linking"), dict) else None
        if not linking:
            return None
        database_type = (
            query_trace.get("adapter_type")
            or logical_plan.get("adapter_type")
            or logical_plan.get("database_type")
            or diagnostics.get("database_type")
        )
        internal_columns = dialect_for_database(database_type).internal_columns()
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
            and item.get("column") not in internal_columns
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
