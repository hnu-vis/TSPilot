"""Request-state helpers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from app.settings import Settings
from core.completion import (
    evaluate_goal_completion,
    normalize_todo_for_completion,
)
from core.intent import build_intent_profile_fallback
from core.database.dialects import query_language_for_database_type
from core.harness.observation_view import public_observation_view
from core.time_range import normalize_time_range
from runtime.artifacts import persist_json_artifact
from runtime.language import detect_response_language
from runtime.token_usage import token_usage_summary
from runtime.trace import TraceEventModel
from schemas.api import ChatRequest, ChatResponse
from schemas.database_context import DatabaseContext
from schemas.output import AnswerSection, FinalAnswer
from schemas.state import ConversationStateModel, RequestStateModel
from core.data_fact import register_data_facts_from_payload
from core.harness import default_capability_registry
from schemas.task_contract import TaskContract
from schemas.tool import ReActTranscriptStep, ToolObservation

if TYPE_CHECKING:
    from tools.registry import ToolSpec


def normalize_chat_request(request: ChatRequest) -> ChatRequest:
    """Normalize legacy database aliases into database_context."""
    if request.database_context is None and request.selected_database:
        request.database_context = DatabaseContext(
            database_id=request.selected_database,
            database_type=request.selected_database_type or "unknown",
            display_name=request.selected_database,
        )
    return request


def build_request_state(request: ChatRequest, settings: Settings) -> RequestStateModel:
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    conversation_run_dir = None
    request_log_dir = None
    if settings.conversation_log_enabled:
        conversation_run_dir = _resolve_conversation_run_dir(settings, conversation_id)
        request_log_dir = conversation_run_dir / "requests" / request_id
        request_log_dir.mkdir(parents=True, exist_ok=True)
        _write_conversation_meta(conversation_run_dir, conversation_id)
    intent_profile = build_intent_profile_fallback(request.message)
    requested_capabilities = list(intent_profile.get("requested_capabilities") or [])
    focus = request.message
    database_context = _with_schema_hint(request.database_context, settings)
    return RequestStateModel(
        request_id=request_id,
        conversation_id=conversation_id,
        conversation_run_dir=str(conversation_run_dir) if conversation_run_dir else None,
        request_log_dir=str(request_log_dir) if request_log_dir else None,
        message=request.message,
        response_language=detect_response_language(request.message),
        database_context=database_context,
        selected_database=request.selected_database,
        selected_database_type=request.selected_database_type,
        time_range=normalize_time_range(request.time_range),
        constraints=request.constraints,
        history=request.history,
        status="running",
        current_intent="chat_analysis",
        intent_profile=intent_profile,
        requested_capabilities=requested_capabilities,
        focus=focus,
        todo_list=[],
        plan_current_step=0,
        planning_complete=False,
        iteration=0,
        max_iterations=settings.max_iterations,
        context_budget={
            "max_prompt_tokens": settings.max_prompt_tokens,
            "max_history_messages": settings.max_history_messages,
            "max_tool_history_items": settings.max_tool_history_items,
            "max_observation_chars": settings.max_observation_chars,
            "max_visible_rows": settings.max_visible_rows,
            "max_visible_points": settings.max_visible_points,
            "overflow_policy": "fail_closed",
        },
        context_status="ok",
        context_overflow_reason=None,
        completion_state={},
        task_contract=None,
        latest_database_evidence=None,
        database_evidence_artifacts={},
        latest_analysis_id=None,
        analysis_artifacts={},
        latest_forecast=None,
        forecast_artifacts={},
        latest_anomaly=None,
        anomaly_artifacts={},
        latest_rag=None,
        latest_skill=None,
        final_answer_draft=None,
        visualizations=[],
        tool_history=[],
        observations=[],
        react_transcript=[],
        errors=[],
        prompt_context_summary=None,
    )


def _resolve_conversation_run_dir(settings: Settings, conversation_id: str) -> Path:
    root = settings.resolved_conversation_log_dir
    root.mkdir(parents=True, exist_ok=True)
    safe_conversation_id = _safe_path_name(conversation_id)
    existing = sorted(root.glob(f"*_{safe_conversation_id}"))
    for path in reversed(existing):
        if path.is_dir():
            return path.resolve()

    local_now = datetime.now().astimezone()
    timestamp = local_now.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / f"{timestamp}_{safe_conversation_id}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = root / f"{timestamp}_{safe_conversation_id}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def _write_conversation_meta(conversation_run_dir: Path, conversation_id: str) -> None:
    meta_path = conversation_run_dir / "conversation.json"
    if meta_path.exists():
        return
    local_now = datetime.now().astimezone()
    payload = {
        "conversation_id": conversation_id,
        "created_at_local": local_now.isoformat(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": local_now.tzname(),
        "run_dir": str(conversation_run_dir),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_path_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return normalized or "unknown"


def build_conversation_state(request: ChatRequest, conversation_id: str) -> ConversationStateModel:
    return ConversationStateModel(
        conversation_id=conversation_id,
        database_context=request.database_context,
        recent_messages=request.history,
        session_summary=None,
        intent_profile={},
        requested_capabilities=[],
        todo_list=[],
        plan_current_step=0,
        planning_complete=False,
        recent_todo_summary=None,
        latest_database_evidence=None,
        database_evidence_artifacts={},
        latest_analysis_id=None,
        analysis_artifacts={},
        latest_forecast=None,
        forecast_artifacts={},
        latest_anomaly=None,
        anomaly_artifacts={},
        latest_rag=None,
        latest_skill=None,
        recent_visualizations=[],
        updated_at=None,
        context_budget=None,
        recent_react_transcript=[],
        task_contract=None,
    )


def _with_schema_hint(database_context: DatabaseContext | None, settings: Settings) -> DatabaseContext | None:
    """Attach bounded datasource structure known from local config before the first tool call."""
    if database_context is None or database_context.schema_hint:
        return database_context
    config = _load_cached_database_config(database_context.database_id, settings)
    if not config:
        return database_context
    schema_hint = _build_schema_hint(config)
    if not schema_hint:
        return database_context
    return database_context.model_copy(update={"schema_hint": schema_hint})


def _load_cached_database_config(database_id: str, settings: Settings) -> dict | None:
    cache_path = Path(settings.tspilot_root) / "cache_data" / "database" / "databases.json"
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    databases = payload.get("databases") if isinstance(payload, dict) else None
    config = databases.get(database_id) if isinstance(databases, dict) else None
    return config if isinstance(config, dict) else None


def _build_schema_hint(config: dict) -> dict:
    database_type = str(config.get("type") or config.get("db_type") or "unknown")
    hint = {
        "source": "local_database_config",
        "database_id": config.get("id") or config.get("name"),
        "database_type": database_type,
        "query_language": _query_language_for_database_type(database_type),
        "tables_or_measurements": [],
    }
    configured_names = config.get("schema_measurement_names") or config.get("schema_metric_names")
    if isinstance(configured_names, str):
        configured_names = [configured_names]
    if isinstance(configured_names, list):
        hint["tables_or_measurements"] = [
            {"name": str(name)}
            for name in configured_names[:20]
            if name not in (None, "")
        ]
    return hint if hint["tables_or_measurements"] else {}


def _query_language_for_database_type(database_type: str) -> str:
    return query_language_for_database_type(database_type)


def append_trace(request_state: RequestStateModel, event_type: str, payload: dict) -> TraceEventModel:
    event = TraceEventModel(event_type=event_type, payload=payload)
    return event


def apply_task_contract(request_state: RequestStateModel, raw_contract) -> TaskContract | None:
    """Persist an LLM-authored task output contract when present.

    Once a user-visible output contract exists, later model turns may refine or
    append outputs, but must not silently drop previously required deliverables.
    This keeps the runtime completion gate monotonic across ReAct turns.
    """

    if raw_contract is None:
        return None
    contract = raw_contract if isinstance(raw_contract, TaskContract) else TaskContract.model_validate(raw_contract)
    if request_state.task_contract is not None:
        contract = _merge_task_contract(request_state.task_contract, contract)
    request_state.task_contract = contract
    request_state.completion_state["task_contract"] = contract.model_dump(mode="json")
    return contract


def _merge_task_contract(existing: TaskContract, update: TaskContract) -> TaskContract:
    existing_outputs = list(existing.required_outputs or [])
    merged_outputs = list(existing_outputs)
    index_by_key = {
        _task_contract_output_key(output): index
        for index, output in enumerate(merged_outputs)
        if _task_contract_output_key(output)
    }
    for output in update.required_outputs or []:
        key = _task_contract_output_key(output)
        if key and key in index_by_key:
            merged_outputs[index_by_key[key]] = output
        else:
            merged_outputs.append(output)
            if key:
                index_by_key[key] = len(merged_outputs) - 1
    return TaskContract(
        source=existing.source,
        goal=update.goal or existing.goal,
        required_outputs=merged_outputs,
        constraints={**(existing.constraints or {}), **(update.constraints or {})},
        assumptions=_dedupe_strings([*(existing.assumptions or []), *(update.assumptions or [])]),
        evidence_quality_notes=_dedupe_strings(
            [*(existing.evidence_quality_notes or []), *(update.evidence_quality_notes or [])]
        ),
    )


def _task_contract_output_key(output) -> str:
    output_id = str(getattr(output, "id", "") or "").strip().lower()
    if output_id:
        return f"id:{output_id}"
    description = str(getattr(output, "description", "") or "").strip().lower()
    if description:
        return f"description:{description}"
    return ""


def _dedupe_strings(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def append_react_transcript_step(
    request_state: RequestStateModel,
    *,
    iteration: int,
    thought: str | None,
    action: str,
    action_input: dict | None,
    observation: ToolObservation | None,
    action_intention: str | None = None,
    action_reason: str | None = None,
    phase: str | None = None,
) -> ReActTranscriptStep:
    """Append a DB-GPT-style structured ReAct memory fragment."""

    step = ReActTranscriptStep(
        iteration=iteration,
        question=request_state.message,
        thought=thought,
        phase=phase,
        action_intention=action_intention,
        action_reason=action_reason,
        action=action,
        action_input=action_input or {},
        observation=observation,
    )
    request_state.react_transcript.append(step)
    return step


def build_final_response(request_state: RequestStateModel, trace_events: list[TraceEventModel]) -> ChatResponse:
    public_trace = _public_response_trace(trace_events)
    if request_state.final_answer_draft is not None:
        public_answer = public_final_answer(request_state.final_answer_draft)
        return ChatResponse(
            conversation_id=request_state.conversation_id or "",
            request_id=request_state.request_id,
            status="partial" if request_state.status == "partial" else "completed",
            response_kind="final_answer",
            used_tools=_visible_used_tools(request_state),
            answer=public_answer,
            trace=public_trace,
            token_usage=token_usage_summary(request_state),
            error=None,
        )

    return ChatResponse(
        conversation_id=request_state.conversation_id or "",
        request_id=request_state.request_id,
        status="failed",
        response_kind="error",
        used_tools=_visible_used_tools(request_state),
        answer=None,
        trace=public_trace,
        token_usage=token_usage_summary(request_state),
        error="Request did not reach a final answer.",
    )


def public_final_answer_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    return public_final_answer(FinalAnswer.model_validate(payload)).model_dump(mode="json")


def public_final_answer(answer: FinalAnswer) -> FinalAnswer:
    return FinalAnswer(
        title=answer.title,
        summary=_public_final_answer_text(answer.summary),
        sections=[
            section.model_copy(
                update={
                    "content": _public_final_answer_section_content(section),
                    "structured_payload": _sanitize_public_value(
                        section.structured_payload,
                        allow_query_fields=section.section_type in {"query", "query_results"},
                    ),
                }
            )
            for section in answer.sections
        ],
        references=[
            reference.model_copy(
                update={
                    "label": _public_final_answer_text(reference.label),
                    "evidence": _sanitize_public_value(
                        reference.evidence,
                        allow_query_fields=reference.source_type == "query",
                    ),
                }
            )
            for reference in answer.references
        ],
        claims=answer.claims,
        visualizations=answer.visualizations,
    )


def _public_final_answer_section_content(section: AnswerSection) -> str:
    if section.section_type in {"query", "query_results"}:
        return section.content
    return _public_final_answer_text(section.content)


def _public_final_answer_text(value: str | None) -> str:
    text = _strip_public_query_text(value)
    if not isinstance(text, str):
        return ""
    text = text.replace("[query omitted]", "")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _public_response_trace(trace_events: list[TraceEventModel]) -> list[TraceEventModel]:
    public_events: list[TraceEventModel] = []
    for event in trace_events:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.event_type == "action_output":
            view = payload.get("view") if isinstance(payload.get("view"), dict) else {}
            timing = _timing_from_action_output_payload(payload)
            result_target = payload.get("meta", {}).get("result_target") if isinstance(payload.get("meta"), dict) else None
            if result_target == "policy":
                public_events.append(
                    TraceEventModel(
                        event_type="policy_decision",
                        payload={
                            "tool": payload.get("tool_name"),
                            "accepted": bool(payload.get("success", False)),
                            "summary": payload.get("content"),
                            "payload_preview": view.get("payload") if isinstance(view.get("payload"), dict) else view,
                            **timing,
                        },
                        timestamp=event.timestamp,
                    )
                )
                continue
            public_events.append(
                TraceEventModel(
                    event_type="tool_result",
                    payload={
                        "tool": payload.get("tool_name"),
                        "success": payload.get("success", False),
                        "summary": payload.get("content"),
                        "payload_preview": view.get("payload") if isinstance(view.get("payload"), dict) else view,
                        "resource_ref": payload.get("resource_ref"),
                        **timing,
                    },
                    timestamp=event.timestamp,
                )
            )
            continue
        if event.event_type == "observation":
            public_view = public_observation_view(payload) or {}
            public_events.append(
                TraceEventModel(
                    event_type="tool_result",
                    payload={
                        "tool": public_view.get("tool_name") or payload.get("tool_name"),
                        "success": public_view.get("success", False),
                        "summary": public_view.get("summary"),
                        "payload_preview": public_view.get("payload") or {},
                        "artifact_ref": public_view.get("artifact_ref"),
                        "payload_ref": public_view.get("payload_ref") or payload.get("payload_ref"),
                    },
                    timestamp=event.timestamp,
                )
            )
            continue
        public_events.append(
            TraceEventModel(
                event_type=event.event_type,
                payload=(
                    public_final_answer_payload(payload)
                    if event.event_type == "final_answer"
                    else _sanitize_public_trace_payload(payload)
                ),
                timestamp=event.timestamp,
            )
        )
    return public_events


def _timing_from_action_output_payload(payload: dict) -> dict:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    return {
        key: meta[key]
        for key in ("started_at", "completed_at", "duration_ms", "elapsed_seconds")
        if key in meta and meta[key] is not None
    }


def _sanitize_public_trace_payload(payload: dict) -> dict:
    sanitized = dict(payload or {})
    observation = sanitized.get("observation")
    if isinstance(observation, dict):
        sanitized["observation"] = public_observation_view(observation) or {}
    if "answer" in sanitized and isinstance(sanitized.get("answer"), dict):
        sanitized["answer"] = public_final_answer_payload(sanitized["answer"])
    return sanitized


def _sanitize_public_value(value, *, allow_query_fields: bool = False, key_name: str | None = None):
    internal_keys = {
        "query_trace",
        "schema_linking",
        "schema_linking_generation",
        "llm_query_generation",
        "query_generation",
        "query_task_contract",
        "executed_query",
        "generated_query",
        "repaired_from_query",
        "previous_error",
        "repair_contract",
    }
    if not allow_query_fields:
        internal_keys.update({"query", "query_language"})
    if isinstance(value, str):
        if allow_query_fields and key_name == "query":
            return value
        return _strip_public_query_text(value)
    if isinstance(value, list):
        return [_sanitize_public_value(item, allow_query_fields=allow_query_fields) for item in value]
    if isinstance(value, dict):
        return {
            key: _sanitize_public_value(item, allow_query_fields=allow_query_fields, key_name=str(key))
            for key, item in value.items()
            if str(key) not in internal_keys
        }
    return value


def _strip_public_query_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return value
    text = value
    for marker in ("Query statement:", "\nQuery statement:", " for query '", " for query `"):
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
    text = re.sub(r"```(?:flux|sql|promql)?\s+.*?```", "[query omitted]", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"from\s*\([^)]*\)(?:\s*\|>.*?)(?=(?:['`\"。；;]|\n\n|$))", "[query omitted]", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\bselect\b\s+.*?\bfrom\b\s+.*?(?=(?:['`\"。；;]|\n\n|$))", "[query omitted]", text, flags=re.IGNORECASE | re.DOTALL)
    return text


def _visible_used_tools(request_state: RequestStateModel) -> list[str]:
    return [
        call.tool_name
        for call in request_state.tool_history
        if call.tool_name != "terminate"
    ]


def apply_observation(
    request_state: RequestStateModel,
    observation: ToolObservation,
    full_payload: dict,
    tool_spec: "ToolSpec",
    *,
    thought: str | None = None,
    action_reason: str | None = None,
) -> ToolObservation:
    from core.harness import StateTransitionEngine

    return StateTransitionEngine().apply(
        request_state,
        observation,
        full_payload,
        tool_spec,
    ).observation


async def apply_observation_async(
    request_state: RequestStateModel,
    observation: ToolObservation,
    full_payload: dict,
    tool_spec: "ToolSpec",
    *,
    thought: str | None = None,
    action_reason: str | None = None,
) -> ToolObservation:
    from core.harness import StateTransitionEngine

    return (
        await StateTransitionEngine().apply_async(
            request_state,
            observation,
            full_payload,
            tool_spec,
        )
    ).observation


def enrich_observation_payload(
    request_state: RequestStateModel,
    observation: ToolObservation,
    full_payload: dict,
    tool_spec: "ToolSpec",
) -> ToolObservation:
    """Replace observation payload with the latest prompt-safe state view."""

    payload = observation.payload
    if tool_spec.result_target == "evidence" and request_state.latest_database_evidence is not None:
        if request_state.latest_database_evidence.evidence_id == full_payload.get("evidence_id"):
            payload = request_state.latest_database_evidence.model_dump(mode="json")
    elif tool_spec.result_target == "analysis":
        if "analysis_id" in full_payload:
            analysis_id = str(full_payload.get("analysis_id"))
            analysis = request_state.analysis_artifacts.get(analysis_id)
            if analysis is not None:
                payload = _build_prompt_safe_analysis(analysis, request_state)
        elif "forecast_id" in full_payload and request_state.latest_forecast is not None:
            payload = request_state.latest_forecast.model_dump(mode="json")
        elif "anomaly_id" in full_payload and request_state.latest_anomaly is not None:
            payload = request_state.latest_anomaly.model_dump(mode="json")
    elif tool_spec.result_target == "presentation" and request_state.final_answer_draft is not None:
        payload = request_state.final_answer_draft.model_dump(mode="json")

    payload = _deduplicate_observation_payload(observation, payload)
    return observation.model_copy(
        update={
            "payload": payload,
            "payload_truncated": observation.payload_truncated or payload != full_payload,
        }
    )


def _build_prompt_safe_failure_observation(observation: ToolObservation) -> ToolObservation:
    payload = _bounded_value(observation.payload, max_string_chars=1600, max_list_items=8, max_dict_items=12)
    payload = _deduplicate_observation_payload(observation, payload)
    return observation.model_copy(
        update={
            "summary": _truncate_text(observation.summary, 1600),
            "error": _truncate_text(observation.error, 1600),
            "payload": payload,
            "payload_truncated": observation.payload_truncated or _is_large_value(observation.payload),
        }
    )


def _deduplicate_observation_payload(observation: ToolObservation, payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    cleaned = dict(payload)
    envelope_values = {
        "summary": observation.summary,
        "error": observation.error,
        "success": observation.success,
        "tool_name": observation.tool_name,
    }
    for key, envelope_value in envelope_values.items():
        if key in cleaned and cleaned.get(key) == envelope_value:
            cleaned.pop(key, None)
    return cleaned


def _truncate_text(value: str | None, max_chars: int) -> str | None:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def _bounded_value(
    value,
    *,
    max_string_chars: int = 800,
    max_list_items: int = 8,
    max_dict_items: int = 12,
):
    if isinstance(value, str):
        return _truncate_text(value, max_string_chars)
    if isinstance(value, list):
        items = [
            _bounded_value(
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
            bounded[key] = _bounded_value(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
        return bounded
    return value


def _is_large_value(value) -> bool:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str)) > 2000
    except Exception:
        return len(str(value)) > 2000


def _apply_todo_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    if isinstance(full_payload.get("task_contract"), dict):
        try:
            apply_task_contract(request_state, full_payload["task_contract"])
        except Exception as exc:
            request_state.completion_state.setdefault("todo_warnings", []).append(
                {
                    "type": "invalid_todowrite_task_contract",
                    "message": str(exc),
                }
            )
    request_state.todo_list = [
        normalize_todo_for_completion(todo)
        for todo in list(full_payload.get("todos", []))
        if isinstance(todo, dict)
    ]
    request_state.plan_current_step = int(full_payload.get("current_step") or 0)
    request_state.planning_complete = bool(full_payload.get("planning_complete", False))
    _activate_next_todo(request_state)
    if request_state.todo_list:
        request_state.max_iterations = max(
            request_state.max_iterations,
            min(20, len(request_state.todo_list) * 3 + 2),
        )
    request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()


def _complete_answer_todo_after_terminal(
    request_state: RequestStateModel,
    tool_name: str,
    full_payload: dict,
) -> None:
    if not request_state.todo_list:
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current_index = next((index for index, todo in enumerate(request_state.todo_list) if todo.get("status") == "in_progress"), None)
    if current_index is None:
        return
    current_todo = dict(request_state.todo_list[current_index])
    if str(current_todo.get("task_type") or "").strip().lower() != "answer":
        request_state.completion_state["latest_step"] = {
            "completed": False,
            "reason": "Terminal payload cannot complete a non-answer todo.",
            "missing_evidence": ["answer_todo"],
            "evidence_refs": [],
            "next_action_hint": "Complete the active evidence or analysis todo before terminating.",
            "tool_name": tool_name,
            "todo_index": current_index,
            "todo": current_todo,
        }
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current_todo = normalize_todo_for_completion(current_todo)
    if not _has_final_answer_payload(full_payload):
        request_state.completion_state["latest_step"] = {
            "completed": False,
            "reason": "Terminal payload did not contain a final answer.",
            "missing_evidence": ["final_answer"],
            "evidence_refs": [],
            "next_action_hint": "Assemble the final answer again after evidence is ready.",
            "tool_name": tool_name,
            "todo_index": current_index,
            "todo": current_todo,
        }
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    ref = "final_answer:latest"
    current_todo = _complete_todo_at_index(
        request_state,
        current_index,
        result_ref=ref,
        reason=f"Tool '{tool_name}' produced the final answer.",
    )
    request_state.completion_state["latest_step"] = {
        "completed": True,
        "reason": current_todo["completion_reason"],
        "missing_evidence": [],
        "evidence_refs": [ref],
        "next_action_hint": None,
        "tool_name": tool_name,
        "todo_index": current_index,
        "todo": current_todo,
    }
    request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()


def _advance_todo_after_artifact(
    request_state: RequestStateModel,
    tool_name: str,
    full_payload: dict,
    result_target: str,
) -> None:
    if not request_state.todo_list:
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    synchronized = _sync_todos_from_artifact_state(request_state)
    if synchronized:
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current_index = next((index for index, todo in enumerate(request_state.todo_list) if todo.get("status") == "in_progress"), None)
    if current_index is None:
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current = normalize_todo_for_completion(dict(request_state.todo_list[current_index]))
    task_type = str(current.get("task_type") or "").strip().lower()
    if not _tool_covers_todo_type(tool_name, result_target, task_type):
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    ref = _artifact_ref_for_tool(tool_name, full_payload)
    if not ref:
        request_state.completion_state["latest_step"] = {
            "completed": False,
            "reason": f"Tool '{tool_name}' succeeded but did not produce a usable artifact ref.",
            "missing_evidence": [f"{tool_name}_artifact"],
            "evidence_refs": [],
            "next_action_hint": default_capability_registry().hint_for_task_type(task_type),
            "tool_name": tool_name,
            "todo_index": current_index,
            "todo": current,
        }
        request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()
        return
    current = _complete_todo_at_index(
        request_state,
        current_index,
        result_ref=ref,
        reason=f"Tool '{tool_name}' produced artifact {ref}.",
    )
    request_state.completion_state["latest_step"] = {
        "completed": True,
        "reason": current["completion_reason"],
        "missing_evidence": [],
        "evidence_refs": [ref] if ref else [],
        "next_action_hint": None,
        "tool_name": tool_name,
        "todo_index": current_index,
        "todo": current,
    }
    request_state.completion_state["latest_goal"] = evaluate_goal_completion(request_state).model_dump()


def _sync_todos_from_artifact_state(request_state: RequestStateModel) -> bool:
    """Complete any non-answer todos already covered by structured artifacts."""

    changed = False
    for index, todo in enumerate(list(request_state.todo_list)):
        if not isinstance(todo, dict) or todo.get("status") == "completed":
            continue
        task_type = str(todo.get("task_type") or "").strip().lower()
        if task_type in {"answer", "plan", "generic", ""}:
            continue
        ref = _state_ref_for_todo_type(request_state, task_type)
        if not ref:
            continue
        completed = normalize_todo_for_completion(dict(todo))
        completed["status"] = "completed"
        completed["result_ref"] = ref
        completed["completion_reason"] = f"Structured artifact state covers todo type '{task_type}'."
        request_state.todo_list[index] = completed
        changed = True
    if changed:
        _activate_next_todo(request_state)
    return changed


def _state_ref_for_todo_type(request_state: RequestStateModel, task_type: str) -> str | None:
    capability = default_capability_registry().normalize_id(task_type)
    if capability in {"query", "database", "database_evidence"} and request_state.latest_database_evidence is not None:
        return f"evidence:{request_state.latest_database_evidence.evidence_id}"
    if capability == "analysis" and request_state.latest_analysis_id:
        return f"analysis:{request_state.latest_analysis_id}"
    if capability == "anomaly" and request_state.latest_anomaly is not None:
        return f"anomaly:{request_state.latest_anomaly.anomaly_id}"
    if capability == "forecast" and request_state.latest_forecast is not None:
        points = getattr(request_state.latest_forecast, "forecast_points", None)
        if not isinstance(points, list) or not points:
            return None
        return f"forecast:{request_state.latest_forecast.forecast_id}"
    if capability == "external_knowledge" and request_state.latest_rag:
        return "rag:latest"
    if capability == "skill" and request_state.latest_skill:
        skill_name = request_state.latest_skill.get("skill_name") or "latest"
        return f"skill:{skill_name}"
    if capability in {"generic", ""}:
        if request_state.latest_analysis_id:
            return f"analysis:{request_state.latest_analysis_id}"
        if request_state.latest_database_evidence is not None:
            return f"evidence:{request_state.latest_database_evidence.evidence_id}"
    return None


def _complete_todo_at_index(
    request_state: RequestStateModel,
    index: int,
    *,
    result_ref: str,
    reason: str,
) -> dict:
    completed = normalize_todo_for_completion(dict(request_state.todo_list[index]))
    completed["status"] = "completed"
    completed["result_ref"] = result_ref
    completed["completion_reason"] = reason
    request_state.todo_list[index] = completed
    _activate_next_todo(request_state)
    return completed


def _activate_next_todo(request_state: RequestStateModel) -> None:
    active_indices = [
        index
        for index, todo in enumerate(request_state.todo_list)
        if isinstance(todo, dict) and todo.get("status") == "in_progress"
    ]
    if active_indices:
        first_active = active_indices[0]
        for index in active_indices[1:]:
            updated = dict(request_state.todo_list[index])
            updated["status"] = "pending"
            request_state.todo_list[index] = updated
        request_state.plan_current_step = first_active + 1
        request_state.planning_complete = False
        return
    next_index = next(
        (
            index
            for index, todo in enumerate(request_state.todo_list)
            if isinstance(todo, dict) and todo.get("status") == "pending"
        ),
        None,
    )
    if next_index is not None:
        next_todo = dict(request_state.todo_list[next_index])
        next_todo["status"] = "in_progress"
        request_state.todo_list[next_index] = next_todo
        request_state.plan_current_step = next_index + 1
        request_state.planning_complete = False
    else:
        request_state.plan_current_step = len(request_state.todo_list)
        request_state.planning_complete = True


def _tool_covers_todo_type(tool_name: str, result_target: str, task_type: str) -> bool:
    if task_type in {"answer", "plan"}:
        return False
    if task_type in {"generic", ""}:
        return result_target in {"evidence", "analysis"}
    return default_capability_registry().action_matches_task_type(tool_name, task_type)


def _artifact_ref_for_tool(tool_name: str, payload: dict) -> str | None:
    return default_capability_registry().artifact_ref_for_payload(tool_name, payload)


def _apply_evidence_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    from schemas.database import DatabaseEvidence

    full_evidence = DatabaseEvidence.model_validate(full_payload)
    diagnostics = dict(full_evidence.diagnostics)
    diagnostics["artifact_kind"] = "database_evidence"
    diagnostics["artifact_ref"] = f"evidence:{full_evidence.evidence_id}"
    diagnostics["snapshot_ref"] = persist_json_artifact(
        artifact_id=full_evidence.evidence_id,
        artifact_kind="database_evidence",
        payload=full_evidence.model_dump(mode="json"),
        directory=_artifact_directory(request_state, "evidence"),
    )
    full_evidence = full_evidence.model_copy(update={"diagnostics": diagnostics})
    request_state.database_evidence_artifacts[full_evidence.evidence_id] = full_evidence
    request_state.latest_database_evidence = _build_prompt_safe_evidence(full_evidence)


def _artifact_directory(request_state: RequestStateModel, artifact_group: str) -> str | Path:
    if request_state.request_log_dir:
        return Path(request_state.request_log_dir) / "artifacts" / artifact_group
    fallback = "analysis_snapshots" if artifact_group == "analysis" else f"{artifact_group}_artifacts"
    return Path(__file__).resolve().parents[1] / "cache_data" / fallback


def _build_prompt_safe_evidence(evidence):
    from schemas.database import DatabaseEvidence

    data = dict(evidence.data)
    summary_stats = {
        "points_count": len(data.get("points", [])) if isinstance(data.get("points"), list) else None,
        "rows_count": len(data.get("rows", [])) if isinstance(data.get("rows"), list) else None,
        "series_count": len(data.get("series", [])) if isinstance(data.get("series"), list) else None,
    }
    if isinstance(data.get("points"), list):
        data["points"] = _sample_edges(data["points"], limit=24)
    if isinstance(data.get("rows"), list):
        data["rows"] = _sample_edges(data["rows"], limit=12)
    if isinstance(data.get("series"), list):
        summarized_series = []
        for series in data["series"][:6]:
            item = dict(series)
            if isinstance(item.get("points"), list):
                item["points"] = _sample_edges(item["points"], limit=12)
                item["points_count"] = len(series.get("points", []))
            summarized_series.append(item)
        data["series"] = summarized_series
    diagnostics = dict(evidence.diagnostics)
    diagnostics["artifact_kind"] = "database_evidence"
    diagnostics["artifact_ref"] = f"evidence:{evidence.evidence_id}"
    diagnostics["summary_stats"] = {key: value for key, value in summary_stats.items() if value is not None}
    visible_counts = {
        "points_count": len(data.get("points", [])) if isinstance(data.get("points"), list) else None,
        "rows_count": len(data.get("rows", [])) if isinstance(data.get("rows"), list) else None,
        "series_count": len(data.get("series", [])) if isinstance(data.get("series"), list) else None,
    }
    full_counts = {key: value for key, value in summary_stats.items() if value is not None}
    diagnostics["prompt_sampling"] = {
        "policy": "head_tail_edges",
        "sampled_for_prompt": any(
            isinstance(full_counts.get(key), int)
            and isinstance(visible_counts.get(key), int)
            and visible_counts[key] < full_counts[key]
            for key in ("points_count", "rows_count", "series_count")
        ),
        "full_counts": full_counts,
        "visible_counts": {key: value for key, value in visible_counts.items() if value is not None},
        "full_artifact_ref": f"evidence:{evidence.evidence_id}",
    }
    if "query_trace" in diagnostics and isinstance(diagnostics["query_trace"], dict):
        query_trace = dict(diagnostics["query_trace"])
        raw_result_summary = dict(query_trace.get("raw_result_summary") or {})
        raw_result_summary.pop("columns", None)
        query_trace["raw_result_summary"] = raw_result_summary
        diagnostics["query_trace"] = query_trace
    return DatabaseEvidence(
        evidence_id=evidence.evidence_id,
        result_type=evidence.result_type,
        database=evidence.database,
        query_language=evidence.query_language,
        query=evidence.query,
        summary=evidence.summary,
        data=data,
        columns=evidence.columns,
        metadata=dict(evidence.metadata),
        diagnostics=diagnostics,
        produced_facts=list(getattr(evidence, "produced_facts", []) or []),
        rejected_facts=list(getattr(evidence, "rejected_facts", []) or []),
        fact_coverage=getattr(evidence, "fact_coverage", None),
    )


def _sample_edges(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return items
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return [*items[:head], *items[-tail:]]


def _summarize_visualization_dict(payload: dict) -> dict:
    item = dict(payload)
    chart = item.get("chart")
    if isinstance(chart, dict):
        x_axis_data = list(chart.get("x_axis_data") or [])
        series_data = list(chart.get("series_data") or [])
        item["chart"] = {
            "x_axis_count": len(x_axis_data),
            "x_axis_preview": [*x_axis_data[:3], *x_axis_data[-3:]] if len(x_axis_data) > 6 else x_axis_data,
            "series_data": [
                {
                    "name": series.get("name"),
                    "points_count": len(series.get("data") or []),
                }
                for series in series_data[:4]
                if isinstance(series, dict)
            ],
        }
    if isinstance(item.get("annotations"), list):
        item["annotations"] = item["annotations"][:12]
    if isinstance(item.get("rows"), list):
        item["rows"] = _sample_edges(item["rows"], limit=12)
    if isinstance(item.get("display_rows"), list):
        item["display_rows"] = _sample_edges(item["display_rows"], limit=12)
    return item


def _build_prompt_safe_analysis(analysis, request_state: RequestStateModel):
    from schemas.analysis import AnalysisResult

    payload = analysis.model_dump(mode="json")
    result = dict(payload.get("result") or {})
    if isinstance(result.get("details"), list):
        result["details"] = _sample_edges([item for item in result["details"] if isinstance(item, dict)], limit=12)
    if isinstance(result.get("rows"), list):
        result["rows"] = _sample_edges([item for item in result["rows"] if isinstance(item, dict)], limit=12)
    payload["result"] = result
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics["artifact_kind"] = "analysis_result"
    diagnostics["artifact_ref"] = f"analysis:{analysis.analysis_id}"
    diagnostics["snapshot_ref"] = persist_json_artifact(
        artifact_id=analysis.analysis_id,
        artifact_kind="analysis_result",
        payload=analysis.model_dump(mode="json"),
        directory=_artifact_directory(request_state, "analysis"),
    )
    payload["diagnostics"] = diagnostics
    return AnalysisResult.model_validate(payload).model_dump(mode="json")


def _build_prompt_safe_forecast(forecast, request_state: RequestStateModel):
    from schemas.timeseries import ForecastResult

    payload = forecast.model_dump(mode="json")
    full_points = payload.get("forecast_points", [])
    if isinstance(full_points, list):
        diagnostics = dict(payload.get("diagnostics") or {})
        diagnostics["forecast_point_count"] = len(full_points)
        if full_points:
            diagnostics["forecast_first_point"] = full_points[0]
            diagnostics["forecast_last_point"] = full_points[-1]
        payload["diagnostics"] = diagnostics
    payload["forecast_points"] = payload.get("forecast_points", [])[:12]
    payload["confidence_interval"] = payload.get("confidence_interval", [])[:12]
    payload["visualizations"] = [
        _summarize_visualization_dict(item)
        for item in payload.get("visualizations", [])[:3]
    ]
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics["artifact_kind"] = "forecast_result"
    diagnostics["artifact_ref"] = f"forecast:{forecast.forecast_id}"
    diagnostics["snapshot_ref"] = persist_json_artifact(
        artifact_id=forecast.forecast_id,
        artifact_kind="forecast_result",
        payload=forecast.model_dump(mode="json"),
        directory=_artifact_directory(request_state, "analysis"),
    )
    payload["diagnostics"] = diagnostics
    return ForecastResult.model_validate(payload)


def _build_prompt_safe_anomaly(anomaly, request_state: RequestStateModel):
    from schemas.timeseries import AnomalyResult

    payload = anomaly.model_dump(mode="json")
    payload["anomaly_points"] = payload.get("anomaly_points", [])[:12]
    payload["anomaly_spans"] = payload.get("anomaly_spans", [])[:12]
    payload["scores"] = payload.get("scores", [])[:12]
    payload["visualizations"] = [
        _summarize_visualization_dict(item)
        for item in payload.get("visualizations", [])[:3]
    ]
    diagnostics = dict(payload.get("diagnostics") or {})
    diagnostics["artifact_kind"] = "anomaly_result"
    diagnostics["artifact_ref"] = f"anomaly:{anomaly.anomaly_id}"
    diagnostics["snapshot_ref"] = persist_json_artifact(
        artifact_id=anomaly.anomaly_id,
        artifact_kind="anomaly_result",
        payload=anomaly.model_dump(mode="json"),
        directory=_artifact_directory(request_state, "analysis"),
    )
    payload["diagnostics"] = diagnostics
    return AnomalyResult.model_validate(payload)


def _apply_analysis_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    if "analysis_id" in full_payload:
        from schemas.analysis import AnalysisResult

        analysis = AnalysisResult.model_validate(full_payload)
        request_state.analysis_artifacts[analysis.analysis_id] = analysis
        request_state.latest_analysis_id = analysis.analysis_id
        return

    if "forecast_id" in full_payload:
        from schemas.timeseries import ForecastResult

        forecast = ForecastResult.model_validate(full_payload)
        request_state.forecast_artifacts[forecast.forecast_id] = forecast
        request_state.latest_forecast = _build_prompt_safe_forecast(forecast, request_state)
        request_state.visualizations.extend(request_state.latest_forecast.visualizations)
        return

    if "anomaly_id" in full_payload:
        from schemas.timeseries import AnomalyResult

        anomaly = AnomalyResult.model_validate(full_payload)
        request_state.anomaly_artifacts[anomaly.anomaly_id] = anomaly
        request_state.latest_anomaly = _build_prompt_safe_anomaly(anomaly, request_state)
        request_state.visualizations.extend(request_state.latest_anomaly.visualizations)
        return

    if "skill_name" in full_payload:
        request_state.latest_skill = full_payload
        return

    if "results" in full_payload:
        request_state.latest_rag = full_payload


def _apply_presentation_payload(request_state: RequestStateModel, full_payload: dict) -> None:
    request_state.final_answer_draft = FinalAnswer.model_validate(full_payload)


def _has_final_answer_payload(payload: dict) -> bool:
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        return True
    sections = payload.get("sections")
    return isinstance(sections, list) and bool(sections)
