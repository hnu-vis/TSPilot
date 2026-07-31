"""DB-GPT-style action output construction for TSPilot tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.harness.observation_view import model_observation_view, public_observation_view
from schemas.action_output import ActionOutput
from schemas.tool import ToolObservation


@dataclass(frozen=True)
class ActionOutputBuildInput:
    tool_name: str
    success: bool
    summary: str
    full_payload: dict
    result_target: str
    action_input: dict
    iteration: int
    request_id: str
    produces_terminal_payload: bool = False
    error: str | None = None


class ActionOutputBuilder:
    """Build separated model/UI/resource views from a validated tool payload."""

    def build(self, item: ActionOutputBuildInput) -> ActionOutput:
        raw_observation = ToolObservation(
            tool_name=item.tool_name,
            success=item.success,
            summary=item.summary,
            payload=item.full_payload,
            error=item.error,
            payload_truncated=True,
            payload_ref=f"obs:{item.request_id}:{item.iteration}:{item.tool_name}",
        )
        model_view = model_observation_view(raw_observation) or {}
        public_view = public_observation_view(raw_observation) or model_view
        resource_ref = self._resource_ref(model_view, public_view, item)
        observations = self._observation_payload(model_view, resource_ref)
        view = self._public_payload(public_view, resource_ref, item)
        memory_fragment = self._memory_fragment(item, observations, resource_ref)
        return ActionOutput(
            tool_name=item.tool_name,
            success=item.success,
            content=str(model_view.get("summary") or item.summary or ""),
            observations=observations,
            view=view,
            resource_type=self._resource_type(item),
            resource_value=item.full_payload,
            resource_ref=resource_ref,
            memory_fragment=memory_fragment,
            error=item.error,
            have_retry=not item.produces_terminal_payload,
            terminate=item.produces_terminal_payload,
            meta={
                "iteration": item.iteration,
                "result_target": item.result_target,
                "raw_payload_ref": raw_observation.payload_ref,
            },
        )

    def from_observation(
        self,
        observation: ToolObservation,
        *,
        result_target: str = "policy",
        action_input: dict | None = None,
        iteration: int = 0,
        request_id: str = "",
    ) -> ActionOutput:
        payload = observation.payload if isinstance(observation.payload, dict) else {}
        return self.build(
            ActionOutputBuildInput(
                tool_name=observation.tool_name,
                success=observation.success,
                summary=observation.summary,
                full_payload=payload,
                result_target=result_target,
                action_input=action_input or {},
                iteration=iteration,
                request_id=request_id,
                error=observation.error,
            )
        )

    def to_tool_observation(self, action_output: ActionOutput) -> ToolObservation:
        payload = action_output.observations if isinstance(action_output.observations, dict) else {"summary": action_output.observations}
        return ToolObservation(
            tool_name=action_output.tool_name,
            success=action_output.success,
            summary=action_output.content,
            payload=payload,
            error=action_output.error,
            payload_truncated=True,
            payload_ref=str((action_output.meta or {}).get("raw_payload_ref") or "")
            or None,
        )

    def _observation_payload(self, model_view: dict, resource_ref: str | None) -> dict:
        payload = dict(model_view.get("payload") or {})
        result = {
            "tool": model_view.get("tool_name"),
            "success": model_view.get("success"),
            "summary": model_view.get("summary"),
            **payload,
        }
        if resource_ref:
            result["resource_ref"] = resource_ref
        if model_view.get("error"):
            result["error"] = model_view.get("error")
        return _drop_empty(result)

    def _public_payload(self, public_view: dict, resource_ref: str | None, item: ActionOutputBuildInput) -> dict:
        result = dict(public_view)
        if item.tool_name == "code_interpreter":
            code_preview = self._code_input_preview(item.action_input)
            if code_preview:
                payload = dict(result.get("payload") or {})
                payload.update(code_preview)
                result["payload"] = payload
        if resource_ref:
            result["resource_ref"] = resource_ref
        result.pop("payload_ref", None)
        return _drop_empty(result)

    def _code_input_preview(self, action_input: dict) -> dict:
        if not isinstance(action_input, dict):
            return {}
        code = str(action_input.get("analysis_code") or action_input.get("code") or "")
        if not code:
            return {}
        return {
            "code_preview": _truncate_text(code, 8000),
            "analysis_code_chars": len(code),
        }

    def _memory_fragment(self, item: ActionOutputBuildInput, observations: dict, resource_ref: str | None) -> dict:
        action_input = self._action_input_memory(item.tool_name, item.action_input)
        return _drop_empty(
            {
                "iteration": item.iteration,
                "action": item.tool_name,
                "action_input": action_input,
                "observation": _bounded_value(observations, max_dict_items=24, max_list_items=8, max_string_chars=900),
                "resource_ref": resource_ref,
                "status": "succeeded" if item.success else "failed",
            }
        )

    def _action_input_memory(self, tool_name: str, action_input: dict) -> dict:
        if not isinstance(action_input, dict):
            return {}
        if tool_name == "sql_query":
            return {
                key: _bounded_value(value, max_string_chars=500)
                for key, value in action_input.items()
                if key not in {"query", "message|query", "query_language", "database_context"}
                and value not in (None, "", [], {})
            }
        if tool_name == "code_interpreter":
            return {
                key: _bounded_value(value, max_string_chars=500)
                for key, value in action_input.items()
                if key not in {"code", "analysis_code", "database_evidence"}
                and value not in (None, "", [], {})
            }
        return {
            key: _bounded_value(value, max_string_chars=500)
            for key, value in action_input.items()
            if value not in (None, "", [], {})
        }

    def _resource_type(self, item: ActionOutputBuildInput) -> str | None:
        if item.result_target == "evidence":
            return "database_evidence"
        if item.tool_name == "forecast":
            return "forecast"
        if item.tool_name == "anomaly":
            return "anomaly"
        if item.result_target == "analysis":
            return "analysis"
        if item.result_target == "presentation":
            return "final_answer"
        if item.result_target == "todo":
            return "todo_plan"
        return item.result_target or None

    def _resource_ref(self, model_view: dict, public_view: dict, item: ActionOutputBuildInput) -> str | None:
        for source in (model_view, public_view):
            ref = source.get("artifact_ref") or source.get("resource_ref")
            if isinstance(ref, str) and ref.strip():
                return ref.strip()
            payload = source.get("payload") if isinstance(source.get("payload"), dict) else {}
            ref = payload.get("artifact_ref") or payload.get("resource_ref")
            if isinstance(ref, str) and ref.strip():
                return ref.strip()
        payload = item.full_payload
        if item.result_target == "evidence" and payload.get("evidence_id"):
            return f"evidence:{payload['evidence_id']}"
        if item.tool_name == "code_interpreter" and payload.get("analysis_id"):
            return f"analysis:{payload['analysis_id']}"
        if item.tool_name == "forecast" and payload.get("forecast_id"):
            return f"forecast:{payload['forecast_id']}"
        if item.tool_name == "anomaly" and payload.get("anomaly_id"):
            return f"anomaly:{payload['anomaly_id']}"
        if item.result_target == "presentation":
            return f"final_answer:{item.request_id}"
        return None


def _bounded_value(
    value: Any,
    *,
    max_string_chars: int = 700,
    max_list_items: int = 8,
    max_dict_items: int = 12,
) -> Any:
    if isinstance(value, str):
        if len(value) <= max_string_chars:
            return value
        return value[:max_string_chars] + f"... [truncated {len(value) - max_string_chars} chars]"
    if isinstance(value, list):
        result = [
            _bounded_value(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            result.append({"truncated_items": len(value) - max_list_items})
        return result
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_dict_items:
                result["truncated_keys"] = len(value) - max_dict_items
                break
            result[key] = _bounded_value(
                item,
                max_string_chars=max_string_chars,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
        return _drop_empty(result)
    return value


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def _drop_empty(payload: dict) -> dict:
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {})
    }
