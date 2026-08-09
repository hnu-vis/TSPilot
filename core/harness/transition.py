"""State transition boundary for runtime tool observations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from schemas.state import RequestStateModel
from schemas.tool import ToolObservation


@dataclass
class StateTransitionResult:
    observation: ToolObservation
    full_payload: dict


class StateTransitionEngine:
    """Apply tool output to request state through one runtime boundary."""

    def apply(
        self,
        request_state: RequestStateModel,
        observation: ToolObservation,
        full_payload: dict,
        tool_spec,
    ) -> StateTransitionResult:
        from runtime import request_state as request_state_runtime

        safe_observation = self._apply_sync(
            request_state_runtime,
            request_state,
            observation,
            full_payload,
            tool_spec,
        )
        return StateTransitionResult(observation=safe_observation, full_payload=full_payload)

    async def apply_async(
        self,
        request_state: RequestStateModel,
        observation: ToolObservation,
        full_payload: dict,
        tool_spec,
    ) -> StateTransitionResult:
        from runtime import request_state as request_state_runtime

        safe_observation = self._apply_sync(
            request_state_runtime,
            request_state,
            observation,
            full_payload,
            tool_spec,
        )
        return StateTransitionResult(observation=safe_observation, full_payload=full_payload)

    def _apply_sync(
        self,
        request_state_runtime,
        request_state: RequestStateModel,
        observation: ToolObservation,
        full_payload: dict,
        tool_spec,
    ) -> ToolObservation:
        if not observation.success:
            safe_observation = request_state_runtime._build_prompt_safe_failure_observation(observation)
            safe_observation = _annotate_failure_repetition(request_state, safe_observation, observation)
            request_state.observations.append(safe_observation)
            return safe_observation

        if tool_spec.result_target == "todo":
            request_state_runtime._apply_todo_payload(request_state, full_payload)
        elif tool_spec.result_target == "evidence":
            request_state_runtime._apply_evidence_payload(request_state, full_payload)
        elif tool_spec.result_target == "analysis":
            request_state_runtime._apply_analysis_payload(request_state, full_payload)
        elif tool_spec.result_target == "presentation":
            request_state_runtime._apply_presentation_payload(request_state, full_payload)
            request_state_runtime._complete_answer_todo_after_terminal(
                request_state,
                observation.tool_name,
                full_payload,
            )

        if tool_spec.result_target in {"evidence", "analysis"}:
            request_state_runtime.register_data_facts_from_payload(
                request_state,
                observation.tool_name,
                full_payload,
            )
            request_state_runtime._advance_todo_after_artifact(
                request_state,
                observation.tool_name,
                full_payload,
                tool_spec.result_target,
            )

        safe_observation = request_state_runtime.enrich_observation_payload(
            request_state,
            observation,
            full_payload,
            tool_spec,
        )
        request_state.observations.append(safe_observation)
        return safe_observation


def _annotate_failure_repetition(
    request_state: RequestStateModel,
    safe_observation: ToolObservation,
    raw_observation: ToolObservation,
) -> ToolObservation:
    payload = raw_observation.payload if isinstance(raw_observation.payload, dict) else {}
    validation_failure = payload.get("validation_failure") if isinstance(payload.get("validation_failure"), dict) else {}
    repair_contract = validation_failure.get("repair_contract") if isinstance(validation_failure.get("repair_contract"), dict) else {}
    identity = {
        "tool": raw_observation.tool_name,
        "error_code": validation_failure.get("error_code") or payload.get("error_type") or "tool_failure",
        "scope": validation_failure.get("scope"),
        "capability": validation_failure.get("capability"),
        "repair_mode": repair_contract.get("mode"),
    }
    encoded = json.dumps(identity, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    signature = hashlib.sha256(encoded).hexdigest()[:16]
    repeated = 1
    for previous in reversed(request_state.observations):
        if previous.success:
            break
        previous_payload = previous.payload if isinstance(previous.payload, dict) else {}
        if previous_payload.get("failure_signature") != signature:
            break
        repeated += 1

    safe_payload = dict(safe_observation.payload or {})
    safe_payload["failure_signature"] = signature
    safe_payload["repeated_failure_count"] = repeated
    return safe_observation.model_copy(update={"payload": safe_payload})
