"""State transition boundary for runtime tool observations."""
from __future__ import annotations

from dataclasses import dataclass

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
