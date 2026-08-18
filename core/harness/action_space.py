"""Registry-driven next action space construction."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.harness.capabilities import CapabilityRegistry, default_capability_registry
from core.harness.observation import ObservationFrame


VALID_ACTIONS = {
    "todowrite",
    "sql_query",
    "code_interpreter",
    "forecast",
    "anomaly",
    "visualization",
    "rag",
    "skill",
    "terminate",
}


@dataclass(frozen=True)
class RequiredAction:
    action: str
    reason: str
    input_guidance: dict = field(default_factory=dict)

    def model_view(self) -> dict:
        payload = {"action": self.action, "reason": self.reason}
        if self.input_guidance:
            payload["input_guidance"] = self.input_guidance
        return payload


@dataclass(frozen=True)
class ActionSpace:
    required_actions: tuple[RequiredAction, ...] = ()
    prohibited_actions: tuple[str, ...] = ()
    missing_outputs: tuple[str, ...] = ()
    reason: str = "No runtime-enforced action constraint is active."

    def model_view(self) -> dict:
        return {
            "required_actions": [item.model_view() for item in self.required_actions],
            "prohibited_actions": list(self.prohibited_actions),
            "missing_outputs": list(self.missing_outputs),
            "reason": self.reason,
        }


class ActionSpaceBuilder:
    def __init__(self, registry: CapabilityRegistry | None = None):
        self._registry = registry or default_capability_registry()

    def build(self, frame: ObservationFrame) -> ActionSpace:
        if frame.requires_initial_todo_plan:
            return ActionSpace(
                required_actions=(
                    RequiredAction(
                        action="todowrite",
                        reason="The user requested a multi-step deliverable and no todo plan exists.",
                        input_guidance={
                            "message": frame.message,
                            "focus": frame.message,
                            "task_contract": {
                                "required": True,
                                "purpose": "Define the user-visible deliverables that must be covered before terminate.",
                                "required_outputs": "One output per independently verifiable user-facing result; set evidence_kind to query, analysis, anomaly, forecast, rag, skill, or conclusion.",
                            },
                        },
                    ),
                ),
                prohibited_actions=tuple(sorted(VALID_ACTIONS - {"todowrite"})),
                missing_outputs=("todo_plan",),
                reason="Create the initial todo plan before querying or answering.",
            )

        capabilities = set(self._registry.normalize_many(frame.requested_capabilities, include_query=False))
        has_evidence = frame.artifacts.has_database_evidence and not frame.latest_database_evidence_empty
        active_task_type = str(frame.active_todo_task_type or "").strip().lower()
        required: list[RequiredAction] = []
        missing: list[str] = []

        repair_action = self._repair_action_for_latest_failure(frame)
        if repair_action is not None:
            required.append(repair_action)
            missing.append("validation_repair")

        if not required and frame.shape_recovery_request:
            recovery = frame.shape_recovery_request
            required.append(
                RequiredAction(
                    action="sql_query",
                    reason="The latest SQL evidence does not satisfy its structured query contract.",
                    input_guidance=recovery,
                )
            )
            missing.append("query_evidence_recovery")

        if not required and frame.pending_source_request:
            request = frame.pending_source_request
            action = str(request.get("required_action") or "").strip()
            if action in VALID_ACTIONS:
                guidance = {
                    "mode": "visualization_source_completion",
                    "source_refs": request.get("input_source_refs", []),
                    "database_evidence": request.get("input_evidence"),
                    "analysis_goal": request.get("purpose"),
                    "constraints": {"visualization_source_request": request},
                }
                if action == "code_interpreter":
                    guidance["insight_requests"] = request.get("insight_requests", [])
                required.append(RequiredAction(
                    action=action,
                    reason="Visualization planning identified a semantic source dependency.",
                    input_guidance={key: value for key, value in guidance.items() if value not in (None, "", [], {})},
                ))
                missing.append("visualization_sources")

        for capability_id, artifact_attr, missing_name in (
            ("anomaly", "has_anomaly", "anomaly"),
            ("forecast", "has_forecast", "forecast"),
        ):
            if required:
                break
            if capability_id not in capabilities:
                continue
            action = self._registry.actions_for_capability(capability_id)[0]
            if active_task_type and active_task_type != "answer" and not self._registry.action_matches_task_type(action, active_task_type):
                continue
            if getattr(frame.artifacts, artifact_attr):
                continue
            required.append(
                RequiredAction(
                    action=action if has_evidence else "sql_query",
                    reason=f"{missing_name.capitalize()} output is required by the structured intent profile.",
                    input_guidance=(
                        {"database_evidence": "latest"}
                        if has_evidence
                        else {"constraints": {"evidence_shape": "raw_timeseries"}}
                    ),
                )
            )
            missing.append(missing_name)
            break

        if not required and frame.downstream_analysis_request and not frame.artifacts.has_analysis:
            required.append(
                RequiredAction(
                    action="code_interpreter",
                    reason="Latest database evidence declares uncovered derived outputs for downstream analysis.",
                    input_guidance={
                        "database_evidence": "latest",
                        "analysis_request": frame.downstream_analysis_request,
                    },
                )
            )
            missing.append("analysis")

        if not required:
            gap_action = self._action_for_completion_gap(frame, has_evidence=has_evidence)
            if gap_action is not None and gap_action.action != "visualization":
                required.append(gap_action)
                missing.extend(frame.completion_missing_outputs)

        if not required and "visualization" in capabilities and frame.artifacts.visualization_count == 0:
            required.append(
                RequiredAction(
                    action="visualization" if has_evidence else "sql_query",
                    reason="A grounded visualization is required before final answer assembly.",
                    input_guidance=(
                        {"message": frame.message}
                        if has_evidence
                        else {"constraints": {"evidence_shape": "raw_timeseries", "full_fidelity": True}}
                    ),
                )
            )
            missing.append("visualization")

        if not required:
            gap_action = self._action_for_completion_gap(frame, has_evidence=has_evidence)
            if gap_action is not None:
                required.append(gap_action)
                missing.extend(frame.completion_missing_outputs)

        prohibited: list[str] = []
        if frame.has_todo_plan:
            prohibited.append("todowrite")
        if frame.has_database_context and "external_knowledge" not in capabilities:
            prohibited.append("rag")
        if "skill" not in capabilities:
            prohibited.append("skill")
        if required:
            if required[0].action != "terminate":
                prohibited.append("terminate")
            else:
                prohibited.extend(VALID_ACTIONS - {"terminate"})
            return ActionSpace(
                required_actions=tuple(required[:1]),
                prohibited_actions=tuple(_dedupe_sorted(prohibited)),
                missing_outputs=tuple(missing),
                reason=required[0].reason,
            )
        return ActionSpace(prohibited_actions=tuple(_dedupe_sorted(prohibited)))

    def _repair_action_for_latest_failure(self, frame: ObservationFrame) -> RequiredAction | None:
        payload = frame.latest_failure_payload if isinstance(frame.latest_failure_payload, dict) else {}
        validation_failure = payload.get("validation_failure") if isinstance(payload.get("validation_failure"), dict) else {}
        repair_contract = validation_failure.get("repair_contract") if isinstance(validation_failure.get("repair_contract"), dict) else None
        retry_policy = validation_failure.get("retry_policy") if isinstance(validation_failure.get("retry_policy"), dict) else {}
        if not repair_contract:
            return None
        repeated = int(payload.get("repeated_failure_count") or 1)
        max_retries = int(retry_policy.get("max_equivalent_retries") or 2)
        action = str(retry_policy.get("required_action") or validation_failure.get("tool") or frame.latest_failed_tool or "").strip()
        if action not in VALID_ACTIONS:
            return None
        if repeated > max_retries and retry_policy.get("terminal_after_exhausted") is True:
            capability = str(validation_failure.get("capability") or action).strip()
            return RequiredAction(
                action="terminate",
                reason=(
                    f"Equivalent {action} repair attempts are exhausted. "
                    "Return the available grounded evidence and mark the blocked output unavailable."
                ),
                input_guidance={
                    "unavailable_outputs": [capability],
                    "unavailable_reason": (
                        f"{action} failed the same validation contract {repeated} times; "
                        "no further equivalent retry is allowed."
                    ),
                },
            )
        return RequiredAction(
            action=action,
            reason=f"Previous {action} output failed validation and must be repaired via structured repair_contract.",
            input_guidance={
                "mode": "repair",
                "repair_contract": repair_contract,
                "validation_failure": validation_failure,
                **(
                    {"requires_code": True}
                    if action == "code_interpreter"
                    and repair_contract.get("mode") in {"generated_code_required", "code_execution_repair", "analysis_artifact_repair"}
                    else {}
                ),
                **(
                    {"constraints": repair_contract.get("constraints")}
                    if isinstance(repair_contract.get("constraints"), dict)
                    else {}
                ),
            },
        )

    def _action_for_completion_gap(self, frame: ObservationFrame, *, has_evidence: bool) -> RequiredAction | None:
        missing = [
            self._registry.normalize_id(item)
            for item in frame.completion_missing_outputs
            if str(item).strip()
        ]
        for capability_id in ("analysis", "anomaly", "forecast", "visualization"):
            if capability_id not in missing:
                continue
            action = self._registry.actions_for_capability(capability_id)[0]
            return RequiredAction(
                action=action if has_evidence else "sql_query",
                reason=frame.completion_reason or f"{capability_id} evidence is required before termination.",
                input_guidance=(
                    {
                        "database_evidence": "latest",
                        **(
                            {
                                "analysis_request": {
                                    "goal": frame.message,
                                    "required_outputs": list(frame.completion_missing_outputs),
                                    "mode": "canonical_timeseries_metrics",
                                }
                            }
                            if capability_id == "analysis"
                            else {}
                        ),
                        **({"message": frame.message} if capability_id == "visualization" else {}),
                    }
                    if has_evidence
                    else {"constraints": {"evidence_shape": "raw_timeseries"}}
                ),
            )
        return None


def build_action_space(frame: ObservationFrame, registry: CapabilityRegistry | None = None) -> ActionSpace:
    return ActionSpaceBuilder(registry).build(frame)


def _dedupe_sorted(values: list[str]) -> list[str]:
    return sorted({str(item).strip() for item in values if str(item).strip()})
