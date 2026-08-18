"""Tool execution runtime."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import re
import time
from typing import Any

from core.harness import ActionOutputBuildInput, ActionOutputBuilder, default_capability_registry
from schemas.action_output import ActionOutput
from schemas.state import ConversationStateModel, RequestStateModel
from schemas.tool import ToolCall, ToolObservation
from runtime.action_policy import runtime_action_constraints
from runtime.output_selection import select_outputs_for_action
from runtime.timeout_policy import TimeoutPolicy, load_timeout_policy
from tools.registry import ToolRegistry, ToolSpec
from tools.base import StructuredToolError


@dataclass
class ExecutionResult:
    """Full execution result."""

    tool_spec: ToolSpec
    observation: ToolObservation
    full_payload: dict
    action_output: ActionOutput


@dataclass
class PreparedAction:
    """Validated runtime action and the exact compact input that will execute."""

    tool_spec: ToolSpec
    validated_input: Any
    action_input: dict


class ToolExecutor:
    """Resolve, validate, and invoke one tool."""

    def __init__(
        self,
        registry: ToolRegistry,
        memory_retriever: Any | None = None,
        timeout_policy: TimeoutPolicy | None = None,
    ):
        self._registry = registry
        self._memory_retriever = memory_retriever
        self._timeout_policy = timeout_policy or load_timeout_policy()
        self._capability_registry = default_capability_registry()
        self._action_output_builder = ActionOutputBuilder()

    async def close(self) -> None:
        await self._registry.close()

    async def execute(
        self,
        action_name: str,
        action_input: dict,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> ExecutionResult:
        budget = self.execution_timeout_seconds(action_name)
        started = time.monotonic()
        prepared = await self.prepare(
            action_name,
            action_input,
            request_state,
            timeout_seconds=budget,
        )
        return await self.execute_prepared(
            prepared,
            request_state,
            conversation_state,
            timeout_seconds=self._remaining_timeout(budget, started),
        )

    def execution_timeout_seconds(self, action_name: str) -> float:
        configured = self._registry.resolve(action_name).execution_timeout_seconds
        if configured is not None and float(configured) > 0:
            return float(configured)
        return float(self._timeout_policy.tool(action_name).execution_seconds)

    async def prepare(
        self,
        action_name: str,
        action_input: dict,
        request_state: RequestStateModel,
        *,
        timeout_seconds: float | None = None,
    ) -> PreparedAction:
        tool_spec = self._registry.resolve(action_name)
        budget = float(timeout_seconds) if timeout_seconds is not None else self.execution_timeout_seconds(action_name)
        task = asyncio.create_task(
            self._prepare_resolved(tool_spec, action_name, action_input, request_state)
        )
        try:
            return await asyncio.wait_for(task, timeout=budget)
        except TimeoutError as exc:
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()
            raise self._timeout_error(action_name, budget, phase="preparation") from exc

    async def _prepare_resolved(
        self,
        tool_spec: ToolSpec,
        action_name: str,
        action_input: dict,
        request_state: RequestStateModel,
    ) -> PreparedAction:
        normalized_input = self._normalize_action_input(action_name, action_input, request_state)
        normalized_input = await self._apply_insight_memory(action_name, normalized_input, request_state)
        if action_name == "code_interpreter":
            normalized_input["insight_requests"] = self._remove_evidence_refs_from_insight_dependencies(
                normalized_input.get("insight_requests"),
                request_state,
            )
        validated = tool_spec.input_model.model_validate(normalized_input)
        compact_input = _compact_input(validated.model_dump(mode="json", exclude_none=True))
        return PreparedAction(
            tool_spec=tool_spec,
            validated_input=validated,
            action_input=compact_input,
        )

    async def execute_prepared(
        self,
        prepared: PreparedAction,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
        *,
        timeout_seconds: float | None = None,
    ) -> ExecutionResult:
        tool_spec = prepared.tool_spec
        budget = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else self.execution_timeout_seconds(tool_spec.tool_name)
        )
        task = asyncio.create_task(
            self._execute_prepared_unbounded(prepared, request_state, conversation_state)
        )
        try:
            return await asyncio.wait_for(task, timeout=budget)
        except TimeoutError as exc:
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()
            raise self._timeout_error(tool_spec.tool_name, budget, phase="execution") from exc

    async def _execute_prepared_unbounded(
        self,
        prepared: PreparedAction,
        request_state: RequestStateModel,
        conversation_state: ConversationStateModel,
    ) -> ExecutionResult:
        tool_spec = prepared.tool_spec
        validated = prepared.validated_input
        request_state.tool_history.append(
            ToolCall(
                tool_name=tool_spec.tool_name,
                tool_input=prepared.action_input,
                iteration=request_state.iteration,
            )
        )

        request_state_view = request_state.model_copy(deep=True)
        conversation_state_view = conversation_state.model_copy(deep=True)
        if tool_spec.runtime_access == "none":
            raw_result = await tool_spec.tool.execute(validated)
        elif tool_spec.runtime_access == "request_state_read":
            raw_result = await tool_spec.tool.execute(validated, request_state=request_state_view)
        else:
            raw_result = await tool_spec.tool.execute(
                validated,
                request_state=request_state_view,
                conversation_state=conversation_state_view,
            )

        full_payload = tool_spec.output_model.model_validate(raw_result).model_dump(mode="json")
        summary = tool_spec.tool.summarize(full_payload)
        action_output = self._action_output_builder.build(
            ActionOutputBuildInput(
                tool_name=tool_spec.tool_name,
                success=True,
                summary=summary,
                full_payload=full_payload,
                result_target=tool_spec.result_target,
                action_input=prepared.action_input,
                iteration=request_state.iteration,
                request_id=request_state.request_id,
                produces_terminal_payload=tool_spec.produces_terminal_payload,
            )
        )
        observation = self._action_output_builder.to_tool_observation(action_output)
        return ExecutionResult(
            tool_spec=tool_spec,
            observation=observation,
            full_payload=full_payload,
            action_output=action_output,
        )

    @staticmethod
    def _remaining_timeout(total_seconds: float, started: float) -> float:
        return max(0.001, float(total_seconds) - (time.monotonic() - started))

    @staticmethod
    def _timeout_error(tool_name: str, timeout_seconds: float, *, phase: str) -> StructuredToolError:
        return StructuredToolError(
            f"Tool '{tool_name}' exceeded its {float(timeout_seconds):g}s {phase} timeout.",
            error_type="tool_execution_timeout",
            retryable=True,
            recommended_next_action=tool_name,
            diagnostics={
                "tool": tool_name,
                "phase": phase,
                "timeout_seconds": float(timeout_seconds),
            },
        )

    def react_action_input(self, prepared: PreparedAction) -> dict:
        """Return the executed semantic contract without runtime-owned context."""

        return self.react_proposed_action_input(
            prepared.tool_spec.tool_name,
            prepared.action_input,
        )

    def react_proposed_action_input(self, action_name: str, action_input: dict) -> dict:
        """Return a compact semantic view for an attempted outer action."""

        fields = self._capability_registry.semantic_input_fields_for_action(action_name)
        return _react_action_input(action_input, semantic_fields=fields)

    def _normalize_action_input(self, action_name: str, action_input: dict, request_state: RequestStateModel) -> dict:
        normalized = dict(action_input or {})
        if action_name == "terminate":
            if normalized.get("unavailable_outputs") in (None, False, ""):
                normalized["unavailable_outputs"] = []
            if isinstance(normalized.get("unavailable_reason"), (dict, list)):
                normalized["unavailable_reason"] = str(normalized["unavailable_reason"])
            return normalized
        if action_name == "visualization":
            normalized.setdefault("message", request_state.message)
            constraints = dict(request_state.constraints or {})
            if isinstance(normalized.get("constraints"), dict):
                constraints.update(normalized["constraints"])
            next_constraints = runtime_action_constraints(request_state)
            for item in next_constraints.get("required_actions", []) or []:
                if not isinstance(item, dict) or item.get("action") != "visualization":
                    continue
                guidance = item.get("input_guidance") if isinstance(item.get("input_guidance"), dict) else {}
                if isinstance(guidance.get("repair_contract"), dict):
                    constraints = {**constraints, "repair_contract": guidance["repair_contract"]}
                if isinstance(guidance.get("constraints"), dict):
                    constraints = {**constraints, **guidance["constraints"]}
            normalized["constraints"] = constraints
            return {
                key: value
                for key, value in normalized.items()
                if key in {"message", "source_refs", "constraints"}
            }
        if normalized.get("constraints") in (None, "", False):
            normalized["constraints"] = {}
        if action_name in {"anomaly", "forecast", "code_interpreter"}:
            evidence_input = normalized.get("database_evidence")
            if isinstance(evidence_input, list) and len(evidence_input) == 1:
                normalized["database_evidence"] = evidence_input[0]
        if "time_range" in normalized:
            normalized["time_range"] = self._normalize_time_range_hint(normalized.get("time_range"), normalized)
        normalized["insight_requests"] = self._normalize_insight_requests(normalized.get("insight_requests"), normalized)
        normalized = self._apply_runtime_input_guidance(action_name, normalized, request_state)
        if action_name == "anomaly":
            self._drop_unselected_optional_choice(normalized, "detector_name")
        if action_name == "forecast":
            self._drop_unselected_optional_choice(normalized, "model_name")
        if action_name == "code_interpreter":
            normalized.setdefault("analysis_goal", request_state.message)
            if normalized.get("source_refs"):
                # Exact refs already select the grounded inputs. A parallel
                # "latest" selector is redundant and can become ambiguous as
                # request state advances.
                normalized.pop("database_evidence", None)
            else:
                normalized.setdefault("database_evidence", "latest")
            return {
                key: value
                for key, value in normalized.items()
                if key in {"database_evidence", "source_refs", "analysis_goal", "insight_requests", "code", "constraints"}
            }
        if action_name == "todowrite":
            normalized.setdefault("message", request_state.message)
            normalized.setdefault("focus", request_state.focus or request_state.message)
            if not normalized.get("requested_capabilities"):
                normalized["requested_capabilities"] = list(request_state.requested_capabilities or [])
        state_database_context = request_state.database_context
        if state_database_context is not None:
            state_payload = state_database_context.model_dump(mode="json")
            raw_context = normalized.get("database_context")
            if isinstance(raw_context, dict):
                normalized["database_context"] = {
                    **state_payload,
                    **{key: value for key, value in raw_context.items() if value not in (None, "", [], {})},
                }
            elif "database_context" not in normalized or raw_context in (None, "", {}, []):
                normalized["database_context"] = state_payload

        return normalized

    def _apply_runtime_input_guidance(self, action_name: str, normalized: dict, request_state: RequestStateModel) -> dict:
        constraints = runtime_action_constraints(request_state)
        for item in constraints.get("required_actions", []) or []:
            if not isinstance(item, dict) or item.get("action") != action_name:
                continue
            guidance = item.get("input_guidance") if isinstance(item.get("input_guidance"), dict) else {}
            for key, value in guidance.items():
                if key in {"mode", "repair_contract", "validation_failure", "constraints", "analysis_request", "requires_code"}:
                    continue
                if value not in (None, "", [], {}) and normalized.get(key) in (None, "", [], {}):
                    normalized[key] = value
            if isinstance(guidance.get("repair_contract"), dict):
                normalized["mode"] = guidance.get("mode") or "repair"
                normalized["repair_contract"] = guidance["repair_contract"]
                merged_constraints = dict(normalized.get("constraints") or {})
                merged_constraints["_repair_contract"] = guidance["repair_contract"]
                if isinstance(guidance.get("validation_failure"), dict):
                    merged_constraints["_validation_failure"] = guidance["validation_failure"]
                normalized["constraints"] = merged_constraints
                if action_name == "sql_query":
                    normalized.pop("query", None)
                if action_name == "code_interpreter":
                    normalized.setdefault("database_evidence", guidance["repair_contract"].get("input_evidence") or "latest")
                    normalized.setdefault("analysis_goal", guidance["repair_contract"].get("analysis_goal") or request_state.message)
                    generated_code_required = (
                        guidance.get("requires_code") is True
                        or guidance["repair_contract"].get("mode") == "generated_code_required"
                    )
                    if generated_code_required and not normalized.get("code"):
                        normalized.pop("analysis_request", None)
                    analysis_request = self._normalize_analysis_request(normalized.get("analysis_request"))
                    analysis_request.setdefault("goal", normalized.get("analysis_goal") or request_state.message)
                    analysis_request.setdefault("mode", guidance["repair_contract"].get("mode") or "analysis_artifact_repair")
                    if isinstance(guidance["repair_contract"].get("required_metrics"), list):
                        analysis_request.setdefault("required_outputs", guidance["repair_contract"]["required_metrics"])
                        if not normalized.get("required_outputs"):
                            normalized["required_outputs"] = guidance["repair_contract"]["required_metrics"]
                    if isinstance(guidance["repair_contract"].get("missing_metrics"), list):
                        analysis_request.setdefault("missing_metrics", guidance["repair_contract"]["missing_metrics"])
                    if isinstance(guidance["repair_contract"].get("required_details_fields"), list):
                        analysis_request.setdefault("required_details_fields", guidance["repair_contract"]["required_details_fields"])
                    if analysis_request and (not generated_code_required or not normalized.get("code")):
                        normalized["analysis_request"] = analysis_request
                    elif normalized.get("code"):
                        normalized.pop("analysis_request", None)
                if action_name in {"anomaly", "forecast"} and guidance["repair_contract"].get("input_evidence"):
                    normalized.setdefault("database_evidence", guidance["repair_contract"]["input_evidence"])
            guidance_constraints = guidance.get("constraints") if isinstance(guidance.get("constraints"), dict) else {}
            if guidance_constraints:
                merged_constraints = dict(normalized.get("constraints") or {})
                merged_constraints.update(guidance_constraints)
                normalized["constraints"] = merged_constraints
            if action_name in {"forecast", "anomaly", "code_interpreter"} and not normalized.get("database_evidence"):
                normalized["database_evidence"] = guidance.get("database_evidence") or "latest"
            if action_name == "code_interpreter" and isinstance(guidance.get("analysis_request"), dict):
                normalized.setdefault("analysis_request", guidance["analysis_request"])
                required_outputs = guidance["analysis_request"].get("required_outputs")
                if isinstance(required_outputs, list) and not normalized.get("required_outputs"):
                    normalized["required_outputs"] = required_outputs
            break
        if action_name == "sql_query":
            if not isinstance(normalized.get("intent_profile"), dict) or not normalized.get("intent_profile"):
                normalized["intent_profile"] = (
                    dict(request_state.intent_profile)
                    if isinstance(request_state.intent_profile, dict)
                    else {}
                )
            if not normalized.get("selected_database") and request_state.selected_database:
                normalized["selected_database"] = request_state.selected_database
            if not normalized.get("selected_database_type") and request_state.selected_database_type:
                normalized["selected_database_type"] = request_state.selected_database_type
            merged_constraints = dict(normalized.get("constraints") or {})
            if self._latest_sql_shape_requested_raw(request_state):
                merged_constraints["evidence_shape"] = "raw_timeseries"
                merged_constraints["dialect_complexity_policy"] = "simple_raw_evidence"
            if merged_constraints:
                normalized["constraints"] = merged_constraints
        return normalized

    def _normalize_analysis_request(self, value) -> dict:
        if value in (None, "", [], {}, False):
            return {}
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, list):
            return {"required_outputs": [str(item).strip() for item in value if str(item).strip()]}
        return {"goal": str(value).strip()}

    def _drop_unselected_optional_choice(self, normalized: dict, key: str) -> None:
        value = str(normalized.get(key) or "").strip().lower()
        if value in {"default", "auto", "none", "null"}:
            normalized.pop(key, None)

    def _latest_sql_shape_requested_raw(self, request_state: RequestStateModel) -> bool:
        latest = request_state.observations[-1] if request_state.observations else None
        if latest is None or latest.tool_name != "sql_query" or latest.success:
            return False
        payload = latest.payload if isinstance(latest.payload, dict) else {}
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else payload
        issues = diagnostics.get("query_shape_issues") if isinstance(diagnostics.get("query_shape_issues"), list) else []
        return any(
            isinstance(issue, dict) and issue.get("recommended_shape") == "raw_series"
            for issue in issues
        )

    def _contract_required_outputs(self, request_state: RequestStateModel) -> list[str]:
        contract = request_state.task_contract
        if contract is not None:
            outputs = getattr(contract, "required_outputs", []) or []
            names = []
            for output in outputs:
                value = getattr(output, "id", None) or getattr(output, "description", None) or output
                if str(value or "").strip():
                    names.append(str(value).strip())
            if names:
                return names
        profile = request_state.intent_profile if isinstance(request_state.intent_profile, dict) else {}
        raw_outputs = profile.get("required_outputs") if isinstance(profile.get("required_outputs"), list) else []
        return [str(item).strip() for item in raw_outputs if str(item).strip()]

    def _action_required_outputs(self, request_state: RequestStateModel, action_name: str) -> list[str]:
        selected = select_outputs_for_action(
            request_state,
            action_name,
            fallback_outputs=self._contract_required_outputs(request_state),
        )
        outputs = selected.get("required_outputs") if isinstance(selected.get("required_outputs"), list) else []
        return [str(item).strip() for item in outputs if str(item).strip()]

    def _normalize_time_range_hint(self, value, normalized_input: dict):
        if value in (None, "", [], {}):
            return None
        if isinstance(value, dict):
            normalized = dict(value)
            if "stop" in normalized and "end" not in normalized:
                normalized["end"] = normalized.pop("stop")
            return normalized
        if isinstance(value, str):
            text = value.strip()
            iso_range = re.match(r"^([^/]+)\s*/\s*([^/]+)$", text)
            if iso_range:
                return {"start": iso_range.group(1).strip(), "end": iso_range.group(2).strip()}
            constraints = normalized_input.setdefault("constraints", {})
            if isinstance(constraints, dict):
                constraints.setdefault("time_range_text", text)
            return None
        return None

    def _normalize_insight_requests(self, value, normalized_input: dict) -> list:
        if value in (None, "", False):
            return []
        if not isinstance(value, list):
            constraints = normalized_input.setdefault("constraints", {})
            if isinstance(constraints, dict):
                constraints.setdefault("insight_request_hints", []).append(value)
            return []
        normalized: list = []
        hints: list = []
        for item in value:
            if isinstance(item, dict):
                if item.get("name") and item.get("insight_type"):
                    normalized.append(item)
                else:
                    hints.append(item)
                continue
            if isinstance(item, str) and item.strip():
                hints.append(item.strip())
        if hints:
            constraints = normalized_input.setdefault("constraints", {})
            if isinstance(constraints, dict):
                existing = constraints.setdefault("insight_request_hints", [])
                if isinstance(existing, list):
                    existing.extend(hints)
        return normalized

    async def _apply_insight_memory(
        self,
        action_name: str,
        normalized_input: dict,
        request_state: RequestStateModel,
    ) -> dict:
        if action_name not in {"sql_query", "code_interpreter"}:
            return normalized_input
        if self._memory_retriever is None:
            return normalized_input
        merged = dict(normalized_input)
        explicit = self._normalize_insight_requests(merged.get("insight_requests"), merged)
        merged["insight_requests"] = explicit
        # An explicit Code Interpreter request is already the exact computation
        # contract selected by the outer agent. Retrieval cannot affect that
        # contract, so do not pay for reranking and reconciliation only to
        # discard their output afterwards.
        if action_name == "code_interpreter" and explicit:
            self._record_insight_memory_diagnostics(
                request_state,
                action_name,
                {
                    "source": "explicit_tool_contract",
                    "memory_enabled": False,
                    "selected_card_ids": [],
                    "insight_request_count": 0,
                    "explicit_contract_authoritative": True,
                },
            )
            return merged
        retrieval = await self._memory_retriever.retrieve(
            request_state=request_state,
            tool_name=action_name,
            action_input=merged,
        )
        selected_card_ids = [hit.card_id for hit in retrieval.hits]
        retrieved_requests = [
            self._retrieved_insight_request_payload(item)
            for item in retrieval.insight_requests
        ]
        merged["insight_requests"] = self._dedupe_insight_requests([*explicit, *retrieved_requests])
        diagnostics = dict(retrieval.diagnostics or {})
        diagnostics["source"] = "tool_scoped_memory_retrieval"
        diagnostics["selected_card_ids"] = selected_card_ids
        diagnostics["insight_request_count"] = len(retrieved_requests)
        self._record_insight_memory_diagnostics(request_state, action_name, diagnostics)
        return merged

    def _record_insight_memory_diagnostics(
        self,
        request_state: RequestStateModel,
        action_name: str,
        diagnostics: dict,
    ) -> None:
        memory_context = request_state.completion_state.setdefault("memory_context", {})
        tool_calls = memory_context.setdefault("tool_calls", []) if isinstance(memory_context, dict) else []
        if isinstance(tool_calls, list):
            tool_calls.append({"tool_name": action_name, **diagnostics})

    def _retrieved_insight_request_payload(self, request) -> dict:
        # Retrieval provenance belongs to completion_state.memory_context, not
        # to the semantic tool contract consumed by the model and tool.
        return request.model_dump(mode="json", exclude_none=True) if hasattr(request, "model_dump") else dict(request)

    def _dedupe_insight_requests(self, requests: list) -> list:
        result: list = []
        seen: set[str] = set()
        seen_insight_keys: set[str] = set()
        for item in requests:
            if hasattr(item, "model_dump"):
                payload = item.model_dump(mode="json", exclude_none=True)
            elif isinstance(item, dict):
                payload = dict(item)
            else:
                continue
            if not payload.get("name") or not payload.get("insight_type"):
                continue
            insight_key = str(payload.get("insight_key") or "").strip()
            if insight_key and insight_key in seen_insight_keys:
                continue
            requirements = dict(payload.get("requirements") or {})
            for metadata_key in ("source", "memory_card_ids", "retrieval_reason", "retrieval_confidence"):
                requirements.pop(metadata_key, None)
            key = json.dumps(
                {
                    "insight_key": payload.get("insight_key"),
                    "name": payload.get("name"),
                    "insight_type": payload.get("insight_type"),
                    "subject": payload.get("subject"),
                    "time_range": payload.get("time_range"),
                    "dimensions": payload.get("dimensions") or {},
                    "derived_from": payload.get("derived_from") or [],
                    "requirements": requirements,
                    "result_shape": payload.get("result_shape"),
                    "expected_item_count": payload.get("expected_item_count"),
                    "semantic_class": payload.get("semantic_class"),
                    "derivation": payload.get("derivation"),
                    "selection": payload.get("selection") or {},
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if key in seen:
                continue
            result.append(payload)
            seen.add(key)
            if insight_key:
                seen_insight_keys.add(insight_key)
        return result

    def _remove_evidence_refs_from_insight_dependencies(
        self,
        requests: list | None,
        request_state: RequestStateModel,
    ) -> list:
        evidence_ids = set(request_state.database_evidence_artifacts.keys())
        if request_state.latest_database_evidence is not None:
            evidence_ids.add(request_state.latest_database_evidence.evidence_id)
        evidence_refs = evidence_ids | {f"evidence:{evidence_id}" for evidence_id in evidence_ids}
        # An LLM may first describe the raw query result as a semantic Insight
        # dependency. SQL correctly rejects row collections as Insights, while the
        # same rows remain valid direct evidence for code_interpreter. Treat
        # those explicitly rejected SQL Insight keys as evidence aliases rather
        # than unresolved parent Insights.
        for call in reversed(request_state.tool_history):
            if call.tool_name != "sql_query":
                continue
            constraints = call.tool_input.get("constraints") if isinstance(call.tool_input, dict) else {}
            unsupported = constraints.get("unsupported_insight_requests") if isinstance(constraints, dict) else []
            for item in unsupported if isinstance(unsupported, list) else []:
                if not isinstance(item, dict):
                    continue
                insight_key = str(item.get("insight_key") or "").strip()
                if insight_key:
                    evidence_refs.add(insight_key)
            break
        normalized: list = []
        for request in requests or []:
            payload = request.model_dump(mode="json", exclude_none=True) if hasattr(request, "model_dump") else dict(request)
            dependencies = payload.get("derived_from")
            if isinstance(dependencies, list):
                payload["derived_from"] = [
                    dependency
                    for dependency in dependencies
                    if str(dependency).strip() not in evidence_refs
                ]
            normalized.append(payload)
        return normalized

    def _truncate_payload(self, payload: dict, request_state: RequestStateModel) -> tuple[dict, bool]:
        max_chars = int(request_state.context_budget.get("max_observation_chars", 1600))
        rendered = str(payload)
        if len(rendered) <= max_chars:
            return payload, False

        visible = dict(payload)
        if isinstance(visible.get("rejected_insights"), list):
            visible["rejected_insights"] = visible["rejected_insights"][:6]
        if isinstance(visible.get("forecast_points"), list):
            visible["forecast_points"] = visible["forecast_points"][:12]
        if isinstance(visible.get("anomaly_points"), list):
            visible["anomaly_points"] = visible["anomaly_points"][:12]
        if isinstance(visible.get("scores"), list):
            visible["scores"] = visible["scores"][:12]
        if isinstance(visible.get("derived_evidence"), list):
            visible["derived_evidence"] = [
                {
                    **item,
                    "rows": item.get("rows", [])[:8] if isinstance(item.get("rows"), list) else item.get("rows"),
                }
                for item in visible["derived_evidence"][:6]
                if isinstance(item, dict)
            ]
        if isinstance(visible.get("result"), dict):
            result = dict(visible["result"])
            if isinstance(result.get("details"), list):
                result["details"] = result["details"][:12]
            if isinstance(result.get("rows"), list):
                result["rows"] = result["rows"][:12]
            visible["result"] = result
        if isinstance(visible.get("data"), dict):
            data = dict(visible["data"])
            if isinstance(data.get("points"), list):
                limit = int(request_state.context_budget.get("max_visible_points", 240))
                data["points"] = data["points"][:limit]
            if isinstance(data.get("rows"), list):
                limit = int(request_state.context_budget.get("max_visible_rows", 60))
                data["rows"] = data["rows"][:limit]
            if isinstance(data.get("series"), list):
                data["series"] = [
                    self._summarize_series_for_observation(series)
                    for series in data["series"][:6]
                    if isinstance(series, dict)
                ]
            visible["data"] = data
        if isinstance(visible.get("visualizations"), list):
            summarized_visualizations = []
            for item in visible["visualizations"][:4]:
                if not isinstance(item, dict):
                    continue
                copy_item = dict(item)
                chart = copy_item.get("chart")
                if isinstance(chart, dict):
                    x_axis = list(chart.get("x_axis_data") or [])
                    series = list(chart.get("series_data") or [])
                    copy_item["chart"] = {
                        "x_axis_count": len(x_axis),
                        "series": [
                            {
                                "name": series_item.get("name"),
                                "points_count": len(series_item.get("data") or []),
                            }
                            for series_item in series[:4]
                            if isinstance(series_item, dict)
                        ],
                    }
                if isinstance(copy_item.get("annotations"), list):
                    copy_item["annotations"] = copy_item["annotations"][:12]
                summarized_visualizations.append(copy_item)
            visible["visualizations"] = summarized_visualizations
        if isinstance(visible.get("diagnostics"), dict):
            diagnostics = dict(visible["diagnostics"])
            visible["diagnostics"] = {
                key: value
                for key, value in diagnostics.items()
                if key in {"artifact_kind", "artifact_ref", "snapshot_ref", "threshold", "series_name", "query_trace", "runtime_ms", "sandbox"}
            }
        return visible, True

    def _summarize_series_for_observation(self, series: dict) -> dict:
        item = {
            key: value
            for key, value in series.items()
            if key not in {"points", "rows"}
        }
        points = series.get("points")
        if isinstance(points, list):
            item["points_count"] = len(points)
            item["points"] = _sample_edges(points, limit=12)
        rows = series.get("rows")
        if isinstance(rows, list):
            item["rows_count"] = len(rows)
            item["rows"] = _sample_edges(rows, limit=12)
        return item


def _compact_input(value):
    """Drop empty defaults while preserving false/zero values and semantics."""
    if isinstance(value, dict):
        compact = {
            str(key): _compact_input(item)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
        return {key: item for key, item in compact.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        return [_compact_input(item) for item in value]
    return value


def _react_action_input(value: dict, *, semantic_fields: set[str]) -> dict:
    selected = (
        {key: item for key, item in value.items() if key in semantic_fields}
        if semantic_fields
        else dict(value)
    )
    constraints = selected.get("constraints")
    if isinstance(constraints, dict):
        runtime_constraint_keys = {
            "memory_diagnostics",
            "unsupported_insight_requests",
            "insight_request_hints",
            "dialect_complexity_policy",
        }
        selected["constraints"] = {
            key: item for key, item in constraints.items()
            if not str(key).startswith("_") and key not in runtime_constraint_keys
        }
    return _compact_react_value(selected)


def _compact_react_value(value):
    """Compact action values and address artifacts by reference, not copies."""

    if isinstance(value, dict):
        evidence_id = str(value.get("evidence_id") or "").strip()
        if evidence_id and ("data" in value or "result_type" in value):
            return f"evidence:{evidence_id}"
        compact = {
            str(key): _compact_react_value(item)
            for key, item in value.items()
            if item not in (None, "", [], {})
        }
        return {key: item for key, item in compact.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        return [_compact_react_value(item) for item in value]
    return value


def _sample_edges(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    head = max(1, limit // 2)
    tail = max(1, limit - head)
    return [*items[:head], *items[-tail:]]
