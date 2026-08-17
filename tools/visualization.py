"""LLM-planned, full-fidelity visualization artifact tool."""
from __future__ import annotations

import json
import time

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.visualization import PresentationCatalog, VisualizationArtifactStore, VisualizationMaterializer
from runtime.token_usage import record_llm_token_usage
from runtime.prompt_locale import prompt_locale_instruction
from schemas.output import VisualGoal
from schemas.state import RequestStateModel
from schemas.visualization import VisualizationPayload
from tools.base import BaseTool, StructuredToolError


class VisualizationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    source_refs: list[str] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)


class VisualizationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_goals: list[VisualGoal] = Field(default_factory=list)
    required_data_request: dict | None = None

    @model_validator(mode="after")
    def require_goal_or_data_request(self):
        if not self.visual_goals and not self.required_data_request:
            raise ValueError("visualization planning must produce visual_goals or required_data_request")
        return self


class VisualizationResult(BaseModel):
    summary: str
    visualization_ids: list[str] = Field(default_factory=list)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class VisualizationTool(BaseTool):
    """Plan semantic layers with an LLM and persist complete renderer data."""

    def __init__(self, *, llm, artifact_store: VisualizationArtifactStore):
        self._llm = llm
        self._artifact_store = artifact_store

    async def execute(
        self,
        validated_input: VisualizationInput,
        *,
        request_state: RequestStateModel,
        **kwargs,
    ) -> dict:
        catalog = PresentationCatalog(request_state)
        inventory = catalog.planner_inventory()
        plan = await self._plan(validated_input, inventory, request_state)
        if plan.required_data_request:
            raise _full_data_required(plan.required_data_request, inventory)
        try:
            complete = VisualizationMaterializer(request_state).materialize_all(plan.visual_goals)
        except ValueError as exc:
            repaired_plan = await self._plan(
                validated_input,
                inventory,
                request_state,
                repair_context={
                    "validation_error": str(exc),
                    "rejected_plan": plan.model_dump(mode="json"),
                },
            )
            if repaired_plan.required_data_request:
                raise _full_data_required(repaired_plan.required_data_request, inventory)
            try:
                complete = VisualizationMaterializer(request_state).materialize_all(repaired_plan.visual_goals)
            except ValueError as repair_exc:
                raise _semantic_error(repair_exc, inventory) from repair_exc
        descriptors = [self._artifact_store.put(item) for item in complete]
        source_refs = list(dict.fromkeys(ref for item in descriptors for ref in item.source_refs))
        return VisualizationResult(
            summary=f"Created {len(descriptors)} grounded visualization artifact(s).",
            visualization_ids=[item.visualization_id for item in descriptors],
            visualizations=descriptors,
            source_refs=source_refs,
        ).model_dump(mode="json")

    async def _plan(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        repair_context: dict | None = None,
    ) -> VisualizationPlan:
        if self._llm is None:
            raise RuntimeError("visualization planning requires an LLM")
        source_filter, unknown = _expand_source_preferences(request.source_refs, inventory)
        if request.source_refs:
            if unknown:
                raise _semantic_error(ValueError(f"unknown requested source refs: {sorted(unknown)}"), inventory)
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You plan grounded time-series visualizations. Return exactly one JSON object matching this schema: "
            "{\"visual_goals\":[{\"purpose\":str,\"title\":str,\"priority\":\"primary\"|\"supporting\","
            "\"summary\":str|null,\"required_roles\":[str],\"layers\":[{\"role\":str,\"source_ref\":str,"
            "\"mark\":str,\"encoding\":object,\"label\":str|null}]}],\"required_data_request\":object|null}. "
            "Use only exact inventory source refs, fields, and marks. A visualization intended to reveal a time-series pattern "
            "must contain the complete requested time range, not an aggregate, preview, head/tail sample, or bounded subset. "
            "A max/min/anomaly/decision highlight must be an additional semantic layer with a timestamp and value; a scalar alone is insufficient. "
            "Every encoding channel must be either an exact field-name string or an object with an exact field key and optional data_type; "
            "do not emit literal color/value constants. Use point or rule for a timestamped highlight; use text only for scalar cards. "
            "If the inventory cannot provide every required base-series or highlight role, return no visual_goals and describe the missing "
            "full-fidelity database evidence in required_data_request using natural language, required_shape, required_fields, and required_properties. "
            "Do not invent a fallback chart and do not weaken the requested purpose.\n"
            f"Visualization request: {request.message}\n"
            f"Preferred source refs: {json.dumps(sorted(source_filter), ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}\n"
            f"Presentation inventory: {json.dumps(inventory, ensure_ascii=False)}"
        )
        started_at = time.perf_counter()
        response = await self._llm.ainvoke([("system", prompt), ("user", request.message)])
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        content = str(content)
        record_llm_token_usage(
            request_state,
            source="visualization.plan",
            response=response,
            messages=[("system", prompt), ("user", request.message)],
            output_text=content,
            duration_ms=duration_ms,
        )
        try:
            return VisualizationPlan.model_validate(json.loads(_json_object(content)))
        except (json.JSONDecodeError, ValueError) as exc:
            if repair_context is None:
                return await self._plan(
                    request,
                    inventory,
                    request_state,
                    repair_context={
                        "validation_error": f"invalid visualization plan: {exc}",
                        "rejected_response": content,
                    },
                )
            raise _semantic_error(ValueError(f"invalid visualization plan: {exc}"), inventory) from exc


def _expand_source_preferences(refs: list[str], inventory: dict) -> tuple[set[str], set[str]]:
    """Resolve stable outer artifact refs to tool-internal presentation views."""
    available = {
        str(item.get("source_ref"))
        for item in inventory.get("sources", [])
        if isinstance(item, dict) and item.get("source_ref")
    }
    expanded: set[str] = set()
    unknown: set[str] = set()
    for raw_ref in refs:
        ref = str(raw_ref or "").strip()
        if not ref:
            continue
        matches: set[str] = set()
        recognized = ref in available
        if ref.startswith("evidence:"):
            evidence_id = ref.split(":", 1)[1]
            matches.update(
                candidate
                for candidate in available
                if candidate.startswith(f"view:evidence:{evidence_id}:")
            )
        elif ref.startswith("analysis:"):
            analysis_id = ref.split(":", 1)[1]
            matches.update(
                candidate
                for candidate in available
                if candidate.startswith(f"view:analysis:{analysis_id}:")
            )
        if not matches and recognized and not ref.startswith("analysis:"):
            matches.add(ref)
        if matches:
            expanded.update(matches)
        elif not recognized:
            unknown.add(ref)
    return expanded, unknown


def _json_object(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response did not contain a JSON object")
    return text[start:end + 1]


def _full_data_required(requirement: dict, inventory: dict) -> StructuredToolError:
    message = "Visualization requires additional full-fidelity database evidence."
    contract = {
        "mode": "full_timeseries_required",
        "message": requirement.get("message") or requirement.get("purpose") or message,
        "purpose": requirement.get("purpose") or "Provide complete data for the requested visualization.",
        "required_shape": requirement.get("required_shape") or "full_timeseries",
        "required_fields": requirement.get("required_fields") or ["timestamp", "value"],
        "required_properties": requirement.get("required_properties") or ["complete requested time range"],
        "constraints": {"evidence_shape": "raw_timeseries", "full_fidelity": True},
    }
    return StructuredToolError(
        message,
        error_type="visualization_data_incomplete",
        retryable=True,
        recommended_next_action="sql_query",
        diagnostics={"required_data_request": contract},
        validation_failure={
            "scope": "visualization_input_data",
            "capability": "visualization",
            "tool": "visualization",
            "error_code": "visualization_data_incomplete",
            "message": message,
            "repair_contract": contract,
            "retry_policy": {
                "required_action": "sql_query",
                "max_equivalent_retries": 2,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        },
    )


def _semantic_error(exc: ValueError, _inventory: dict) -> StructuredToolError:
    message = f"Visualization semantic validation failed: {exc}"
    contract = {
        "mode": "visualization_semantic_repair",
        "instruction": (
            "Retry visualization; the tool will rebuild its internal source inventory and re-plan "
            "using the validation error below."
        ),
        "validation_error": str(exc),
    }
    return StructuredToolError(
        message,
        error_type="visualization_semantic_validation",
        retryable=True,
        recommended_next_action="visualization",
        diagnostics={"validation_error": str(exc)},
        validation_failure={
            "scope": "visualization_plan",
            "capability": "visualization",
            "tool": "visualization",
            "error_code": "visualization_semantic_validation",
            "message": message,
            "repair_contract": contract,
            "retry_policy": {
                "required_action": "visualization",
                "max_equivalent_retries": 1,
                "allow_same_action": True,
                "terminal_after_exhausted": True,
            },
        },
    )
