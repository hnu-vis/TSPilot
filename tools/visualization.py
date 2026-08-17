"""LLM-planned, full-fidelity visualization artifact tool."""
from __future__ import annotations

import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.visualization import PresentationCatalog, VisualizationArtifactStore, VisualizationMaterializer
from runtime.token_usage import record_llm_token_usage
from runtime.prompt_locale import prompt_locale_instruction
from schemas.output import VisualGoal
from schemas.key_insight import KeyInsightRequest
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
    required_data_request: "VisualizationEvidenceRequest | None" = None

    @model_validator(mode="after")
    def require_goal_or_data_request(self):
        if not self.visual_goals and not self.required_data_request:
            raise ValueError("visualization planning must produce visual_goals or required_data_request")
        return self


class VisualizationEvidenceRequest(BaseModel):
    """A planner-selected request for the tool that owns a missing visual source."""

    model_config = ConfigDict(extra="forbid")

    required_action: Literal["sql_query", "anomaly", "code_interpreter"]
    purpose: str = Field(min_length=1)
    message: str | None = None
    required_shape: str = Field(min_length=1)
    required_fields: list[str] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    input_evidence: str | None = None
    insight_requests: list[KeyInsightRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action_contract(self):
        if self.required_action == "code_interpreter" and not self.insight_requests:
            raise ValueError("code_interpreter visualization dependency requires insight_requests")
        return self


VisualizationPlan.model_rebuild()


class VisualizationPlanAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    revised_visual_goals: list[VisualGoal] = Field(default_factory=list)
    required_data_request: VisualizationEvidenceRequest | None = None

    @model_validator(mode="after")
    def validate_resolution(self):
        if self.approved:
            if self.revised_visual_goals or self.required_data_request is not None:
                raise ValueError("approved audit cannot include a revision or dependency request")
        elif bool(self.revised_visual_goals) == (self.required_data_request is not None):
            raise ValueError("rejected audit requires exactly one revised plan or dependency request")
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
            raise _missing_evidence_required(plan.required_data_request, inventory)
        audit = await self._audit_plan(validated_input, plan, inventory, request_state)
        if audit.required_data_request:
            raise _missing_evidence_required(audit.required_data_request, inventory)
        if audit.revised_visual_goals:
            plan = plan.model_copy(update={"visual_goals": audit.revised_visual_goals})
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
                raise _missing_evidence_required(repaired_plan.required_data_request, inventory)
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

    async def _audit_plan(
        self,
        request: VisualizationInput,
        plan: VisualizationPlan,
        inventory: dict,
        request_state: RequestStateModel,
        contract_repair_attempted: bool = False,
    ) -> VisualizationPlanAudit:
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You independently audit a grounded visualization plan against the full user request. "
            "Return exactly one JSON object with approved, revised_visual_goals, and required_data_request. "
            "Approve only if every explicitly requested base series, anomaly, decision point, comparison, and annotation has a grounded layer. "
            "A point or rule layer is grounded only when its selected source inventory exposes a timestamp and numeric value through item_refs, locator_fields, or schema_fields; a summary object with nested timestamps is not a renderable locator. "
            "Do not accept a plan that weakens or omits part of the request. If all required sources exist but the plan omitted a layer, return a complete revised_visual_goals list. "
            "Treat visually expressible analytical conclusions as claims that require visual verification, even when the user did not explicitly ask for a chart. "
            "A valid verification plan contains both the conclusion layer and enough contextual data across the complete user analysis interval, at the granularity needed to inspect that conclusion. "
            "If a source is missing, return required_data_request using the owning required_action: sql_query for raw rows/full contextual series, anomaly for anomaly artifacts, "
            "or code_interpreter for calculated/filtered/optimization results. For code_interpreter include exact non-empty insight_requests. "
            "When anomaly or code_interpreter must operate on a particular source, set input_evidence to its evidence id. "
            "materialization_complete only means the executed query result was stored without truncation; never use it alone to infer coverage of the analysis interval. "
            "Use source time_range, query_context, row_count, shape, and lineage to decide visual-verification completeness.\n"
            f"User request: {request.message}\n"
            f"Audit constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Proposed plan: {json.dumps(plan.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Presentation inventory: {json.dumps(inventory, ensure_ascii=False)}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response = await self._llm.ainvoke(messages)
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        content = str(content)
        record_llm_token_usage(
            request_state,
            source="visualization.audit",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=duration_ms,
        )
        try:
            return VisualizationPlanAudit.model_validate(json.loads(_json_object(content)))
        except (json.JSONDecodeError, ValueError) as exc:
            if not contract_repair_attempted:
                repair_request = request.model_copy(update={
                    "constraints": {
                        **request.constraints,
                        "audit_contract_repair": (
                            f"The previous audit violated its schema: {exc}. "
                            "Return a corrected audit object. input_evidence must be one exact database evidence id or null, never a list."
                        ),
                    },
                })
                return await self._audit_plan(
                    repair_request, plan, inventory, request_state, contract_repair_attempted=True,
                )
            raise _semantic_error(ValueError(f"invalid visualization plan audit: {exc}"), inventory) from exc

    async def _plan(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        repair_context: dict | None = None,
        contract_repair_attempted: bool = False,
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
            "Use only exact inventory source refs, fields, and marks. Every visually expressible analytical conclusion must be verified by a grounded semantic layer plus contextual data. "
            "The context must cover the complete user analysis interval at the granularity necessary to inspect the conclusion, not an unrelated aggregate, preview, head/tail sample, or bounded subset. "
            "A max/min/anomaly/decision highlight must be an additional semantic layer with a timestamp and value; a scalar alone is insufficient. "
            "Never select an insight as a point/rule source when its inventory has no item_refs and no timestamp/value locator_fields. Request code_interpreter or the owning tool to publish locators instead. "
            "Every encoding channel must be either an exact field-name string or an object with an exact field key and optional data_type; "
            "do not emit literal color/value constants. Use point or rule for a timestamped highlight; use text only for scalar cards. "
            "If the inventory cannot provide every required base-series or highlight role, return no visual_goals and describe the missing "
            "missing evidence in required_data_request using required_action, purpose, required_shape, required_fields, required_properties, input_evidence when applicable, and insight_requests. "
            "Choose required_action=sql_query only when raw database rows or a complete base series are missing. Choose anomaly when an authoritative anomaly set is missing. "
            "Choose code_interpreter when contextual series evidence exists but a calculated or filtered semantic layer is missing; include exact non-empty insight_requests with insight_key, name, and insight_type. "
            "materialization_complete describes storage of the executed query result, not semantic time-range completeness. A source can be materialized completely while its query uses LIMIT or returns only an aggregate. "
            "The inventory preview is bounded, while row_count and time_range describe the full persisted artifact. "
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
            if not contract_repair_attempted:
                return await self._plan(
                    request,
                    inventory,
                    request_state,
                    repair_context={
                        **(repair_context or {}),
                        "validation_error": f"invalid visualization plan: {exc}",
                        "rejected_response": content,
                        "schema_instruction": (
                            "Return a corrected plan object. input_evidence must be one exact database evidence id or null, never a list."
                        ),
                    },
                    contract_repair_attempted=True,
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


def _missing_evidence_required(requirement: VisualizationEvidenceRequest, inventory: dict) -> StructuredToolError:
    action = requirement.required_action
    modes = {
        "sql_query": "full_timeseries_required",
        "anomaly": "anomaly_evidence_required",
        "code_interpreter": "derived_evidence_required",
    }
    error_types = {
        "sql_query": "visualization_data_incomplete",
        "anomaly": "visualization_anomaly_missing",
        "code_interpreter": "visualization_analysis_missing",
    }
    messages = {
        "sql_query": "Visualization requires additional full-fidelity database evidence.",
        "anomaly": "Visualization requires an authoritative anomaly artifact.",
        "code_interpreter": "Visualization requires calculated semantic evidence.",
    }
    message = messages[action]
    contract = requirement.model_dump(mode="json", exclude_none=True)
    contract["mode"] = modes[action]
    if action == "sql_query":
        contract["constraints"] = {"evidence_shape": "raw_timeseries", "full_fidelity": True}
    return StructuredToolError(
        message,
        error_type=error_types[action],
        retryable=True,
        recommended_next_action=action,
        diagnostics={"required_data_request": contract},
        validation_failure={
            "scope": "visualization_input_data",
            "capability": "visualization",
            "tool": "visualization",
            "error_code": error_types[action],
            "message": message,
            "repair_contract": contract,
            "retry_policy": {
                "required_action": action,
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
