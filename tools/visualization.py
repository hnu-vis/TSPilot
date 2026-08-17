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

    decision: Literal["approve", "revise", "need_data"]
    revised_visual_goals: list[VisualGoal] = Field(default_factory=list)
    required_data_request: VisualizationEvidenceRequest | None = None

    @model_validator(mode="after")
    def validate_resolution(self):
        has_revision = bool(self.revised_visual_goals)
        has_request = self.required_data_request is not None
        expected = {
            "approve": (False, False),
            "revise": (True, False),
            "need_data": (False, True),
        }[self.decision]
        if (has_revision, has_request) != expected:
            raise ValueError(f"audit decision '{self.decision}' has inconsistent payload")
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
        source_preferences, unknown = catalog.expand_preferences(validated_input.source_refs)
        if unknown:
            raise _semantic_error(ValueError(f"unknown requested source refs: {sorted(unknown)}"), inventory)
        plan = await self._plan(
            validated_input, inventory, request_state, source_preferences=source_preferences,
        )
        complete = None
        for materialization_attempt in range(3):
            plan = await self._audit_until_approved(
                validated_input, plan, inventory, request_state,
            )
            try:
                complete = VisualizationMaterializer(request_state).materialize_all(plan.visual_goals)
                break
            except ValueError as exc:
                if materialization_attempt >= 2:
                    raise _semantic_error(exc, inventory) from exc
                plan = await self._plan(
                    validated_input,
                    inventory,
                    request_state,
                    source_preferences=source_preferences,
                    repair_context={
                        "validation_error": str(exc),
                        "rejected_plan": plan.model_dump(mode="json"),
                    },
                )
        if complete is None:
            raise _semantic_error(ValueError("visualization did not materialize a verified plan"), inventory)
        descriptors = [self._artifact_store.put(item) for item in complete]
        source_refs = list(dict.fromkeys(ref for item in descriptors for ref in item.source_refs))
        return VisualizationResult(
            summary=f"Created {len(descriptors)} grounded visualization artifact(s).",
            visualization_ids=[item.visualization_id for item in descriptors],
            visualizations=descriptors,
            source_refs=source_refs,
        ).model_dump(mode="json")

    async def _audit_until_approved(
        self,
        request: VisualizationInput,
        plan: VisualizationPlan,
        inventory: dict,
        request_state: RequestStateModel,
    ) -> VisualizationPlan:
        if plan.required_data_request:
            raise _missing_evidence_required(plan.required_data_request, inventory)
        audit = await self._audit_plan(request, plan, inventory, request_state)
        if audit.decision == "need_data":
            raise _missing_evidence_required(audit.required_data_request, inventory)
        if audit.decision == "approve":
            return plan
        return plan.model_copy(update={"visual_goals": audit.revised_visual_goals})

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
            "Return exactly one JSON object with decision, revised_visual_goals, and required_data_request. "
            "decision=approve carries neither revision nor data request; decision=revise carries a complete revised_visual_goals list; "
            "decision=need_data carries exactly one required_data_request. "
            "revised_visual_goals is always an array, never null. "
            "Audit whether the visualization as a whole lets the user inspect the analytical conclusion; do not require one chart layer per claim or mirror the analysis artifact structure. "
            "A contextual full series plus scalar text/table conclusions can jointly verify interval change, growth rate, or other aggregate analysis. "
            "Require a timestamped point only when the user explicitly needs a located event or when the chosen visual expression itself needs that point. "
            "A point or rule layer is grounded only when its selected source exposes timestamped numeric rows. "
            "Reject any point/rule/time-encoded layer whose source render_capabilities says scalar_only or timestamped_numeric=false. "
            "Encoding is only for fields listed by the selected source. Explanatory formulas and conclusions belong in the goal summary or layer label, not in an invented encoding field. "
            "A scalar text layer may use an empty encoding because the materialized metric and grounded Insight statement carry its content. "
            "If all required sources exist but a clearer grounded expression is needed, choose revise and return the complete replacement goals. "
            "Treat visually expressible analytical conclusions as claims that require visual verification, even when the user did not explicitly ask for a chart. "
            "A valid verification plan contains both the conclusion layer and enough contextual data across the complete user analysis interval, at the granularity needed to inspect that conclusion. "
            "If a source is missing, return required_data_request using the owning required_action: sql_query for raw rows/full contextual series, anomaly for anomaly artifacts, "
            "or code_interpreter for calculated/filtered/optimization results. For code_interpreter include exact non-empty insight_requests. "
            "When anomaly or code_interpreter must operate on a particular source, set input_evidence to its evidence id. "
            "materialization_complete only means the executed query result was stored without truncation; never use it alone to infer coverage of the analysis interval. "
            "Use source time_range, query_context, row_count, shape, and lineage to decide visual-verification completeness. "
            "Do not request new analysis merely to make an existing scalar conclusion chart-shaped when contextual data already makes it visually inspectable.\n"
            f"User request: {request.message}\n"
            f"Audit constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Proposed plan: {json.dumps(plan.model_dump(mode='json'), ensure_ascii=False)}\n"
            f"Presentation inventory: {json.dumps(inventory, ensure_ascii=False)}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm, VisualizationPlanAudit, messages,
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        record_llm_token_usage(
            request_state,
            source="visualization.audit",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=duration_ms,
        )
        try:
            if parse_error is not None:
                raise parse_error
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            if not contract_repair_attempted:
                repair_request = request.model_copy(update={
                    "constraints": {
                        **request.constraints,
                        "audit_contract_repair": (
                            f"The previous audit violated its schema: {exc}. "
                            "Return a corrected audit object with exactly one decision state. "
                            "input_evidence must be one exact database evidence id or null, never a list."
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
        source_preferences: set[str],
        repair_context: dict | None = None,
        contract_repair_attempted: bool = False,
    ) -> VisualizationPlan:
        if self._llm is None:
            raise RuntimeError("visualization planning requires an LLM")
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You plan grounded time-series visualizations. Return exactly one JSON object matching this schema: "
            "{\"visual_goals\":[{\"purpose\":str,\"title\":str,\"priority\":\"primary\"|\"supporting\","
            "\"summary\":str|null,\"required_roles\":[str],\"layers\":[{\"role\":str,\"source_ref\":str,"
            "\"mark\":str,\"encoding\":object,\"label\":str|null}]}],\"required_data_request\":object|null}. "
            "Use only exact inventory source refs, fields, and marks. Every visually expressible analytical conclusion must be verified by a grounded semantic layer plus contextual data. "
            "Choose the visual expression yourself from the user's intent; do not translate every insight into a separate point or annotation layer. "
            "The inventory contains renderable sources only; outer evidence/analysis/artifact refs have already been resolved into these candidates. "
            "The context must cover the complete user analysis interval at the granularity necessary to inspect the conclusion, not an unrelated aggregate, preview, head/tail sample, or bounded subset. "
            "For interval changes, rates, totals, and similar scalar conclusions, combine a text/table layer with the complete contextual series; the scalar does not need its own timestamp. "
            "For an explicitly located event such as a requested max/min/anomaly/decision highlight, use a timestamped source when available; request missing data only if that located event is essential to the requested visual purpose. "
            "Never select an insight as a point/rule source when its inventory is not timestamped_numeric. "
            "A source whose render_capabilities.scalar_only is true may be used only with text or table and must never be assigned a timestamp encoding. "
            "For a scalar text layer prefer an empty encoding and communicate its meaning with the grounded Insight plus label/summary. "
            "Never put a formula, explanation, or other metadata name into encoding unless it is explicitly listed in that source's schema_fields or locator_fields. "
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
            f"Preferred source refs: {json.dumps(sorted(source_preferences), ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}\n"
            f"Presentation inventory: {json.dumps(inventory, ensure_ascii=False)}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm, VisualizationPlan, messages,
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        record_llm_token_usage(
            request_state,
            source="visualization.plan",
            response=response,
            messages=messages,
            output_text=content,
            duration_ms=duration_ms,
        )
        try:
            if parse_error is not None:
                raise parse_error
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            if not contract_repair_attempted:
                return await self._plan(
                    request,
                    inventory,
                    request_state,
                    source_preferences=source_preferences,
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


async def _invoke_structured(llm, schema, messages):
    response = None
    content = ""
    try:
        if hasattr(llm, "with_structured_output"):
            runnable = llm.with_structured_output(schema, method="json_mode", include_raw=True)
            bundle = await runnable.ainvoke(messages)
            if isinstance(bundle, dict):
                response = bundle.get("raw")
                content = _llm_content(response)
                parsed = bundle.get("parsed")
                if parsed is None:
                    error = bundle.get("parsing_error") or ValueError("structured output was not parsed")
                    return response, content, None, ValueError(str(error))
                parsed = parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
                return response, content, parsed, None
            parsed = bundle if isinstance(bundle, schema) else schema.model_validate(bundle)
            return bundle, _llm_content(bundle), parsed, None
        response = await llm.ainvoke(messages)
        content = _llm_content(response)
        parsed = schema.model_validate(json.loads(_json_object(content)))
        return response, content, parsed, None
    except (json.JSONDecodeError, ValueError) as exc:
        return response, content, None, exc


def _llm_content(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()


def _expand_source_preferences(
    refs: list[str], catalog: PresentationCatalog,
) -> tuple[set[str], set[str]]:
    """Compatibility wrapper around the catalog-owned artifact-to-view resolver."""
    return catalog.expand_preferences(refs)


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
