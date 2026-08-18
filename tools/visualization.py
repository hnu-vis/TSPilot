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
        if bool(self.visual_goals) == bool(self.required_data_request):
            raise ValueError("chart planning must produce either visual_goals or required_data_request, never both")
        return self


class SemanticFieldPlan(BaseModel):
    """One LLM-authored semantic column backed by an existing source value."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    semantic_role: str = Field(min_length=1)
    source_path: str = Field(min_length=1)


class SemanticViewPlan(BaseModel):
    """A grounded semantic view prepared for independent chart planning."""

    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    grain: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    record_path: str | None = None
    fields: list[SemanticFieldPlan] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_semantic_columns(self):
        names = [item.name for item in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("semantic view field names must be unique")
        return self


class SemanticProjectionPlan(BaseModel):
    """First-stage LLM decision: semantic views or an upstream evidence request."""

    model_config = ConfigDict(extra="forbid")

    semantic_views: list[SemanticViewPlan] = Field(default_factory=list)
    required_data_request: "VisualizationEvidenceRequest | None" = None

    @model_validator(mode="after")
    def require_views_or_data_request(self):
        if bool(self.semantic_views) == bool(self.required_data_request):
            raise ValueError("semantic projection must produce either semantic_views or required_data_request, never both")
        return self


class VisualizationEvidenceRequest(BaseModel):
    """A planner-selected request for the tool that owns a missing visual source."""

    model_config = ConfigDict(extra="forbid")

    required_action: Literal["sql_query", "anomaly", "forecast", "code_interpreter"]
    purpose: str = Field(min_length=1)
    message: str | None = None
    required_shape: str = Field(min_length=1)
    required_fields: list[str] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    input_evidence: str | None = None
    input_source_refs: list[str] = Field(default_factory=list)
    insight_requests: list[KeyInsightRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action_contract(self):
        if self.input_evidence and not self.input_source_refs:
            self.input_source_refs.append(self.input_evidence)
        if self.required_action == "code_interpreter" and not self.insight_requests:
            raise ValueError("code_interpreter visualization dependency requires insight_requests")
        if self.required_action != "code_interpreter":
            self.insight_requests = []
        return self


VisualizationPlan.model_rebuild()
SemanticProjectionPlan.model_rebuild()


class VisualizationResult(BaseModel):
    status: Literal["created", "needs_sources"] = "created"
    summary: str
    visualization_ids: list[str] = Field(default_factory=list)
    visualizations: list[VisualizationPayload] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    required_data_request: VisualizationEvidenceRequest | None = None


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
        requested_refs = _resolve_visualization_lineage_refs(validated_input.source_refs, request_state)
        source_preferences, unknown = catalog.expand_preferences(requested_refs)
        inventory = catalog.planner_inventory(source_preferences)
        if unknown:
            raise _semantic_error(ValueError(f"unknown requested source refs: {sorted(unknown)}"), inventory)
        projection = None
        projection_error = ""
        semantic_refs: list[str] = []
        for projection_attempt in range(3):
            projection = await self._project(
                validated_input,
                inventory,
                request_state,
                source_preferences=source_preferences,
                repair_context=(
                    None
                    if projection_attempt == 0
                    else {
                        "execution_error": projection_error,
                        "rejected_projection": projection.model_dump(mode="json") if projection else None,
                        "allowed_default_source_paths": _default_source_paths(inventory),
                    }
                ),
            )
            if projection.required_data_request:
                requirement = projection.required_data_request
                try:
                    requirement = _normalize_requirement_input(requirement, catalog)
                except ValueError as exc:
                    projection_error = str(exc)
                    if projection_attempt >= 2:
                        raise _semantic_error(exc, inventory, scope="semantic_projection") from exc
                    continue
                return _needs_sources_result(requirement)
            try:
                semantic_refs = catalog.materialize_semantic_views(projection.semantic_views)
                break
            except ValueError as exc:
                projection_error = str(exc)
                if projection_attempt >= 2:
                    raise _semantic_error(exc, inventory, scope="semantic_projection") from exc
        if not semantic_refs:
            raise _semantic_error(
                ValueError("visualization semantic projection produced no executable views"),
                inventory,
                scope="semantic_projection",
            )

        semantic_inventory = catalog.semantic_inventory(semantic_refs)
        plan = await self._plan(
            validated_input, semantic_inventory, request_state,
        )
        complete = None
        for materialization_attempt in range(3):
            if plan.required_data_request:
                requirement = plan.required_data_request
                try:
                    requirement = _normalize_requirement_input(requirement, catalog)
                except ValueError as exc:
                    if materialization_attempt >= 2:
                        raise _semantic_error(exc, inventory) from exc
                    plan = await self._plan(
                        validated_input,
                        semantic_inventory,
                        request_state,
                        repair_context={
                            "requirement_grounding_error": str(exc),
                            "rejected_plan": plan.model_dump(mode="json"),
                        },
                    )
                    continue
                return _needs_sources_result(requirement)
            try:
                complete = VisualizationMaterializer(
                    request_state,
                    catalog=catalog,
                    visual_constraints=validated_input.constraints,
                ).materialize_all(plan.visual_goals)
                break
            except ValueError as exc:
                if materialization_attempt >= 2:
                    raise _semantic_error(exc, inventory) from exc
                plan = await self._plan(
                    validated_input,
                    semantic_inventory,
                    request_state,
                    repair_context={
                        "materialization_error": str(exc),
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

    async def _project(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        source_preferences: set[str],
        repair_context: dict | None = None,
        contract_repair_attempt: int = 0,
    ) -> SemanticProjectionPlan:
        if self._llm is None:
            raise RuntimeError("visualization planning requires an LLM")
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You are the semantic projection stage for grounded visualization. Understand the user's analytical goal, "
            "the meaning of each Evidence or verified Insight, its nested data structure, examples, statement, labels, and lineage. "
            "Return exactly one JSON object matching: "
            "{\"semantic_views\":[{\"view_id\":str,\"name\":str,\"purpose\":str,\"grain\":str,"
            "\"source_ref\":str,\"record_path\":str|null,\"fields\":[{\"name\":str,\"semantic_role\":str,\"source_path\":str}]}],"
            "\"required_data_request\":{\"required_action\":\"sql_query\"|\"anomaly\"|\"forecast\"|\"code_interpreter\","
            "\"purpose\":str,\"message\":str|null,\"required_shape\":str,\"required_fields\":[str],"
            "\"required_properties\":[str],\"input_evidence\":str|null,\"input_source_refs\":[str],\"insight_requests\":[{"
            "\"name\":str,\"insight_type\":str,\"insight_key\":str|null}]}|null}. "
            "Create semantic views that make all visually relevant existing values explicit: temporal context, measures, "
            "central estimates, interval bounds, categories, series identities, event labels, and located decisions as appropriate. "
            "This is semantic interpretation, not blind field extraction: choose fields from their meaning in the request and Insight, "
            "and give the projected columns clear semantic names and roles. For sources with multiple grains, use record_path on "
            "projection_root to select the object or array that defines one output row (for example $.items, $.value, or $.records); "
            "then use source_path relative to each selected record. Do not repeat the record_path prefix inside source_path. "
            "For example, record_path $.records with source_path $.metric selects each record's metric. "
            "Leave record_path null only when using the source's default records. "
            "Paths use $.field.nested syntax. A view uses one exact grounded source_ref; later chart planning can compose multiple views. "
            "With record_path null, use the top-level names shown in schema_fields (for example $.timestamp or $.value). "
            "Never add a $.value prefix unless projection_root explicitly shows value as an object containing that field. "
            "Keep every semantic view at one record grain. Every source_path must exist inside the structure selected by record_path; "
            "do not reach into a sibling summary from an item row. If another preferred source exposes a required value more directly, "
            "create a separate semantic view from that source and let chart planning compose the views. "
            "Prefer the owning artifact's complete series or interval view over a downstream Insight that merely summarizes or samples it. "
            "Use verified Insights for calculated conclusions and located annotations, not as a substitute for a complete upstream series. "
            "For a forecast or prediction visualization, historical actuals are the default visual baseline. When the inventory contains "
            "the forecast's historical evidence ancestor, project both the historical actual series and the forecast series so chart "
            "planning can join them at the forecast boundary; source preferences are hints, not permission to omit that context. If the "
            "forecast exists but its required historical actual series is genuinely unavailable, request sql_query for that context instead "
            "of silently producing a forecast-only chart. "
            "A prediction line is sufficient when the user and visual contract do not request uncertainty. Do not request or invent "
            "confidence intervals merely because the source is a forecast; require interval data only when uncertainty, bounds, or a "
            "confidence band is explicitly part of the requested visual meaning. "
            "You may select, rename, and reorganize existing values. Never define formulas, aggregate, rescale, predict, infer, "
            "or manufacture values. Never replace a requested decision/forecast/anomaly role with a merely similar field. "
            "If the grounded inventory truly lacks a required business value, return semantic_views=[] and required_data_request. "
            "Return exactly one branch: either non-empty semantic_views with required_data_request null, or an empty semantic_views "
            "list with one complete required_data_request. code_interpreter requests require at least one insight_request containing "
            "When requesting data, semantic_views must be the literal empty list: do not include partial views, placeholder views, "
            "null source_refs, or empty fields alongside required_data_request. "
            "name and insight_type. input_evidence must be one exact semantic source_ref from the inventory or null, never prose. "
            "Choose sql_query for missing raw context, anomaly for authoritative anomaly detection or when suspicious source values "
            "must be assessed before a specialized model is rerun, forecast for missing or invalidated prediction outputs, and "
            "code_interpreter only for calculations over valid existing artifacts with exact non-empty insight_requests. "
            "Never ask code_interpreter to generate, clean, repair, or replace forecast/anomaly outputs. If forecast output appears "
            "contaminated and no matching anomaly artifact exists, request anomaly on the forecast's evidence ancestor; when that "
            "anomaly artifact exists but the forecast quality lineage does not consume it, request forecast. A forecast rerun on the "
            "same evidence is valid only after a matching anomaly artifact or materially different evidence exists; otherwise the "
            "same specialized model will repeat the invalid output, so request anomaly first. Only code_interpreter dependency requests "
            "may contain insight_requests; use an empty list for sql_query, anomaly, and forecast. Do not produce a fallback view.\n"
            "When Repair context contains an execution error, change the rejected source_ref, record_path, or source_path as needed "
            "after re-reading projection_root; never repeat a path that the executor reported unavailable.\n"
            f"Visualization request: {request.message}\n"
            f"Authoritative visual contract: {json.dumps(_visual_contract(request_state), ensure_ascii=False)}\n"
            f"Preferred source refs: {json.dumps(sorted(source_preferences), ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}\n"
            f"Allowed paths when record_path is null: {json.dumps(_default_source_paths(inventory), ensure_ascii=False)}\n"
            f"Grounded source inventory: {json.dumps(inventory, ensure_ascii=False)}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm, SemanticProjectionPlan, messages,
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        record_llm_token_usage(
            request_state,
            source="visualization.semantic_projection",
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
            if contract_repair_attempt < 2:
                return await self._project(
                    request,
                    inventory,
                    request_state,
                    source_preferences=source_preferences,
                    repair_context={
                        **(repair_context or {}),
                        "contract_error": f"invalid semantic projection plan: {exc}",
                        "rejected_response": content,
                        "schema_instruction": (
                            "Return a corrected plan object. input_evidence must be one exact source_ref from the inventory or null."
                        ),
                    },
                    contract_repair_attempt=contract_repair_attempt + 1,
                )
            raise _semantic_error(
                ValueError(f"invalid semantic projection plan: {exc}"), inventory, scope="semantic_projection",
            ) from exc

    async def _plan(
        self,
        request: VisualizationInput,
        inventory: dict,
        request_state: RequestStateModel,
        repair_context: dict | None = None,
        contract_repair_attempt: int = 0,
    ) -> VisualizationPlan:
        if self._llm is None:
            raise RuntimeError("visualization planning requires an LLM")
        prompt = prompt_locale_instruction(request_state.response_language) + (
            "You are the chart-planning stage for grounded visualization. The semantic projection stage has already interpreted "
            "the raw artifacts and organized their existing values into semantic views. Return exactly one JSON object matching: "
            "{\"visual_goals\":[{\"purpose\":str,\"title\":str,\"priority\":\"primary\"|\"supporting\","
            "\"summary\":str|null,\"required_roles\":[str],\"presentation\":object,\"layers\":[{\"role\":str,"
            "\"source_ref\":str,\"mark\":str,\"encoding\":object,\"transform\":[object],"
            "\"presentation\":object,\"label\":str|null}]}],\"required_data_request\":{"
            "\"required_action\":\"sql_query\"|\"anomaly\"|\"forecast\"|\"code_interpreter\",\"purpose\":str,"
            "\"message\":str|null,\"required_shape\":str,\"required_fields\":[str],"
            "\"required_properties\":[str],\"input_evidence\":str|null,\"input_source_refs\":[str],\"insight_requests\":[{"
            "\"name\":str,\"insight_type\":str,\"insight_key\":str|null}]}|null}. "
            "Design the visual expression from the user's goal and the semantic meaning of the views. Compose as many views and "
            "layers as needed for context, conclusions, intervals, events, and comparisons. Use exact semantic source refs and column "
            "names from the inventory. mark is any ECharts-native series type. required_roles are your own concise description of "
            "what the completed chart expresses; they are not matched by a separate business validator. "
            "Every forecast chart must include a historical-actual layer whenever the supplied semantic views contain its historical "
            "ancestor, plus a distinct forecast layer connected at the prediction boundary. Do not replace the historical baseline with "
            "a scalar direction/change Insight; Insights may annotate the two series. If a forecast view is present but its historical "
            "actual view is missing, return required_data_request for sql_query rather than a forecast-only visual goal. "
            "A layer's role and label must be entailed by the field_semantics of its encoded columns. Never present a central estimate "
            "as a lower bound, upper bound, interval, anomaly, or decision; styling, duplicate layers, and renamed roles do not create "
            "missing semantics. If any user-required visual meaning is absent from the semantic views, return visual_goals=[] and one "
            "complete required_data_request instead. Return exactly one branch, and include name plus insight_type in every "
            "code_interpreter insight_request. Use anomaly for missing anomaly results, forecast for missing or invalidated prediction "
            "outputs, and code_interpreter only for derived calculations over valid existing artifacts; code must never replace a "
            "specialized forecast or anomaly owner. input_evidence must be one exact semantic source_ref "
            "from the inventory or null, never prose. "
            "Chart-level presentation owns axes, coordinate systems, visualMap, dataZoom, brush, toolbox, legend, and tooltip. "
            "Layer presentation owns series styling and interaction. Presentation must not contain data/source/dataset/dimensions/series/encode. "
            "Filters may select existing semantic-view rows but may not calculate or modify values. Filter operator must be exactly one "
            "of eq, neq, in, not_in, exists, not_exists, gt, gte, lt, lte, or between; do not use SQL symbols such as =. "
            "Text cards and tables are unsupported. "
            "Do not weaken the requested purpose and do not invent a fallback chart.\n"
            f"Visualization request: {request.message}\n"
            f"Authoritative visual contract: {json.dumps(_visual_contract(request_state), ensure_ascii=False)}\n"
            f"Constraints: {json.dumps(request.constraints, ensure_ascii=False)}\n"
            f"Repair context: {json.dumps(repair_context, ensure_ascii=False) if repair_context else 'none'}\n"
            f"Semantic view inventory: {json.dumps(inventory, ensure_ascii=False)}"
        )
        messages = [("system", prompt), ("user", request.message)]
        started_at = time.perf_counter()
        response, content, parsed, parse_error = await _invoke_structured(
            self._llm, VisualizationPlan, messages,
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        record_llm_token_usage(
            request_state,
            source="visualization.chart_plan",
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
            if contract_repair_attempt < 2:
                return await self._plan(
                    request,
                    inventory,
                    request_state,
                    repair_context={
                        **(repair_context or {}),
                        "contract_error": f"invalid chart plan: {exc}",
                        "rejected_response": content,
                    },
                    contract_repair_attempt=contract_repair_attempt + 1,
                )
            raise _semantic_error(ValueError(f"invalid chart plan: {exc}"), inventory) from exc


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


def _resolve_visualization_lineage_refs(
    refs: list[str], request_state: RequestStateModel,
) -> list[str]:
    """Dereference an existing presentation artifact to its grounded data lineage."""
    visualizations = {
        str(getattr(visualization, "visualization_id", "") or ""): visualization
        for visualization in request_state.visualizations
    }
    resolved: list[str] = []
    for raw_ref in refs:
        ref = str(raw_ref or "").strip()
        if not ref:
            continue
        visualization_id = ref.removeprefix("visualization:")
        visualization = visualizations.get(visualization_id)
        if visualization is None:
            resolved.append(ref)
            continue
        resolved.extend(
            str(source_ref).strip()
            for source_ref in getattr(visualization, "source_refs", []) or []
            if str(source_ref).strip()
        )
    return list(dict.fromkeys(resolved))


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


def _visual_contract(request_state: RequestStateModel) -> list[dict]:
    """Expose authoritative user-visible visual deliverables without storage details."""
    contract = request_state.task_contract
    if contract is None:
        return []
    visual_outputs = []
    for output in contract.required_outputs:
        if not output.required:
            continue
        if output.output_type != "visualization" and output.evidence_kind != "visualization":
            continue
        visual_outputs.append({
            "id": output.id,
            "description": output.description,
            "success_criteria": output.success_criteria,
            "measures": output.measures,
            "dimensions": output.dimensions,
            "time_scope": output.time_scope,
        })
    return visual_outputs


def _default_source_paths(inventory: dict) -> dict[str, list[str]]:
    """Expose exact executable top-level paths as repair feedback to the LLM."""

    result: dict[str, list[str]] = {}
    for source in inventory.get("sources", []) if isinstance(inventory.get("sources"), list) else []:
        if not isinstance(source, dict):
            continue
        ref = str(source.get("source_ref") or "").strip()
        fields = source.get("schema_fields") if isinstance(source.get("schema_fields"), list) else []
        paths = [
            f"$.{field.get('name')}"
            for field in fields
            if isinstance(field, dict) and str(field.get("name") or "").strip()
        ]
        if ref and paths:
            result[ref] = paths
    return result


def _missing_evidence_required(requirement: VisualizationEvidenceRequest, inventory: dict) -> StructuredToolError:
    action = requirement.required_action
    modes = {
        "sql_query": "full_timeseries_required",
        "anomaly": "anomaly_evidence_required",
        "forecast": "forecast_evidence_required",
        "code_interpreter": "derived_evidence_required",
    }
    error_types = {
        "sql_query": "visualization_data_incomplete",
        "anomaly": "visualization_anomaly_missing",
        "forecast": "visualization_forecast_missing",
        "code_interpreter": "visualization_analysis_missing",
    }
    messages = {
        "sql_query": "Visualization requires additional full-fidelity database evidence.",
        "anomaly": "Visualization requires an authoritative anomaly artifact.",
        "forecast": "Visualization requires an authoritative forecast artifact.",
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


def _normalize_requirement_input(
    requirement: VisualizationEvidenceRequest,
    catalog: PresentationCatalog,
) -> VisualizationEvidenceRequest:
    refs = list(requirement.input_source_refs)
    if requirement.input_evidence and requirement.input_evidence not in refs:
        refs.insert(0, requirement.input_evidence)
    if not refs:
        return requirement
    source_refs = catalog.analysis_input_source_refs(refs)
    return requirement.model_copy(update={
        "input_source_refs": source_refs,
        "input_evidence": next(
            (ref.split(":", 1)[1] for ref in source_refs if ref.startswith("evidence:")),
            None,
        ),
    })


def _needs_sources_result(requirement: VisualizationEvidenceRequest) -> dict:
    return VisualizationResult(
        status="needs_sources",
        summary="Visualization planning identified additional semantic sources required before materialization.",
        required_data_request=requirement,
    ).model_dump(mode="json")


def _semantic_error(
    exc: ValueError,
    _inventory: dict,
    *,
    scope: str = "chart_plan",
) -> StructuredToolError:
    message = f"Visualization planning failed: {exc}"
    contract = {
        "mode": "visualization_llm_repair",
        "instruction": (
            "Retry visualization and let the responsible LLM stage re-plan from the execution feedback below."
        ),
        "execution_error": str(exc),
        "failed_stage": scope,
    }
    return StructuredToolError(
        message,
        error_type="visualization_planning_failed",
        retryable=True,
        recommended_next_action="visualization",
        diagnostics={"execution_error": str(exc), "failed_stage": scope},
        validation_failure={
            "scope": scope,
            "capability": "visualization",
            "tool": "visualization",
            "error_code": "visualization_planning_failed",
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
